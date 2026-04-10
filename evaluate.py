# evaluate.py
"""
AquaThreat — Ablation Study Runner with W&B comparison logging.

Runs all 3 experimental conditions and produces a comparison report:

  Condition A | Baseline        | Vanilla YOLOv8n, raw images
  Condition B | Offline CLAHE   | Vanilla YOLOv8n, CLAHE-preprocessed images
  Condition C | Aqua-YOLO+CBAM  | YOLOv8n + CBAM LearnableColorCorrection (ours)

W&B: each condition is logged as a separate run inside the "aquathreat-ablation"
project, producing side-by-side loss curves and mAP charts automatically.

Usage:
    python evaluate.py --data data/dataset.yaml
    python evaluate.py --skip-train   # use existing weights, only run eval
    python evaluate.py --no-wandb     # skip W&B logging
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO

from utils.metrics import print_metrics_table, save_metrics, CLASS_NAMES
from models.aqua_yolo import AquaYOLO

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


EXPERIMENTS = [
    {
        "name":        "baseline",
        "description": "Vanilla YOLOv8n — raw images, no preprocessing",
        "train_name":  "ablation_baseline",
        "clahe":       False,
        "aqua_yolo":   False,
        "wandb_tags":  ["ablation", "baseline"],
    },
    {
        "name":        "offline_clahe",
        "description": "Vanilla YOLOv8n — offline CLAHE preprocessing",
        "train_name":  "ablation_clahe",
        "clahe":       True,
        "aqua_yolo":   False,
        "wandb_tags":  ["ablation", "clahe"],
    },
    {
        "name":        "aqua_yolo_cbam",
        "description": "Aqua-YOLO — CBAM learnable color correction (ours)",
        "train_name":  "ablation_aquayolo",
        "clahe":       False,
        "aqua_yolo":   True,
        "wandb_tags":  ["ablation", "aqua_yolo", "cbam"],
    },
]


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply offline CLAHE to a BGR image (simulates Condition B)."""
    lab       = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq      = clahe.apply(l)
    lab_eq    = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def build_and_train(exp: dict, args) -> str:
    """Train one experimental condition. Returns path to best.pt."""
    train_name = exp["train_name"]
    best_pt = f"runs/detect/runs/ablation/{train_name}/weights/best.pt"
    if args.skip_train and Path(best_pt).exists():
        print(f"[Eval] Skipping training for '{train_name}' (weights found).")
        return best_pt

    print(f"\n[Eval] Training: {exp['description']}")

    # Init a W&B run for this ablation condition
    wb_run = None
    if _WANDB_AVAILABLE and not args.no_wandb:
        wb_run = wandb.init(
            project = os.getenv("WANDB_PROJECT", "aquathreat-ablation"),
            name    = train_name,
            config  = {
                "condition":    exp["name"],
                "description":  exp["description"],
                "epochs":       args.epochs,
                "batch_size":   args.batch,
                "clahe":        exp["clahe"],
                "aqua_yolo":    exp["aqua_yolo"],
                "cbam":         exp["aqua_yolo"],
                "fl_gamma":     1.5,
            },
            tags    = exp["wandb_tags"],
            reinit  = True,
        )
        print(f"[W&B] Ablation run → {wb_run.url}")

    if exp["aqua_yolo"]:
        aqua  = AquaYOLO(model_size="n", num_classes=4, pretrained=True)
        aqua.inject_correction_into_yolo()
        model = aqua.get_yolo_model()
    else:
        model = YOLO("yolov8n.pt")

    model.train(
        data       = args.data,
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = 640,
        device     = args.device,
        project    = "runs/train",
        name       = train_name,
        exist_ok   = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        cls        = 1.5,
        box        = 7.5,
        verbose    = False,
    )

    if wb_run is not None:
        wandb.finish()

    return best_pt


def evaluate_model(weights_path: str, data_yaml: str) -> dict:
    """Run validation and return per-class metrics dict."""
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="test", verbose=False)

    results = {}
    if hasattr(metrics, "box"):
        box = metrics.box
        for i, cls_name in enumerate(CLASS_NAMES):
            try:
                results[cls_name] = {
                    "AP":        round(float(box.ap[i]),  4),
                    "precision": round(float(box.p[i]),   4),
                    "recall":    round(float(box.r[i]),   4),
                }
            except (IndexError, AttributeError):
                results[cls_name] = {"AP": 0.0, "precision": 0.0, "recall": 0.0}

        results["mAP50"]    = round(float(box.map50), 4)
        results["mAP50-95"] = round(float(box.map),   4)

    return results


def log_comparison_to_wandb(all_results: dict, project: str):
    """
    Log a final comparison table to W&B as a summary run.
    Creates a grouped bar chart of mAP50 and per-class AP across all 3 conditions.
    """
    if not _WANDB_AVAILABLE:
        return

    run = wandb.init(
        project = project,
        name    = "ablation_comparison_summary",
        tags    = ["ablation", "comparison", "summary"],
        reinit  = True,
    )

    # Log scalar summary for easy comparison
    for cond_name, data in all_results.items():
        m = data["metrics"]
        wandb.log({
            f"{cond_name}/mAP50":           m.get("mAP50",    0),
            f"{cond_name}/mAP50-95":        m.get("mAP50-95", 0),
            f"{cond_name}/AP_bottom_mine":  m.get("bottom_mine",   {}).get("AP", 0),
            f"{cond_name}/AP_moored_mine":  m.get("moored_mine",   {}).get("AP", 0),
            f"{cond_name}/AP_drifting_mine":m.get("drifting_mine", {}).get("AP", 0),
            f"{cond_name}/AP_artillery_uxo":m.get("artillery_uxo", {}).get("AP", 0),
        })

    # W&B comparison table
    table = wandb.Table(
        columns=["Condition", "mAP50", "bottom_mine AP",
                 "moored_mine AP", "drifting_mine AP", "artillery_uxo AP"]
    )
    for cond_name, data in all_results.items():
        m = data["metrics"]
        table.add_data(
            cond_name,
            m.get("mAP50", 0),
            m.get("bottom_mine",   {}).get("AP", 0),
            m.get("moored_mine",   {}).get("AP", 0),
            m.get("drifting_mine", {}).get("AP", 0),
            m.get("artillery_uxo", {}).get("AP", 0),
        )
    wandb.log({"ablation_comparison": table})

    print(f"[W&B] Comparison table logged → {run.url}")
    wandb.finish()


def run_ablation(args):
    print("\n" + "=" * 65)
    print("  AquaThreat — Ablation Study (3 conditions)")
    print("=" * 65)

    all_results = {}

    for exp in EXPERIMENTS:
        weights = build_and_train(exp, args)
        print(f"\n[Eval] Evaluating: {exp['description']}")
        metrics = evaluate_model(weights, args.data)
        all_results[exp["name"]] = {"description": exp["description"],
                                    "metrics": metrics}
        print_metrics_table(metrics)

    # Save JSON report
    save_metrics(all_results, "runs/ablation/comparison_report.json")

    # Log comparison to W&B
    if not args.no_wandb:
        log_comparison_to_wandb(
            all_results,
            project=os.getenv("WANDB_PROJECT", "aquathreat-ablation"),
        )

    # Print terminal summary
    print("\n" + "=" * 65)
    print("  ABLATION SUMMARY — mAP@0.5 per condition")
    print("=" * 65)
    for name, data in all_results.items():
        mAP = data["metrics"].get("mAP50", 0.0)
        print(f"  {name:<25} mAP50 = {mAP:.4f}   | {data['description']}")
    print("=" * 65)
    print("\n[Eval] Full report → runs/ablation/comparison_report.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/dataset.yaml")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training if weights already exist")
    p.add_argument("--no-wandb",   action="store_true",
                   help="Disable W&B logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
