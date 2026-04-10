# train.py
"""
AquaThreat — Main Training Script

Trains the Aqua-YOLO model (YOLOv8 + CBAM LearnableColorCorrection) using the
Ultralytics training API with Focal Loss, CIoU box loss, and W&B tracking.

Usage:
    python train.py                          # default (Aqua-YOLO + W&B)
    python train.py --model-size s           # use YOLOv8-small
    python train.py --epochs 100 --batch 16
    python train.py --baseline              # train vanilla YOLO (ablation A)
    python train.py --no-wandb              # disable W&B for this run
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from ultralytics import YOLO

from models.color_correction import LearnableColorCorrection
from models.aqua_yolo import AquaYOLO


# ── Weights & Biases setup ────────────────────────────────────────────────────
# W&B is optional. If not installed or WANDB_DISABLED=true, training still runs.
# To enable: pip install wandb  then  wandb login
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def init_wandb(args, run_name: str):
    """
    Initialise a W&B run for this training job.

    Logs hyperparameters, model architecture choice, and dataset path.
    W&B then automatically receives all Ultralytics metrics (loss, mAP, etc.)
    via the built-in Ultralytics-W&B integration — no extra code needed.

    Set WANDB_PROJECT env var to customise the project name (default: aquathreat).
    Set WANDB_DISABLED=true to suppress W&B even if installed.
    """
    if not _WANDB_AVAILABLE:
        print("[W&B] wandb not installed — tracking disabled.")
        print("      To enable: pip install wandb && wandb login")
        return None

    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        print("[W&B] WANDB_DISABLED=true — tracking disabled.")
        return None

    run = wandb.init(
        project = os.getenv("WANDB_PROJECT", "aquathreat"),
        name    = run_name,
        config  = {
            "model_size":      args.model_size,
            "mode":            "baseline" if args.baseline else "aqua_yolo",
            "epochs":          args.epochs,
            "batch_size":      args.batch,
            "image_size":      args.imgsz,
            "optimizer":       "AdamW",
            "lr0":             0.001,
            "lrf":             0.01,
            "fl_gamma":        1.5,
            "cls_loss_weight": 1.5,
            "box_loss_weight": 7.5,
            "dfl_loss_weight": 1.5,
            "pretrained":      args.pretrained,
            "dataset":         args.data,
            "cbam_attention":  not args.baseline,
            "focal_loss":      True,
            "ciou_box_loss":   True,
        },
        tags = [
            "aquathreat",
            "yolov8",
            "baseline" if args.baseline else "aqua_yolo+cbam",
            f"yolov8{args.model_size}",
        ],
        notes = (
            "Ablation A: Baseline YOLOv8 — raw images, no preprocessing."
            if args.baseline
            else "Aqua-YOLO: YOLOv8 + CBAM learnable colour correction block."
        ),
    )
    print(f"[W&B] Run initialised → {run.url}")
    return run


# ── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AquaThreat detector")
    p.add_argument("--model-size",  default="n",    choices=["n", "s", "m"],
                   help="YOLOv8 size: n=nano, s=small, m=medium")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch",       type=int, default=16)
    p.add_argument("--imgsz",       type=int, default=640)
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--device",      default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data",        default="data/dataset.yaml")
    p.add_argument("--project",     default="runs/train")
    p.add_argument("--name",        default="aqua_yolo")
    p.add_argument("--pretrained",  action="store_true", default=True)
    p.add_argument("--baseline",    action="store_true",
                   help="Train vanilla YOLOv8 without color correction (ablation A)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no-wandb",    action="store_true",
                   help="Disable W&B tracking for this run")
    return p.parse_args()


# ── Training Functions ────────────────────────────────────────────────────────

def build_model(args) -> YOLO:
    """
    Build and return the YOLO model.
    - Baseline mode: plain YOLOv8 (no preprocessing)
    - Default mode:  YOLOv8 + CBAM LearnableColorCorrection injected
    """
    weights = f"yolov8{args.model_size}.pt" if args.pretrained else \
              f"yolov8{args.model_size}.yaml"

    if args.resume:
        print(f"[Train] Resuming from: {args.resume}")
        return YOLO(args.resume)

    model = YOLO(weights)

    if not args.baseline:
        aqua = AquaYOLO(model_size=args.model_size, num_classes=4,
                        pretrained=args.pretrained)
        aqua.inject_correction_into_yolo()
        model = aqua.get_yolo_model()
        print("[Train] Aqua-YOLO: CBAM LearnableColorCorrection injected ✓")
    else:
        print("[Train] Baseline mode: plain YOLOv8 (no color correction)")

    return model


def train(args):
    print("\n" + "=" * 60)
    print("  AquaThreat — Underwater Mine Detection Training")
    print("=" * 60)
    print(f"  Model     : YOLOv8-{args.model_size}  |  Mode: "
          f"{'Baseline' if args.baseline else 'Aqua-YOLO + CBAM'}")
    print(f"  Epochs    : {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print(f"  Device    : {args.device}")
    print(f"  Dataset   : {args.data}")
    print(f"  W&B       : {'disabled (--no-wandb)' if args.no_wandb else 'enabled'}")
    print("=" * 60 + "\n")

    # Disable W&B if flag set
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    run_name = f"{'baseline' if args.baseline else 'aquayolo'}_yolov8{args.model_size}"
    wb_run   = init_wandb(args, run_name)

    model = build_model(args)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # fl_gamma  > 0  → Focal Loss (forces model to learn rare drifting_mine class)
    # box       7.5  → CIoU box regression loss weight (Ultralytics default for v8)
    # cls       1.5  → higher than default 0.5 to sharpen 4-class separation
    train_kwargs = dict(
        data         = args.data,
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        workers      = args.workers,
        device       = args.device,
        project      = "runs/train",
        name         = "aqua_yolo",
        exist_ok     = True,

        # Optimizer
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,

        # Loss — CIoU is Ultralytics default for box; Focal for cls
        cls          = 1.5,
        box          = 7.5,
        dfl          = 1.5,

        # Augmentation
        hsv_h        = 0.015,
        hsv_s        = 0.5,
        hsv_v        = 0.3,
        flipud       = 0.1,
        fliplr       = 0.5,
        mosaic       = 0.8,
        mixup        = 0.1,

        # Output / checkpointing
        save         = True,
        save_period  = 10,
        plots        = True,
        verbose      = True,
    )

    results = model.train(**train_kwargs)

    # Finish W&B run cleanly
    if wb_run is not None:
        wandb.finish()

    print("\n[Train] Training complete.")
    print(f"[Train] Best weights → {args.project}/{args.name}/weights/best.pt")
    print(f"[Train] Results      → {args.project}/{args.name}/results.csv")
    if wb_run is not None:
        print(f"[Train] W&B report   → {wb_run.url}")

    return results


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
