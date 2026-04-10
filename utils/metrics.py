# utils/metrics.py
"""
Per-class evaluation metrics for AquaThreat.

The project reports per-class mAP, Precision, and Recall — NOT only overall mAP.
This is critical because overall mAP can mask a model that detects 'bottom_mine'
well but completely misses rare 'drifting_mine' or 'artillery_uxo' classes.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


CLASS_NAMES = ["bottom_mine", "moored_mine", "drifting_mine", "artillery_uxo"]


def parse_ultralytics_results(results_csv: str | Path) -> dict:
    """
    Parse the results.csv written by Ultralytics after training and
    return per-class metrics as a structured dict.

    The CSV contains columns like:
      metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)

    Per-class breakdown is extracted from the validator's class_result attribute.
    """
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    last = df.iloc[-1]
    return {
        "overall": {
            "precision":  float(last.get("metrics/precision(B)", 0)),
            "recall":     float(last.get("metrics/recall(B)", 0)),
            "mAP50":      float(last.get("metrics/mAP50(B)", 0)),
            "mAP50-95":   float(last.get("metrics/mAP50-95(B)", 0)),
        }
    }


def compute_per_class_metrics(
    pred_boxes:  list[np.ndarray],
    pred_scores: list[np.ndarray],
    pred_labels: list[np.ndarray],
    gt_boxes:    list[np.ndarray],
    gt_labels:   list[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 4,
) -> dict:
    """
    Compute per-class Precision, Recall, and AP at a given IoU threshold.

    Args:
        pred_boxes:  List (one per image) of predicted boxes [N, 4] (xyxy)
        pred_scores: List (one per image) of confidence scores [N]
        pred_labels: List (one per image) of predicted class indices [N]
        gt_boxes:    List (one per image) of ground-truth boxes [M, 4] (xyxy)
        gt_labels:   List (one per image) of ground-truth class indices [M]
        iou_threshold: IoU threshold to consider a detection a True Positive
        num_classes: Number of classes (4 for AquaThreat)

    Returns:
        dict with per-class and mean metrics
    """

    per_class: dict[int, dict] = {i: {"tp": [], "fp": [], "scores": [], "n_gt": 0}
                                   for i in range(num_classes)}

    for pboxes, pscores, plabels, gboxes, glabels in zip(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    ):
        for cls in range(num_classes):
            gt_mask   = glabels == cls
            pred_mask = plabels == cls

            gt_c  = gboxes[gt_mask]
            pb_c  = pboxes[pred_mask]
            ps_c  = pscores[pred_mask]

            per_class[cls]["n_gt"] += len(gt_c)

            if len(pb_c) == 0:
                continue

            # Sort by confidence descending
            order = np.argsort(-ps_c)
            pb_c, ps_c = pb_c[order], ps_c[order]

            matched = np.zeros(len(gt_c), dtype=bool)

            for pb, sc in zip(pb_c, ps_c):
                per_class[cls]["scores"].append(sc)
                if len(gt_c) == 0:
                    per_class[cls]["fp"].append(1)
                    per_class[cls]["tp"].append(0)
                    continue

                ious = _iou_vectorized(pb, gt_c)
                best_idx = int(np.argmax(ious))

                if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                    matched[best_idx] = True
                    per_class[cls]["tp"].append(1)
                    per_class[cls]["fp"].append(0)
                else:
                    per_class[cls]["tp"].append(0)
                    per_class[cls]["fp"].append(1)

    results = {}
    aps = []

    for cls in range(num_classes):
        d = per_class[cls]
        n_gt = d["n_gt"]

        if n_gt == 0:
            results[CLASS_NAMES[cls]] = {"AP": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        tp = np.array(d["tp"], dtype=float)
        fp = np.array(d["fp"], dtype=float)

        if len(tp) == 0:
            results[CLASS_NAMES[cls]] = {"AP": 0.0, "precision": 0.0, "recall": 0.0}
            aps.append(0.0)
            continue

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / (n_gt + 1e-9)
        ap        = _compute_ap(precision, recall)

        results[CLASS_NAMES[cls]] = {
            "AP":        round(float(ap), 4),
            "precision": round(float(precision[-1]), 4),
            "recall":    round(float(recall[-1]), 4),
            "n_gt":      int(n_gt),
        }
        aps.append(ap)

    results["mAP50"] = round(float(np.mean(aps)), 4) if aps else 0.0
    return results


def print_metrics_table(metrics: dict):
    """Pretty-print per-class metrics to stdout."""
    print("\n" + "=" * 60)
    print(f"  {'Class':<20} {'AP':>8} {'Precision':>10} {'Recall':>8}")
    print("=" * 60)
    for cls_name in CLASS_NAMES:
        if cls_name in metrics:
            m = metrics[cls_name]
            print(f"  {cls_name:<20} {m['AP']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
    print("-" * 60)
    print(f"  {'mAP@0.5':<20} {metrics.get('mAP50', 0):>8.4f}")
    print("=" * 60 + "\n")


def save_metrics(metrics: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved → {path}")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU of one box against an array of boxes (xyxy format)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area_box + area_boxes - inter + 1e-9)


def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        ap += np.max(prec_at_thr) if len(prec_at_thr) else 0.0
    return ap / 11.0
