# predict.py
"""
AquaThreat — Inference + Threat Assessment

Runs the trained Aqua-YOLO model on an image/video/folder, then passes
each frame's detections through the ThreatAgent for actionable output.

Usage:
    python predict.py --source data/images/test/frame_001.jpg
    python predict.py --source data/images/test/              # whole folder
    python predict.py --source 0                              # webcam / live feed
    python predict.py --source video.mp4 --save-video
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from agents.threat_agent import Detection, ThreatAgent, ThreatLevel


# ── Color map for bounding box visualization ─────────────────────────────────
CLASS_COLORS = {
    0: (0, 165, 255),    # Bottom Mine   → Orange
    1: (0, 255, 0),      # Moored Mine   → Green
    2: (0, 0, 255),      # Drifting Mine → Red (danger!)
    3: (255, 0, 255),    # Artillery UXO → Magenta
}

THREAT_COLORS = {
    ThreatLevel.LOW:      (0, 200, 0),
    ThreatLevel.MEDIUM:   (0, 165, 255),
    ThreatLevel.HIGH:     (0, 0, 255),
    ThreatLevel.CRITICAL: (0, 0, 180),
}


def parse_args():
    p = argparse.ArgumentParser(description="AquaThreat inference")
    p.add_argument("--weights",    default="runs/train/aqua_yolo/weights/best.pt")
    p.add_argument("--source",     required=True, help="Image / folder / video / webcam index")
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--device",     default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--depth",      type=float, default=None,
                   help="Simulated depth in metres (passed to ThreatAgent)")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def draw_overlay(frame: np.ndarray, result, assessment) -> np.ndarray:
    """Draw bounding boxes + threat banner onto the frame."""
    h, w = frame.shape[:2]

    # ── Bounding boxes ────────────────────────────────────────────────────
    if result.boxes is not None:
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{result.names[cls]} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Threat banner ─────────────────────────────────────────────────────
    banner_color = THREAT_COLORS.get(assessment.level, (128, 128, 128))
    cv2.rectangle(frame, (0, 0), (w, 60), banner_color, -1)

    threat_text = f"THREAT: {assessment.level.value}  |  Score: {assessment.score:.2f}"
    cv2.putText(frame, threat_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    evasion_short = assessment.evasion_vector[:80]
    cv2.putText(frame, evasion_short, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def run_predict(args):
    print(f"[Predict] Loading weights: {args.weights}")
    model  = YOLO(args.weights)
    agent  = ThreatAgent()

    results_gen = model.predict(
        source  = args.source,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device,
        stream  = True,         # memory-efficient streaming
        verbose = False,
    )

    video_writer = None
    frame_idx    = 0

    for result in results_gen:
        frame = result.orig_img.copy()

        # ── Convert YOLO detections → Detection objects ───────────────────
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls    = int(box.cls[0])
                conf   = float(box.conf[0])
                xywhn  = box.xywhn[0].tolist()   # normalized [xc, yc, w, h]
                detections.append(Detection(class_id=cls, confidence=conf, bbox=xywhn))

        # ── Threat assessment ─────────────────────────────────────────────
        assessment = agent.evaluate(detections, simulated_depth_m=args.depth)

        # ── Print to console ──────────────────────────────────────────────
        print(f"\n[Frame {frame_idx:04d}]")
        print(f"  Detections : {len(detections)}")
        print(f"  Threat     : {assessment.level.value} (score={assessment.score:.3f})")
        print(f"  Evasion    : {assessment.evasion_vector[:70]}...")

        # ── Visual overlay ────────────────────────────────────────────────
        annotated = draw_overlay(frame, result, assessment)

        if not args.no_display:
            cv2.imshow("AquaThreat", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save_video:
            if video_writer is None:
                h, w = annotated.shape[:2]
                video_writer = cv2.VideoWriter(
                    "output_threat.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25, (w, h),
                )
            video_writer.write(annotated)

        frame_idx += 1

    if video_writer:
        video_writer.release()
        print("[Predict] Saved → output_threat.mp4")

    cv2.destroyAllWindows()
    print(f"\n[Predict] Processed {frame_idx} frame(s). Done.")


if __name__ == "__main__":
    args = parse_args()
    run_predict(args)
