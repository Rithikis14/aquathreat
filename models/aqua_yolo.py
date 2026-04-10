# models/aqua_yolo.py
"""
Aqua-YOLO: YOLOv8 + LearnableColorCorrection prepended to the backbone.

How it works:
  Raw underwater image
      ↓
  LearnableColorCorrection  ← trained end-to-end with the rest of the model
      ↓
  YOLOv8 backbone + neck + detection heads
      ↓
  [class, confidence, bbox] per detection
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from models.color_correction import LearnableColorCorrection


class AquaYOLO(nn.Module):
    """
    Wraps a standard YOLOv8 model with the learnable preprocessing block.

    Args:
        model_size:  YOLOv8 variant — 'n' (nano), 's' (small), 'm' (medium)
        num_classes: Must match dataset.yaml nc (default 4)
        pretrained:  Load ImageNet-pretrained YOLO weights before fine-tuning
    """

    def __init__(
        self,
        model_size: str = "n",
        num_classes: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Learnable preprocessing block ──────────────────────────────────
        self.color_correction = LearnableColorCorrection()

        # ── YOLOv8 backbone ────────────────────────────────────────────────
        weights = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        self.yolo = YOLO(weights)

        # Update the model's class count to our 4-class taxonomy
        # (ultralytics handles head re-initialization internally on first train call)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        """
        Forward pass: color correction → YOLO detection.
        Note: During training, call model.yolo.train() directly via the
        Ultralytics Trainer. This forward() is used for custom inference.
        """
        x = self.color_correction(x)
        return self.yolo.model(x)

    def get_yolo_model(self) -> YOLO:
        """Return the inner YOLO object for use with Ultralytics Trainer API."""
        return self.yolo

    def inject_correction_into_yolo(self):
        """
        Patch the YOLO model's internal forward pass to run color correction
        first. This integrates the block into Ultralytics' training loop
        without modifying the library.
        """
        correction = self.color_correction
        original_forward = self.yolo.model.forward

        def patched_forward(x, *args, **kwargs):
            x = correction(x)
            return original_forward(x, *args, **kwargs)

        self.yolo.model.forward = patched_forward
        print("[AquaYOLO] Color correction block injected into YOLO forward pass.")

    def save(self, path: str):
        """Save the correction block weights separately (YOLO saves its own)."""
        torch.save(self.color_correction.state_dict(), path)
        print(f"[AquaYOLO] Color correction weights saved → {path}")

    def load_correction(self, path: str):
        """Load previously saved correction block weights."""
        self.color_correction.load_state_dict(torch.load(path))
        print(f"[AquaYOLO] Color correction weights loaded ← {path}")


# ── Quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AquaYOLO(model_size="n", num_classes=4, pretrained=False)
    model.inject_correction_into_yolo()

    dummy = torch.randn(1, 3, 640, 640)
    out = model(dummy)
    print(f"[AquaYOLO] Raw output type: {type(out)}")
    print("[AquaYOLO] Model ready.")
