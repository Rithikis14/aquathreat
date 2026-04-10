# utils/augment.py
"""
Underwater-specific augmentation pipeline using Albumentations.

These augmentations simulate the degradation effects of water:
  - Blue/green color cast  (water absorbs red light with depth)
  - Turbidity blur          (suspended particles scatter light)
  - Low contrast            (color washes out underwater)
  - Noise                   (sonar/camera sensor noise)
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_underwater_augmentation(image_size: int = 640, mode: str = "train") -> A.Compose:
    """
    Returns an Albumentations Compose pipeline.

    Args:
        image_size: Target square resolution (default 640 for YOLO)
        mode:       'train' (heavy augmentation) | 'val' (minimal)
    """

    if mode == "train":
        return A.Compose(
            [
                # ── Spatial transforms ──────────────────────────────────────
                A.RandomResizedCrop(height=image_size, width=image_size,
                                    scale=(0.6, 1.0), ratio=(0.75, 1.33)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.3),

                # ── Underwater color simulation ─────────────────────────────
                UnderwaterColorShift(p=0.7),

                # ── Turbidity / visibility simulation ───────────────────────
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),   # particle scatter
                    A.MotionBlur(blur_limit=7, p=1.0),           # AUV motion
                ], p=0.5),

                # ── Contrast & brightness degradation ───────────────────────
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.1),   # bias toward darker (deeper water)
                    contrast_limit=(-0.3, 0.2),
                    p=0.6,
                ),

                # ── Sensor noise ─────────────────────────────────────────────
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),

                # ── Normalize to ImageNet stats (used by YOLO pretrained) ────
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",          # [x_center, y_center, width, height] normalized
                label_fields=["class_labels"],
                min_area=100,           # Drop tiny boxes created by crop
                min_visibility=0.3,
            ),
        )

    else:  # val / test — only resize + normalize
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )


class UnderwaterColorShift(A.ImageOnlyTransform):
    """
    Custom Albumentations transform that simulates underwater color absorption.

    Physics: Red light is absorbed within ~5m, green within ~20m, blue persists
    the longest. We randomly attenuate the red channel and boost blue/green.
    """

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.astype(np.float32)

        # Random attenuation factors per channel (R most affected, B least)
        red_scale   = np.random.uniform(0.5, 0.85)
        green_scale = np.random.uniform(0.8, 1.05)
        blue_scale  = np.random.uniform(0.9, 1.15)

        img[:, :, 0] *= red_scale    # R
        img[:, :, 1] *= green_scale  # G
        img[:, :, 2] *= blue_scale   # B (OpenCV is BGR, albumentations expects RGB)

        # Add a subtle green/blue haze
        haze_intensity = np.random.uniform(0, 30)
        haze_channel   = np.random.choice([1, 2])  # green or blue
        img[:, :, haze_channel] = np.clip(
            img[:, :, haze_channel] + haze_intensity, 0, 255
        )

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


# ── Quick visual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_img  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [[0.5, 0.5, 0.2, 0.2]]   # one YOLO-format box
    dummy_cls  = [0]

    pipeline = get_underwater_augmentation(mode="train")
    result   = pipeline(image=dummy_img, bboxes=dummy_bbox, class_labels=dummy_cls)

    print(f"[Augment] Output tensor shape : {result['image'].shape}")
    print(f"[Augment] Transformed boxes   : {result['bboxes']}")
    print(f"[Augment] Pipeline OK.")
