#!/usr/bin/env python3
# setup_project.py
"""
Run this ONCE to create the full project folder structure.

Usage:
    python setup_project.py
"""

from pathlib import Path

DIRS = [
    "data/images/train",
    "data/images/val",
    "data/images/test",
    "data/labels/train",
    "data/labels/val",
    "data/labels/test",
    "models",
    "agents",
    "utils",
    "runs/train",
    "runs/ablation",
]

INITS = ["models/__init__.py", "agents/__init__.py", "utils/__init__.py"]

if __name__ == "__main__":
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created  {d}/")

    for f in INITS:
        Path(f).touch()
        print(f"  Touched  {f}")

    print("\n✓ Project structure ready.")
    print("\nNext steps:")
    print("  1.  pip install -r requirements.txt")
    print("  2.  Add images → data/images/{train,val,test}/")
    print("  3.  Add labels → data/labels/{train,val,test}/  (YOLO .txt format)")
    print("  4.  python train.py")
    print("  5.  python predict.py --source data/images/test/your_image.jpg")
    print("  6.  python evaluate.py --skip-train   # ablation study")
