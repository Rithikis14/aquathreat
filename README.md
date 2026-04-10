# AquaThreat - Underwater Mine Detection

## Folder Structure
```
aquathreat/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml
├── models/
│   ├── aqua_yolo.py          # Custom YOLOv8 + learnable CNN block
│   └── color_correction.py   # Learnable preprocessing module
├── agents/
│   └── threat_agent.py       # Agentic threat evaluation layer
├── utils/
│   ├── augment.py            # Underwater-specific augmentations
│   └── metrics.py            # Per-class evaluation metrics
├── train.py                  # Main training script
├── predict.py                # Inference + threat assessment
├── evaluate.py               # Ablation study runner
└── requirements.txt
```
