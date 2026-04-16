# 🌊 AquaThreat: Underwater Mine & Artillery Detection

[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)](https://github.com/Rithikis14/aquathreat)
[![Computer Vision](https://img.shields.io/badge/Model-YOLOv8%2Fv11-00FFFF)](https://github.com/Rithikis14/aquathreat)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/Rithikis14/aquathreat)

**AquaThreat** is an end-to-end learnable architecture designed for the autonomous detection and classification of underwater mines and artillery. Using state-of-the-art Computer Vision, this project addresses the challenges of low visibility, light refraction, and complex maritime environments to provide reliable threat detection.

---

## 🚀 Key Features

- **Class-Specific Detection**: Specialized in identifying various types of underwater ordnance (mines, shells, torpedoes).
- **Data Augmentation via GANs**: Utilizes Generative Adversarial Networks (GANs) to improve existing training datasets, compensating for the scarcity of high-quality underwater imagery.
- **Optimized for Real-time**: Built on the YOLO architecture for high-speed inference, suitable for AUV (Autonomous Underwater Vehicle) integration.
- **Environmental Robustness**: Architecture tuned to handle the unique spectral distortions of underwater environments using custom preprocessing pipelines.

---

## 🏗️ Project Structure

```text
AquaThreat/
├── data/                    # Dataset configuration and samples
├── models/                  # Custom model architectures and weights
├── notebooks/               # Research experiments and training logs
├── src/                     # Core Source Code
│   ├── detection/           # YOLO inference and detection logic
│   ├── preprocessing/       # Image enhancement and noise reduction
│   └── training/            # Custom training loops and GAN augmentation
├── requirements.txt         # Project dependencies (PyTorch, Ultralytics, etc.)
└── train.py                 # Main training script
```

---

🛠️ Installation & Setup
------------------------

### Prerequisites

*   Python 3.9+
    
*   NVIDIA GPU with CUDA support (highly recommended)
    
*   pip or conda
    

### 1\. Clone the Repository

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone [https://github.com/Rithikis14/aquathreat.git](https://github.com/Rithikis14/aquathreat.git)  cd aquathreat   `

### 2\. Environment Setup

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create a virtual environment  python -m venv venv  # Activate it (Windows)  .\venv\Scripts\activate  # Install dependencies  pip install -r requirements.txt   `

### 3\. Data Preparation

Place your dataset in the data/ directory following the YOLO format:

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   /datasets/    /images/      train/      val/    /labels/      train/      val/   `

📈 Usage
--------

### Training the Model

To start training on your custom underwater dataset:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train.py --data data/config.yaml --epochs 100 --imgsz 640   `

### Running Inference

To run detection on a video stream or image:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python detect.py --source path/to/underwater_video.mp4 --weights models/best.pt   `

🔬 Methodology Note
-------------------

> **Data Enhancement**: In this project, GANs are used exclusively to **improve the training dataset** by generating varied environmental conditions. They are not used during live inference to ensure the raw sensor data is processed with maximum fidelity and minimum latency.

🤝 Contributing
---------------

1.  Fork the repository.
    
2.  Create your branch (git checkout -b feature/NewDetectionMethod).
    
3.  Commit your changes (git commit -m 'Add new detection layer').
    
4.  Push to the branch (git push origin feature/NewDetectionMethod).
    
5.  Open a Pull Request.
    
---

**Developed by** [**Rithik V Kumar**](https://www.google.com/search?q=https://github.com/Rithikis14) _Aspiring AI & Software Development Engineer_
