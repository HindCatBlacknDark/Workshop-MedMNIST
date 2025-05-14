# Workshop on Biology-driven Trustworthy AI -  Applications in Medecine, Agriculture and Environment

- This repo supports the final 45 min hands‑on: training a CNN on MedMNIST PneumoniaMNIST & visualizing predictions with Grad-CAM. We provide two notebook demos:

- Colab demo (notebooks/colab_demo.ipynb): zero‑setup, runs entirely in Google Colab.

- Local demo (notebooks/local_demo.ipynb): run on participants’ machines.

- This repository contains the code for the hands-on session and slides materials for the workshop titeled "IA de Confiance & Traitement d’Images : Médecine, Agriculture, Environnement".
- This is an open source and free workshop as part of the Meet-up program held online on 15th May 2025 by the Zone01 Dakar AI_community x GalsenAI .

## Structure of the repository
```plaintext
workshop-medmnist/
├── README.md
├── requirements.txt         # for local demo
├── environment.yml         # optional conda env for local
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   └── gradcam.py
└── notebooks/
    ├── colab_demo.ipynb     # Google Colab–compatible demo
    └── local_demo.ipynb     # Local Jupyter demo
```

## Prerequisites

- Python 3.8+ with basic familiarity
- Git and command-line comfort
- Recommended: GPU-enabled machine (CPU is OK for demo)



## Slides
Download: slides/Workshop-Slides.pdf

## Hands-on Session
### Code Components
src/data.py – load & preprocess PneumoniaMNIST
src/models.py – simple CNN
src/train.py – training & checkpointing
src/gradcam.py – Grad‑CAM implementation

### Demo
1. **Colab**
   - Open `notebooks/colab_demo.ipynb` in Google Colab
   - Runtime: Python 3, GPU recommended
   - Run first cell to install dependencies automatically

2. **Local**
   ```bash
   git clone https://github.com/<org>/workshop-medmnist.git
   cd workshop-medmnist
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   jupyter notebook notebooks/local_demo.ipynb
   ```
