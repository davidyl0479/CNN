# Convolutional Neural Networks – CIFAR-10 Edition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Hands-on deep-learning project that uses **Convolutional Neural Networks
(CNNs) to classify images from the CIFAR-10 data-set** (airplane,
automobile, bird, cat, deer, dog, frog, horse, ship, truck).  
The project follows the Cookiecutter-Data-Science structure so it is
reproducible and production-ready.

---

## 🎯 Project Goals
- Learn core CNN concepts through real code.
- Build three architectures (Tiny, Simple, Improved) and compare them.
- Achieve ≥ 85 % accuracy on CIFAR-10 with the Improved model.
- Document a professional ML workflow: data → training → evaluation →
  visualisation.

---

## 🚀 Quick Start

### 1 — Create & activate the environment
```bash
# conda example (CPU wheels)
conda create -n cnn-env python=3.11 -y
conda activate cnn-env
pip install -r requirements.txt
```

### 2 — Launch the notebook
```bash
jupyter notebook notebooks/test1.ipynb
# or
jupyter lab
```

### 3 — Reproduce full experiment (CLI)
```bash
python -m typer convolutional_neural_networks.dataset download-cifar10
python -m typer convolutional_neural_networks.modeling.train train \
       --model improved
```

---

## 📊 Dataset – CIFAR-10

| Property | Value |
| -------- | ----- |
| Images   | 60 000 colour images |
| Size     | 32 × 32 px, RGB |
| Classes  | 10 (6 000 images each) |
| Source   | [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Licence  | MIT |

The dataset is downloaded automatically into `data/raw/` the first time
you run the notebook or the CLI command:

```bash
python -m typer convolutional_neural_networks.dataset download-cifar10
```

---

### Optional – GPU wheels (CUDA ≥ 12.1)

If `nvidia-smi` shows a CUDA runtime 12.1+:

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
            torchaudio==2.2.2+cu121 \
            --index-url https://download.pytorch.org/whl/cu121
```
Replace `cu121` with `cu122`, `cu124`, … to match your driver.

---

## 📁 Project Organisation

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         convolutional_neural_networks and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── convolutional_neural_networks   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes convolutional_neural_networks a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## 🤝 Contributing
This is a learning project, but feedback and suggestions are welcome!

---
*Built with ❤️ for learning CNNs*