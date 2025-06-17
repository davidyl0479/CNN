from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Clean organized path dictionary for easy access
DATA_PATHS = {
    "raw": RAW_DATA_DIR,  # Original downloaded data
    "interim": INTERIM_DATA_DIR,  # Intermediate transformed data
    "processed": PROCESSED_DATA_DIR,  # Final canonical datasets
    "external": EXTERNAL_DATA_DIR,  # Third party data sources
    "models": MODELS_DIR,  # Trained models and predictions
    "figures": FIGURES_DIR,  # Generated plots and charts
    "reports": REPORTS_DIR,  # Generated analysis reports
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ============================================================================
# CIFAR-10 PROJECT SPECIFIC CONFIGURATIONS
# ============================================================================

import torch

# CIFAR-10 Dataset Configuration
CIFAR10_CONFIG = {
    "num_classes": 10,  # CIFAR-10 has 10 different categories
    "input_channels": 3,  # RGB images (Red, Green, Blue = 3 channels)
    "input_size": (32, 32),  # CIFAR-10 images are 32x32 pixels
    "class_names": [  # The 10 categories in CIFAR-10
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}

# Model Configuration
MODEL_CONFIG = {
    "learning_rate": 0.001,  # How fast the model learns (too high = unstable, too low = slow)
    "epochs": 50,  # How many times to go through entire dataset
    "dropout_rate": 0.5,  # Randomly ignore 50% of neurons (prevents overfitting)
    "weight_decay": 1e-4,  # Regularization technique (keeps weights small)
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 64,  # Process 64 images at once (faster training)
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    "num_workers": 4,  # How many CPU cores to use for loading data
    "pin_memory": True,  # Speed optimization for GPU
    "shuffle": True,  # Randomize data order each epoch
    "validation_split": 0.1,  # Use 10% of training data for validation
}

# Data Augmentation Settings
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,  # 50% chance to flip image horizontally
    "rotation_degrees": 10,  # Rotate images up to 10 degrees
    "normalize_mean": [0.4914, 0.4822, 0.4465],  # CIFAR-10 specific values
    "normalize_std": [0.2023, 0.1994, 0.2010],  # for normalizing pixel values
}
