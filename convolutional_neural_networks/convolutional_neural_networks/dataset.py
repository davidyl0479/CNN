from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

# Add these new imports
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import typer

from convolutional_neural_networks.config import (
    AUGMENTATION_CONFIG,
    CIFAR10_CONFIG,
    DATA_PATHS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAINING_CONFIG,
)

app = typer.Typer()

# ============================================================================
# CIFAR-10 DATA LOADING CLASS (ADD THIS)
# ============================================================================


class CIFAR10DataLoader:
    """Handle CIFAR-10 data loading and preprocessing."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_PATHS["raw"]
        self.class_names = CIFAR10_CONFIG["class_names"]
        logger.info(f"CIFAR-10 data will be stored in: {self.data_dir}")

    def get_transforms(self, train=True):
        """Get data transforms for training or testing."""
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip_prob"]),
                    transforms.RandomRotation(AUGMENTATION_CONFIG["rotation_degrees"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        AUGMENTATION_CONFIG["normalize_mean"], AUGMENTATION_CONFIG["normalize_std"]
                    ),
                ]
            )
            logger.info("Created training transforms with augmentation")
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        AUGMENTATION_CONFIG["normalize_mean"], AUGMENTATION_CONFIG["normalize_std"]
                    ),
                ]
            )
            logger.info("Created test transforms without augmentation")
        return transform

    def load_data(self):
        """Load CIFAR-10 dataset."""
        logger.info("Loading CIFAR-10 dataset...")

        # Training data with augmentation
        train_transform = self.get_transforms(train=True)
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        logger.success(f"Loaded {len(train_dataset)} training samples")

        # Test data without augmentation
        test_transform = self.get_transforms(train=False)
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        logger.success(f"Loaded {len(test_dataset)} test samples")

        return train_dataset, test_dataset

    def get_data_loaders(self):
        """Get train, validation, and test data loaders."""
        train_dataset, test_dataset = self.load_data()

        # Split training data into train and validation
        train_size = int((1 - TRAINING_CONFIG["validation_split"]) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        logger.info(f"Split data: {train_size} train, {val_size} validation")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=TRAINING_CONFIG["shuffle"],
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=TRAINING_CONFIG["pin_memory"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=TRAINING_CONFIG["pin_memory"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAINING_CONFIG["num_workers"],
            pin_memory=TRAINING_CONFIG["pin_memory"],
        )

        logger.success("Created all data loaders successfully")
        return train_loader, val_loader, test_loader

    def visualize_samples(self, data_loader, num_samples=8, save_path=None):
        """Visualize sample images from the dataset."""
        logger.info(f"Creating visualization of {num_samples} samples...")

        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        # Denormalize images for visualization
        mean = torch.tensor(AUGMENTATION_CONFIG["normalize_mean"])
        std = torch.tensor(AUGMENTATION_CONFIG["normalize_std"])
        images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        images = torch.clamp(images, 0, 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(num_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(images[i].permute(1, 2, 0))
            axes[row, col].set_title(f"{self.class_names[labels[i]]}")
            axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.success(f"Saved visualization to: {save_path}")

        return fig


# ============================================================================
# COMMAND LINE INTERFACE (ENHANCE YOUR EXISTING COMMANDS)
# ============================================================================


@app.command()
def download_cifar10():
    """Download and prepare CIFAR-10 dataset."""
    logger.info("Starting CIFAR-10 dataset download and preparation...")

    data_loader = CIFAR10DataLoader()
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Create a sample visualization
    data_loader.visualize_samples(
        train_loader, save_path=DATA_PATHS["figures"] / "cifar10_samples.png"
    )

    logger.success("CIFAR-10 dataset ready for training!")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
