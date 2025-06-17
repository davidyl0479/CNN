from pathlib import Path

from loguru import logger

# Add these new imports (removed unused ones)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import typer

from convolutional_neural_networks.config import (
    CIFAR10_CONFIG,
    DATA_PATHS,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

# ============================================================================
# CNN VISUALIZATION CLASS
# ============================================================================


class CNNVisualizer:
    """Handle all visualization needs for CNN project."""

    def __init__(self):
        self.class_names = CIFAR10_CONFIG["class_names"]
        plt.style.use("seaborn-v0_8")  # Nice default style
        logger.info("CNN Visualizer initialized")

    def plot_training_history(self, history, save_path=None):
        """Plot training and validation loss/accuracy curves."""
        logger.info("Creating training history plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(history["train_loss"], label="Training Loss", color="blue")
        ax1.plot(history["val_loss"], label="Validation Loss", color="red")
        ax1.set_title("Model Loss Over Time")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(history["train_acc"], label="Training Accuracy", color="blue")
        ax2.plot(history["val_acc"], label="Validation Accuracy", color="red")
        ax2.set_title("Model Accuracy Over Time")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.success(f"Training history saved to: {save_path}")

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix."""
        logger.info("Creating confusion matrix...")

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.success(f"Confusion matrix saved to: {save_path}")

        return plt.gcf()

    def plot_sample_predictions(self, model, dataloader, num_samples=16, save_path=None):
        """Plot sample predictions with true vs predicted labels."""
        logger.info(f"Creating sample predictions plot with {num_samples} samples...")

        model.eval()

        # Get a batch of data
        data_iter = iter(dataloader)
        images, labels = next(data_iter)

        # Make predictions
        with torch.no_grad():
            outputs = model(images.to(next(model.parameters()).device))
            _, predicted = torch.max(outputs.cpu(), 1)

        # Denormalize images for display
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i in range(min(num_samples, len(images))):
            row, col = i // 4, i % 4

            # Display image
            axes[row, col].imshow(images[i].permute(1, 2, 0))

            # Create title with true vs predicted
            true_label = self.class_names[labels[i]]
            pred_label = self.class_names[predicted[i]]

            color = "green" if labels[i] == predicted[i] else "red"
            axes[row, col].set_title(
                f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10
            )
            axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.success(f"Sample predictions saved to: {save_path}")

        return fig

    def plot_class_distribution(self, dataset, save_path=None):
        """Plot class distribution in the dataset."""
        logger.info("Creating class distribution plot...")

        class_counts = torch.zeros(CIFAR10_CONFIG["num_classes"])

        for _, label in dataset:
            class_counts[label] += 1

        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_counts.numpy())
        plt.title("Class Distribution in Dataset")
        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{int(count)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.success(f"Class distribution saved to: {save_path}")

        return plt.gcf()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


@app.command()
def create_sample_plots():
    """Create sample visualization plots for demonstration."""
    logger.info("Creating sample CNN visualization plots...")

    visualizer = CNNVisualizer()

    # Create sample training history
    sample_history = {
        "train_loss": [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5],
        "val_loss": [2.4, 1.9, 1.5, 1.2, 1.0, 0.9, 0.8, 0.8],
        "train_acc": [20, 35, 50, 65, 75, 82, 87, 90],
        "val_acc": [18, 32, 48, 62, 72, 78, 82, 85],
    }

    # Create and save plots (using the returned figures)
    training_fig = visualizer.plot_training_history(
        sample_history, save_path=FIGURES_DIR / "sample_training_history.png"
    )
    plt.close(training_fig)  # Close to free memory

    # Create sample confusion matrix data
    np.random.seed(42)  # For reproducible results
    y_true = np.random.randint(0, 10, 1000)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(1000, 200, replace=False)
    y_pred[error_indices] = np.random.randint(0, 10, 200)

    confusion_fig = visualizer.plot_confusion_matrix(
        y_true, y_pred, save_path=FIGURES_DIR / "sample_confusion_matrix.png"
    )
    plt.close(confusion_fig)  # Close to free memory

    logger.success("Sample plots created successfully!")


@app.command()
def analyze_training_results(
    model_path: Path = typer.Option(..., help="Path to trained model"),
    data_path: Path = typer.Option(DATA_PATHS["processed"], help="Path to test data"),
):
    """Analyze results from a trained model."""
    logger.info(f"Analyzing model results from: {model_path}")
    logger.info(f"Using data from: {data_path}")

    # This would load your trained model and create analysis plots
    # For now, just demonstrate the structure
    logger.info("Model analysis would be performed here...")
    logger.info("- Loading trained model")
    logger.info("- Running inference on test data")
    logger.info("- Creating confusion matrix")
    logger.info("- Generating sample predictions")
    logger.info("- Saving all plots to figures directory")

    logger.success("Analysis complete!")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
