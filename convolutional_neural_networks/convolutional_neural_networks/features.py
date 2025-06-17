from pathlib import Path
import pickle

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA

# Add these new imports
import torch
from tqdm import tqdm
import typer

from convolutional_neural_networks.config import CIFAR10_CONFIG, DATA_PATHS, PROCESSED_DATA_DIR

app = typer.Typer()

# ============================================================================
# CIFAR-10 FEATURE ANALYSIS CLASS
# ============================================================================


class CIFAR10FeatureAnalyzer:
    """Advanced feature analysis for trained CNN models."""

    # Extracts the internal representations (features) from any layer of your trained CNN.

    def __init__(self):
        self.class_names = CIFAR10_CONFIG["class_names"]

    def extract_features_from_layer(self, model, dataloader, layer_name, max_batches=10):
        """Extract features from a specific layer of the model."""
        logger.info(f"Extracting features from layer: {layer_name}")

        features = []
        labels = []

        def hook_fn(module, input, output):
            # Hook Function: This is the "spy" that captures data flowing through the layer
            features.append(output.detach().cpu())

        # Register hook
        # Register Hook: Attaches our "spy" to the specific layer
        layer = dict(model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
                if i >= max_batches:
                    break

                data = data.to(next(model.parameters()).device)
                _ = model(data)  # Forward pass triggers hook
                labels.extend(target.tolist())

        hook.remove()
        all_features = torch.cat(features, dim=0)
        logger.success(f"Extracted features shape: {all_features.shape}")

        return all_features.numpy(), np.array(labels)

    def analyze_feature_importance(self, features, labels):
        """Analyze which features are most important for classification."""
        logger.info("Analyzing feature importance...")

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features.reshape(features.shape[0], -1))

        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_

        logger.success(
            f"Top 10 components explain {explained_variance[:10].sum():.2%} of variance"
        )

        return features_pca, explained_variance, pca

    def calculate_class_weights(self, dataset):
        """Calculate class weights for imbalanced datasets."""
        logger.info("Calculating class weights...")

        class_counts = torch.zeros(CIFAR10_CONFIG["num_classes"])
        for _, label in tqdm(dataset, desc="Counting classes"):
            class_counts[label] += 1

        total_samples = len(dataset)
        class_weights = total_samples / (CIFAR10_CONFIG["num_classes"] * class_counts)

        logger.info(f"Class weights: {class_weights.tolist()}")
        return class_weights


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


@app.command()
def extract_cnn_features(
    model_path: Path = DATA_PATHS["models"] / "best_model.pth",
    layer_name: str = "conv3",
    output_path: Path = PROCESSED_DATA_DIR / "cnn_features.pkl",
    max_batches: int = 50,
):
    """Extract features from a trained CNN model."""
    logger.info(f"Extracting features from {layer_name} layer...")

    # This would need to load your trained model and dataloader
    # For now, just show the structure
    logger.info(f"Model path: {model_path}")
    logger.info(f"Layer: {layer_name}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Max batches: {max_batches}")

    # TODO: Implement actual feature extraction when model is trained
    logger.warning("Feature extraction will be implemented after model training")

    # Placeholder for actual implementation:
    # analyzer = CIFAR10FeatureAnalyzer()
    # model = load_model(model_path)
    # dataloader = get_dataloader()
    # features, labels = analyzer.extract_features_from_layer(model, dataloader, layer_name, max_batches)
    #
    # # Save features
    # with open(output_path, 'wb') as f:
    #     pickle.dump({'features': features, 'labels': labels}, f)

    logger.success("Feature extraction command ready!")


@app.command()
def analyze_features(
    features_path: Path = PROCESSED_DATA_DIR / "cnn_features.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "feature_analysis.pkl",
):
    """Analyze extracted CNN features."""
    logger.info("Analyzing CNN features...")

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Run 'extract-cnn-features' command first")
        return

    # Load features
    with open(features_path, "rb") as f:
        data = pickle.load(f)
        features = data["features"]
        labels = data["labels"]

    logger.info(f"Loaded features shape: {features.shape}")

    # Analyze features
    analyzer = CIFAR10FeatureAnalyzer()
    features_pca, explained_variance, pca = analyzer.analyze_feature_importance(features, labels)

    # Save analysis results
    analysis_results = {
        "features_pca": features_pca,
        "explained_variance": explained_variance,
        "pca_model": pca,
        "labels": labels,
    }

    with open(output_path, "wb") as f:
        pickle.dump(analysis_results, f)

    logger.success(f"Feature analysis saved to: {output_path}")


@app.command()
def calculate_weights(output_path: Path = PROCESSED_DATA_DIR / "class_weights.pkl"):
    """Calculate class weights for CIFAR-10 dataset."""
    logger.info("Calculating class weights for CIFAR-10...")

    try:
        # Load actual dataset
        from convolutional_neural_networks.dataset import CIFAR10DataLoader

        analyzer = CIFAR10FeatureAnalyzer()
        data_loader = CIFAR10DataLoader()
        train_dataset, _ = data_loader.load_data()
        class_weights = analyzer.calculate_class_weights(train_dataset)

        logger.success("Calculated actual class weights from dataset")

    except ImportError as e:
        logger.warning(f"Could not load dataset: {e}")
        logger.info("Using balanced weights as fallback")
        # Fallback to balanced weights
        class_weights = torch.ones(CIFAR10_CONFIG["num_classes"])

    # Save weights
    with open(output_path, "wb") as f:
        pickle.dump(class_weights.tolist(), f)

    logger.success(f"Class weights saved to: {output_path}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
