"""CNN model architectures for CIFAR-10 classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CIFAR10_CONFIG, MODEL_CONFIG


class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 classification - great for learning!"""

    def __init__(self, num_classes=None):
        super(SimpleCNN, self).__init__()

        # Use config values instead of hardcoded numbers
        num_classes = num_classes or CIFAR10_CONFIG["num_classes"]
        input_channels = CIFAR10_CONFIG["input_channels"]

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(MODEL_CONFIG["dropout_rate"])

    def forward(self, x):
        """Forward pass - how data flows through the network."""
        # Conv block 1: 32x32 → 16x16
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 16x16 → 8x8
        x = self.pool(F.relu(self.conv2(x)))

        # Conv block 3: 8x8 → 4x4
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    ### DATA FLOW VISUALIZATION
    # Input: (batch, 3, 32, 32)     # RGB images
    # ↓ conv1 + relu + pool
    # (batch, 32, 16, 16)           # 32 feature maps
    # ↓ conv2 + relu + pool
    # (batch, 64, 8, 8)             # 64 feature maps
    # ↓ conv3 + relu + pool
    # (batch, 128, 4, 4)            # 128 feature maps
    # ↓ flatten
    # (batch, 2048)                 # 1D vector
    # ↓ fc1 + relu + dropout
    # (batch, 512)                  # Hidden layer
    # ↓ fc2
    # (batch, 10)                   # Class predictions
    ###


class ImprovedCNN(nn.Module):
    """An improved CNN with batch normalization and more layers."""

    def __init__(self, num_classes=None):
        super(ImprovedCNN, self).__init__()

        # Use config values
        num_classes = num_classes or CIFAR10_CONFIG["num_classes"]
        input_channels = CIFAR10_CONFIG["input_channels"]

        # First conv block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Second conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Third conv block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(MODEL_CONFIG["dropout_rate"])

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass with batch normalization."""
        # First conv block: 32x32 → 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Second conv block: 16x16 → 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Third conv block: 8x8 → 4x4
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        # Flatten and fully connected
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    ### DATA FLOW VISUALIZATION
    # Input: (batch, 3, 32, 32)     RGB images
    # ↓ conv1 + bn1 + relu
    # (batch, 64, 32, 32)           64 basic features
    # ↓ conv2 + bn2 + relu + pool
    # (batch, 64, 16, 16)           64 refined features, smaller
    # ↓ conv3 + bn3 + relu
    # (batch, 128, 16, 16)          128 intermediate features
    # ↓ conv4 + bn4 + relu + pool
    # (batch, 128, 8, 8)            128 refined features, smaller
    # ↓ conv5 + bn5 + relu
    # (batch, 256, 8, 8)            256 complex features
    # ↓ conv6 + bn6 + relu + pool
    # (batch, 256, 4, 4)            256 most complex features, smallest
    # ↓ flatten
    # (batch, 4096)                 1D feature vector
    # ↓ fc1 + relu + dropout
    # (batch, 512)                  Hidden representation
    # ↓ fc2
    # (batch, 10)                   Class probabilities
    ###


class TinyCNN(nn.Module):
    """A very small CNN for quick experiments and learning."""

    def __init__(self, num_classes=None):
        super(TinyCNN, self).__init__()

        # Use config values
        num_classes = num_classes or CIFAR10_CONFIG["num_classes"]
        input_channels = CIFAR10_CONFIG["input_channels"]

        # Just 2 conv layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Smaller fully connected layer
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Conv layers: 32x32 → 16x16 → 8x8
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten and classify
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    ### DATA FLOW VISUALIZATION
    # Input: (batch, 3, 32, 32)     # RGB images
    # ↓ conv1 + relu + pool
    # (batch, 16, 16, 16)           # 16 feature maps
    # ↓ conv2 + relu + pool
    # (batch, 32, 8, 8)             # 32 feature maps
    # ↓ flatten
    # (batch, 2048)                 # 1D vector
    # ↓ fc1 + relu + dropout
    # (batch, 128)                  # Hidden layer
    # ↓ fc2
    # (batch, 10)                   # Class predictions
    ###


def get_model(model_type="simple", num_classes=None):
    """Factory function to get different model architectures."""
    num_classes = num_classes or CIFAR10_CONFIG["num_classes"]

    models = {"simple": SimpleCNN, "improved": ImprovedCNN, "tiny": TinyCNN}

    if model_type not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    return models[model_type](num_classes)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=None):
    """Print a summary of the model architecture."""
    if input_size is None:
        # Use CIFAR-10 input size from config
        input_size = (1, CIFAR10_CONFIG["input_channels"], *CIFAR10_CONFIG["input_size"])

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Test with dummy input to see output shapes
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    return model


# Example usage and testing
if __name__ == "__main__":
    # Test all models
    for model_name in ["tiny", "simple", "improved"]:
        print(f"\n{'=' * 50}")
        model = get_model(model_name)
        model_summary(model)
