"""Basic usage examples for torchcachex.

This demonstrates the simplest use case: caching a pretrained feature extractor
during training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator


# 1. Define your dataset (must provide stable cache_ids)
class ImageDataset(Dataset):
    """Example dataset with stable sample IDs."""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # Simulate image data
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "cache_ids": f"sample_{idx}",  # Stable ID (required!)
        }


# 2. Define your expensive feature extractor
class PretrainedFeatureExtractor(nn.Module):
    """Simulates a heavy pretrained model (e.g., ResNet50)."""

    def __init__(self):
        super().__init__()
        # Simulate expensive pretrained layers (frozen)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all parameters (required for caching)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract features (expensive operation to cache)."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        return x.flatten(1)  # Shape: (batch_size, 256)


# 3. Define your trainable classifier
class SimpleClassifier(nn.Module):
    """Lightweight classifier head (trainable)."""

    def __init__(self, num_features=256, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, features):
        return self.fc(features)


def main():
    print("=== torchcachex Basic Usage Example ===\n")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    dataset = ImageDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create cache backend
    print("Creating cache backend...")
    backend = ArrowIPCCacheBackend(
        cache_dir="./cache",
        module_id="pretrained_features_v1",  # Stable ID for this module
        lru_size=4096,  # Keep 4096 samples in memory
        async_write=True,  # Non-blocking writes
        flush_every=512,  # Flush to disk every 512 samples
    )
    print(f"Cache location: {backend.cache_dir}\n")

    # Wrap feature extractor with caching
    feature_extractor = PretrainedFeatureExtractor().to(device)
    cached_extractor = CacheModuleDecorator(
        module=feature_extractor,
        cache_backend=backend,
        enabled=True,
        enforce_stateless=True,  # Ensure no trainable params
    )

    # Create trainable classifier
    classifier = SimpleClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            cache_ids = batch["cache_ids"]

            # Extract features with caching
            # First epoch: computes and caches
            # Later epochs: loads from cache (fast!)
            with torch.no_grad():
                features = cached_extractor(images, cache_ids=cache_ids)

            # Train classifier
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # Force flush at end of epoch
        backend.flush()
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Average Loss: {avg_loss:.4f}\n")

    print("Training complete!")
    print(
        "\nNote: First epoch computed features, subsequent epochs used cache."
    )
    print("Run this script again - all epochs will be fast (cache reused)!")


if __name__ == "__main__":
    main()
