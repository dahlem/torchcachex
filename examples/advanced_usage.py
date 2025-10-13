"""Advanced usage patterns for torchcachex.

Demonstrates:
- K-fold cross-validation with shared cache
- DDP (distributed) training
- Multiple models sharing cache
- Complex output structures
"""

import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator


class ImageDataset(Dataset):
    """Example dataset."""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "cache_ids": f"sample_{idx}",
        }


class FeatureExtractor(nn.Module):
    """Pretrained feature extractor."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return x.flatten(1)


class MultiOutputFeatureExtractor(nn.Module):
    """Feature extractor with multiple outputs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Returns dict with multiple feature types."""
        conv_out = torch.relu(self.conv(x))
        pooled = self.pool(conv_out).flatten(1)

        return {
            "features": pooled,  # Shape: (B, 256)
            "spatial": conv_out.mean(dim=[2, 3]),  # Shape: (B, 256)
            "stats": torch.stack([pooled.mean(dim=1), pooled.std(dim=1)], dim=1),
        }


def example_kfold_cv():
    """K-fold cross-validation with shared cache across folds."""
    print("\n=== Example 1: K-Fold Cross-Validation ===\n")

    dataset = ImageDataset(num_samples=500)
    num_folds = 5
    fold_size = len(dataset) // num_folds

    # Single cache shared across all folds
    backend = ArrowIPCCacheBackend(
        cache_dir="./cache_kfold",
        module_id="features_v1",
        async_write=False,  # Sync for demo
    )

    feature_extractor = FeatureExtractor()
    cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

    for fold in range(num_folds):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split data
        val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_indices = [i for i in range(len(dataset)) if i not in val_indices]

        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=32)

        # Train on fold (features cached progressively)
        for batch in train_loader:
            features = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
            # ... train classifier ...

        backend.flush()

        # Validate (reuses cached features from overlapping samples)
        for batch in val_loader:
            features = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
            # ... evaluate ...

        print(f"  Fold {fold + 1} complete\n")

    print("All folds reused cache for overlapping samples!\n")


def example_ddp_training():
    """Distributed training with single-writer pattern."""
    print("\n=== Example 2: DDP Training ===\n")

    # Simulate DDP ranks (in real DDP, use torch.distributed)
    rank = int(os.getenv("RANK", 0))  # Current process rank
    world_size = int(os.getenv("WORLD_SIZE", 1))

    print(f"Simulating DDP: rank {rank}/{world_size}")

    # Only rank 0 writes to cache
    backend = ArrowIPCCacheBackend(
        cache_dir="./cache_ddp",
        module_id="features_v1",
        writer_rank=0,  # Only rank 0 writes
        current_rank=rank,  # Current rank
        async_write=True,
    )

    feature_extractor = FeatureExtractor()
    cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

    dataset = ImageDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=32)

    print("Training (all ranks compute, only rank 0 writes cache)...")
    for batch in loader:
        # All ranks compute features
        features = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
        # ... train on features ...

    backend.flush()

    print(f"Rank {rank}: Cache populated by rank 0, all ranks can read in next epoch\n")


def example_multiple_models_shared_cache():
    """Multiple models sharing the same feature cache."""
    print("\n=== Example 3: Multiple Models Sharing Cache ===\n")

    # Single cache for shared feature extractor
    backend = ArrowIPCCacheBackend(
        cache_dir="./cache_shared",
        module_id="resnet50_features_v1",  # Same module_id!
        async_write=False,
    )

    # Model A: Binary classifier
    class ModelA(nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.features = feature_extractor
            self.classifier = nn.Linear(256, 2)  # Binary

        def forward(self, x, cache_ids):
            features = self.features(x, cache_ids=cache_ids)
            return self.classifier(features)

    # Model B: Multi-class classifier
    class ModelB(nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.features = feature_extractor
            self.classifier = nn.Linear(256, 10)  # 10 classes

        def forward(self, x, cache_ids):
            features = self.features(x, cache_ids=cache_ids)
            return self.classifier(features)

    # Create cached extractor
    feature_extractor = FeatureExtractor()
    cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

    # Both models use the same cached features!
    model_a = ModelA(cached_extractor)
    model_b = ModelB(cached_extractor)

    dataset = ImageDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=32)

    print("Training Model A (populates cache)...")
    for batch in loader:
        logits = model_a(batch["image"], cache_ids=batch["cache_ids"])
        # ... train model A ...
    backend.flush()

    print("Training Model B (reuses Model A's cache)...")
    for batch in loader:
        logits = model_b(batch["image"], cache_ids=batch["cache_ids"])
        # ... train model B ...

    print("Model B reused all features from Model A's cache!\n")


def example_complex_outputs():
    """Caching modules with complex output structures."""
    print("\n=== Example 4: Complex Output Structures ===\n")

    backend = ArrowIPCCacheBackend(
        cache_dir="./cache_complex",
        module_id="multi_output_v1",
        async_write=False,
    )

    # Multi-output extractor
    extractor = MultiOutputFeatureExtractor()
    cached_extractor = CacheModuleDecorator(extractor, backend, enabled=True)

    dataset = ImageDataset(num_samples=100)
    loader = DataLoader(dataset, batch_size=32)

    print("Caching complex dict outputs...")
    for batch in loader:
        outputs = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])

        # Access different outputs
        features = outputs["features"]  # Main features
        spatial = outputs["spatial"]  # Spatial features
        stats = outputs["stats"]  # Statistics

        print(
            f"  Features: {features.shape}, Spatial: {spatial.shape}, Stats: {stats.shape}"
        )

    backend.flush()

    print("\nAll output tensors cached with correct dtypes and shapes!")


def example_progressive_enrichment():
    """Progressive cache enrichment across multiple runs."""
    print("\n=== Example 5: Progressive Enrichment ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ArrowIPCCacheBackend(
            cache_dir=tmpdir,
            module_id="features_v1",
            async_write=False,
        )

        extractor = FeatureExtractor()
        cached_extractor = CacheModuleDecorator(extractor, backend, enabled=True)

        dataset = ImageDataset(num_samples=100)

        # Run 1: Process first 30 samples
        print("Run 1: Cache first 30 samples")
        loader1 = DataLoader(Subset(dataset, range(30)), batch_size=10)
        for batch in loader1:
            _ = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
        backend.flush()
        print("  30 samples cached\n")

        # Run 2: Process samples 20-60 (overlap with run 1)
        print("Run 2: Cache samples 20-60 (20-30 already cached)")
        loader2 = DataLoader(Subset(dataset, range(20, 60)), batch_size=10)
        for batch in loader2:
            _ = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
        backend.flush()
        print("  Only new 30 samples computed, 10 reused from cache\n")

        # Run 3: Process all 100 samples
        print("Run 3: Process all 100 samples")
        loader3 = DataLoader(dataset, batch_size=10)
        for batch in loader3:
            _ = cached_extractor(batch["image"], cache_ids=batch["cache_ids"])
        backend.flush()
        print("  60 from cache, 40 newly computed\n")

        print("Cache progressively enriched across runs!")


def main():
    print("=== torchcachex Advanced Usage Examples ===")

    # Run examples
    example_kfold_cv()
    example_ddp_training()
    example_multiple_models_shared_cache()
    example_complex_outputs()
    example_progressive_enrichment()

    print("\n=== All Examples Complete ===")


if __name__ == "__main__":
    main()
