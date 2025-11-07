"""Minimal working example for torchcachex.

This is the simplest possible use case - just 40 lines of code.
Perfect for copy-pasting into your own projects.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator

# 1. Your frozen feature extractor (pretrained model)
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
)
for param in feature_extractor.parameters():
    param.requires_grad = False

# 2. Setup cache
backend = ArrowIPCCacheBackend(
    cache_dir="./minimal_cache",
    module_id="my_features_v1",
)

cached_extractor = CacheModuleDecorator(
    feature_extractor,
    backend,
    enabled=True,
)

# 3. Create dummy dataset with cache IDs
images = torch.randn(100, 3, 32, 32)
labels = torch.randint(0, 10, (100,))
cache_ids = [f"sample_{i}" for i in range(100)]  # Stable IDs!

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=10)

# 4. Training loop
print("Training...")
for epoch in range(3):
    print(f"  Epoch {epoch + 1}/3")
    for batch_idx, (batch_images, _batch_labels) in enumerate(loader):
        # Get cache IDs for this batch
        start_idx = batch_idx * 10
        batch_cache_ids = cache_ids[start_idx : start_idx + 10]

        # Extract features (cached after first epoch!)
        features = cached_extractor(batch_images, cache_ids=batch_cache_ids)

        # ... rest of your training code ...
        # logits = classifier(features)
        # loss.backward()
        # etc.

    backend.flush()  # Persist cache at end of epoch

print("Done! Run again to see cache speedup.")
