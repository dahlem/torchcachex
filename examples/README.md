# torchcachex Examples

This directory contains usage examples demonstrating different patterns and use cases.

## Quick Start

### Basic Usage

The simplest pattern: cache a frozen feature extractor during training.

```bash
python basic_usage.py
```

**What it demonstrates:**
- Creating a cache backend
- Wrapping a feature extractor with caching
- Using cached features in a training loop
- First epoch populates cache, later epochs reuse it

**Key takeaway:** Run the script multiple times - subsequent runs will be much faster as they reuse the cache!

### Advanced Usage

More complex patterns for real-world scenarios.

```bash
python advanced_usage.py
```

**What it demonstrates:**

1. **K-Fold Cross-Validation**: Share cache across folds for overlapping samples
2. **DDP Training**: Single-writer pattern for distributed training
3. **Multiple Models**: Multiple models sharing the same feature cache
4. **Complex Outputs**: Caching modules with dict/tuple outputs
5. **Progressive Enrichment**: Cache grows across multiple runs

## Usage Patterns

### Pattern 1: Frozen Feature Extractor

**Use case:** You have a pretrained model (ResNet, BERT, etc.) and want to cache its outputs.

```python
from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator

# Create backend
backend = ArrowIPCCacheBackend(
    cache_dir="./cache",
    module_id="resnet50_imagenet_v1",  # Stable ID
    lru_size=4096,
    async_write=True,
)

# Wrap frozen feature extractor
feature_extractor = ResNet50Pretrained()  # Your frozen model
cached_extractor = CacheModuleDecorator(
    module=feature_extractor,
    cache_backend=backend,
    enabled=True,
    enforce_stateless=True,  # Verify no trainable params
)

# Use in training
for batch in dataloader:
    # Cache lookup (or compute if miss)
    features = cached_extractor(
        batch["images"],
        cache_ids=batch["sample_ids"]  # Must be stable!
    )

    # Train your classifier
    logits = classifier(features)
    loss = criterion(logits, batch["labels"])
    # ...
```

### Pattern 2: K-Fold Cross-Validation

**Use case:** Run K-fold CV without recomputing features for overlapping samples.

```python
# Single cache shared across all folds
backend = ArrowIPCCacheBackend(
    cache_dir="./cache",
    module_id="features_v1",
)

cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

for fold in range(K):
    train_loader, val_loader = get_fold_loaders(fold)

    # Train fold (features cached progressively)
    for batch in train_loader:
        features = cached_extractor(batch["input"], cache_ids=batch["ids"])
        # ... train ...

    # Validate (reuses cached features)
    for batch in val_loader:
        features = cached_extractor(batch["input"], cache_ids=batch["ids"])
        # ... evaluate ...
```

**Benefit:** Fold N reuses all features computed in folds 0..N-1.

### Pattern 3: Distributed Training (DDP)

**Use case:** Train with multiple GPUs, cache on shared filesystem.

```python
import os

backend = ArrowIPCCacheBackend(
    cache_dir="/shared/cache",  # Shared across ranks
    module_id="features_v1",
    writer_rank=0,  # Only rank 0 writes
    current_rank=int(os.getenv("RANK", 0)),  # From DDP
)

cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

# All ranks compute, only rank 0 writes cache
for batch in dataloader:
    features = cached_extractor(batch["input"], cache_ids=batch["ids"])
    # ... train ...
```

**Benefit:** First epoch is same speed, all subsequent epochs are fast on all ranks.

### Pattern 4: Multiple Models

**Use case:** Train multiple models (e.g., ensembles) that share the same features.

```python
# Single cache for shared features
backend = ArrowIPCCacheBackend(
    cache_dir="./cache",
    module_id="resnet50_features_v1",  # Same ID!
)

# Multiple models use the same cached features
cached_extractor = CacheModuleDecorator(feature_extractor, backend, enabled=True)

model_a = ModelA(features=cached_extractor)
model_b = ModelB(features=cached_extractor)
model_c = ModelC(features=cached_extractor)

# Train model A (populates cache)
train(model_a, train_loader)

# Train models B and C (reuse cache from A)
train(model_b, train_loader)  # Fast!
train(model_c, train_loader)  # Fast!
```

### Pattern 5: Complex Output Structures

**Use case:** Your module returns dicts, tuples, or mixed types.

```python
class MultiHeadExtractor(nn.Module):
    def forward(self, x):
        return {
            "visual": self.visual_head(x),    # Tensor
            "semantic": self.semantic_head(x), # Tensor
            "metadata": {"width": x.shape[-1]}, # Non-tensor (pickled)
        }

# Decorator handles complex structures automatically
cached = CacheModuleDecorator(
    MultiHeadExtractor(),
    backend,
    enabled=True
)

outputs = cached(batch["input"], cache_ids=batch["ids"])
# outputs["visual"] → Tensor (native storage, dtype preserved)
# outputs["semantic"] → Tensor (native storage, dtype preserved)
# outputs["metadata"] → dict (pickled)
```

## Common Pitfalls

### 1. Non-Stable Cache IDs

**❌ Wrong:**
```python
# This breaks caching across runs!
cache_ids = [f"sample_{time.time()}_{i}" for i in range(len(batch))]
```

**✅ Correct:**
```python
# Use dataset index or UUID
cache_ids = [f"sample_{dataset_idx}" for dataset_idx in batch["indices"]]
```

### 2. Changing module_id

**❌ Wrong:**
```python
# Different ID every run!
module_id = f"features_{datetime.now()}"
```

**✅ Correct:**
```python
# Semantic versioning
module_id = "resnet50_imagenet_v1"  # Change v2 when model changes
```

### 3. Not Flushing

**❌ Wrong:**
```python
for epoch in range(10):
    for batch in loader:
        cached_module(batch["input"], cache_ids=batch["ids"])
# Cache may be lost if process crashes!
```

**✅ Correct:**
```python
for epoch in range(10):
    for batch in loader:
        cached_module(batch["input"], cache_ids=batch["ids"])
    backend.flush()  # Force persist at end of epoch
```

### 4. Caching Trainable Modules

**❌ Wrong:**
```python
# This module has trainable parameters!
trainable_module = nn.Linear(512, 10)
cached = CacheModuleDecorator(trainable_module, backend, enabled=True)
# Will raise error if enforce_stateless=True
```

**✅ Correct:**
```python
# Freeze module first
for param in trainable_module.parameters():
    param.requires_grad = False

cached = CacheModuleDecorator(trainable_module, backend, enabled=True)
```

## Performance Tips

### LRU Sizing

```python
# Small dataset (< 10k): cache everything in memory
backend = ArrowIPCCacheBackend(..., lru_size=10000)

# Large dataset: size for your working set
backend = ArrowIPCCacheBackend(..., lru_size=4096)

# Very large dataset: minimal LRU, rely on disk cache
backend = ArrowIPCCacheBackend(..., lru_size=1024)
```

### Flush Frequency

```python
# Large batches: flush more frequently
backend = ArrowIPCCacheBackend(..., flush_every=512)

# Small batches: flush less frequently
backend = ArrowIPCCacheBackend(..., flush_every=4096)
```

### Async Writes

```python
# Training (non-blocking writes)
backend = ArrowIPCCacheBackend(..., async_write=True)

# Testing/debugging (immediate persistence)
backend = ArrowIPCCacheBackend(..., async_write=False)
```

## Next Steps

- See `../benchmark.py` for performance benchmarks
- See `../ARCHITECTURE.md` for technical deep-dive
- See `../README.md` for full API reference
