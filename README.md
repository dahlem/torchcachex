# SHADE I/O

Modular I/O layer for SHADE feature extraction and caching.

## Overview

SHADE I/O provides a clean, modular architecture for feature computation, caching, and management. It separates I/O concerns from the core SHADE logic, enabling:

- **Clean Architecture**: Separation of computation, storage, and filtering concerns
- **Decorator Pattern**: Composable feature transformations (filtering, caching, logging)
- **Multi-level Caching**: Hierarchical storage with memory and file backends
- **Backward Compatibility**: Adapters for gradual migration from v1

## Installation

```bash
# Install from local directory
pip install -e /path/to/shade-io

# With optional Arrow support
pip install -e "/path/to/shade-io[arrow]"

# Development installation
pip install -e "/path/to/shade-io[dev]"
```

## Quick Start

### Basic Feature Computation

```python
from shade_io import SimpleFeatureSet, AttentionData
import torch

# Create a feature set
feature_set = SimpleFeatureSet(
    name="my_features",
    extractors=[...],  # Your feature extractors
)

# Prepare attention data
attention_data = AttentionData(
    attention_matrices=torch.randn(12, 12, 100, 100),
    model_name="gpt2",
    dataset_name="test",
)

# Compute features
result = feature_set.compute_features(attention_data)
print(f"Features: {result.features.shape}")
```

### With Filtering

```python
from shade_io import FilteredFeatureSet
from shade_io.feature_sets.filters import RemoveConstantFeaturesFilter

# Wrap with filter decorator
filtered = FilteredFeatureSet(
    base=feature_set,
    filters=[RemoveConstantFeaturesFilter(threshold=1e-10)],
)

result = filtered.compute_features(attention_data)
```

### With Caching

```python
from shade_io import CachedFeatureSet, FileFeatureStore

# Create file store
store = FileFeatureStore(cache_dir="./cache", format="npz")

# Wrap with cache decorator
cached = CachedFeatureSet(base=filtered, store=store)

# First call computes and caches
result = cached.compute_features(attention_data)

# Second call loads from cache
result = cached.compute_features(attention_data)
```

### Multi-level Cache

```python
from shade_io import CompositeStore, MemoryFeatureStore

# Create composite store with L1 (memory) and L2 (file) cache
store = CompositeStore([
    MemoryFeatureStore(max_size_mb=50),
    FileFeatureStore(cache_dir="./cache"),
])
```

## Architecture

### Core Interfaces

- `IFeatureSet`: Interface for feature computation
- `IFeatureStore`: Interface for feature persistence
- `IFeatureFilter`: Interface for feature filtering

### Feature Sets

- `SimpleFeatureSet`: Basic feature set with extractors
- `CompositeFeatureSet`: Combines multiple feature sets
- `LazyCompositeFeatureSet`: Loads components from cache when available

### Decorators

- `FilteredFeatureSet`: Applies filters to features
- `CachedFeatureSet`: Adds caching behavior
- `LoggedFeatureSet`: Adds logging and metrics
- `ValidatedFeatureSet`: Adds input/output validation

### Stores

- `MemoryFeatureStore`: In-memory storage
- `FileFeatureStore`: File-based persistence (PyTorch, NumPy, Arrow)
  - **Streaming Support**: Arrow format supports incremental writes via `StreamingArrowWriter`
  - **Memory Efficiency**: Async writes avoid memory accumulation for large datasets
- `CompositeStore`: Multi-level cache hierarchy

### Feature Processing

- `FeatureProcessor`: Core processing engine with multiple modes:
  - **Standard Processing**: In-memory batch processing for smaller datasets
  - **Chunked Processing**: Disk-based chunking for memory-constrained environments
  - **Streaming Processing**: Direct Arrow writes for maximum memory efficiency

## Hydra Configuration

SHADE I/O is designed to work with Hydra configuration:

```yaml
# feature_set configuration
feature_set:
  _target_: shade_io.SimpleFeatureSet
  name: my_features
  extractors:
    - _target_: MyExtractor

# store configuration
feature_store:
  _target_: shade_io.FileFeatureStore
  cache_dir: ${paths.cache_dir}
  format: arrow
```

## Migration from v1

Use the provided adapters for backward compatibility:

```python
from shade_io.adapters import V1FeatureSetAdapter

# Wrap v2 feature set for v1 compatibility
v1_compatible = V1FeatureSetAdapter(feature_set, store)
```

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
ruff format .

# Type checking
mypy src/
```

## License

MIT License - See LICENSE file for details.