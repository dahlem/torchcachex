# torchcachex

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/dahlem/torchcachex/actions/workflows/ci.yml/badge.svg)](https://github.com/dahlem/torchcachex/actions)
[![codecov](https://codecov.io/gh/dahlem/torchcachex/branch/main/graph/badge.svg)](https://codecov.io/gh/dahlem/torchcachex)

**Drop-in PyTorch module caching with Arrow IPC + in-memory index backend**

`torchcachex` provides transparent, per-sample caching for non-trainable PyTorch modules with:
- ✅ **O(1) append-only writes** via incremental Arrow IPC segments
- ✅ **O(1) batched lookups** via in-memory index + Arrow memory-mapping
- ✅ **Native tensor storage** with automatic dtype preservation
- ✅ **LRU hot cache** for in-process hits
- ✅ **Async writes** (non-blocking forward pass)
- ✅ **DDP-safe** single-writer pattern
- ✅ **Progressive enrichment** (resume from partial caches)
- ✅ **Device-agnostic** (store CPU, return on input device)
- ✅ **MPS (Apple Silicon) support** with automatic synchronization
- ✅ **Scales to billions of samples** with constant memory usage

## Why torchcachex?

**There's no existing open-source library that provides drop-in, dataset-aware, HF-backed, per-sample PyTorch caching** with async persistence and cross-run reuse out of the box.

### The Gap in the Ecosystem

| Category | Representative Tools | What They Cover | Why They Fall Short |
|----------|---------------------|-----------------|-------------------|
| **Dataset-level caching** | 🤗 `datasets` (Arrow cache), `webdataset`, `torchdata.datapipes` | Automatic caching of raw samples or dataset shards on disk | Works at dataset granularity, not **per-module forward outputs** |
| **Feature store frameworks** | Feast, LakeFS, Metaflow, Tecton | Persistent key-value or feature tables | Heavy-weight, external infrastructure; not PyTorch-native nor transparent in `forward()` |
| **Intermediate caching in ML pipelines** | Hydra's launcher caching, DVC, ZenML, Metaflow, Ploomber | Cache *steps* or *scripts* by input hash | Operates at script/task level, not within the training graph |
| **PyTorch-specific accelerators** | `torchdata.datapipes.iter.FileOpener`, Lightning Fabric, HuggingFace Accelerate | Handle I/O and multi-process, not semantic caching of feature modules | - |
| **In-memory caching libs** | `cachetools`, `joblib.Memory`, `functools.lru_cache` | Memory-only or per-function pickling | Don't persist large tensors efficiently, no async or Arrow integration |
| **Reusable embedding caches** | OpenCLIP, SentenceTransformers' `encode` caching | Ad-hoc; usually write `.npy` or `.pt` feature dumps | Single-use, not generalizable as a decorator |

### What torchcachex Provides

| Feature | Existing Tools | torchcachex |
|---------|---------------|-------------|
| Per-sample granularity | ❌ | ✅ |
| Drop-in `nn.Module` decorator | ❌ | ✅ |
| Arrow persistence (native tensors) | ✅ (`datasets`) | ✅ |
| O(1) writes (scale-independent) | ❌ | ✅ |
| Batched push-down lookup | ❌ | ✅ |
| Async write-back | Partial (`joblib`) | ✅ |
| Cross-run progressive cache | ❌ | ✅ |
| Shared across models (module_id) | ❌ | ✅ |
| DDP-aware single-writer | ❌ | ✅ |
| Scales to billions of samples | ❌ | ✅ |
| Transparent to training loop | Partial (`functools.cache`) | ✅ |

**torchcachex fills the gap** between dataset-level caching (like HF Datasets) and ML pipeline tools (like DVC/Feast) by providing **module-level caching** that's especially valuable for:
- Heavy feature extractors (pretrained vision/text models)
- Large-scale datasets with expensive preprocessing
- Distributed training scenarios
- K-fold cross-validation with overlapping samples

## Installation

```bash
pip install torchcachex
```

Or from source:
```bash
git clone https://github.com/dahlem/torchcachex.git
cd torchcachex
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
from torchcachex import CacheModuleDecorator, ArrowIPCCacheBackend

# Define your feature extractor
class FeatureExtractor(nn.Module):
    def forward(self, x):
        # Expensive computation
        return torch.nn.functional.relu(x @ x.t())

# Create cache backend
backend = ArrowIPCCacheBackend(
    cache_dir="./cache",
    module_id="feature_extractor_v1",  # Stable ID
    lru_size=4096,
    async_write=True,
)

# Wrap module with caching
feature_extractor = FeatureExtractor()
cached_extractor = CacheModuleDecorator(
    module=feature_extractor,
    cache_backend=backend,
    enabled=True,
    enforce_stateless=True,  # Ensure no trainable params
)

# Use in training loop
for batch in dataloader:
    # Automatically caches per sample
    features = cached_extractor(
        batch["input"],
        cache_ids=batch["cache_ids"]  # Required: stable sample IDs
    )
```

## Core Concepts

### Module ID

A **stable identifier** for your module that determines cache location. Use semantic versioning:

```python
# ✅ Good: semantic, versioned
module_id = "resnet50_features_v1"
module_id = "bert_embeddings_layer12_v2"

# ❌ Bad: includes run-specific info
module_id = f"features_{datetime.now()}"  # Different each run!
```

### Cache IDs

**Stable per-sample identifiers** that persist across runs. Your dataset must provide these:

```python
# Example dataset
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {
            "input": self.data[idx],
            "label": self.labels[idx],
            "cache_ids": f"sample_{idx}",  # Stable ID
        }
```

**Requirements:**
- Must be stable across runs
- Must be unique per sample
- Can be `str` or `int`

### Cache Keys

Internally constructed as: `{module_id}:{sample_cache_id}`

This allows:
- Different modules to have separate caches
- Same module to be reused across parent models
- K-fold CV to share cache across folds

## Usage Patterns

### Basic Feature Caching

```python
from torchcachex import CacheModuleDecorator, ArrowIPCCacheBackend

backend = ArrowIPCCacheBackend(
    cache_dir="./cache/my_features",
    module_id="my_feature_extractor_v1",
)

cached_module = CacheModuleDecorator(my_module, backend, enabled=True)

# First epoch: computes and caches
for batch in dataloader:
    features = cached_module(batch["input"], cache_ids=batch["ids"])

# Second epoch: 90%+ cache hits, 3x+ speedup
for batch in dataloader:
    features = cached_module(batch["input"], cache_ids=batch["ids"])
```

### DDP Training

```python
import os

# Only rank 0 writes to cache
backend = ArrowIPCCacheBackend(
    cache_dir="./cache/shared",
    module_id="my_features_v1",
    writer_rank=0,
    current_rank=int(os.getenv("RANK", 0)),  # From DDP
)

cached_module = CacheModuleDecorator(my_module, backend, enabled=True)

# All ranks can read, only rank 0 writes
# Safe for DDP without coordination
```

### K-Fold Cross-Validation

```python
# Same cache shared across folds
backend = ArrowIPCCacheBackend(
    cache_dir="./cache/shared",
    module_id="my_features_v1",
)

for fold in range(5):
    train_loader = get_fold_loader(fold, split="train")
    val_loader = get_fold_loader(fold, split="val")

    # Fold 0 populates cache
    # Folds 1-4 reuse cache for overlapping samples
    for batch in train_loader:
        features = cached_module(batch["input"], cache_ids=batch["ids"])
```

### Multiple Parent Models

```python
# Two models share the same feature extractor cache
backend = ArrowIPCCacheBackend(
    cache_dir="./cache/shared",
    module_id="resnet50_features_v1",  # Same ID!
)

# Model A
model_a = ParentModelA(features=CacheModuleDecorator(resnet50, backend))

# Model B (reuses Model A's cache)
model_b = ParentModelB(features=CacheModuleDecorator(resnet50, backend))
```

### Nested Output Structures

```python
class ComplexModule(nn.Module):
    def forward(self, x):
        return {
            "features": x @ self.W,
            "attention": torch.softmax(x @ x.t(), dim=-1),
            "metadata": [x.mean(), x.std()],
        }

# Decorator handles arbitrary output structures
cached = CacheModuleDecorator(ComplexModule(), backend, enabled=True)
out = cached(x, cache_ids=ids)
# out is dict with same structure
```

## Configuration

### Backend Options

```python
ArrowIPCCacheBackend(
    cache_dir="./cache",           # Root directory for cache
    module_id="my_module_v1",      # Stable module identifier
    lru_size=4096,                 # In-memory LRU cache size
    async_write=True,              # Non-blocking writes
    max_workers=2,                 # Async write threads
    flush_every=2048,              # Samples before disk flush
    writer_rank=0,                 # Which rank writes (DDP)
    current_rank=0,                # Current rank (DDP)
)
```

### Decorator Options

```python
CacheModuleDecorator(
    module=my_module,              # Module to wrap
    cache_backend=backend,         # Cache backend
    enabled=True,                  # Enable/disable caching
    key_from_batch_fn=None,        # Custom cache_id extractor
    enforce_stateless=True,        # Check for trainable params
    map_location_on_read="cpu",    # Device for cached data
)
```

## Performance Tips

### LRU Sizing

```python
# Small datasets (< 10k samples): cache everything
backend = ArrowIPCCacheBackend(..., lru_size=10000)

# Large datasets: size for working set
backend = ArrowIPCCacheBackend(..., lru_size=4096)

# Very large datasets: minimal LRU
backend = ArrowIPCCacheBackend(..., lru_size=1024)
```

### Flush Cadence

```python
# Small batches: flush less frequently
backend = ArrowIPCCacheBackend(..., flush_every=4096)

# Large batches: flush more frequently
backend = ArrowIPCCacheBackend(..., flush_every=1024)
```

### Manual Flush

```python
# Force flush at end of epoch
for batch in dataloader:
    features = cached_module(batch["input"], cache_ids=batch["ids"])

backend.flush()  # Ensure all pending writes complete
```

## Examples and Benchmarks

### Usage Examples

See `examples/` directory for comprehensive examples:

```bash
# Basic usage - frozen feature extractor
python examples/basic_usage.py

# Advanced patterns - K-fold CV, DDP, multi-model, etc.
python examples/advanced_usage.py
```

**Examples cover:**
- Basic feature caching workflow
- K-fold cross-validation with shared cache
- DDP (distributed) training
- Multiple models sharing cache
- Complex output structures (dict, tuple, mixed types)
- Progressive cache enrichment

See [examples/README.md](examples/README.md) for detailed documentation and common pitfalls.

### Performance Benchmarks

Run the benchmark suite to measure performance on your system:

```bash
# Full benchmark (generates BENCHMARK.md report)
python benchmark.py

# Skip slow benchmarks
python benchmark.py --skip-write-scaling --skip-memory

# Custom output file
python benchmark.py --output my_results.md
```

**Benchmark measures:**
- Write scaling (O(1) verification)
- Read performance across cache sizes
- Memory usage
- Cache speedup (cached vs uncached)
- Dtype preservation

The benchmark generates a markdown report with:
- Performance metrics and throughput
- O(1) verification with statistical analysis
- Speedup measurements (typically 3-10x for cached epochs)
- Interpretation and recommendations

## Development

### Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/dahlem/torchcachex.git
cd torchcachex
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchcachex --cov-report=term-missing

# Run specific test file
pytest tests/test_backend.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## API Reference

### `CacheModuleDecorator`

Wraps a PyTorch module to add transparent per-sample caching.

**Parameters:**
- `module` (nn.Module): Module to wrap
- `cache_backend` (ArrowIPCCacheBackend): Cache backend
- `enabled` (bool): Whether caching is enabled
- `key_from_batch_fn` (Optional[Callable]): Custom cache ID extractor
- `enforce_stateless` (bool): Check for trainable params
- `map_location_on_read` (str): Device to load cached tensors to

**Methods:**
- `forward(*args, **kwargs)`: Forward pass with caching
- `state_dict()`: Get inner module's state dict
- `load_state_dict(state_dict)`: Load inner module's state dict

### `ArrowIPCCacheBackend`

Persistent cache using Arrow IPC segments with in-memory index for O(1) operations.

**Storage Format:**
```
cache_dir/module_id/
  segments/
    segment_000000.arrow  # Incremental Arrow IPC files
    segment_000001.arrow
    ...
  index.pkl             # Pickled dict: key → (segment_id, row_offset)
  schema.json           # Auto-inferred Arrow schema
```

**Parameters:**
- `cache_dir` (str): Root directory for cache storage
- `module_id` (str): Stable identifier for the module
- `lru_size` (int): Size of in-memory LRU cache (default: 4096)
- `async_write` (bool): Enable asynchronous writes (default: True)
- `max_workers` (int): Number of async write threads (default: 2)
- `flush_every` (int): Samples before disk flush (default: 2048)
- `writer_rank` (int): Which rank writes in DDP (default: 0)
- `current_rank` (Optional[int]): Current process rank (default: None)

**Methods:**
- `get_batch(keys, map_location="cpu")`: O(1) batch lookup via in-memory index + memory-mapped Arrow
- `put_batch(items)`: O(1) append-only write to pending buffer
- `flush()`: Force flush pending writes to new Arrow segment

**Features:**
- **O(1) writes**: New data appended to incremental segments, no rewrites
- **O(1) reads**: In-memory dict index points directly to (segment_id, row_offset)
- **Native tensors**: Automatic dtype preservation via Arrow's type system
- **Schema inference**: Automatically detects structure on first write
- **Crash safety**: Automatic index rebuild from segments on corruption
- **No database dependencies**: Simple pickle-based index persistence

## Architecture

### Storage Design

torchcachex uses a hybrid Arrow IPC + in-memory index architecture optimized for billion-scale caching:

**Components:**

1. **Arrow IPC Segments** (`segments/segment_*.arrow`)
   - Immutable, append-only files
   - Columnar storage with native tensor dtypes
   - Memory-mapped for zero-copy reads
   - Each segment contains a batch of cached samples

2. **Pickle Index** (`index.pkl`)
   - In-memory Python dict backed by pickle persistence
   - Maps cache keys to (segment_id, row_offset)
   - O(1) lookups via dict access
   - Atomic persistence with temp file swap
   - Auto-rebuilds from segments on corruption

3. **Schema File** (`schema.json`)
   - Auto-inferred from first forward pass
   - Preserves tensor dtypes and output structure
   - Supports tensors, dicts, tuples, lists, and mixed types

**Write Path:**

```
put_batch() → pending buffer → flush() → {
  1. Create Arrow RecordBatch
  2. Write to temp segment file
  3. Update in-memory index dict
  4. Atomic rename temp → final
  5. Persist index.pkl (atomic)
}
```

**Read Path:**

```
get_batch() → {
  1. Check LRU cache (in-memory)
  2. Query in-memory index for (segment_id, row_offset)
  3. Memory-map Arrow segment
  4. Extract rows (zero-copy)
  5. Reconstruct tensors with correct dtype
}
```

**Scalability Properties:**

- **Writes**: O(1) - append new segment, update index
- **Reads**: O(1) - direct dict lookup + memory-map
- **Memory**: O(working set) - only LRU + current segment in memory
- **Disk**: O(N) - one entry per sample across segments
- **Crash Recovery**: Atomic - incomplete segments ignored, index auto-rebuilds from segments if corrupted

### Schema Inference

On the first `put_batch()` call, the backend automatically infers the Arrow schema from the module output:

**Single Tensor:**
```python
output = torch.randn(10)
# → Schema: {key: string, data: list<float32>, shape: list<int64>}
```

**Dict of Tensors:**
```python
output = {"features": torch.randn(10), "logits": torch.randn(5)}
# → Schema: {key: string, features_data: list<float32>, features_shape: list<int64>,
#           logits_data: list<float32>, logits_shape: list<int64>}
```

**Mixed Types:**
```python
output = {"tensor": torch.randn(10), "metadata": "foo"}
# → Tensors stored natively, non-tensors pickled
```

This eliminates the need for manual schema definition while preserving full dtype information.

### Important Constraints

torchcachex handles two fundamental constraints automatically - users don't need to think about them:

**1. Concurrent Writers** (Single-Writer Pattern)

Arrow IPC is append-only but not transactional. The solution is built-in:

```python
# DDP: only rank 0 writes
backend = ArrowIPCCacheBackend(
    cache_dir="/shared/cache",
    writer_rank=0,  # Default: rank 0 writes
    current_rank=int(os.getenv("RANK", 0)),
)
```

**2. GPU Tensor Handling** (Automatic Conversion)

GPU tensors are automatically moved to CPU for storage, then restored to input device:

```python
# Transparent handling - user sees tensors on correct device
input_cuda = batch["images"].to("cuda")
features = cached_extractor(input_cuda, cache_ids=batch["ids"])
assert features.device.type == "cuda"  # ✓ Same device as input
```

**3. MPS (Apple Silicon) Support**

MPS devices are fully supported with automatic synchronization:

```python
# Transparent MPS handling - syncs automatically before transfers
input_mps = batch["images"].to("mps")
features = cached_extractor(input_mps, cache_ids=batch["ids"])
assert features.device.type == "mps"  # ✓ Same device as input
```

The decorator handles the asynchronous nature of MPS by calling `torch.mps.synchronize()` before CPU transfers, preventing potential hangs.

See **[ARCHITECTURE.md § Operational Caveats](ARCHITECTURE.md#operational-caveats)** for detailed explanation of both constraints and their implementations.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Format code (`black .` and `isort .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please ensure:
- Code follows the existing style (Black, isort, ruff)
- All tests pass with good coverage (>90%)
- Documentation is updated for new features
- Type hints are included for new functions

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use torchcachex in your research, please cite:

```bibtex
@software{torchcachex,
  title = {torchcachex: Drop-in PyTorch Module Caching},
  author = {Dahlem, Dominik},
  year = {2025},
  url = {https://github.com/dahlem/torchcachex}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/dahlem/torchcachex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dahlem/torchcachex/discussions)

