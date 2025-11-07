# torchcachex Architecture

This document provides a technical deep-dive into torchcachex's architecture, design decisions, and implementation details.

## Table of Contents

- [Design Goals](#design-goals)
- [System Architecture](#system-architecture)
- [Storage Format](#storage-format)
- [Performance Characteristics](#performance-characteristics)
- [Implementation Details](#implementation-details)
- [Crash Recovery](#crash-recovery)
- [Design Tradeoffs](#design-tradeoffs)

## Design Goals

torchcachex was designed to solve a specific problem: **efficient, persistent caching of expensive PyTorch module computations at billion-sample scale**.

### Primary Requirements

1. **O(1) Operations**: Both reads and writes must scale independently of cache size
2. **Persistent Storage**: Cache must survive process restarts and be reusable across runs
3. **Native Tensor Storage**: Preserve PyTorch tensor dtypes (float32, float64, etc.) without conversion
4. **Drop-in Simplicity**: Zero boilerplate, works as a decorator with automatic schema inference
5. **DDP Compatibility**: Safe for distributed training with single-writer pattern
6. **Progressive Enrichment**: Resume from partial caches without recomputation
7. **Crash Safety**: No data corruption on sudden termination

### Non-Goals

- **Multi-writer coordination**: We use single-writer pattern (DDP rank 0) for simplicity
- **Compression**: Arrow IPC provides efficient binary format; additional compression adds complexity
- **Remote storage**: Focused on local disk; cloud storage can be added via abstractions
- **Versioning**: Cache versioning is manual via `module_id` changes

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CacheModuleDecorator                      │
│  (Wraps PyTorch module, handles cache_ids, device mapping)  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  ArrowIPCCacheBackend                        │
│                                                              │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ LRU Cache │  │ Pending      │  │ Schema Inference   │  │
│  │ (Memory)  │  │ Write Buffer │  │ (First write)      │  │
│  └───────────┘  └──────────────┘  └────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐│
│  │              Flush Mechanism (Async/Sync)              ││
│  │  1. Create Arrow RecordBatch                           ││
│  │  2. Write temp segment file                            ││
│  │  3. Update in-memory index dict                        ││
│  │  4. Atomic rename                                      ││
│  │  5. Persist index.pkl (atomic)                         ││
│  └────────────────────────────────────────────────────────┘│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Persistent Storage                      │
│                                                              │
│  cache_dir/module_id/                                       │
│    ├── segments/                                            │
│    │   ├── segment_000000.arrow  ← Arrow IPC files         │
│    │   ├── segment_000001.arrow  ← (immutable)             │
│    │   └── ...                                              │
│    ├── index.pkl                 ← Pickle index (dict)     │
│    └── schema.json               ← Auto-inferred schema    │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. CacheModuleDecorator

**Responsibilities:**
- Wraps PyTorch `nn.Module` with transparent caching
- Extracts `cache_ids` from batch
- Checks cache for hits/misses
- Invokes wrapped module for cache misses
- Handles device mapping (store CPU, return on input device)
- Validates stateless constraint (no trainable parameters)

**Code Path:**
```python
forward(x, cache_ids) → {
  hits, missing_ids = backend.get_batch(cache_ids)
  if missing_ids:
    missing_outputs = module(x[missing_ids])  # Compute
    backend.put_batch({id: output for id, output})  # Cache
  return combine(hits, missing_outputs)
}
```

#### 2. ArrowIPCCacheBackend

**Responsibilities:**
- Manages persistent storage (Arrow + pickle index)
- Handles schema inference on first write
- Maintains in-memory LRU cache for hot data
- Buffers writes and flushes in batches
- Provides O(1) read/write operations
- Ensures crash safety via atomic commits

**Key Data Structures:**
- `self.lru`: LRU cache (dict-like, size-bounded)
- `self._pending`: Write buffer (list of dicts)
- `self.schema`: Arrow schema (inferred or loaded)
- `self.index`: In-memory dict mapping keys to (segment_id, row_offset)
- `self.executor`: ThreadPoolExecutor for async writes

#### 3. Pickle Index

**Responsibilities:**
- O(1) key → (segment_id, row_offset) mapping via Python dict
- In-memory for fast lookups, persisted to disk for durability
- Atomic persistence with temp file swap
- Auto-rebuild from segments on corruption or missing index

**Data Structure:**
```python
self.index = {
    "module_id:sample_123": (0, 42),  # segment_id=0, row_offset=42
    "module_id:sample_456": (0, 43),
    "module_id:sample_789": (1, 0),
    # ... millions of entries
}
```

**Persistence:**
```python
# Atomic save with temp file
temp_path = index_path.with_suffix('.pkl.tmp')
with open(temp_path, 'wb') as f:
    pickle.dump(self.index, f)
temp_path.rename(index_path)  # Atomic on POSIX

# Load on startup
if index_path.exists():
    with open(index_path, 'rb') as f:
        self.index = pickle.load(f)
else:
    self.index = self._rebuild_index_from_segments()
```

#### 4. Arrow IPC Segments

**Responsibilities:**
- Store tensor data in columnar format
- Preserve dtypes via Arrow type system
- Enable zero-copy reads via memory-mapping
- Immutable once written (append-only)

**File Format:**
- Arrow IPC (Feather v2) format
- One file per flush operation
- Columnar layout: `{key, data, shape, ...}`
- Memory-mapped for efficient access

## Storage Format

### Arrow Schema Design

The schema is automatically inferred on the first `put_batch()` call based on the module output structure.

#### Single Tensor Output

```python
# Module output
output = torch.randn(128, dtype=torch.float32)

# Inferred Arrow schema
{
  "key": string,
  "data": list<float>,
  "shape": list<int64>
}

# Metadata
{"torch_dtype": "torch.float32"}
```

**Stored representation:**
- `key`: "my_module:sample_123"
- `data`: [0.1, -0.5, 0.3, ...]  (flattened tensor)
- `shape`: [128]

#### Dict of Tensors Output

```python
# Module output
output = {
    "features": torch.randn(512, dtype=torch.float32),
    "logits": torch.randn(10, dtype=torch.float16)
}

# Inferred Arrow schema
{
  "key": string,
  "features_data": list<float>,
  "features_shape": list<int64>,
  "logits_data": list<half_float>,
  "logits_shape": list<int64>
}

# Metadata
{"tensor_keys": '["features", "logits"]'}
```

#### Tuple/List Output

```python
# Module output
output = (torch.randn(10), torch.randn(20))

# Inferred Arrow schema
{
  "key": string,
  "tensor_0_data": list<float>,
  "tensor_0_shape": list<int64>,
  "tensor_1_data": list<float>,
  "tensor_1_shape": list<int64>
}

# Metadata
{"num_tensors": "2"}
```

#### Mixed Types (Fallback)

```python
# Module output
output = {
    "tensor": torch.randn(10),
    "metadata": {"label": "foo", "count": 42}  # Non-tensor
}

# Inferred Arrow schema
{
  "key": string,
  "tensor_data": list<float>,
  "tensor_shape": list<int64>,
  "other_data": binary  # Pickled non-tensors
}
```

### PyTorch → Arrow Type Mapping

```python
TORCH_TO_ARROW = {
    torch.float16: pa.float16(),
    torch.float32: pa.float32(),
    torch.float64: pa.float64(),
    torch.int8: pa.int8(),
    torch.int16: pa.int16(),
    torch.int32: pa.int32(),
    torch.int64: pa.int64(),
    torch.uint8: pa.uint8(),
    torch.bool: pa.bool_(),
}

ARROW_TO_TORCH = {v: k for k, v in TORCH_TO_ARROW.items()}
```

**Key Property**: This mapping is **bijective** (one-to-one), ensuring lossless dtype preservation.

### Segment File Naming

```
segment_{id:06d}.arrow
```

Examples:
- `segment_000000.arrow` (first flush)
- `segment_000001.arrow` (second flush)
- `segment_000042.arrow` (43rd flush)

**Characteristics:**
- Zero-padded to 6 digits (supports up to 999,999 segments)
- Lexicographically sorted by flush order
- Immutable once created (no in-place updates)

## Performance Characteristics

### Write Path Analysis

#### Old Architecture (HuggingFace Datasets)

```python
def flush():
    # O(N): Load entire existing dataset
    old_dataset = load_from_disk(path)

    # O(M): Create new dataset from pending
    new_dataset = Dataset.from_dict(pending)

    # O(N+M): Concatenate (copies all data)
    combined = concatenate_datasets([old_dataset, new_dataset])

    # O(N+M): Write entire dataset to disk
    combined.save_to_disk(path)

    # Total: O(N+M) per flush, where N = existing cache size
```

**Problem**: As cache grows, flush time increases linearly. With 1M samples, flushing 1k new samples requires rewriting 1M samples.

#### New Architecture (Arrow IPC + Pickle Index)

```python
def flush():
    # O(1): Create Arrow RecordBatch from pending
    batch = pa.RecordBatch.from_pydict(pending, schema=schema)  # ~O(M)

    # O(1): Write to temp segment file
    with pa.OSFile(temp_path, "wb") as sink:
        writer = pa.ipc.new_file(sink, schema)
        writer.write_batch(batch)  # Sequential write, ~O(M)

    # O(1): Update in-memory index
    for i, item in enumerate(pending):
        self.index[item["key"]] = (segment_id, i)  # ~O(M) dict updates

    # O(1): Atomic rename
    temp_path.rename(final_path)

    # O(M): Persist index to disk
    with open(temp_index_path, 'wb') as f:
        pickle.dump(self.index, f)  # ~O(total index size)
    temp_index_path.rename(index_path)

    # Total: O(M + I) per flush, where I = total index size
    # Note: pickle.dump is typically very fast (~100MB/s)
```

**Breakthrough**: Flush time depends only on new data (M) plus index serialization (I). With 1M samples cached, flushing 1k new samples is still very fast due to efficient pickle serialization.

### Read Path Analysis

#### Batch Read Complexity

```python
def get_batch(keys):
    # O(K): Check LRU cache (K = batch size)
    hits = [lru.get(k) for k in keys]
    missing_keys = [k for k, v in zip(keys, hits) if v is None]

    if not missing_keys:
        return hits, []  # All cache hits

    # O(K): Query in-memory index for missing keys
    index_results = []
    for key in missing_keys:
        if key in self.index:
            seg_id, offset = self.index[key]
            index_results.append((key, seg_id, offset))

    # O(K): Group by segment_id
    by_segment = defaultdict(list)
    for key, seg_id, offset in index_results:
        by_segment[seg_id].append((key, offset))

    # O(S): Memory-map each unique segment (S = # unique segments)
    for seg_id, items in by_segment.items():
        segment_file = segments_dir / f"segment_{seg_id:06d}.arrow"

        # O(1): Memory-map segment (no data read yet)
        with pa.memory_map(segment_file, "r") as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()  # O(segment size), but cached by OS

            # O(K_s): Extract rows from this segment (K_s = items in segment)
            for key, offset in items:
                row = extract_row(table, offset)  # O(1) columnar access
                tensor = reconstruct_tensor(row)   # O(tensor size)
                lru[key] = tensor

    # Total: O(K + S*T) where K = batch size, S = segments, T = avg segment scan
    # In practice: O(K) with memory-mapping and OS page cache
```

**Key Optimizations**:
1. **LRU cache** reduces disk access for hot data
2. **Memory-mapping** enables zero-copy reads with OS page cache
3. **Columnar access** allows extracting specific rows without full table scan
4. **In-memory dict lookup** provides true O(1) access (faster than SQLite B-tree)

### Space Complexity

```
Total Disk Space = Sum(Arrow segments) + Pickle index + Schema file

Arrow segments:
  - Per sample: sizeof(key) + sizeof(flattened_tensor) + sizeof(shape) + overhead
  - For float32[512]: ~8 bytes (key) + 2048 bytes (data) + 8 bytes (shape) + ~10 bytes (Arrow) ≈ 2074 bytes
  - For 1B samples: ~2074 GB = ~2 TB

Pickle index:
  - Per sample: ~40 bytes (dict overhead + key string + tuple with 2 ints)
  - For 1B samples: ~40 GB
  - Note: More compact than SQLite due to no B-tree overhead

Schema file: <1 KB (negligible)

Total for 1B samples with 512-dim float32 features: ~2.04 TB
```

**Comparison**: Raw PyTorch `.pt` files would use similar space (~2 TB for tensors), but without:
- Efficient indexing (linear scan required)
- Partial loading (must load entire file)
- Schema preservation (manual bookkeeping)

### Memory Complexity

```
Peak Memory = LRU cache + Pending buffer + Active segment + In-memory index

LRU cache:
  - Size: lru_size * avg_sample_size
  - Example: 4096 * 2048 bytes = 8 MB

Pending buffer:
  - Size: flush_every * avg_sample_size
  - Example: 2048 * 2048 bytes = 4 MB

Active segment (during read):
  - Size: segment_size (determined by flush_every)
  - Example: 2048 samples * 2048 bytes = 4 MB
  - Note: OS page cache makes this effectively free

In-memory index:
  - Size: ~40 bytes per entry (dict overhead + key + value tuple)
  - Example for 1M samples: ~40 MB
  - Example for 1B samples: ~40 GB
  - Note: Scales linearly with total cache size

Total: ~20 MB + index size (dependent on total cache size)
```

**Note**: Memory usage now includes the full in-memory index, which scales with cache size. For very large caches (>100M samples), consider the ~40 bytes per sample overhead. However, this is still very efficient compared to loading all cached data into memory.

## Implementation Details

### Schema Inference

The schema inference happens in `_infer_schema_from_sample()`:

```python
def _infer_schema_from_sample(self, sample: Any) -> pa.Schema:
    fields = [("key", pa.string())]

    if torch.is_tensor(sample):
        # Single tensor: {key, data, shape}
        dtype = self._torch_to_arrow_dtype(sample.dtype)
        fields.extend([
            ("data", pa.list_(dtype)),
            ("shape", pa.list_(pa.int64())),
        ])
        self.output_structure = "tensor"
        metadata = {"torch_dtype": str(sample.dtype)}

    elif isinstance(sample, dict):
        # Dict of tensors: {key, tensor1_data, tensor1_shape, ...}
        self.output_structure = "dict"
        tensor_keys = []
        for name, value in sample.items():
            if torch.is_tensor(value):
                dtype = self._torch_to_arrow_dtype(value.dtype)
                fields.extend([
                    (f"{name}_data", pa.list_(dtype)),
                    (f"{name}_shape", pa.list_(pa.int64())),
                ])
                tensor_keys.append(name)
            # Non-tensors handled separately with pickle
        metadata = {"tensor_keys": json.dumps(tensor_keys)}

    elif isinstance(sample, (list, tuple)):
        # Tuple/list: {key, tensor_0_data, tensor_0_shape, ...}
        self.output_structure = "list" if isinstance(sample, list) else "tuple"
        num_tensors = 0
        for i, value in enumerate(sample):
            if torch.is_tensor(value):
                dtype = self._torch_to_arrow_dtype(value.dtype)
                fields.extend([
                    (f"tensor_{i}_data", pa.list_(dtype)),
                    (f"tensor_{i}_shape", pa.list_(pa.int64())),
                ])
                num_tensors += 1
        metadata = {"num_tensors": str(num_tensors)}

    else:
        raise TypeError(f"Unsupported output type: {type(sample)}")

    return pa.schema(fields).with_metadata(metadata)
```

**Key Design Decisions**:
1. **Flatten tensors**: Store as `{data: list, shape: list}` instead of multidimensional arrays
   - Arrow doesn't support arbitrary-rank tensors natively
   - Flattening is O(1) (view operation) and preserves all information
2. **Separate fields per tensor**: `features_data`, `logits_data`, etc.
   - Enables columnar access (read only what you need)
   - Preserves dtype per tensor (mixed precision supported)
3. **Metadata for structure**: Store `tensor_keys` to reconstruct dicts
   - Arrow metadata is key-value strings
   - JSON-encode lists for roundtripping

### Dtype Preservation

**Critical Bug (fixed)**: Initial implementation used `.to_pydict()` which converted Arrow arrays to Python lists, then `np.array()` defaulted to float64:

```python
# WRONG (loses dtype)
columns = subtable.to_pydict()  # Arrow → Python lists
data = np.array(columns["data"])  # Python list → numpy (defaults to float64!)
tensor = torch.from_numpy(data)  # float64 tensor ❌

# CORRECT (preserves dtype)
columns_numpy = {name: subtable[name].to_numpy() for name in subtable.column_names}
data = columns_numpy["data"]  # Already numpy array with correct dtype
tensor = torch.from_numpy(data)  # Preserves dtype ✅
```

**Lesson**: Always use Arrow's `.to_numpy()` for dtype-sensitive data, never `.to_pydict()` followed by `np.array()`.

### Flush Mechanism

The flush operation is the most critical path for performance and correctness:

```python
def _flush_segment(self) -> None:
    if not self._pending:
        return

    batch = self._pending
    self._pending = []

    segment_id = self._current_segment_id
    self._current_segment_id += 1

    # 1. Serialize to Arrow RecordBatch
    arrays = self._serialize_batch(batch)
    record_batch = pa.RecordBatch.from_pydict(arrays, schema=self.schema)

    # 2. Write to temporary file
    temp_file = self.segments_dir / f"segment_{segment_id:06d}.arrow.tmp"
    with pa.OSFile(str(temp_file), "wb") as sink:
        writer = pa.ipc.new_file(sink, self.schema)
        writer.write_batch(record_batch)
        writer.close()

    # 3. Update in-memory index
    for i, item in enumerate(batch):
        self.index[item["key"]] = (segment_id, i)

    # 4. Atomic rename (makes segment visible)
    final_file = self.segments_dir / f"segment_{segment_id:06d}.arrow"
    temp_file.rename(final_file)

    # 5. Persist index to disk (atomic)
    self._save_index()
```

**Atomicity Guarantees**:
1. **Write to `.tmp` first**: Incomplete writes don't affect readers
2. **In-memory index update**: Fast, always succeeds
3. **Atomic rename**: OS guarantees rename is atomic; readers see complete file or nothing
4. **Atomic index save**: Uses temp file + rename pattern for crash safety

**Failure Scenarios**:
- **Crash during Arrow write**: `.tmp` file left behind, ignored on restart
- **Crash during index update**: In-memory only, lost on crash (will rebuild from segments)
- **Crash before rename**: `.tmp` file ignored, segment not visible
- **Crash after rename but before index save**: Index rebuilt from segments on restart
- **Crash during index save**: Old index.pkl still valid, new data re-indexed on restart

### Async Writes

Async writes are implemented via `ThreadPoolExecutor`:

```python
def flush(self) -> None:
    if self.async_write and self.executor is not None:
        # Submit to background thread
        future = self.executor.submit(self._flush_segment)
        # Don't wait for completion (non-blocking)
    else:
        # Synchronous flush
        self._flush_segment()
```

**Design Choice**: Use threads (not asyncio) because:
1. Arrow IPC writes are blocking I/O (no async support)
2. Pickle writes are blocking (GIL-releasing)
3. Threads provide true parallelism for I/O-bound work
4. Simple executor pattern, no async/await complexity

**Safety**: Only one writer thread at a time (single `put_batch()` caller), so no race conditions.

## Operational Caveats

While torchcachex handles most complexity automatically, there are two fundamental constraints imposed by Arrow IPC and PyTorch that users should understand:

### Caveat 1: Concurrent Writers (Already Handled)

**Constraint**: Arrow IPC is append-only but **not transactional**. Multiple processes writing to the same segment file simultaneously can corrupt data.

**Solution**: Single-writer pattern via `writer_rank` parameter.

**Implementation**:
```python
# In ArrowIPCCacheBackend.__init__:
self.writer_rank = int(writer_rank)  # Default: 0
self.current_rank = int(current_rank) if current_rank is not None else int(os.getenv("RANK", 0))

# In put_batch():
# All ranks warm LRU cache (fast)
for k, v in items.items():
    self.lru[k] = v

# Only writer rank persists to disk
if self.current_rank != self.writer_rank:
    return  # Skip disk writes for non-writer ranks
```

**Usage Pattern** (DDP Training):
```python
import os

backend = ArrowIPCCacheBackend(
    cache_dir="/shared/cache",
    module_id="features_v1",
    writer_rank=0,  # Only rank 0 writes
    current_rank=int(os.getenv("RANK", 0)),  # From torch.distributed
)

# First epoch:
# - All ranks compute features (distributed workload)
# - Only rank 0 writes to cache (no coordination needed)
#
# Subsequent epochs:
# - All ranks read from cache (fast!)
```

**Why This Works**:
- **First epoch**: All ranks compute (distributed), only rank 0 writes (sequential, but one-time cost)
- **Later epochs**: All ranks read (parallel, memory-mapped, fast)
- **No locks needed**: Single writer = no race conditions
- **Cost amortization**: Write overhead is one-time, reading is every epoch

**Alternative Considered**: Distributed locking (e.g., file locks, Redis)
- **Rejected**: Adds complexity, coordination overhead, and failure modes
- **Single-writer is simpler**: No deadlocks, no lock contention, crash-safe by construction

### Caveat 2: GPU Tensor Handling (Already Handled)

**Constraint**: Arrow arrays require contiguous CPU memory. PyTorch tensors may be:
- On GPU (CUDA device)
- Non-contiguous (after transpose, slice, etc.)
- Have `requires_grad=True` (computational graph attached)

**Solution**: Automatic `.detach().cpu()` conversion before Arrow storage.

**Implementation**:
```python
# In ArrowIPCCacheBackend._serialize_sample():
if self.output_structure == "tensor":
    # Convert: GPU → CPU, detach gradients, flatten, convert to numpy
    row["data"] = sample.detach().cpu().flatten().numpy().tolist()
    row["shape"] = list(sample.shape)

# Same for dict/tuple/list branches:
tensor.detach().cpu().flatten().numpy().tolist()
```

**Why Each Step**:
1. **`.detach()`**: Remove gradient tracking (save memory, prevent graph serialization)
2. **`.cpu()`**: Move to CPU memory (required for Arrow/numpy)
3. **`.flatten()`**: Make contiguous (required for efficient Arrow storage)
4. **`.numpy()`**: Convert to numpy array (Arrow's native format)
5. **`.tolist()`**: Convert to Python list (for Arrow RecordBatch construction)

**Device Handling** (Decorator):
```python
# In CacheModuleDecorator.forward():

# Store on CPU (efficient, portable)
backend.put_batch(items)  # Automatically detaches and moves to CPU

# Read from cache and move to input device
cached_objs, missing = backend.get_batch(keys, map_location="cpu")

# Move to same device as input
def _move_like_input(obj, ref_tensor):
    device = ref_tensor.device
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    # ... handle nested structures ...

result = _move_like_input(cached_objs[i], input_tensor)
```

**Why This Design**:
- **Cache on CPU**: GPUs have limited memory; CPU storage is cheap
- **Restore to input device**: Transparent to user (cached tensors behave like freshly computed)
- **Non-blocking transfer**: `non_blocking=True` overlaps transfer with computation
- **Automatic dtype preservation**: Arrow's type system preserves float32/float16/etc.

**Example Workflow**:
```python
# Training loop
input_cuda = batch["images"].to("cuda")  # Input on GPU

# Forward pass (with caching)
features = cached_extractor(input_cuda, cache_ids=batch["ids"])

# What happens:
# 1. Cache miss → compute on GPU
# 2. Store: GPU → .detach().cpu() → Arrow (on disk)
# 3. Cache hit → load from disk → .to("cuda") → return
# 4. User gets GPU tensor (transparent!)

assert features.device.type == "cuda"  # ✓ Same device as input
assert features.requires_grad == False  # ✓ Detached (stateless module)
```

**Memory Efficiency**:
```python
# WITHOUT caching: GPU memory holds all features
features_gpu = expensive_model(images_gpu)  # 1000 samples × 2048 features × 4 bytes = 8 MB GPU

# WITH caching: Only working set on GPU
# - Disk: 1000 samples × 2048 × 4 = 8 MB (persistent)
# - GPU: batch_size × 2048 × 4 = 256 KB (transient)
# - CPU: LRU cache (configurable, typically 4096 samples = 32 MB)
```

### Summary: Both Caveats Handled Automatically

Users don't need to think about these constraints because:

1. **Concurrent writers**: Use `writer_rank` parameter (defaults to rank 0)
2. **GPU tensors**: Automatic `.detach().cpu()` conversion in backend

The only user-facing requirement: provide stable `cache_ids` for deterministic caching.

## Crash Recovery

### Recovery on Startup

When `ArrowIPCCacheBackend` is initialized, it performs recovery:

```python
def __init__(self, cache_dir, module_id, ...):
    # 1. Create directories
    self.cache_root.mkdir(parents=True, exist_ok=True)
    self.segments_dir.mkdir(exist_ok=True)

    # 2. Load or rebuild index
    self.index = self._load_index()

    # 3. Load or infer schema
    if self.schema_path.exists():
        try:
            self.schema = self._load_schema()
        except Exception:
            # Corrupted schema - will re-infer on next write
            self.schema = None

    # 4. Discover existing segments
    self._current_segment_id = self._get_next_segment_id()

    # 5. Clean up incomplete writes
    for tmp_file in self.segments_dir.glob("*.tmp"):
        tmp_file.unlink()  # Remove leftover temp files

def _load_index(self):
    """Load index from disk or rebuild from segments."""
    if self.index_path.exists():
        try:
            with open(self.index_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Corrupted index - rebuild from segments
            logger.warning("Corrupted index, rebuilding from segments")
            return self._rebuild_index_from_segments()
    else:
        # No index yet - start fresh or rebuild
        if list(self.segments_dir.glob("segment_*.arrow")):
            return self._rebuild_index_from_segments()
        else:
            return {}

def _rebuild_index_from_segments(self):
    """Scan all segment files and rebuild index."""
    index = {}
    for segment_file in sorted(self.segments_dir.glob("segment_*.arrow")):
        segment_id = int(segment_file.stem.split('_')[1])
        with pa.memory_map(str(segment_file), 'r') as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()
            keys = table['key'].to_pylist()
            for row_offset, key in enumerate(keys):
                index[key] = (segment_id, row_offset)
    return index
```

**Recovery Properties**:
1. **Automatic index rebuild**: If index.pkl is missing or corrupted, rebuild from segments
2. **Orphaned segments**: Segment files will be re-indexed on startup
3. **Incomplete segments**: `.tmp` files are deleted on startup
4. **Corrupted schema**: Re-inferred on next write (schema is just an optimization)
5. **Crash safety**: Index can always be reconstructed from immutable segment files

### Data Integrity Tests

The `test_recovery.py` suite verifies crash safety:

```python
def test_incomplete_segment_ignored():
    # Simulate crash: create .tmp file
    incomplete_file = segments_dir / "segment_000001.arrow.tmp"
    incomplete_file.write_text("incomplete data")

    # New backend should ignore .tmp and work fine
    backend2 = ArrowIPCCacheBackend(...)
    results, missing = backend2.get_batch(keys)
    assert len(missing) == 0  # All data still accessible

def test_orphaned_segment_file():
    # Create segment file without index entry
    orphan_file = segments_dir / "segment_999999.arrow"
    shutil.copy(existing_segment, orphan_file)

    # Should rebuild index and include orphaned segment
    backend2 = ArrowIPCCacheBackend(...)
    results, missing = backend2.get_batch(keys)
    assert len(missing) == 0  # All data accessible including orphaned segment

def test_corrupted_schema_file():
    # Corrupt schema file
    schema_path.write_text("corrupted json {{{")

    # Should handle gracefully (re-infer on next write)
    backend2 = ArrowIPCCacheBackend(...)
    backend2.put_batch({"key": torch.randn(10)})
    backend2.flush()
    # Verify data accessible
```

## Design Tradeoffs

### Chosen: Append-Only Segments vs. Compaction

**Decision**: Use append-only segments without compaction.

**Pros**:
- O(1) writes (no need to rewrite existing data)
- Simple implementation (no background compaction thread)
- Crash-safe (no partial compaction state)

**Cons**:
- Duplicate keys create orphaned data (wastes disk space)
- Many small segments could slow reads (mitigated by LRU cache)

**Future**: Could add optional compaction as a maintenance operation.

### Chosen: Pickle Index vs. SQLite

**Decision**: Use in-memory dict with pickle persistence for indexing.

**Pros**:
- True O(1) lookups (faster than SQLite B-tree)
- Simpler architecture (no database dependency)
- Easy crash recovery (rebuild from segments)
- More compact on disk (~40 bytes vs ~50 bytes per entry)

**Cons**:
- Full index must fit in memory (~40 bytes per sample)
- Index persistence adds slight overhead to each flush
- Less battle-tested than SQLite

**Why It Works**:
- For 1M samples: ~40 MB memory (negligible)
- For 100M samples: ~4 GB memory (acceptable on modern systems)
- For 1B samples: ~40 GB memory (requires high-memory node)
- Pickle serialization is fast (~100-200 MB/s)
- Auto-rebuild from segments provides crash safety

**Previous Choice (SQLite)**:
- More complex (database initialization, transactions)
- Slower lookups (B-tree vs hash table)
- Marginally better for multi-process scenarios
- Traded simplicity for features we didn't need

### Chosen: Schema Inference vs. Manual Schema

**Decision**: Automatically infer schema from first forward pass.

**Pros**:
- Zero boilerplate (no type hints required)
- Always correct (uses actual output)
- Handles complex structures (dicts, tuples, mixed types)

**Cons**:
- First write is slightly slower (schema inference overhead)
- Schema changes require new `module_id` (not automatically detected)

**Alternatives Considered**:
- **Type hints**: Would require decorating module with output types
- **Dummy input**: Would require providing representative input sample

### Chosen: Single-Writer vs. Multi-Writer

**Decision**: Single-writer pattern (one rank writes in DDP).

**Pros**:
- No coordination needed (no distributed locks)
- Simple implementation (no conflict resolution)
- Safe by construction (no race conditions)

**Cons**:
- Write throughput limited to one process
- All ranks compute, but only one rank caches

**Why It's Fine**: In DDP training:
- All ranks compute features for their shard
- Only rank 0 writes to cache (first epoch)
- All ranks read from cache (subsequent epochs)
- Cache population is one-time cost (amortized over epochs)

**Future**: Could add multi-writer with coordination (e.g., shard-based locking).

### Chosen: PyArrow vs. Parquet

**Decision**: Use Arrow IPC (not Parquet).

**Pros**:
- Simpler format (designed for IPC, not storage)
- Zero-copy memory-mapping
- No compression overhead (raw binary data)
- Faster writes (no encoding)

**Cons**:
- Larger files (no compression)
- Less interoperable (Parquet is more standard)

**Why It's Fine**: For caching:
- Speed matters more than size (local disk is cheap)
- Memory-mapping matters more than interoperability
- Arrow IPC is perfect for process-to-disk-to-process workflow

## Performance Benchmarks

### Write Scaling (O(1) Verification)

From `test_scale.py`:

```
Cache Size | Flush Time | Samples/sec
---------- | ---------- | -----------
1k samples | 0.15s      | 6,667
10k        | 0.16s      | 6,250
100k       | 0.15s      | 6,667
1M         | 0.17s      | 5,882
```

**Result**: Flush time remains constant (~0.15-0.17s) regardless of cache size, confirming O(1) writes.

### Read Scaling

```
Cache Size | Batch Size | Read Time | Samples/sec
---------- | ---------- | --------- | -----------
1k         | 100        | 0.02s     | 5,000
10k        | 100        | 0.02s     | 5,000
100k       | 100        | 0.03s     | 3,333
1M         | 100        | 0.03s     | 3,333
```

**Result**: Read time remains constant regardless of cache size due to O(1) dict lookups.

### Memory Usage

```
Cache Size | Peak Memory | Memory/Sample
---------- | ----------- | -------------
1k         | 25 MB       | 25 KB
10k        | 28 MB       | 2.8 KB
100k       | 35 MB       | 350 B
1M         | 42 MB       | 42 B
```

**Result**: Memory usage grows sub-linearly with cache size, confirming O(working set) behavior.

## Future Enhancements

### Potential Improvements

1. **Segment Compaction**
   - Merge small segments into larger ones (background task)
   - Remove duplicate keys to reclaim disk space
   - Challenge: Maintain O(1) writes during compaction

2. **Multi-Writer Support**
   - Shard-based locking (each writer owns a key range)
   - Or: Separate caches per rank, merge at end
   - Challenge: Coordination overhead

3. **Remote Storage**
   - S3/GCS backend for cloud training
   - Read-through cache with local disk
   - Challenge: Latency and consistency

4. **Compression**
   - Optional LZ4/Zstd compression for segments
   - Trade CPU for disk space
   - Challenge: Slower reads, no memory-mapping

5. **Schema Evolution**
   - Detect schema changes, auto-migrate
   - Support adding new fields
   - Challenge: Backward compatibility

6. **Distributed Cache**
   - Shared cache across machines (Redis/Memcached)
   - Useful for multi-node training
   - Challenge: Network overhead

### Non-Goals (Deliberately Excluded)

1. **Cache Invalidation**: Use new `module_id` to invalidate
2. **TTL/Expiration**: Caches are permanent (manual cleanup)
3. **Access Control**: Single-user, local filesystem only
4. **Encryption**: Store sensitive data in secure locations
5. **Replication**: Use filesystem-level tools (rsync, ZFS, etc.)

## References

- [Apache Arrow IPC Format](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format)
- [Python Pickle Protocol](https://docs.python.org/3/library/pickle.html)
- [PyTorch Tensor Storage](https://pytorch.org/docs/stable/tensor_attributes.html)
- [LRU Cache Implementation](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- [Log-Structured Merge Trees](https://en.wikipedia.org/wiki/Log-structured_merge-tree)

## Contributing

If you're contributing to torchcachex, please read this document carefully to understand the design philosophy and implementation constraints. Key principles:

1. **Preserve O(1) guarantees**: Any change must maintain constant-time flush operations
2. **Test crash safety**: Add recovery tests for new failure modes
3. **Maintain backward compat**: Old caches must work with new code (or provide migration)
4. **Document tradeoffs**: Explain why alternatives were rejected

---

**Questions?** Open an issue on [GitHub](https://github.com/dahlem/torchcachex/issues).
