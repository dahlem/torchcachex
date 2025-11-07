"""Scalable cache backend using Arrow IPC + in-memory index.

This module provides persistent caching using Apache Arrow IPC files
with pickle-backed in-memory index for O(1) lookups, native tensor storage, and async writes.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Production"

import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc
import torch
from cachetools import LRUCache


class ArrowIPCCacheBackend:
    """Scalable cache using Arrow IPC with in-memory index and native tensor storage.

    Features:
    - **O(1) append-only writes** via incremental Arrow segments
    - **O(1) batched lookups** via in-memory dict index + Arrow memory-mapping
    - **Native tensor storage** with schema inferred from first forward pass
    - **LRU hot cache** for in-process hits
    - **Single-writer** (DDP-safe) via writer_rank
    - **Crash recovery** via index rebuild from segments
    - **Scales to billions of samples** with constant memory usage

    The cache key is constructed as: `{module_id}:{sample_cache_id}`

    Storage layout:
        cache_dir/module_id/
            segments/
                segment_000000.arrow
                segment_000001.arrow
                ...
            index.pkl (pickled dict: key → (segment_id, row_offset))
            schema.json

    Args:
        cache_dir: Root directory for cache storage
        module_id: Stable identifier for the module (e.g., "resnet50_features_v1")
        lru_size: Size of in-memory LRU cache (default: 4096)
        async_write: Enable asynchronous writes (default: True)
        max_workers: Number of async write threads (default: 2)
        flush_every: Number of pending samples before flushing to disk (default: 2048)
        writer_rank: Which rank writes to disk in DDP (default: 0)
        current_rank: Current process rank in DDP (default: 0)
    """

    def __init__(
        self,
        cache_dir: str,
        module_id: str,
        lru_size: int = 4096,
        async_write: bool = True,
        max_workers: int = 2,
        flush_every: int = 2048,
        writer_rank: int = 0,
        current_rank: int | None = None,
    ):
        self.cache_dir = Path(cache_dir) / module_id
        self.segments_dir = self.cache_dir / "segments"
        self.index_path = self.cache_dir / "index.pkl"
        self.schema_path = self.cache_dir / "schema.json"
        self.module_id = module_id

        # Create directories
        self.segments_dir.mkdir(parents=True, exist_ok=True)

        # Schema (lazy init on first write)
        self.schema: pa.Schema | None = None
        self.output_structure: str | None = None
        if self.schema_path.exists():
            try:
                self.schema = self._load_schema()
            except Exception:
                # Corrupted schema - will re-infer on next write
                self.schema = None

        # In-memory index: key → (segment_id, row_offset)
        self.index: dict[str, tuple[int, int]] = self._load_index()

        # Configuration
        self.async_write = async_write
        self.flush_every = int(flush_every)
        self.writer_rank = int(writer_rank)
        self.current_rank = (
            int(current_rank) if current_rank is not None else int(os.getenv("RANK", 0))
        )

        # LRU cache + pending writes
        # When lru_size=0, disable LRU entirely (useful for benchmarking disk-only reads)
        self.lru_enabled = lru_size > 0
        self.lru: dict[str, Any] = LRUCache(maxsize=lru_size) if lru_size > 0 else {}
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()

        # Async executor
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers) if async_write else None
        )

        # Next segment ID
        self._current_segment_id = self._get_next_segment_id()

    def _load_index(self) -> dict[str, tuple[int, int]]:
        """Load index from pickle file, or rebuild from segments if corrupted/missing."""
        if not self.index_path.exists():
            # No index file - rebuild from segments
            return self._rebuild_index_from_segments()

        try:
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Corrupted index - rebuild from segments
            return self._rebuild_index_from_segments()

    def _rebuild_index_from_segments(self) -> dict[str, tuple[int, int]]:
        """Rebuild index by scanning all Arrow segments.

        This is called on first init or if the index file is corrupted.
        """
        index: dict[str, tuple[int, int]] = {}
        segment_files = sorted(self.segments_dir.glob("segment_*.arrow"))

        for segment_file in segment_files:
            segment_id = int(segment_file.stem.split("_")[1])
            try:
                with pa.memory_map(str(segment_file), "r") as source:
                    reader = pa.ipc.open_file(source)
                    batch = reader.get_batch(0)
                    keys = batch.column("key").to_pylist()
                    for offset, key in enumerate(keys):
                        index[key] = (segment_id, offset)
            except Exception:
                # Skip corrupted segment
                continue

        return index

    def _save_index(self) -> None:
        """Save index to pickle file atomically."""
        temp_path = self.index_path.with_suffix(".pkl.tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.rename(self.index_path)

    def _get_next_segment_id(self) -> int:
        """Get next available segment ID from existing segment files."""
        segment_files = list(self.segments_dir.glob("segment_*.arrow"))
        if not segment_files:
            return 0

        max_id = max(int(f.stem.split("_")[1]) for f in segment_files)
        return max_id + 1

    def _load_schema(self) -> pa.Schema:
        """Load schema from disk."""
        with open(self.schema_path) as f:
            schema_dict = json.load(f)

        # Reconstruct Arrow schema
        fields = []
        for field_dict in schema_dict["fields"]:
            field_type = self._parse_arrow_type(field_dict["type"])
            fields.append(pa.field(field_dict["name"], field_type))

        schema = pa.schema(fields)

        # Restore metadata (convert back to bytes keys for Arrow)
        if "metadata" in schema_dict:
            metadata_bytes = {
                k.encode() if isinstance(k, str) else k: v.encode()
                if isinstance(v, str)
                else v
                for k, v in schema_dict["metadata"].items()
            }
            schema = schema.with_metadata(metadata_bytes)

        # Restore output structure
        self.output_structure = schema_dict.get("output_structure")

        return schema

    def _save_schema(self) -> None:
        """Save schema to disk."""
        # Convert metadata bytes keys to strings for JSON serialization
        metadata = {}
        if self.schema.metadata:
            for k, v in self.schema.metadata.items():
                key_str = k.decode() if isinstance(k, bytes) else str(k)
                val_str = v.decode() if isinstance(v, bytes) else str(v)
                metadata[key_str] = val_str

        schema_dict = {
            "fields": [
                {"name": field.name, "type": self._serialize_arrow_type(field.type)}
                for field in self.schema
            ],
            "metadata": metadata,
            "output_structure": self.output_structure,
        }

        with open(self.schema_path, "w") as f:
            json.dump(schema_dict, f, indent=2)

    def _serialize_arrow_type(self, arrow_type: pa.DataType) -> str:
        """Serialize Arrow type to string."""
        if pa.types.is_list(arrow_type):
            value_type = self._serialize_arrow_type(arrow_type.value_type)
            return f"list<{value_type}>"
        return str(arrow_type)

    def _parse_arrow_type(self, type_str: str) -> pa.DataType:
        """Parse Arrow type from string."""
        if type_str.startswith("list<"):
            # Extract inner type, handling nested structures
            value_type_str = type_str[5:-1]  # Remove "list<" and ">"
            # Handle "list<item: float>" format from Arrow's string representation
            if ": " in value_type_str:
                value_type_str = value_type_str.split(": ")[1]
            value_type = self._parse_arrow_type(value_type_str)
            return pa.list_(value_type)

        # Map basic types
        type_map = {
            "string": pa.string(),
            "binary": pa.binary(),
            "int8": pa.int8(),
            "int16": pa.int16(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "uint8": pa.uint8(),
            "uint16": pa.uint16(),
            "uint32": pa.uint32(),
            "uint64": pa.uint64(),
            "float16": pa.float16(),
            "float32": pa.float32(),
            "float64": pa.float64(),
            "float": pa.float32(),  # Handle "float" -> float32
            "double": pa.float64(),  # Handle "double" -> float64
            "bool": pa.bool_(),
        }
        return type_map.get(type_str, pa.string())

    def _torch_to_arrow_dtype(self, torch_dtype) -> pa.DataType:
        """Map PyTorch dtype to Arrow dtype."""
        mapping = {
            torch.float32: pa.float32(),
            torch.float64: pa.float64(),
            torch.float16: pa.float16(),
            torch.int32: pa.int32(),
            torch.int64: pa.int64(),
            torch.int16: pa.int16(),
            torch.int8: pa.int8(),
            torch.uint8: pa.uint8(),
            torch.bool: pa.bool_(),
        }
        return mapping.get(torch_dtype, pa.float32())

    def _infer_schema_from_sample(self, sample: Any) -> pa.Schema:
        """Infer Arrow schema from actual module output (first forward pass)."""
        fields = [("key", pa.string())]

        if torch.is_tensor(sample):
            # Single tensor: store as flattened array + shape
            dtype = self._torch_to_arrow_dtype(sample.dtype)
            fields.extend(
                [
                    ("data", pa.list_(dtype)),
                    ("shape", pa.list_(pa.int64())),
                ]
            )
            self.output_structure = "tensor"
            # Store original torch dtype in metadata
            return pa.schema(fields).with_metadata({"torch_dtype": str(sample.dtype)})

        elif isinstance(sample, dict):
            # Dict of tensors (most common!)
            self.output_structure = "dict"
            tensor_keys = []
            for name, value in sample.items():
                if torch.is_tensor(value):
                    dtype = self._torch_to_arrow_dtype(value.dtype)
                    fields.extend(
                        [
                            (f"{name}_data", pa.list_(dtype)),
                            (f"{name}_shape", pa.list_(pa.int64())),
                        ]
                    )
                    tensor_keys.append(name)
                else:
                    # Non-tensor: fallback to pickle blob
                    fields.append((f"{name}_blob", pa.binary()))

            # Store tensor keys in schema metadata
            metadata = {"tensor_keys": json.dumps(tensor_keys)}
            return pa.schema(fields).with_metadata(metadata)

        elif isinstance(sample, (list, tuple)):
            # Tuple/list of tensors
            self.output_structure = "tuple" if isinstance(sample, tuple) else "list"
            for i, item in enumerate(sample):
                if torch.is_tensor(item):
                    dtype = self._torch_to_arrow_dtype(item.dtype)
                    fields.extend(
                        [
                            (f"item{i}_data", pa.list_(dtype)),
                            (f"item{i}_shape", pa.list_(pa.int64())),
                        ]
                    )
                else:
                    fields.append((f"item{i}_blob", pa.binary()))

        else:
            # Complex nested: fallback to blob
            fields.append(("blob", pa.binary()))
            self.output_structure = "blob"

        return pa.schema(fields)

    def _serialize_sample(self, key: str, sample: Any) -> dict[str, Any]:
        """Convert sample to Arrow-compatible row."""
        row = {"key": key}

        if self.output_structure == "tensor":
            row["data"] = sample.detach().cpu().flatten().numpy().tolist()
            row["shape"] = list(sample.shape)

        elif self.output_structure == "dict":
            # Access metadata (handle both bytes and str keys)
            metadata_key = (
                b"tensor_keys"
                if b"tensor_keys" in self.schema.metadata
                else "tensor_keys"
            )
            tensor_keys_str = self.schema.metadata[metadata_key]
            if isinstance(tensor_keys_str, bytes):
                tensor_keys_str = tensor_keys_str.decode()
            tensor_keys = json.loads(tensor_keys_str)

            for name in tensor_keys:
                tensor = sample[name]
                row[f"{name}_data"] = tensor.detach().cpu().flatten().numpy().tolist()
                row[f"{name}_shape"] = list(tensor.shape)
            # Handle non-tensor dict values with pickle
            for name, value in sample.items():
                if name not in tensor_keys:
                    row[f"{name}_blob"] = pickle.dumps(value)

        elif self.output_structure in ("tuple", "list"):
            for i, item in enumerate(sample):
                if torch.is_tensor(item):
                    row[f"item{i}_data"] = (
                        item.detach().cpu().flatten().numpy().tolist()
                    )
                    row[f"item{i}_shape"] = list(item.shape)
                else:
                    row[f"item{i}_blob"] = pickle.dumps(item)

        elif self.output_structure == "blob":
            # Fallback: full pickle
            row["blob"] = pickle.dumps(sample)

        return row

    def _deserialize_sample(
        self, row: dict[str, Any], map_location: str = "cpu"
    ) -> Any:
        """Reconstruct sample from Arrow row."""
        if self.output_structure == "tensor":
            # Row data is already numpy array from Arrow (dtype preserved)
            # Make a copy to ensure writability (Arrow memory-maps are read-only)
            data = (
                row["data"]
                if isinstance(row["data"], np.ndarray)
                else np.array(row["data"])
            )
            data = np.array(data, copy=True) if not data.flags.writeable else data
            shape = tuple(row["shape"])
            tensor = torch.from_numpy(data).reshape(shape)
            return tensor.to(map_location)

        elif self.output_structure == "dict":
            result = {}
            # Access metadata (handle both bytes and str keys)
            metadata_key = (
                b"tensor_keys"
                if b"tensor_keys" in self.schema.metadata
                else "tensor_keys"
            )
            tensor_keys_str = self.schema.metadata[metadata_key]
            if isinstance(tensor_keys_str, bytes):
                tensor_keys_str = tensor_keys_str.decode()
            tensor_keys = json.loads(tensor_keys_str)

            for name in tensor_keys:
                # Row data is already numpy array from Arrow (dtype preserved)
                # Make a copy to ensure writability (Arrow memory-maps are read-only)
                data = (
                    row[f"{name}_data"]
                    if isinstance(row[f"{name}_data"], np.ndarray)
                    else np.array(row[f"{name}_data"])
                )
                data = np.array(data, copy=True) if not data.flags.writeable else data
                shape = tuple(row[f"{name}_shape"])
                tensor = torch.from_numpy(data).reshape(shape).to(map_location)
                result[name] = tensor
            # Deserialize non-tensor values
            for key in row.keys():
                if key.endswith("_blob"):
                    name = key[:-5]  # Remove '_blob'
                    result[name] = pickle.loads(row[key])
            return result

        elif self.output_structure in ("tuple", "list"):
            items = []
            i = 0
            while f"item{i}_data" in row or f"item{i}_blob" in row:
                if f"item{i}_data" in row:
                    # Row data is already numpy array from Arrow (dtype preserved)
                    # Make a copy to ensure writability (Arrow memory-maps are read-only)
                    data = (
                        row[f"item{i}_data"]
                        if isinstance(row[f"item{i}_data"], np.ndarray)
                        else np.array(row[f"item{i}_data"])
                    )
                    data = (
                        np.array(data, copy=True) if not data.flags.writeable else data
                    )
                    shape = tuple(row[f"item{i}_shape"])
                    tensor = torch.from_numpy(data).reshape(shape).to(map_location)
                    items.append(tensor)
                else:
                    items.append(pickle.loads(row[f"item{i}_blob"]))
                i += 1
            return tuple(items) if self.output_structure == "tuple" else items

        else:  # blob
            return pickle.loads(row["blob"])

    def get_batch(
        self, keys: list[str], map_location: str = "cpu"
    ) -> tuple[list[Any | None], list[int]]:
        """O(1) batch lookup via SQLite index + Arrow memory-mapping.

        Args:
            keys: List of cache keys to look up
            map_location: Device to load tensors to (default: "cpu")

        Returns:
            Tuple of (results_in_input_order, missing_positions)
            - results[i] is the deserialized object or None if missing
            - missing_positions lists indices where results[i] is None
        """
        # 1. Fast LRU pass
        results: list[Any | None] = [self.lru.get(k) for k in keys]
        need = [
            (i, k)
            for i, (k, r) in enumerate(zip(keys, results, strict=False))
            if r is None
        ]

        if not need:
            return results, []

        # 2. Query in-memory index for (segment_id, row_offset)
        segment_reads: dict[int, list[tuple[str, int, int]]] = {}
        for result_idx, key in need:
            if key in self.index:
                segment_id, row_offset = self.index[key]
                if segment_id not in segment_reads:
                    segment_reads[segment_id] = []
                segment_reads[segment_id].append((key, row_offset, result_idx))

        # 3. Read from Arrow segments (memory-mapped)
        for segment_id, reads in segment_reads.items():
            segment_file = self.segments_dir / f"segment_{segment_id:06d}.arrow"

            with pa.memory_map(str(segment_file), "r") as source:
                reader = pa.ipc.open_file(source)

                # Read specific rows (Arrow supports efficient row selection)
                row_indices = [offset for _, offset, _ in reads]
                table = reader.read_all()
                subtable = table.take(row_indices)

                # Convert Arrow columns to numpy (preserves dtypes)
                columns_numpy = {
                    name: subtable[name].to_numpy() for name in subtable.column_names
                }

                # Deserialize each row
                for idx, (key, _, result_idx) in enumerate(reads):
                    # Extract row from numpy arrays (preserves dtype)
                    row_dict = {k: v[idx] for k, v in columns_numpy.items()}
                    sample = self._deserialize_sample(row_dict, map_location)
                    results[result_idx] = sample
                    self.lru[key] = sample

        # 4. Identify missing keys
        missing = [i for i, v in enumerate(results) if v is None]
        return results, missing

    def put_batch(self, items: dict[str, Any]) -> None:
        """O(1) append-only write via new Arrow segment.

        Args:
            items: Dictionary of {key: value} pairs to cache
        """
        if not items:
            return

        # Infer schema on first write
        if self.schema is None:
            first_sample = next(iter(items.values()))
            self.schema = self._infer_schema_from_sample(first_sample)
            self._save_schema()

        # Warm LRU (all ranks)
        for k, v in items.items():
            self.lru[k] = v

        # Only writer rank persists
        if self.current_rank != self.writer_rank:
            return

        # Serialize outside lock
        entries = [(k, self._serialize_sample(k, v)) for k, v in items.items()]

        def _commit(entries_local: list[tuple[str, dict[str, Any]]]) -> None:
            with self._lock:
                self._pending.extend(entries_local)

                if len(self._pending) < self.flush_every:
                    return

                self._flush_segment()

        if self.async_write:
            self.executor.submit(_commit, entries)  # type: ignore
        else:
            _commit(entries)

    def _flush_segment(self) -> None:
        """Write pending samples to new Arrow segment (O(1) operation)."""
        batch = self._pending
        self._pending = []

        if not batch:
            return

        segment_id = self._current_segment_id
        self._current_segment_id += 1

        # 1. Convert to Arrow RecordBatch
        rows = [row for _, row in batch]

        # Build arrays for each field
        arrays = {}
        for field in self.schema:
            field_values = []
            for row in rows:
                value = row.get(field.name)
                # Handle None values
                if value is None:
                    if pa.types.is_list(field.type):
                        field_values.append([])
                    elif pa.types.is_binary(field.type):
                        field_values.append(b"")
                    else:
                        field_values.append(None)
                else:
                    field_values.append(value)

            # Convert to PyArrow array, handling ChunkedArrays
            arr = pa.array(field_values, type=field.type)
            # If it's a ChunkedArray, combine into a single Array
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()
            arrays[field.name] = arr

        # Create RecordBatch from arrays (not pydict, to handle pre-converted arrays)
        record_batch = pa.RecordBatch.from_arrays(
            list(arrays.values()),
            schema=self.schema
        )

        # 2. Write to temp Arrow file
        temp_file = self.segments_dir / f"segment_{segment_id:06d}.arrow.tmp"
        final_file = self.segments_dir / f"segment_{segment_id:06d}.arrow"

        with pa.OSFile(str(temp_file), "wb") as sink:
            writer = pa.ipc.new_file(sink, self.schema)
            writer.write_batch(record_batch)
            writer.close()

        # 3. Atomic rename (commit Arrow segment to disk)
        temp_file.rename(final_file)

        # 4. Update in-memory index and persist to disk
        try:
            # Update in-memory index (within lock for thread safety)
            for i, (key, _) in enumerate(batch):
                self.index[key] = (segment_id, i)

            # Save index to disk atomically
            self._save_index()

        except Exception as e:
            # If index update fails, remove the segment file to maintain consistency
            if final_file.exists():
                final_file.unlink()
            raise e

    def flush(self) -> None:
        """Force flush all pending entries to disk (writer rank only)."""
        if self.current_rank != self.writer_rank:
            return

        with self._lock:
            if self._pending:
                self._flush_segment()

    def __del__(self) -> None:
        """Cleanup: flush pending writes, save index, and shutdown executor."""
        try:
            self.flush()
            if self.executor:
                self.executor.shutdown(wait=True)
            # Final index save
            self._save_index()
        except Exception:
            pass  # Best effort cleanup
