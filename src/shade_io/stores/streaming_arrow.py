"""Streaming Arrow writer for memory-efficient feature storage."""

import json
import logging
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StreamingArrowWriter:
    """Streaming Arrow/Parquet writer for incremental feature writes.

    This class enables writing features to Arrow/Parquet files incrementally,
    avoiding memory accumulation. Supports async writes for better performance.
    """

    def __init__(
        self,
        path: Path,
        feature_names: list[str],
        compression: str | None = "snappy",
        buffer_size: int = 1000,
        write_queue_size: int = 10,
        enable_async: bool = True,
    ):
        """Initialize streaming Arrow writer.

        Args:
            path: Output file path
            feature_names: Names of features to write
            compression: Arrow compression (snappy, gzip, brotli, lz4, zstd, None)
            buffer_size: Number of samples to buffer before writing
            write_queue_size: Maximum batches in write queue
            enable_async: Whether to enable async writes
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self._pa = pa
            self._pq = pq
        except ImportError as err:
            raise ImportError("pyarrow is required for streaming Arrow writes") from err

        self.path = Path(path)
        # Ensure parent directory exists even if the filename encodes subpaths
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create parent directory for {self.path}: {e}")
            raise
        self.feature_names = feature_names
        self.compression = compression

        # Validate and set buffer_size with defensive checks
        if buffer_size > 100:
            logger.warning(f"Large buffer_size ({buffer_size}) may cause memory issues. Consider using smaller values (e.g., 5-10 batches)")
        self.buffer_size = buffer_size
        self.enable_async = enable_async

        # Create schema
        self.schema = self._create_schema()

        # Initialize writer
        self.writer = None
        self.buffer = []
        self.total_samples = 0

        # Async components
        self._error = None  # Always initialize error tracking
        if enable_async:
            self.write_queue = Queue(maxsize=write_queue_size)
            self.write_thread = None
            self._shutdown = threading.Event()

        logger.info(f"StreamingArrowWriter initialized for {path}")
        logger.info(f"Features: {len(feature_names)}, Compression: {compression}")
        logger.info(f"Buffer size: {buffer_size}, Async: {enable_async}")

    def _create_schema(self) -> "pa.Schema":
        """Create Arrow schema for features."""
        fields = []

        # Add feature columns (all float32)
        for name in self.feature_names:
            fields.append(self._pa.field(name, self._pa.float32()))

        # Add metadata columns
        fields.extend(
            [
                self._pa.field("split", self._pa.string()),
                self._pa.field("sample_hash", self._pa.string()),
                self._pa.field("original_index", self._pa.int64()),
                self._pa.field("label", self._pa.int32()),
            ]
        )

        schema = self._pa.schema(fields)

        # Add metadata
        metadata = {
            b"feature_names": json.dumps(self.feature_names).encode(),
            b"streaming_writer": b"true",
        }
        schema = schema.with_metadata(metadata)

        return schema

    def start_async_writer(self) -> None:
        """Start background thread for async writes."""
        if not self.enable_async:
            return

        if self.write_thread is not None:
            logger.warning("Async writer already started")
            return

        logger.info("Starting async writer thread")
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()

    def _write_worker(self) -> None:
        """Background worker for async writes."""
        logger.info("Async write worker started")

        try:
            while not self._shutdown.is_set():
                try:
                    # Get batch from queue (with timeout)
                    batch_data = self.write_queue.get(timeout=1.0)

                    if batch_data is None:  # Sentinel for shutdown
                        logger.info("Received shutdown signal")
                        self.write_queue.task_done()  # Mark sentinel as done
                        break

                    # Write batch synchronously in background thread
                    self._write_batch_sync(batch_data)
                    self.write_queue.task_done()

                except Empty:
                    # Timeout is normal, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in async write worker: {e}")
                    self._error = e
                    break

        except Exception as e:
            logger.error(f"Fatal error in write worker: {e}")
            self._error = e

        logger.info("Async write worker finished")

    def write_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        splits: list[str] | None = None,
        sample_hashes: list[str] | None = None,
        original_indices: list[int] | None = None,
    ) -> None:
        """Write a batch of features to the file.

        Args:
            features: Feature tensor (batch_size, n_features)
            labels: Labels for samples (optional)
            splits: Split assignments for samples (optional)
            sample_hashes: Sample hashes for samples (optional)
            original_indices: Original indices for samples (optional)
        """
        if self._error:
            raise RuntimeError(f"Async writer encountered error: {self._error}")

        batch_size = features.shape[0]

        # Convert to CPU numpy if needed (detach first to break computation graph)
        if features.is_cuda or features.device.type == "mps":
            features = features.detach().cpu()
        else:
            features = features.detach()
        feature_array = features.numpy().astype(np.float32)

        # Handle optional fields
        if labels is not None:
            if labels.is_cuda or labels.device.type == "mps":
                labels = labels.detach().cpu()
            else:
                labels = labels.detach()
            labels_array = labels.numpy().astype(np.int32)
        else:
            labels_array = np.zeros(batch_size, dtype=np.int32)

        if splits is None:
            splits = ["unknown"] * batch_size

        if sample_hashes is None:
            sample_hashes = [""] * batch_size

        if original_indices is None:
            original_indices = list(
                range(self.total_samples, self.total_samples + batch_size)
            )

        # Create batch data
        batch_data = {
            "features": feature_array,
            "labels": labels_array,
            "splits": splits,
            "sample_hashes": sample_hashes,
            "original_indices": original_indices,
        }

        # Add to buffer
        self.buffer.append(batch_data)
        self.total_samples += batch_size

        # Write if buffer is full
        if len(self.buffer) >= self.buffer_size:
            logger.debug(f"Buffer full ({len(self.buffer)}/{self.buffer_size}), flushing")
            self._flush_buffer()

        # Add warning if buffer is getting very large (should not happen with correct buffer_size)
        if len(self.buffer) > 50:
            logger.warning(f"Buffer has grown to {len(self.buffer)} batches, consider reducing buffer_size")

    def _flush_buffer(self) -> None:
        """Flush current buffer to disk."""
        if not self.buffer:
            return

        # Combine all batches in buffer
        combined_data = self._combine_buffer()

        if self.enable_async:
            # Add to async queue
            if self._error:
                raise RuntimeError(f"Async writer encountered error: {self._error}")

            try:
                self.write_queue.put(combined_data, timeout=5.0)
            except:
                logger.error("Failed to add batch to write queue")
                raise
        else:
            # Write synchronously
            self._write_batch_sync(combined_data)

        # Clear buffer
        self.buffer.clear()

    def _combine_buffer(self) -> dict[str, Any]:
        """Combine all batches in buffer into single batch."""
        if not self.buffer:
            return {}

        combined = {}

        # Combine features
        feature_arrays = [batch["features"] for batch in self.buffer]
        combined["features"] = np.vstack(feature_arrays)

        # Combine labels
        label_arrays = [batch["labels"] for batch in self.buffer]
        combined["labels"] = np.concatenate(label_arrays)

        # Combine other fields
        combined["splits"] = []
        combined["sample_hashes"] = []
        combined["original_indices"] = []

        for batch in self.buffer:
            combined["splits"].extend(batch["splits"])
            combined["sample_hashes"].extend(batch["sample_hashes"])
            combined["original_indices"].extend(batch["original_indices"])

        return combined

    def _write_batch_sync(self, batch_data: dict[str, Any]) -> None:
        """Write batch data synchronously to Arrow file."""
        if not batch_data:
            return

        # Initialize writer if needed
        if self.writer is None:
            # Safety: ensure parent directory still exists (in case of external cleanup)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to (re)create parent directory for {self.path}: {e}")
                raise
            self.writer = self._pq.ParquetWriter(
                self.path, self.schema, compression=self.compression
            )
            logger.info(f"Opened Arrow writer for {self.path}")

        # Create Arrow arrays
        arrays = []

        # Add feature columns
        features = batch_data["features"]  # Shape: (batch_size, n_features)
        for i, _feature_name in enumerate(self.feature_names):
            arrays.append(self._pa.array(features[:, i]))

        # Add metadata columns
        arrays.extend(
            [
                self._pa.array(batch_data["splits"]),
                self._pa.array(batch_data["sample_hashes"]),
                self._pa.array(batch_data["original_indices"]),
                self._pa.array(batch_data["labels"]),
            ]
        )

        # Create record batch
        record_batch = self._pa.RecordBatch.from_arrays(arrays, schema=self.schema)

        # Write to file
        self.writer.write_batch(record_batch)

        logger.debug(f"Wrote batch with {len(batch_data['labels'])} samples")

    def close(self) -> None:
        """Close the writer and finalize the file."""
        logger.info("Closing streaming Arrow writer")

        # Flush remaining buffer
        if self.buffer:
            logger.info(f"Flushing final buffer with {len(self.buffer)} batches")
            self._flush_buffer()

        # Shutdown async writer if enabled
        if self.enable_async and self.write_thread is not None:
            logger.info("Shutting down async writer")

            # Wait for queue to be empty (all pending writes processed)
            if hasattr(self, "write_queue"):
                logger.info(
                    f"Waiting for {self.write_queue.qsize()} pending writes to complete"
                )
                self.write_queue.join()  # Wait for all tasks to be processed

            # Signal shutdown
            self._shutdown.set()
            self.write_queue.put(None)  # Sentinel

            # Wait for completion
            self.write_thread.join(timeout=10.0)
            if self.write_thread.is_alive():
                logger.warning("Async writer thread did not shutdown cleanly")

        # Close writer
        if self.writer is not None:
            self.writer.close()
            logger.info(f"Closed Arrow writer, wrote {self.total_samples} samples")

        # Check for errors
        if self._error:
            raise RuntimeError(f"Streaming writer encountered error: {self._error}")

    def __enter__(self):
        """Context manager entry."""
        if self.enable_async:
            self.start_async_writer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
