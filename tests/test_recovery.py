"""Crash recovery and integrity tests for torchcachex.

Tests that verify data integrity and crash recovery mechanisms.
"""

import shutil
import tempfile
from pathlib import Path

import torch

from torchcachex import ArrowIPCCacheBackend


class TestCrashRecovery:
    """Test crash recovery scenarios."""

    def test_incomplete_segment_ignored(self):
        """Test that incomplete segment files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False, flush_every=10
            )

            # Write some data
            batch1 = {f"key_{i}": torch.randn(10) for i in range(10)}
            backend.put_batch(batch1)
            backend.flush()

            # Simulate incomplete write: create .tmp file that would be left by crash
            segments_dir = Path(tmpdir) / "test" / "segments"
            incomplete_file = segments_dir / "segment_000001.arrow.tmp"
            incomplete_file.write_text("incomplete data")

            del backend

            # New backend should ignore .tmp file and work fine
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            results, missing = backend2.get_batch([f"key_{i}" for i in range(10)])
            assert len(missing) == 0
            assert all(r is not None for r in results)

    def test_orphaned_segment_file(self):
        """Test handling of segment file without SQLite entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Write data
            batch = {f"key_{i}": torch.randn(10) for i in range(10)}
            backend.put_batch(batch)
            backend.flush()

            # Manually create orphaned segment file
            segments_dir = Path(tmpdir) / "test" / "segments"
            orphan_file = segments_dir / "segment_999999.arrow"

            # Copy existing segment as orphan
            existing = list(segments_dir.glob("segment_*.arrow"))[0]
            shutil.copy(existing, orphan_file)

            del backend

            # New backend should handle orphan gracefully
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Should still be able to read original data
            results, missing = backend2.get_batch([f"key_{i}" for i in range(10)])
            assert len(missing) == 0

    def test_corrupted_schema_file(self):
        """Test handling of corrupted schema file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            batch = {"key": torch.randn(10)}
            backend.put_batch(batch)
            backend.flush()
            del backend

            # Corrupt schema file
            schema_path = Path(tmpdir) / "test" / "schema.json"
            schema_path.write_text("corrupted json {{{")

            # Should handle gracefully (will re-infer on next write)
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Can still write new data (will re-infer schema)
            batch2 = {"key2": torch.randn(10)}
            backend2.put_batch(batch2)
            backend2.flush()

            # Verify new data is accessible
            results, missing = backend2.get_batch(["key2"])
            assert len(missing) == 0

    def test_graceful_shutdown_flushes_pending(self):
        """Test that graceful shutdown (del) flushes pending data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False, flush_every=10
            )

            # Write initial data
            batch1 = {f"key_{i}": torch.randn(10) for i in range(10)}
            backend.put_batch(batch1)
            backend.flush()

            # Add more data but don't reach flush threshold
            batch2 = {f"key_{i}": torch.randn(10) for i in range(10, 15)}
            backend.put_batch(batch2)
            # Don't manually flush

            # Delete backend - __del__ should auto-flush pending data
            del backend

            # Recovery: new backend should have ALL data (including pending)
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Original data should be intact
            results, missing = backend2.get_batch([f"key_{i}" for i in range(10)])
            assert len(missing) == 0

            # Pending data should also be persisted (auto-flushed by __del__)
            results2, missing2 = backend2.get_batch([f"key_{i}" for i in range(10, 15)])
            assert len(missing2) == 0  # All data was auto-flushed


class TestDataIntegrity:
    """Test data integrity guarantees."""

    def test_no_data_loss_across_flushes(self):
        """Test that no data is lost across multiple flushes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False, flush_every=10
            )

            all_data = {}

            # Write in multiple batches
            for round_idx in range(10):
                batch = {f"key_{round_idx}_{i}": torch.randn(20) for i in range(15)}
                all_data.update(batch)
                backend.put_batch(batch)
                backend.flush()

            # Verify all data is accessible
            all_keys = list(all_data.keys())
            results, missing = backend.get_batch(all_keys)

            assert len(missing) == 0, f"Missing keys: {[all_keys[i] for i in missing]}"
            assert len(results) == len(all_keys)

            # Verify values match
            for key, result in zip(all_keys, results, strict=False):
                assert torch.allclose(result, all_data[key])

    def test_atomic_flush(self):
        """Test that flush is atomic (all or nothing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Write and flush
            batch1 = {f"key_{i}": torch.randn(10) for i in range(10)}
            backend.put_batch(batch1)
            backend.flush()

            # Check segment and index are consistent
            segments_dir = Path(tmpdir) / "test" / "segments"
            _ = list(segments_dir.glob("segment_*.arrow"))  # Verify directory exists

            # Count entries in SQLite
            cursor = backend.conn.execute("SELECT COUNT(*) FROM cache")
            cache_count = cursor.fetchone()[0]

            # Count entries in segments metadata
            cursor = backend.conn.execute("SELECT SUM(num_rows) FROM segments")
            segment_count = cursor.fetchone()[0]

            assert cache_count == segment_count == 10

    def test_index_segment_consistency(self):
        """Test that SQLite index and Arrow segments stay consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False, flush_every=20
            )

            # Write multiple batches
            for i in range(5):
                batch = {
                    f"key_{j}": torch.randn(15) for j in range(i * 20, (i + 1) * 20)
                }
                backend.put_batch(batch)
                backend.flush()

            # Verify index count matches actual segments
            cursor = backend.conn.execute("SELECT COUNT(*) FROM cache")
            total_keys = cursor.fetchone()[0]

            # Count keys across all segments
            segments_dir = Path(tmpdir) / "test" / "segments"
            segment_files = sorted(segments_dir.glob("segment_*.arrow"))

            total_rows = 0
            for seg_file in segment_files:
                import pyarrow as pa

                with pa.memory_map(str(seg_file), "r") as source:
                    reader = pa.ipc.open_file(source)
                    table = reader.read_all()
                    total_rows += len(table)

            assert total_keys == total_rows

    def test_schema_consistency_across_restarts(self):
        """Test that schema remains consistent across backend restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            batch1 = {"key1": torch.randn(10, dtype=torch.float32)}
            backend1.put_batch(batch1)
            backend1.flush()

            schema1 = backend1.schema
            output_structure1 = backend1.output_structure

            del backend1

            # Second instance
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Schema should be loaded from disk
            assert backend2.schema is not None
            assert backend2.output_structure == output_structure1
            assert str(backend2.schema) == str(schema1)

            # Adding more data with same schema should work
            batch2 = {"key2": torch.randn(10, dtype=torch.float32)}
            backend2.put_batch(batch2)
            backend2.flush()

            # Verify both keys accessible
            results, missing = backend2.get_batch(["key1", "key2"])
            assert len(missing) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_cache_operations(self):
        """Test operations on empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Get from empty cache
            results, missing = backend.get_batch(["nonexistent"])
            assert len(missing) == 1
            assert results[0] is None

            # Flush empty pending
            backend.flush()  # Should not crash

    def test_single_sample_operations(self):
        """Test with single sample batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Write single sample multiple times
            for i in range(5):
                backend.put_batch({f"key_{i}": torch.randn(10)})
                backend.flush()

            # Verify all accessible
            results, missing = backend.get_batch([f"key_{i}" for i in range(5)])
            assert len(missing) == 0

    def test_duplicate_key_overwrites(self):
        """Test that duplicate keys overwrite previous values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Write same key twice
            value1 = torch.randn(10)
            value2 = torch.randn(10)

            backend.put_batch({"key": value1})
            backend.flush()

            backend.put_batch({"key": value2})
            backend.flush()

            # Should get latest value
            results, missing = backend.get_batch(["key"])
            assert len(missing) == 0
            assert torch.allclose(results[0], value2)
            assert not torch.allclose(results[0], value1)

    def test_very_long_keys(self):
        """Test handling of very long cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Create very long key
            long_key = "key_" + "x" * 1000
            batch = {long_key: torch.randn(10)}

            backend.put_batch(batch)
            backend.flush()

            # Should be retrievable
            results, missing = backend.get_batch([long_key])
            assert len(missing) == 0
