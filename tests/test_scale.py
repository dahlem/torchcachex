"""Scale tests for torchcachex backend.

Tests that verify O(1) performance characteristics with large caches.
"""

import tempfile
import time

import pytest
import torch

from torchcachex import ArrowIPCCacheBackend


class TestScalePerformance:
    """Test performance characteristics at scale."""

    def test_flush_time_constant_with_cache_size(self):
        """Test that flush time is O(1) regardless of existing cache size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=False,
                flush_every=100,
            )

            flush_times = []

            # Populate cache with increasing amounts of data
            for round_idx in range(10):
                # Add 100 samples per round (1000 total)
                batch = {f"key_{round_idx}_{i}": torch.randn(100) for i in range(100)}

                start = time.time()
                backend.put_batch(batch)
                backend.flush()
                elapsed = time.time() - start

                flush_times.append(elapsed)
                print(
                    f"Round {round_idx}: {len(backend.key_to_idx)} total samples, "
                    f"flush took {elapsed:.4f}s"
                )

            # Flush time should not grow significantly with cache size
            # Later flushes should be within 2x of first flush
            avg_early = sum(flush_times[:3]) / 3
            avg_late = sum(flush_times[-3:]) / 3

            assert avg_late < avg_early * 2.0, (
                f"Flush time grew too much: {avg_early:.4f}s -> {avg_late:.4f}s"
            )

    def test_large_cache_operations(self):
        """Test cache with 10k samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=False,
                flush_every=1000,
            )

            # Write 10k samples in batches
            for i in range(100):
                batch = {
                    f"key_{j}": torch.randn(50) for j in range(i * 100, (i + 1) * 100)
                }
                backend.put_batch(batch)
                if (i + 1) % 10 == 0:  # Flush every 1000 samples
                    backend.flush()

            backend.flush()  # Final flush

            # Verify all samples are accessible
            keys = [f"key_{i}" for i in range(10000)]
            results, missing = backend.get_batch(keys[:100])  # Sample 100
            assert len(missing) == 0
            assert all(r is not None for r in results)

            # Verify cache size
            assert len(backend.key_to_idx) == 10000

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not installed"),
        reason="psutil required for memory tests",
    )
    def test_memory_usage_constant_during_flush(self):
        """Test that memory usage doesn't grow with cache size during flush."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=False,
                flush_every=500,
            )

            # Populate with 5k samples
            for i in range(10):
                batch = {
                    f"key_{j}": torch.randn(100) for j in range(i * 500, (i + 1) * 500)
                }
                backend.put_batch(batch)
                backend.flush()

            # Measure memory before and during flush of new data
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Add more data
            batch = {f"key_new_{i}": torch.randn(100) for i in range(500)}
            backend.put_batch(batch)
            backend.flush()

            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            # Memory increase should be small (< 50MB) even with 5k existing samples
            mem_increase = mem_after - mem_before
            print(f"Memory increase during flush: {mem_increase:.2f} MB")
            assert mem_increase < 50, (
                f"Memory increased by {mem_increase:.2f} MB during flush"
            )

    def test_many_segments_read_performance(self):
        """Test read performance with many small segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=False,
                flush_every=50,  # Create many small segments
            )

            # Create 100 segments (5000 samples)
            for i in range(100):
                batch = {
                    f"key_{j}": torch.randn(20) for j in range(i * 50, (i + 1) * 50)
                }
                backend.put_batch(batch)
                backend.flush()

            # Read should still be fast even with 100 segments
            keys = [
                f"key_{i}" for i in range(0, 5000, 50)
            ]  # Sample across all segments

            start = time.time()
            results, missing = backend.get_batch(keys)
            elapsed = time.time() - start

            assert len(missing) == 0
            assert elapsed < 1.0, f"Read took {elapsed:.4f}s (should be < 1s)"

    @pytest.mark.parametrize("cache_size", [1000, 5000, 10000])
    def test_read_latency_independent_of_cache_size(self, cache_size):
        """Test that read latency doesn't depend on total cache size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=False,
                flush_every=1000,
            )

            # Populate cache
            for i in range(cache_size // 100):
                batch = {
                    f"key_{j}": torch.randn(50) for j in range(i * 100, (i + 1) * 100)
                }
                backend.put_batch(batch)
                if (i + 1) % 10 == 0:
                    backend.flush()
            backend.flush()

            # Measure read time for same batch size
            keys = [f"key_{i}" for i in range(100)]

            start = time.time()
            results, missing = backend.get_batch(keys)
            elapsed = time.time() - start

            assert len(missing) == 0
            print(f"Cache size: {cache_size}, read 100 keys in {elapsed:.4f}s")
            # Should be fast regardless of cache size
            assert elapsed < 0.5


class TestLargeDataTypes:
    """Test handling of large tensors and batches."""

    def test_large_tensor_storage(self):
        """Test caching of large tensors (e.g., images)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False, flush_every=10
            )

            # Simulate ResNet50 features: (B, 2048, 7, 7)
            large_tensors = {f"key_{i}": torch.randn(2048, 7, 7) for i in range(20)}

            backend.put_batch(large_tensors)
            backend.flush()

            # Read back
            keys = list(large_tensors.keys())
            results, missing = backend.get_batch(keys)

            assert len(missing) == 0
            for i, result in enumerate(results):
                assert result.shape == (2048, 7, 7)
                assert torch.allclose(result, large_tensors[keys[i]])

    def test_mixed_tensor_sizes(self):
        """Test caching tensors with varying sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Different sized tensors
            batch = {
                "small": torch.randn(10),
                "medium": torch.randn(100, 50),
                "large": torch.randn(512, 512),
            }

            backend.put_batch(batch)
            backend.flush()

            results, missing = backend.get_batch(list(batch.keys()))
            assert len(missing) == 0
            assert results[0].shape == (10,)
            assert results[1].shape == (100, 50)
            assert results[2].shape == (512, 512)


class TestConcurrentAccess:
    """Test behavior with concurrent reads (future: concurrent writes)."""

    def test_read_while_writing(self):
        """Test that reads work while async writes are happening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test",
                async_write=True,  # Enable async
                flush_every=500,
            )

            # Write initial data
            batch1 = {f"key_{i}": torch.randn(50) for i in range(500)}
            backend.put_batch(batch1)
            backend.flush()

            # Start async writes
            batch2 = {f"key_{i}": torch.randn(50) for i in range(500, 1000)}
            backend.put_batch(batch2)  # Will flush async

            # Immediate read of old data should work
            results, missing = backend.get_batch([f"key_{i}" for i in range(10)])
            assert len(missing) == 0

            # Wait for async flush
            if backend.executor:
                backend.executor.shutdown(wait=True)

            # Verify new data is also accessible
            results2, missing2 = backend.get_batch(
                [f"key_{i}" for i in range(500, 510)]
            )
            assert len(missing2) == 0


class TestDifferentDtypes:
    """Test handling of different tensor dtypes."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ],
    )
    def test_dtype_preservation(self, dtype):
        """Test that different dtypes are preserved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Create tensor with specific dtype
            if dtype == torch.bool:
                tensor = torch.tensor([True, False, True, False])
            else:
                tensor = torch.randn(10).to(dtype)

            backend.put_batch({"key": tensor})
            backend.flush()
            del backend

            # Load in new backend instance
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            results, missing = backend2.get_batch(["key"])

            assert len(missing) == 0
            assert results[0].dtype == dtype
            if dtype != torch.bool:
                assert torch.allclose(results[0], tensor, rtol=1e-5, atol=1e-7)
            else:
                assert torch.equal(results[0], tensor)
