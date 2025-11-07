"""Tests for ArrowIPCCacheBackend."""

import tempfile

import torch

from torchcachex.backend import ArrowIPCCacheBackend


class TestBackendBasics:
    """Test basic backend functionality."""

    def test_initialization(self):
        """Test backend initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_module", lru_size=10
            )
            assert backend.cache_dir.exists()
            assert backend.segments_dir.exists()
            assert backend.index_path.parent.exists()  # Check cache_dir exists
            assert isinstance(backend.index, dict)  # Check index is initialized

    def test_put_and_get_single(self):
        """Test putting and getting a single item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                lru_size=10,
                async_write=False,
            )

            # Put item
            x = torch.randn(3, 224, 224)
            backend.put_batch({"key1": x})
            backend.flush()

            # Get item
            results, missing = backend.get_batch(["key1"])
            assert len(missing) == 0
            assert torch.allclose(x, results[0])

    def test_put_and_get_batch(self):
        """Test putting and getting multiple items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                lru_size=10,
                async_write=False,
            )

            # Put batch
            items = {f"key{i}": torch.randn(3, 224, 224) for i in range(5)}
            backend.put_batch(items)
            backend.flush()

            # Get batch
            keys = [f"key{i}" for i in range(5)]
            results, missing = backend.get_batch(keys)
            assert len(missing) == 0
            for i, result in enumerate(results):
                assert torch.allclose(items[f"key{i}"], result)

    def test_missing_keys(self):
        """Test getting missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                lru_size=10,
                async_write=False,
            )

            # Put some items
            backend.put_batch({"key1": torch.randn(10), "key2": torch.randn(10)})
            backend.flush()

            # Get mix of existing and missing
            results, missing = backend.get_batch(
                ["key1", "missing1", "key2", "missing2"]
            )
            assert missing == [1, 3]
            assert results[0] is not None
            assert results[1] is None
            assert results[2] is not None
            assert results[3] is None

    def test_lru_cache(self):
        """Test that LRU cache works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                lru_size=2,  # Small LRU
                async_write=False,
            )

            # Put items
            backend.put_batch({"key1": torch.randn(10)})
            backend.flush()

            # First access - from disk
            results1, _ = backend.get_batch(["key1"])

            # Second access - should be from LRU (check that it's in LRU)
            assert "key1" in backend.lru
            results2, _ = backend.get_batch(["key1"])
            assert torch.allclose(results1[0], results2[0])


class TestBackendPersistence:
    """Test persistence across instances."""

    def test_persistence(self):
        """Test that data persists across backend instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance - write data
            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            items = {f"key{i}": torch.randn(10) for i in range(5)}
            backend1.put_batch(items)
            backend1.flush()
            del backend1

            # Second instance - read data
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            keys = [f"key{i}" for i in range(5)]
            results, missing = backend2.get_batch(keys)
            assert len(missing) == 0
            for i, result in enumerate(results):
                assert torch.allclose(items[f"key{i}"], result)

    def test_progressive_enrichment(self):
        """Test progressive enrichment (append-only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First pass - partial data
            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            batch1 = {f"key{i}": torch.randn(10) for i in range(3)}
            backend1.put_batch(batch1)
            backend1.flush()
            del backend1

            # Second pass - add more data
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            batch2 = {f"key{i}": torch.randn(10) for i in range(3, 6)}
            backend2.put_batch(batch2)
            backend2.flush()
            del backend2

            # Third pass - read all
            backend3 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            all_keys = [f"key{i}" for i in range(6)]
            results, missing = backend3.get_batch(all_keys)
            assert len(missing) == 0


class TestBackendNestedStructures:
    """Test backend with nested tensor structures."""

    def test_nested_dict(self):
        """Test caching nested dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )

            # Put nested structure
            x = {"img": torch.randn(3, 224, 224), "label": torch.tensor(5)}
            backend.put_batch({"key1": x})
            backend.flush()

            # Get nested structure
            results, missing = backend.get_batch(["key1"])
            assert len(missing) == 0
            assert torch.allclose(x["img"], results[0]["img"])
            assert x["label"].item() == results[0]["label"].item()

    def test_nested_list(self):
        """Test caching nested lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )

            # Put nested structure
            x = [torch.randn(10), torch.randn(20), torch.randn(30)]
            backend.put_batch({"key1": x})
            backend.flush()

            # Get nested structure
            results, missing = backend.get_batch(["key1"])
            assert len(missing) == 0
            assert len(results[0]) == 3
            for i in range(3):
                assert torch.allclose(x[i], results[0][i])


class TestIndexPersistence:
    """Test in-memory index persistence and recovery."""

    def test_index_persists_to_disk(self):
        """Test that index is saved to pickle file."""
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )

            # Write data
            backend.put_batch({"key1": torch.randn(10)})
            backend.flush()

            # Check index file exists
            index_path = Path(tmpdir) / "test_module" / "index.pkl"
            assert index_path.exists(), "Index file should be created after flush"

    def test_index_rebuild_from_segments(self):
        """Test that index can be rebuilt from Arrow segments."""
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write data
            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )
            test_data = {"key1": torch.randn(10), "key2": torch.randn(20)}
            backend1.put_batch(test_data)
            backend1.flush()
            del backend1

            # Delete index file to force rebuild
            index_path = Path(tmpdir) / "test_module" / "index.pkl"
            index_path.unlink()

            # Reopen - should rebuild index from segments
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir,
                module_id="test_module",
                async_write=False,
            )

            # Verify data is still accessible
            results, missing = backend2.get_batch(["key1", "key2"])
            assert len(missing) == 0, "All keys should be found after index rebuild"
            assert torch.allclose(test_data["key1"], results[0])
            assert torch.allclose(test_data["key2"], results[1])
