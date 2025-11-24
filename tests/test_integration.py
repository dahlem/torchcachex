"""Integration tests for torchcachex end-to-end scenarios."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input": self.data[idx],
            "cache_ids": f"sample_{idx}",
            "label": idx % 10,
        }


class ExpensiveFeatureExtractor(nn.Module):
    """Simulate an expensive feature extraction module."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        # Simulate expensive computation
        return torch.matmul(x.unsqueeze(1), x.unsqueeze(2)).squeeze()


class TestFullTrainingLoop:
    """Test complete training loop scenario."""

    def test_single_epoch_caching(self):
        """Test caching behavior over a single epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DummyDataset(size=32)
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )

            feature_extractor = ExpensiveFeatureExtractor()
            cached_extractor = CacheModuleDecorator(
                feature_extractor, backend, enabled=True
            )

            # First epoch - all misses
            for batch in loader:
                features = cached_extractor(
                    batch["input"], cache_ids=batch["cache_ids"]
                )
                assert features is not None

            backend.flush()

            # Check that module was called once per batch (4 batches of 8)
            # The decorator batches missing samples efficiently
            assert feature_extractor.call_count == 4

    def test_multi_epoch_caching(self):
        """Test that second epoch reuses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DummyDataset(size=32)
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )

            feature_extractor = ExpensiveFeatureExtractor()
            cached_extractor = CacheModuleDecorator(
                feature_extractor, backend, enabled=True
            )

            # First epoch
            epoch1_outputs = []
            for batch in loader:
                features = cached_extractor(
                    batch["input"], cache_ids=batch["cache_ids"]
                )
                epoch1_outputs.append(features)

            backend.flush()
            first_epoch_calls = feature_extractor.call_count

            # Second epoch - should all be cache hits
            epoch2_outputs = []
            for batch in loader:
                features = cached_extractor(
                    batch["input"], cache_ids=batch["cache_ids"]
                )
                epoch2_outputs.append(features)

            # Module should not have been called again
            assert feature_extractor.call_count == first_epoch_calls

            # Outputs should be identical
            for out1, out2 in zip(epoch1_outputs, epoch2_outputs, strict=False):
                assert torch.allclose(out1, out2)


class TestCrossRunCaching:
    """Test caching across different runs/processes."""

    def test_cache_reuse_across_runs(self):
        """Test that cache persists and is reused across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DummyDataset(size=32)
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            # First run - populate cache
            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )
            feature_extractor1 = ExpensiveFeatureExtractor()
            cached_extractor1 = CacheModuleDecorator(
                feature_extractor1, backend1, enabled=True
            )

            run1_outputs = []
            for batch in loader:
                features = cached_extractor1(
                    batch["input"], cache_ids=batch["cache_ids"]
                )
                run1_outputs.append(features)

            backend1.flush()
            del backend1, cached_extractor1

            # Second run - reuse cache
            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )
            feature_extractor2 = ExpensiveFeatureExtractor()
            cached_extractor2 = CacheModuleDecorator(
                feature_extractor2, backend2, enabled=True
            )

            run2_outputs = []
            for batch in loader:
                features = cached_extractor2(
                    batch["input"], cache_ids=batch["cache_ids"]
                )
                run2_outputs.append(features)

            # Module in second run should never be called
            assert feature_extractor2.call_count == 0

            # Outputs should match
            for out1, out2 in zip(run1_outputs, run2_outputs, strict=False):
                assert torch.allclose(out1, out2)


class TestProgressiveEnrichment:
    """Test progressive enrichment scenario."""

    def test_progressive_cache_building(self):
        """Test building cache progressively over multiple runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            full_dataset = DummyDataset(size=32)

            # Run 1 - cache first half
            subset1_indices = list(range(16))
            loader1 = DataLoader(
                torch.utils.data.Subset(full_dataset, subset1_indices),
                batch_size=8,
                shuffle=False,
            )

            backend1 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )
            feature_extractor1 = ExpensiveFeatureExtractor()
            cached_extractor1 = CacheModuleDecorator(
                feature_extractor1, backend1, enabled=True
            )

            for batch in loader1:
                cached_extractor1(batch["input"], cache_ids=batch["cache_ids"])

            backend1.flush()
            first_run_calls = feature_extractor1.call_count
            # 16 samples / 8 batch size = 2 batches
            assert first_run_calls == 2
            del backend1, cached_extractor1

            # Run 2 - cache second half
            subset2_indices = list(range(16, 32))
            loader2 = DataLoader(
                torch.utils.data.Subset(full_dataset, subset2_indices),
                batch_size=8,
                shuffle=False,
            )

            backend2 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )
            feature_extractor2 = ExpensiveFeatureExtractor()
            cached_extractor2 = CacheModuleDecorator(
                feature_extractor2, backend2, enabled=True
            )

            for batch in loader2:
                cached_extractor2(batch["input"], cache_ids=batch["cache_ids"])

            backend2.flush()
            # Should only compute second half (16 samples / 8 batch size = 2 batches)
            assert feature_extractor2.call_count == 2
            del backend2, cached_extractor2

            # Run 3 - use full dataset, all should be cached
            loader3 = DataLoader(full_dataset, batch_size=8, shuffle=False)

            backend3 = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )
            feature_extractor3 = ExpensiveFeatureExtractor()
            cached_extractor3 = CacheModuleDecorator(
                feature_extractor3, backend3, enabled=True
            )

            for batch in loader3:
                cached_extractor3(batch["input"], cache_ids=batch["cache_ids"])

            # Should be all cache hits
            assert feature_extractor3.call_count == 0


class TestDevicePlacement:
    """Test device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_to_cpu_caching(self):
        """Test that CUDA tensors are cached on CPU and moved back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test_features", async_write=False
            )

            module = ExpensiveFeatureExtractor()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            # Input on CUDA
            x = torch.randn(4, 10, device="cuda")
            cache_ids = ["id1", "id2", "id3", "id4"]

            # First call - compute
            out1 = cached(x, cache_ids=cache_ids)
            backend.flush()
            assert out1.device.type == "cuda"

            # Second call - from cache
            out2 = cached(x, cache_ids=cache_ids)
            assert out2.device.type == "cuda"
            assert torch.allclose(out1.cpu(), out2.cpu())


class TestModuleWithDifferentOutputTypes:
    """Test modules with various output types."""

    def test_tuple_output(self):
        """Test caching with tuple outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class TupleModule(nn.Module):
                def forward(self, x):
                    return (x * 2, x * 3, x.sum(dim=-1))

            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = TupleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            out1 = cached(x, cache_ids=cache_ids)
            backend.flush()

            out2 = cached(x, cache_ids=cache_ids)

            assert len(out2) == 3
            assert torch.allclose(out1[0], out2[0])
            assert torch.allclose(out1[1], out2[1])
            assert torch.allclose(out1[2], out2[2])

    def test_list_output(self):
        """Test caching with tuple outputs (lists reserved for variable-length)."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class TupleModule(nn.Module):
                def forward(self, x):
                    # Use tuple for structural outputs (list reserved for variable-length)
                    return (x * 2, x * 3, x.sum(dim=-1))

            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = TupleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            out1 = cached(x, cache_ids=cache_ids)
            backend.flush()

            out2 = cached(x, cache_ids=cache_ids)

            assert len(out2) == 3
            for o1, o2 in zip(out1, out2, strict=False):
                assert torch.allclose(o1, o2)
