"""Tests for CacheModuleDecorator."""

import tempfile

import pytest
import torch
import torch.nn as nn

from torchcachex import ArrowIPCCacheBackend, CacheModuleDecorator


class SimpleModule(nn.Module):
    """Simple test module that doubles input."""

    def forward(self, x):
        return x * 2


class NestedOutputModule(nn.Module):
    """Module that returns nested output."""

    def forward(self, x):
        return {"double": x * 2, "triple": x * 3}


class TestDecoratorBasics:
    """Test basic decorator functionality."""

    def test_disabled_caching(self):
        """Test that disabled caching bypasses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = SimpleModule()
            cached = CacheModuleDecorator(module, backend, enabled=False)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            out = cached(x, cache_ids=cache_ids)
            expected = x * 2
            assert torch.allclose(out, expected)

    def test_simple_caching(self):
        """Test basic caching behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = SimpleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            # First call - should compute and cache
            out1 = cached(x, cache_ids=cache_ids)
            backend.flush()

            # Second call - should hit cache
            out2 = cached(x, cache_ids=cache_ids)

            assert torch.allclose(out1, out2)

    def test_partial_cache_hit(self):
        """Test behavior with partial cache hits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = SimpleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            # First batch - cache 2 samples
            x1 = torch.randn(2, 10)
            cache_ids1 = ["id1", "id2"]
            out1 = cached(x1, cache_ids=cache_ids1)
            backend.flush()

            # Second batch - 1 hit, 2 misses
            x2 = torch.randn(3, 10)
            cache_ids2 = ["id1", "id3", "id4"]  # id1 should hit
            out2 = cached(x2, cache_ids=cache_ids2)
            backend.flush()

            # Verify first sample matches
            assert torch.allclose(out1[0], out2[0])


class TestDecoratorNestedOutputs:
    """Test decorator with nested output structures."""

    def test_dict_output(self):
        """Test caching with dict outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = NestedOutputModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            # First call
            out1 = cached(x, cache_ids=cache_ids)
            backend.flush()

            # Second call - from cache
            out2 = cached(x, cache_ids=cache_ids)

            assert torch.allclose(out1["double"], out2["double"])
            assert torch.allclose(out1["triple"], out2["triple"])


class TestDecoratorCacheIdExtraction:
    """Test different ways of providing cache IDs."""

    def test_kwargs_cache_ids(self):
        """Test cache_ids in kwargs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = SimpleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)
            out = cached(x, cache_ids=["id1", "id2", "id3", "id4"])
            backend.flush()

            assert out.shape == (4, 10)

    def test_batch_dict_cache_ids(self):
        """Test cache_ids in batch dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Module that takes dict input
            class DictModule(nn.Module):
                def forward(self, batch_dict):
                    return batch_dict["data"] * 2

            module = DictModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            batch = {"data": torch.randn(4, 10), "cache_ids": ["id1", "id2", "id3", "id4"]}
            out = cached(batch)
            backend.flush()

            assert out.shape == (4, 10)

    def test_missing_cache_ids(self):
        """Test that missing cache_ids raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )
            module = SimpleModule()
            cached = CacheModuleDecorator(module, backend, enabled=True)

            x = torch.randn(4, 10)

            with pytest.raises(ValueError, match="cache_ids"):
                cached(x)


class TestDecoratorStatelessEnforcement:
    """Test stateless enforcement."""

    def test_stateless_enforcement(self):
        """Test that trainable params trigger error when enforce_stateless=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            # Module with trainable params
            class TrainableModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def forward(self, x):
                    return self.linear(x)

            module = TrainableModule()
            module.train()  # Set to training mode
            cached = CacheModuleDecorator(module, backend, enabled=True, enforce_stateless=True)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            with pytest.raises(RuntimeError, match="trainable parameters"):
                cached(x, cache_ids=cache_ids)

    def test_allow_trainable_with_flag(self):
        """Test that trainable params are allowed when enforce_stateless=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            class TrainableModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def forward(self, x):
                    return self.linear(x)

            module = TrainableModule()
            module.eval()  # Set to eval mode
            cached = CacheModuleDecorator(module, backend, enabled=True, enforce_stateless=False)

            x = torch.randn(4, 10)
            cache_ids = ["id1", "id2", "id3", "id4"]

            # Should not raise
            out = cached(x, cache_ids=cache_ids)
            assert out.shape == (4, 10)


class TestDecoratorStateDict:
    """Test state dict save/load."""

    def test_state_dict(self):
        """Test that state_dict only contains inner module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ArrowIPCCacheBackend(
                cache_dir=tmpdir, module_id="test", async_write=False
            )

            class ModuleWithParams(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def forward(self, x):
                    return self.linear(x)

            module = ModuleWithParams()
            cached = CacheModuleDecorator(module, backend, enabled=True, enforce_stateless=False)

            state = cached.state_dict()
            # Should only have inner module's params
            assert "linear.weight" in state
            assert "linear.bias" in state
            # Should not have cache-related state
            assert "cache" not in str(state.keys())
