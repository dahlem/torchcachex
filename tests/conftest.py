"""Pytest configuration and fixtures for torchcachex tests."""
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory that's cleaned up after tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def simple_module():
    """Create a simple test module for caching."""

    class SimpleModule(torch.nn.Module):
        def forward(self, x):
            # Simple transformation that's easy to verify
            return x * 2 + 1

    return SimpleModule()


@pytest.fixture
def complex_module():
    """Create a module with complex (nested) outputs."""

    class ComplexModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            return {
                "features": x @ self.W,
                "norm": x.norm(dim=-1),
                "metadata": [x.mean().item(), x.std().item()],
            }

    return ComplexModule()


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "input": torch.randn(8, 4),
        "cache_ids": [f"sample_{i}" for i in range(8)],
    }
