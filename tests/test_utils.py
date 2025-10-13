"""Tests for tree manipulation and serialization utilities."""

import torch

from torchcachex.utils import (
    _deserialize_sample,
    _serialize_sample,
    _tree_index,
    _tree_stack,
)


class TestTreeIndex:
    """Test _tree_index function."""

    def test_tensor_indexing(self):
        """Test indexing a simple tensor."""
        x = torch.randn(4, 3, 224, 224)
        sample = _tree_index(x, 1)
        assert sample.shape == (3, 224, 224)
        assert torch.allclose(sample, x[1])

    def test_list_indexing(self):
        """Test indexing a list of tensors."""
        x = [torch.randn(4, 3, 224, 224), torch.randn(4, 10)]
        sample = _tree_index(x, 2)
        assert isinstance(sample, list)
        assert len(sample) == 2
        assert sample[0].shape == (3, 224, 224)
        assert sample[1].shape == (10,)

    def test_tuple_indexing(self):
        """Test indexing a tuple of tensors."""
        x = (torch.randn(4, 3, 224, 224), torch.randn(4, 10))
        sample = _tree_index(x, 0)
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_dict_indexing(self):
        """Test indexing a dict of tensors."""
        x = {"img": torch.randn(4, 3, 224, 224), "label": torch.tensor([1, 2, 3, 4])}
        sample = _tree_index(x, 1)
        assert isinstance(sample, dict)
        assert sample["img"].shape == (3, 224, 224)
        assert sample["label"].item() == 2

    def test_nested_structure(self):
        """Test indexing a deeply nested structure."""
        x = {
            "data": [torch.randn(4, 10), torch.randn(4, 20)],
            "meta": {"ids": torch.arange(4), "labels": [1, 2, 3, 4]},
        }
        sample = _tree_index(x, 2)
        assert sample["data"][0].shape == (10,)
        assert sample["data"][1].shape == (20,)
        assert sample["meta"]["ids"].item() == 2
        assert sample["meta"]["labels"] == [1, 2, 3, 4]  # Non-tensor leaf


class TestTreeStack:
    """Test _tree_stack function."""

    def test_tensor_stacking(self):
        """Test stacking simple tensors."""
        samples = [torch.randn(3, 224, 224) for _ in range(4)]
        batch = _tree_stack(samples)
        assert batch.shape == (4, 3, 224, 224)

    def test_list_stacking(self):
        """Test stacking lists of tensors."""
        samples = [[torch.randn(10), torch.randn(20)] for _ in range(4)]
        batch = _tree_stack(samples)
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert batch[0].shape == (4, 10)
        assert batch[1].shape == (4, 20)

    def test_tuple_stacking(self):
        """Test stacking tuples of tensors."""
        samples = [(torch.randn(10), torch.randn(20)) for _ in range(4)]
        batch = _tree_stack(samples)
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        assert batch[0].shape == (4, 10)

    def test_dict_stacking(self):
        """Test stacking dicts of tensors."""
        samples = [
            {"img": torch.randn(3, 224, 224), "label": torch.tensor(i)}
            for i in range(4)
        ]
        batch = _tree_stack(samples)
        assert isinstance(batch, dict)
        assert batch["img"].shape == (4, 3, 224, 224)
        assert batch["label"].shape == (4,)

    def test_roundtrip(self):
        """Test that index→stack is identity."""
        original = torch.randn(4, 3, 224, 224)
        samples = [_tree_index(original, i) for i in range(4)]
        reconstructed = _tree_stack(samples)
        assert torch.allclose(original, reconstructed)


class TestSerialization:
    """Test serialization/deserialization."""

    def test_tensor_serialization(self):
        """Test serializing a simple tensor."""
        x = torch.randn(3, 224, 224)
        blob = _serialize_sample(x)
        y = _deserialize_sample(blob)
        assert torch.allclose(x, y)

    def test_nested_serialization(self):
        """Test serializing nested structures."""
        x = {
            "img": torch.randn(3, 224, 224),
            "label": torch.tensor(5),
            "meta": [torch.randn(10), torch.randn(20)],
        }
        blob = _serialize_sample(x)
        y = _deserialize_sample(blob)
        assert torch.allclose(x["img"], y["img"])
        assert x["label"].item() == y["label"].item()
        assert torch.allclose(x["meta"][0], y["meta"][0])

    def test_device_agnostic(self):
        """Test that serialization moves to CPU."""
        if torch.cuda.is_available():
            x = torch.randn(3, 224, 224, device="cuda")
            blob = _serialize_sample(x)
            y = _deserialize_sample(blob)
            assert y.device == torch.device("cpu")
            assert torch.allclose(x.cpu(), y)
