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
        """Test indexing a list structure (not list of tensors)."""
        # Use tuple instead of list for nested structures to avoid ambiguity
        x = (torch.randn(4, 3, 224, 224), torch.randn(4, 10))
        sample = _tree_index(x, 2)
        assert isinstance(sample, tuple)
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
        # Use tuple instead of list for batched tensors to avoid ambiguity
        x = {
            "data": (torch.randn(4, 10), torch.randn(4, 20)),
            "meta": {"ids": torch.arange(4), "labels": [1, 2, 3, 4]},
        }
        sample = _tree_index(x, 2)
        assert sample["data"][0].shape == (10,)
        assert sample["data"][1].shape == (20,)
        assert sample["meta"]["ids"].item() == 2
        assert sample["meta"]["labels"] == [1, 2, 3, 4]  # Non-tensor leaf

    def test_list_of_tensors_indexing(self):
        """Test indexing a list of tensors (from variable-length stacking)."""
        # This simulates the output of _tree_stack with variable-length tensors
        tensor_list = [
            torch.randn(12, 30, 30),
            torch.randn(12, 50, 50),
            torch.randn(12, 40, 40),
        ]

        # Index should return the i-th tensor directly
        sample = _tree_index(tensor_list, 1)
        assert torch.is_tensor(sample)
        assert sample.shape == (12, 50, 50)
        assert sample is tensor_list[1]  # Same object

    def test_variable_length_roundtrip(self):
        """Test that variable-length tensors can be indexed from list result."""
        # Create variable-length samples
        samples = [
            torch.randn(12, 30, 30),
            torch.randn(12, 50, 50),
            torch.randn(12, 40, 40),
        ]

        # Stack returns list
        stacked = _tree_stack(samples)
        assert isinstance(stacked, list)

        # Index into the list should return original tensors
        for i in range(3):
            indexed = _tree_index(stacked, i)
            assert torch.allclose(indexed, samples[i])


class TestTreeStack:
    """Test _tree_stack function."""

    def test_tensor_stacking(self):
        """Test stacking simple tensors."""
        samples = [torch.randn(3, 224, 224) for _ in range(4)]
        batch = _tree_stack(samples)
        assert batch.shape == (4, 3, 224, 224)

    def test_list_stacking(self):
        """Test stacking tuples of tensors (lists are reserved for variable-length)."""
        samples = [(torch.randn(10), torch.randn(20)) for _ in range(4)]
        batch = _tree_stack(samples)
        assert isinstance(batch, tuple)
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

    def test_variable_length_tensors_return_list(self):
        """Test that variable-length tensors return list instead of stacking."""
        # Create samples with different sequence lengths (e.g., attention matrices)
        samples = [
            torch.randn(12, 30, 30),  # seq_len=30
            torch.randn(12, 50, 50),  # seq_len=50
            torch.randn(12, 40, 40),  # seq_len=40
        ]

        result = _tree_stack(samples)

        # Should return list (not stacked tensor) due to shape mismatch
        assert isinstance(result, list), "Expected list for variable-length tensors"
        assert len(result) == 3
        assert result[0].shape == (12, 30, 30)
        assert result[1].shape == (12, 50, 50)
        assert result[2].shape == (12, 40, 40)

        # Verify tensors are the same objects (no copying)
        assert result[0] is samples[0]
        assert result[1] is samples[1]
        assert result[2] is samples[2]

    def test_uniform_tensors_still_stack(self):
        """Test that uniform-shape tensors still get stacked (backward compatibility)."""
        samples = [
            torch.randn(12, 30, 30),
            torch.randn(12, 30, 30),
            torch.randn(12, 30, 30),
        ]

        result = _tree_stack(samples)

        # Should stack normally
        assert torch.is_tensor(result), "Expected stacked tensor for uniform shapes"
        assert result.shape == (3, 12, 30, 30)

    def test_variable_length_in_dict(self):
        """Test variable-length tensors inside dict structures."""
        samples = [
            {
                "attention": torch.randn(12, 30, 30),
                "logits": torch.randn(30, 50257),
            },
            {
                "attention": torch.randn(12, 50, 50),
                "logits": torch.randn(50, 50257),
            },
        ]

        result = _tree_stack(samples)

        # Dict should be preserved
        assert isinstance(result, dict)

        # Attention should be list (variable shapes)
        assert isinstance(result["attention"], list)
        assert len(result["attention"]) == 2
        assert result["attention"][0].shape == (12, 30, 30)
        assert result["attention"][1].shape == (12, 50, 50)

        # Logits should also be list (variable first dimension)
        assert isinstance(result["logits"], list)
        assert len(result["logits"]) == 2
        assert result["logits"][0].shape == (30, 50257)
        assert result["logits"][1].shape == (50, 50257)

    def test_mixed_stackable_and_variable(self):
        """Test dict with both stackable and variable-length tensors."""
        samples = [
            {
                "attention": torch.randn(12, 30, 30),  # Variable (30 vs 50)
                "labels": torch.tensor(1),  # Uniform (scalar)
                "embeddings": torch.randn(768),  # Uniform (768)
            },
            {
                "attention": torch.randn(12, 50, 50),  # Variable
                "labels": torch.tensor(0),  # Uniform
                "embeddings": torch.randn(768),  # Uniform
            },
        ]

        result = _tree_stack(samples)

        assert isinstance(result, dict)

        # Variable-length should be list
        assert isinstance(result["attention"], list)
        assert result["attention"][0].shape == (12, 30, 30)
        assert result["attention"][1].shape == (12, 50, 50)

        # Uniform tensors should be stacked
        assert torch.is_tensor(result["labels"])
        assert result["labels"].shape == (2,)
        assert torch.is_tensor(result["embeddings"])
        assert result["embeddings"].shape == (2, 768)


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
