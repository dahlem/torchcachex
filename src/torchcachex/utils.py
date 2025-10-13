"""Utility functions for tree manipulation and serialization.

This module provides functions for working with nested tensor structures
(trees of tensors, lists, tuples, and dicts) and serialization helpers.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Production"

import io
from collections.abc import Mapping, Sequence
from typing import Any

import torch

# Type alias for nested tensor structures
TensorLike = torch.Tensor | Sequence | Mapping


def _tree_index(x: Any, i: int) -> Any:
    """Extract the i-th sample from a nested structure.

    Supports:
    - torch.Tensor: indexes on dim 0
    - list/tuple: recursively indexes each element
    - dict: recursively indexes each value
    - other: returns as-is (non-tensor leaf)

    Args:
        x: Nested structure to index
        i: Sample index

    Returns:
        The i-th sample from the structure

    Examples:
        >>> tensor = torch.randn(4, 3, 224, 224)
        >>> sample = _tree_index(tensor, 0)  # shape: (3, 224, 224)

        >>> nested = {"img": torch.randn(4, 3, 224, 224), "label": torch.tensor([1, 2, 3, 4])}
        >>> sample = _tree_index(nested, 1)  # {"img": (3, 224, 224), "label": 2}
    """
    if torch.is_tensor(x):
        return x[i]
    if isinstance(x, list):
        return [_tree_index(xx, i) for xx in x]
    if isinstance(x, tuple):
        return tuple(_tree_index(xx, i) for xx in x)
    if isinstance(x, dict):
        return {k: _tree_index(v, i) for k, v in x.items()}
    # Non-tensor leaf (e.g., int, str) - return as-is
    return x


def _tree_stack(samples: list[Any]) -> Any:
    """Stack a list of samples into a batch.

    Inverse operation of _tree_index. Supports the same nested structures.

    Args:
        samples: List of samples to stack

    Returns:
        Batched structure

    Examples:
        >>> samples = [torch.randn(3, 224, 224) for _ in range(4)]
        >>> batch = _tree_stack(samples)  # shape: (4, 3, 224, 224)

        >>> samples = [{"img": torch.randn(3, 224, 224), "label": i} for i in range(4)]
        >>> batch = _tree_stack(samples)  # {"img": (4, 3, 224, 224), "label": [0,1,2,3]}
    """
    if not samples:
        raise ValueError("Cannot stack empty list of samples")

    x0 = samples[0]

    if torch.is_tensor(x0):
        return torch.stack(samples, dim=0)

    if isinstance(x0, list):
        # Stack each position in the list
        n_elements = len(x0)
        cols = [[sample[j] for sample in samples] for j in range(n_elements)]
        return [_tree_stack(col) for col in cols]

    if isinstance(x0, tuple):
        # Stack each position in the tuple
        n_elements = len(x0)
        cols = [[sample[j] for sample in samples] for j in range(n_elements)]
        return tuple(_tree_stack(col) for col in cols)

    if isinstance(x0, dict):
        # Stack each value in the dict
        return {k: _tree_stack([s[k] for s in samples]) for k in x0.keys()}

    # Non-tensor leaves: return list as-is (e.g., list of ints/strings)
    return samples


def _serialize_sample(obj: Any) -> bytes:
    """Serialize a sample to bytes using torch.save.

    Stores a CPU version to ensure device-agnostic caching.

    Args:
        obj: Sample to serialize (tensor or nested structure)

    Returns:
        Serialized bytes
    """

    def to_cpu(o: Any) -> Any:
        """Recursively move tensors to CPU."""
        if torch.is_tensor(o):
            return o.detach().cpu()
        if isinstance(o, (list, tuple)):
            return type(o)(to_cpu(v) for v in o)
        if isinstance(o, dict):
            return {k: to_cpu(v) for k, v in o.items()}
        return o

    bio = io.BytesIO()
    torch.save(to_cpu(obj), bio)
    return bio.getvalue()


def _deserialize_sample(blob: bytes, map_location: str = "cpu") -> Any:
    """Deserialize a sample from bytes.

    Args:
        blob: Serialized bytes
        map_location: Device to load tensors to (default: "cpu")

    Returns:
        Deserialized sample
    """
    bio = io.BytesIO(blob)
    return torch.load(bio, map_location=map_location, weights_only=False)
