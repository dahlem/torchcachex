"""Utility functions for tree manipulation and serialization.

This module provides functions for working with nested tensor structures
(trees of tensors, lists, tuples, and dicts) and serialization helpers.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Production"

import io
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Type alias for nested tensor structures
TensorLike = torch.Tensor | Sequence | Mapping


def _tree_index(x: Any, i: int) -> Any:
    """Extract the i-th sample from a nested structure.

    Supports:
    - torch.Tensor: indexes on dim 0
    - list of tensors (from variable-length stacking): extracts i-th tensor directly
    - list/tuple of other structures: recursively indexes each element
    - dict: recursively indexes each value
    - other: returns as-is (non-tensor leaf)

    Args:
        x: Nested structure to index
        i: Sample index

    Returns:
        The i-th sample from the structure

    Examples:
        >>> # Stacked batch tensor
        >>> tensor = torch.randn(4, 3, 224, 224)
        >>> sample = _tree_index(tensor, 0)  # shape: (3, 224, 224)

        >>> # List of variable-length tensors (from _tree_stack with variable shapes)
        >>> tensor_list = [torch.randn(12, 30, 30), torch.randn(12, 50, 50)]
        >>> sample = _tree_index(tensor_list, 1)  # shape: (12, 50, 50)

        >>> nested = {"img": torch.randn(4, 3, 224, 224), "label": torch.tensor([1, 2, 3, 4])}
        >>> sample = _tree_index(nested, 1)  # {"img": (3, 224, 224), "label": 2}
    """
    if torch.is_tensor(x):
        return x[i]
    if isinstance(x, list):
        # Check if it's a list of tensors (from variable-length stacking)
        if x and torch.is_tensor(x[0]):
            # List of tensors - return i-th tensor directly
            return x[i]
        # List of other structures - recursively index each element
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

    For tensors with uniform shapes, returns a stacked batch tensor.
    For tensors with variable shapes (e.g., different sequence lengths),
    returns a list of tensors to preserve exact shapes.

    Args:
        samples: List of samples to stack

    Returns:
        Batched structure (stacked tensor if shapes match, list otherwise)

    Examples:
        >>> # Uniform shapes - returns stacked batch
        >>> samples = [torch.randn(3, 224, 224) for _ in range(4)]
        >>> batch = _tree_stack(samples)  # shape: (4, 3, 224, 224)

        >>> # Variable shapes - returns list
        >>> samples = [torch.randn(12, 30, 30), torch.randn(12, 50, 50)]
        >>> batch = _tree_stack(samples)  # List[Tensor] with shapes [(12,30,30), (12,50,50)]

        >>> samples = [{"img": torch.randn(3, 224, 224), "label": i} for i in range(4)]
        >>> batch = _tree_stack(samples)  # {"img": (4, 3, 224, 224), "label": [0,1,2,3]}
    """
    if not samples:
        raise ValueError("Cannot stack empty list of samples")

    x0 = samples[0]

    if torch.is_tensor(x0):
        # Check if all tensors have the same shape
        shapes = [s.shape for s in samples]
        if all(shape == shapes[0] for shape in shapes):
            # All same shape - stack normally for efficient batch processing
            return torch.stack(samples, dim=0)
        else:
            # Variable shapes - return list to preserve exact dimensions
            # This is seamless for variable-length sequence processing
            logger.debug(
                f"Variable tensor shapes detected in batch: {shapes}. "
                "Returning list instead of stacked batch to preserve exact shapes."
            )
            return samples

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
