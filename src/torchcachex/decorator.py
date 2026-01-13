"""PyTorch module decorator for transparent per-sample caching.

This module provides a drop-in wrapper for any nn.Module to add efficient,
transparent caching with batched lookups and progressive enrichment.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Production"

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from .backend import ArrowIPCCacheBackend
from .utils import _tree_index, _tree_stack

logger = logging.getLogger(__name__)


class CacheModuleDecorator(nn.Module):
    """Wraps an nn.Module to add read-through caching & async write-back per sample.

    This decorator transparently adds caching to any PyTorch module. It:
    - Extracts `cache_ids` from batch kwargs/dict inputs
    - Performs batched lookups with LRU + Arrow push-down
    - Computes only missing samples
    - Stores results asynchronously (non-blocking forward pass)
    - Handles arbitrary output structures (tensors, tuples, dicts)

    Assumptions:
    - First positional arg is batch-like tensor indexable on dim 0
    - Batch carries `cache_ids` either as kwarg or inside dict-like arg
    - Module is stateless (no trainable params during caching)

    Args:
        module: PyTorch module to wrap
        cache_backend: Cache backend for persistence
        enabled: Whether caching is enabled (default: True)
        key_from_batch_fn: Optional function to extract cache IDs from batch
        enforce_stateless: Check that module has no trainable params (default: True)
        map_location_on_read: Device to load cached tensors to (default: "cpu")

    Example:
        >>> feature_extractor = MyFeatureExtractor()
        >>> backend = ArrowIPCCacheBackend(cache_dir="./cache", module_id="my_features_v1")
        >>> cached = CacheModuleDecorator(feature_extractor, backend, enabled=True)
        >>>
        >>> # In training loop
        >>> for batch in dataloader:
        >>>     features = cached(batch["input"], cache_ids=batch["cache_ids"])
    """

    def __init__(
        self,
        module: nn.Module,
        cache_backend: ArrowIPCCacheBackend,
        enabled: bool = True,
        key_from_batch_fn: Callable[..., list[str]] | None = None,
        enforce_stateless: bool = True,
        map_location_on_read: str = "cpu",
    ):
        super().__init__()
        self.module = module
        self.cache = cache_backend
        self.enabled = enabled
        self.key_from_batch_fn = key_from_batch_fn
        self.enforce_stateless = enforce_stateless
        self.map_location_on_read = map_location_on_read

        # Log module decoration for visibility
        logger.info(
            f"Decorated module: {type(module).__name__} "
            f"(cache_key: {cache_backend.module_id}, enabled: {enabled})"
        )

    def _extract_keys(self, args: tuple, kwargs: dict) -> tuple[list[str], list[int]]:
        """Extract cache keys from batch.

        Tries in order:
        1. kwargs['cache_ids']
        2. args[0]['cache_ids'] if args[0] is dict
        3. Custom key_from_batch_fn if provided
        4. Raises ValueError if none found

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Tuple of (keys, positions) where len == batch_size
        """
        # 1) kwargs
        if "cache_ids" in kwargs and kwargs["cache_ids"] is not None:
            ids = list(kwargs["cache_ids"])
            return ids, list(range(len(ids)))

        # 2) batch dict
        if args and isinstance(args[0], dict) and "cache_ids" in args[0]:
            ids = list(args[0]["cache_ids"])
            return ids, list(range(len(ids)))

        # 3) custom user function
        if self.key_from_batch_fn is not None:
            ids = list(self.key_from_batch_fn(*args, **kwargs))
            return ids, list(range(len(ids)))

        raise ValueError(
            "CacheModuleDecorator requires `cache_ids` in kwargs or batch dict, "
            "or provide `key_from_batch_fn` to derive per-sample keys."
        )

    def _move_like_input(self, obj: Any, ref: torch.Tensor) -> Any:
        """Move cached object to same device as input tensor.

        Args:
            obj: Object to move (can be nested structure)
            ref: Reference tensor for device

        Returns:
            Object moved to target device
        """
        device = ref.device if torch.is_tensor(ref) else torch.device("cpu")
        # MPS doesn't fully support non_blocking transfers - can cause hangs
        use_non_blocking = device.type not in ("mps",)

        def move(o: Any) -> Any:
            if torch.is_tensor(o):
                return o.to(device=device, non_blocking=use_non_blocking)
            if isinstance(o, list):
                return [move(v) for v in o]
            if isinstance(o, tuple):
                return tuple(move(v) for v in o)
            if isinstance(o, dict):
                return {k: move(v) for k, v in o.items()}
            return o

        return move(obj)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with caching.

        Args:
            *args: Positional arguments to wrapped module
            **kwargs: Keyword arguments (must include cache_ids or be extractable)

        Returns:
            Module output (from cache or freshly computed)
        """
        # Filter out cache_ids from kwargs before passing to wrapped module
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "cache_ids"}

        if not self.enabled:
            return self.module(*args, **kwargs_clean)

        # Extract keys and batch size
        keys, positions = self._extract_keys(args, kwargs)

        # Batch lookups (LRU + Arrow push-down)
        cached_objs, missing_pos = self.cache.get_batch(
            keys, map_location=self.map_location_on_read
        )

        # Stateless guardrail
        if self.enforce_stateless and self.training:
            for p in self.module.parameters(recurse=True):
                if p.requires_grad:
                    raise RuntimeError(
                        "CacheModuleDecorator: wrapped module has trainable parameters "
                        "while caching is enabled. Set enforce_stateless=False to allow."
                    )

        if not missing_pos:
            # All hits: reassemble batch on input device
            ref = args[0] if args else None
            batch = _tree_stack(
                [self._move_like_input(cached_objs[i], ref) for i in positions]
            )
            return batch

        # Build sub-batch for misses (assume args[0] is indexable on dim 0)
        miss_idx = torch.tensor(missing_pos, dtype=torch.long)

        if args and torch.is_tensor(args[0]):
            x = args[0]
            x_miss = x.index_select(0, miss_idx.to(device=x.device))
            args_miss = (x_miss, *args[1:])
        else:
            # Fallback: send full args; compute and index outputs per-sample
            args_miss = args

        with torch.no_grad():
            out_miss = self.module(*args_miss, **kwargs_clean)

        # MPS requires sync before CPU copy to avoid hangs
        # Check if output contains MPS tensors and sync once
        def _has_mps_tensor(obj: Any) -> bool:
            if torch.is_tensor(obj):
                return obj.device.type == "mps"
            if isinstance(obj, dict):
                return any(_has_mps_tensor(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return any(_has_mps_tensor(v) for v in obj)
            return False

        if _has_mps_tensor(out_miss):
            torch.mps.synchronize()

        # Helper to recursively move tensors to CPU for safe caching
        def _to_cpu(obj: Any) -> Any:
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_cpu(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_cpu(v) for v in obj)
            return obj

        # Slice per sample, write to cache, assemble final batch
        new_entries = {}
        per_sample = {}

        for j, pos in enumerate(missing_pos):
            sample_out = _tree_index(out_miss, j)  # j-th in miss sub-batch
            per_sample[pos] = sample_out
            # Move to CPU for safe caching (avoids MPS serialization hangs)
            new_entries[keys[pos]] = _to_cpu(sample_out)

        self.cache.put_batch(new_entries)

        # Merge hits + misses in input order; stack back to batch
        ref = args[0] if args else None

        # MPS sync before moving tensors (per_sample may contain MPS tensors)
        has_mps_samples = any(
            _has_mps_tensor(per_sample.get(i)) for i in positions if i in per_sample
        )
        if has_mps_samples:
            torch.mps.synchronize()

        merged = []
        for i in positions:
            obj = per_sample[i] if i in per_sample else cached_objs[i]
            merged.append(self._move_like_input(obj, ref))

        result = _tree_stack(merged)

        # Final MPS sync to ensure all transfers are complete
        # This prevents MPS state corruption after return
        if has_mps_samples:
            torch.mps.synchronize()

        return result

    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """Save only the inner module's weights (cache state is out-of-band).

        Args:
            *args: Arguments to module.state_dict
            **kwargs: Keyword arguments to module.state_dict

        Returns:
            State dict of wrapped module
        """
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: dict, strict: bool = True, assign: bool = False
    ) -> Any:
        """Load state dict into wrapped module.

        Args:
            state_dict: State dict to load
            strict: Whether to strictly enforce key matching
            assign: Whether to assign (PyTorch 2.0+)

        Returns:
            Result from module.load_state_dict
        """
        return self.module.load_state_dict(state_dict, strict=strict, assign=assign)
