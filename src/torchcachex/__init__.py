"""torchcachex: Drop-in PyTorch module caching with Arrow IPC + in-memory index backend.

This library provides transparent, per-sample caching for non-trainable PyTorch modules
with O(1) append-only writes, native tensor storage, batched lookups, LRU hot cache,
async writes, and DDP safety. Scales to billions of samples with constant memory usage.

Basic usage:
    >>> from torchcachex import CacheModuleDecorator, ArrowIPCCacheBackend
    >>>
    >>> # Create backend
    >>> backend = ArrowIPCCacheBackend(
    ...     cache_dir="./cache",
    ...     module_id="my_features_v1",
    ...     lru_size=4096
    ... )
    >>>
    >>> # Wrap module
    >>> cached_module = CacheModuleDecorator(
    ...     module=MyFeatureExtractor(),
    ...     cache_backend=backend,
    ...     enabled=True
    ... )
    >>>
    >>> # Use in forward pass (cache_ids required)
    >>> output = cached_module(batch_input, cache_ids=batch["cache_ids"])
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Production"
__version__ = "0.2.0"

from .backend import ArrowIPCCacheBackend
from .decorator import CacheModuleDecorator

__all__ = [
    "ArrowIPCCacheBackend",
    "CacheModuleDecorator",
    "__version__",
]
