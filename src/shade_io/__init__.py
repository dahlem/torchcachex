"""SHADE I/O - Modular I/O layer for SHADE feature extraction and caching.

This package provides a clean, modular architecture for:
- Feature extraction and computation
- Feature caching and storage
- Model and feature registries
- Integration with existing SHADE components
"""

__version__ = "0.1.0"

# Multi-candidate cached feature sets
# Feature sets
from shade_io.feature_sets.base import (
    CompositeFeatureSet,
    LazyCompositeFeatureSet,
    MultiCandidateCachedFeatureSet,
    SimpleFeatureSet,
)

# Decorators
from shade_io.feature_sets.decorators import (
    CachedFeatureSet,
    FilteredFeatureSet,
    LoggedFeatureSet,
    ValidatedFeatureSet,
)

# Core interfaces
from shade_io.interfaces.core import (
    AttentionData,
    FeatureKey,
    FeatureMetadata,
    FeatureResult,
    IFeatureFilter,
    IFeatureSet,
    IFeatureStore,
)

# Processor
from shade_io.processor.feature_processor import (
    FeatureProcessor,
    ParallelFeatureProcessor,
)

# Registries
from shade_io.registry import (
    FeatureInfo,
    ModelInfo,
    ModelRegistry,
    PCAModelInfo,
)

# Unified Registry
from shade_io.registry.unified_registry import (
    FeatureMatch,
    ModelMatch,
    StorageManager,
    UnifiedMatch,
    UnifiedRegistry,
)
from shade_io.stores.composite import CompositeStore

# Stores
from shade_io.stores.file import FileFeatureStore
from shade_io.stores.memory import MemoryFeatureStore

__all__ = [
    # Version
    "__version__",
    # Interfaces
    "IFeatureSet",
    "IFeatureStore",
    "IFeatureFilter",
    "AttentionData",
    "FeatureResult",
    "FeatureKey",
    "FeatureMetadata",
    # Feature Sets
    "SimpleFeatureSet",
    "CompositeFeatureSet",
    "LazyCompositeFeatureSet",
    # Decorators
    "FilteredFeatureSet",
    "CachedFeatureSet",
    "LoggedFeatureSet",
    "ValidatedFeatureSet",
    # Multi-candidate cached feature sets
    "MultiCandidateCachedFeatureSet",
    # Stores
    "FileFeatureStore",
    "MemoryFeatureStore",
    "CompositeStore",
    # Processor
    "FeatureProcessor",
    "ParallelFeatureProcessor",
    # Registries
    "ModelRegistry",
    "ModelInfo",
    "FeatureInfo",
    "PCAModelInfo",
    # Unified Registry
    "UnifiedRegistry",
    "FeatureMatch",
    "PCAMatch",
    "ModelMatch",
    "UnifiedMatch",
    "StorageManager",
]
