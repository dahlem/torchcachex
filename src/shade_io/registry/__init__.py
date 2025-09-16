"""Registry components for shade-io."""

from shade_io.registry.feature_registry import (
    FeatureInfo,
    FeatureRegistry,
    PCAModelInfo,
)
from shade_io.registry.model_registry import ModelInfo, ModelRegistry
from shade_io.registry.unified_registry import UnifiedRegistry

__all__ = [
    "ModelRegistry",
    "ModelInfo",
    "FeatureRegistry",
    "FeatureInfo",
    "PCAModelInfo",
    "UnifiedRegistry",
]
