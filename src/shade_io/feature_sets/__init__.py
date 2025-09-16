"""Feature set implementations for SHADE I/O."""

from shade_io.feature_sets.base import (
    CompositeFeatureSet,
    LazyCompositeFeatureSet,
    SimpleFeatureSet,
)
from shade_io.feature_sets.decorators import (
    CachedFeatureSet,
    FilteredFeatureSet,
    LoggedFeatureSet,
    ValidatedFeatureSet,
)

# Import filters if they exist
try:
    from shade_io.feature_sets.filters import (
        RemoveConstantFeaturesFilter,
        RemoveCorrelatedFeaturesFilter,
        RemoveFirstKFilter,
    )

    __all__ = [
        # Base
        "SimpleFeatureSet",
        "CompositeFeatureSet",
        "LazyCompositeFeatureSet",
        # Decorators
        "FilteredFeatureSet",
        "CachedFeatureSet",
        "LoggedFeatureSet",
        "ValidatedFeatureSet",
        # Filters
        "RemoveConstantFeaturesFilter",
        "RemoveFirstKFilter",
        "RemoveCorrelatedFeaturesFilter",
    ]
except ImportError:
    __all__ = [
        # Base
        "SimpleFeatureSet",
        "CompositeFeatureSet",
        "LazyCompositeFeatureSet",
        # Decorators
        "FilteredFeatureSet",
        "CachedFeatureSet",
        "LoggedFeatureSet",
        "ValidatedFeatureSet",
    ]
