"""Decorator implementations for feature sets.

These decorators add cross-cutting concerns like filtering, caching,
and logging to any feature set implementation.
"""

import logging
import time

import torch

from shade_io.interfaces.core import (
    AttentionData,
    FeatureKey,
    FeatureMetadata,
    FeatureResult,
    IFeatureFilter,
    IFeatureSet,
    IFeatureStore,
)

logger = logging.getLogger(__name__)


class FilteredFeatureSet(IFeatureSet):
    """Decorator that applies filters to another feature set's output.

    This implements the decorator pattern, allowing filters to be
    composed and nested arbitrarily.
    """

    def __init__(
        self,
        base: IFeatureSet,
        filters: list[IFeatureFilter],
        name: str | None = None,
    ):
        """Initialize filtered feature set.

        Args:
            base: Base feature set to decorate
            filters: List of filters to apply sequentially
            name: Optional name override (defaults to base_filtered)
        """
        self.base = base
        self.filters = filters
        self._name = name or f"{base.name}_filtered"
        self._cached_metadata: FeatureMetadata | None = None

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self._name

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features and apply filters.

        Args:
            attention_data: Input attention data

        Returns:
            Filtered features
        """
        # Get base features
        result = self.base.compute_features(attention_data)

        # Apply filters sequentially
        for filter_obj in self.filters:
            initial_count = result.feature_dim
            result = filter_obj.apply(result)
            final_count = result.feature_dim

            logger.info(
                f"Applied {filter_obj.name}: {initial_count} -> {final_count} features"
            )

        # Update metadata
        result.metadata["filtered"] = True
        result.metadata["filters_applied"] = [f.name for f in self.filters]
        result.metadata["original_feature_count"] = self.base.feature_dim
        result.metadata["filtered_feature_count"] = result.feature_dim

        return result

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about filtered features.

        This is challenging because we don't know which features
        will be filtered until we actually run the filters.
        """
        if self._cached_metadata:
            return self._cached_metadata

        # Get base metadata
        base_meta = self.base.get_metadata()

        # We can't know exact features without running filters
        # Return base metadata with a flag
        return FeatureMetadata(
            name=self.name,
            feature_dim=base_meta.feature_dim,  # Upper bound
            feature_names=base_meta.feature_names,  # May change
            description=f"Filtered version of {base_meta.name}",
            configuration={
                "base": base_meta.name,
                "filters": [f.name for f in self.filters],
                "requires_computation": True,  # Can't predict without running
            },
        )

    @property
    def feature_dim(self) -> int:
        """Get feature dimension.

        Note: This returns the base dimension as we can't know
        the filtered dimension without actually running the filters.
        """
        return self.base.feature_dim


class CachedFeatureSet(IFeatureSet):
    """Decorator that adds caching to any feature set.

    This transparently adds caching behavior to any feature set,
    checking the cache before computing and saving after computation.
    """

    def __init__(
        self,
        base: IFeatureSet,
        store: IFeatureStore,
        cache_ttl: int | None = None,
        force_recompute: bool = False,
    ):
        """Initialize cached feature set.

        Args:
            base: Base feature set to decorate
            store: Storage backend for caching
            cache_ttl: Optional time-to-live in seconds
            force_recompute: If True, always recompute
        """
        self.base = base
        self.store = store
        self.cache_ttl = cache_ttl
        self.force_recompute = force_recompute

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self.base.name

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features with caching.

        Args:
            attention_data: Input attention data

        Returns:
            Cached or computed features
        """
        # Build cache key
        key = FeatureKey(
            feature_set_name=self.name,
            model_name=attention_data.model_name,
            dataset_name=attention_data.dataset_name,
            metadata=attention_data.metadata,
        )

        # Check cache first
        if not self.force_recompute:
            cached = self.store.load(key)
            if cached:
                logger.info(f"Loaded features from cache: {key.to_string()}")
                cached.metadata["from_cache"] = True
                return cached

        # Compute if not cached
        logger.info(f"Computing features: {key.to_string()}")
        result = self.base.compute_features(attention_data)

        # Save to cache
        self.store.save(key, result)
        logger.info(f"Saved features to cache: {key.to_string()}")

        result.metadata["from_cache"] = False
        return result

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata from base set."""
        meta = self.base.get_metadata()
        meta.configuration["cached"] = True
        meta.configuration["cache_ttl"] = self.cache_ttl
        return meta

    @property
    def feature_dim(self) -> int:
        """Get feature dimension from base set."""
        return self.base.feature_dim


class LoggedFeatureSet(IFeatureSet):
    """Decorator that adds logging and metrics to feature computation.

    This is useful for debugging and performance monitoring.
    """

    def __init__(
        self,
        base: IFeatureSet,
        log_level: str = "INFO",
        log_timing: bool = True,
        log_shapes: bool = True,
    ):
        """Initialize logged feature set.

        Args:
            base: Base feature set to decorate
            log_level: Logging level
            log_timing: Whether to log computation time
            log_shapes: Whether to log tensor shapes
        """
        self.base = base
        self.log_level = getattr(logging, log_level)
        self.log_timing = log_timing
        self.log_shapes = log_shapes

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self.base.name

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features with logging.

        Args:
            attention_data: Input attention data

        Returns:
            Computed features
        """
        logger.log(self.log_level, f"Computing features for {self.name}")

        if self.log_shapes:
            logger.log(
                self.log_level,
                f"Input shape: {attention_data.attention_matrices.shape}",
            )

        start_time = time.time() if self.log_timing else None

        # Compute features
        result = self.base.compute_features(attention_data)

        if self.log_timing:
            elapsed = time.time() - start_time
            logger.log(self.log_level, f"Computation took {elapsed:.2f} seconds")

        if self.log_shapes:
            logger.log(
                self.log_level,
                f"Output shape: {result.features.shape}",
            )
            logger.log(
                self.log_level,
                f"Feature dimension: {result.feature_dim}",
            )

        return result

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata from base set."""
        return self.base.get_metadata()

    @property
    def feature_dim(self) -> int:
        """Get feature dimension from base set."""
        return self.base.feature_dim


class ValidatedFeatureSet(IFeatureSet):
    """Decorator that adds validation to feature computation.

    This ensures inputs and outputs meet expected constraints.
    """

    def __init__(
        self,
        base: IFeatureSet,
        validate_input: bool = True,
        validate_output: bool = True,
        check_nans: bool = True,
        check_shape: bool = True,
    ):
        """Initialize validated feature set.

        Args:
            base: Base feature set to decorate
            validate_input: Whether to validate input
            validate_output: Whether to validate output
            check_nans: Whether to check for NaN values
            check_shape: Whether to validate shapes
        """
        self.base = base
        self.validate_input_flag = validate_input
        self.validate_output_flag = validate_output
        self.check_nans = check_nans
        self.check_shape = check_shape

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self.base.name

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features with validation.

        Args:
            attention_data: Input attention data

        Returns:
            Validated features

        Raises:
            ValueError: If validation fails
        """
        # Validate input
        if self.validate_input_flag:
            self.validate_input(attention_data)

            if self.check_nans:
                if torch.isnan(attention_data.attention_matrices).any():
                    raise ValueError("Input contains NaN values")

        # Compute features
        result = self.base.compute_features(attention_data)

        # Validate output
        if self.validate_output_flag:
            if self.check_nans:
                if torch.isnan(result.features).any():
                    raise ValueError("Output contains NaN values")

            if self.check_shape:
                expected_dim = self.base.feature_dim
                actual_dim = result.feature_dim
                if expected_dim > 0 and actual_dim != expected_dim:
                    raise ValueError(
                        f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
                    )

                if len(result.feature_names) != actual_dim:
                    raise ValueError(
                        f"Feature names mismatch: {len(result.feature_names)} names for {actual_dim} features"
                    )

        return result

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata from base set."""
        meta = self.base.get_metadata()
        meta.configuration["validated"] = True
        return meta

    @property
    def feature_dim(self) -> int:
        """Get feature dimension from base set."""
        return self.base.feature_dim


# Example filter implementations that work with the decorator pattern
class RemoveConstantFeaturesFilter(IFeatureFilter):
    """Filter that removes features with zero or near-zero variance."""

    def __init__(self, threshold: float = 1e-10):
        """Initialize filter.

        Args:
            threshold: Variance threshold below which features are removed
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Get filter name."""
        return f"RemoveConstants(threshold={self.threshold})"

    def apply(self, features: FeatureResult) -> FeatureResult:
        """Remove constant features.

        Args:
            features: Input features

        Returns:
            Filtered features
        """
        # Calculate variance across batch dimension
        if features.features.dim() == 2:
            variance = features.features.var(dim=0)
        else:
            # Single sample, can't compute variance
            return features

        # Find non-constant features
        keep_mask = variance > self.threshold
        keep_indices = torch.where(keep_mask)[0]

        # Filter features and names
        filtered_features = features.features[:, keep_indices]
        filtered_names = [features.feature_names[i] for i in keep_indices.tolist()]

        # Create filtered result
        result = FeatureResult(
            features=filtered_features,
            feature_names=filtered_names,
            metadata=features.metadata.copy(),
            labels=features.labels,
            splits=features.splits,
        )

        # Add filter metadata
        result.metadata[f"filter_{self.name}"] = {
            "removed": len(features.feature_names) - len(filtered_names),
            "kept": len(filtered_names),
        }

        return result


class RemoveCorrelatedFeaturesFilter(IFeatureFilter):
    """Filter that removes highly correlated features."""

    def __init__(self, threshold: float = 0.95):
        """Initialize filter.

        Args:
            threshold: Correlation threshold above which features are removed
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Get filter name."""
        return f"RemoveCorrelated(threshold={self.threshold})"

    def apply(self, features: FeatureResult) -> FeatureResult:
        """Remove correlated features.

        Args:
            features: Input features

        Returns:
            Filtered features with reduced correlation
        """
        if features.features.dim() != 2 or features.features.shape[0] < 2:
            # Need multiple samples to compute correlation
            return features

        # Compute correlation matrix
        corr_matrix = torch.corrcoef(features.features.T)

        # Find features to keep (greedy approach)
        n_features = corr_matrix.shape[0]
        keep_mask = torch.ones(n_features, dtype=torch.bool)

        for i in range(n_features):
            if not keep_mask[i]:
                continue
            # Find features highly correlated with i
            high_corr = (torch.abs(corr_matrix[i, :]) > self.threshold) & (
                torch.arange(n_features) > i
            )
            # Remove them
            keep_mask[high_corr] = False

        keep_indices = torch.where(keep_mask)[0]

        # Filter features and names
        filtered_features = features.features[:, keep_indices]
        filtered_names = [features.feature_names[i] for i in keep_indices.tolist()]

        # Create filtered result
        result = FeatureResult(
            features=filtered_features,
            feature_names=filtered_names,
            metadata=features.metadata.copy(),
            labels=features.labels,
            splits=features.splits,
        )

        # Add filter metadata
        result.metadata[f"filter_{self.name}"] = {
            "removed": len(features.feature_names) - len(filtered_names),
            "kept": len(filtered_names),
        }

        return result
