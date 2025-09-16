"""Filter implementations for feature sets."""

import logging

import numpy as np

from shade_io.interfaces.core import FeatureResult, IFeatureFilter

logger = logging.getLogger(__name__)


class RemoveConstantFeaturesFilter(IFeatureFilter):
    """Remove features with near-zero variance."""

    def __init__(self, threshold: float = 1e-10):
        """Initialize filter.

        Args:
            threshold: Variance threshold below which to remove features
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Get filter name."""
        return f"RemoveConstant(threshold={self.threshold})"

    def apply(self, features: FeatureResult) -> FeatureResult:
        """Apply filter to remove constant features.

        Args:
            features: Features to filter

        Returns:
            Filtered features
        """
        # Calculate variance for each feature
        if features.features.dim() == 2:
            # Batched features
            variances = features.features.var(dim=0)
        else:
            # Single sample - can't compute variance
            logger.warning("Cannot remove constant features from single sample")
            return features

        # Find non-constant features
        mask = variances > self.threshold
        n_removed = (~mask).sum().item()

        if n_removed > 0:
            logger.info(f"Removing {n_removed} constant features")

        # Filter features and names
        filtered_features = features.features[:, mask]
        filtered_names = [
            name
            for name, keep in zip(features.feature_names, mask, strict=False)
            if keep
        ]

        # Update metadata
        metadata = features.metadata.copy()
        metadata["filters_applied"] = metadata.get("filters_applied", []) + [self.name]
        metadata["n_features_removed"] = (
            metadata.get("n_features_removed", 0) + n_removed
        )

        return FeatureResult(
            features=filtered_features,
            feature_names=filtered_names,
            metadata=metadata,
            labels=features.labels,
            splits=features.splits,
        )


class RemoveFirstKFilter(IFeatureFilter):
    """Remove first k features (e.g., to skip uninformative eigenvalues)."""

    def __init__(self, k: int = 1):
        """Initialize filter.

        Args:
            k: Number of features to remove from the beginning
        """
        self.k = k

    @property
    def name(self) -> str:
        """Get filter name."""
        return f"RemoveFirstK(k={self.k})"

    def apply(self, features: FeatureResult) -> FeatureResult:
        """Apply filter to remove first k features.

        Args:
            features: Features to filter

        Returns:
            Filtered features
        """
        if features.feature_dim <= self.k:
            logger.warning(
                f"Cannot remove {self.k} features from {features.feature_dim} total"
            )
            return features

        # Remove first k features
        if features.features.dim() == 2:
            filtered_features = features.features[:, self.k :]
        else:
            filtered_features = features.features[self.k :]

        filtered_names = features.feature_names[self.k :]

        # Update metadata
        metadata = features.metadata.copy()
        metadata["filters_applied"] = metadata.get("filters_applied", []) + [self.name]
        metadata["n_features_removed"] = metadata.get("n_features_removed", 0) + self.k

        logger.info(f"Removed first {self.k} features")

        return FeatureResult(
            features=filtered_features,
            feature_names=filtered_names,
            metadata=metadata,
            labels=features.labels,
            splits=features.splits,
        )


class RemoveCorrelatedFeaturesFilter(IFeatureFilter):
    """Remove highly correlated features to reduce redundancy."""

    def __init__(self, threshold: float = 0.95):
        """Initialize filter.

        Args:
            threshold: Correlation threshold above which to remove features
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Get filter name."""
        return f"RemoveCorrelated(threshold={self.threshold})"

    def apply(self, features: FeatureResult) -> FeatureResult:
        """Apply filter to remove correlated features.

        Args:
            features: Features to filter

        Returns:
            Filtered features with highly correlated features removed
        """
        if features.features.dim() != 2:
            logger.warning("Cannot remove correlated features from single sample")
            return features

        # Compute correlation matrix
        features_np = features.features.numpy()
        corr_matrix = np.corrcoef(features_np.T)

        # Find features to keep
        n_features = corr_matrix.shape[0]
        keep_mask = np.ones(n_features, dtype=bool)

        for i in range(n_features):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > self.threshold:
                    keep_mask[j] = False

        n_removed = (~keep_mask).sum()

        if n_removed > 0:
            logger.info(f"Removing {n_removed} correlated features")

        # Filter features and names
        filtered_features = features.features[:, keep_mask]
        filtered_names = [
            name
            for name, keep in zip(features.feature_names, keep_mask, strict=False)
            if keep
        ]

        # Update metadata
        metadata = features.metadata.copy()
        metadata["filters_applied"] = metadata.get("filters_applied", []) + [self.name]
        metadata["n_features_removed"] = (
            metadata.get("n_features_removed", 0) + n_removed
        )

        return FeatureResult(
            features=filtered_features,
            feature_names=filtered_names,
            metadata=metadata,
            labels=features.labels,
            splits=features.splits,
        )
