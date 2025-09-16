"""Adapter for shade's FeatureExtractor classes to work with shade-io.

This adapter allows shade's feature extractors to be used within
shade-io's SimpleFeatureSet by adapting their interface.
"""

from typing import Any

import torch


class FeatureExtractorAdapter:
    """Adapts shade's FeatureExtractor to work with shade-io's expected interface.

    shade-io's SimpleFeatureSet expects extractors to be callable or have an
    extract() method. This adapter wraps shade's FeatureExtractor classes
    which use a forward() method.
    """

    def __init__(self, extractor: Any) -> None:
        """Initialize the adapter with a shade FeatureExtractor.

        Args:
            extractor: An instance of a shade FeatureExtractor
        """
        self.extractor = extractor

    def __call__(self, intermediate_results: dict[str, torch.Tensor]) -> torch.Tensor:
        """Make the extractor callable as shade-io expects.

        Args:
            intermediate_results: Dictionary of intermediate computation results

        Returns:
            Feature tensor from the wrapped extractor
        """
        # Call the forward method with default architecture
        return self.extractor.forward(
            intermediate_results,
            architecture=intermediate_results.get("architecture", "decoder"),
        )

    def get_feature_names(self) -> list[str]:
        """Get feature names from the wrapped extractor.

        Returns:
            List of feature names
        """
        # Try post-forward name generation first (after forward pass when dimensions are available)
        if hasattr(self.extractor, "generate_feature_names_post_forward"):
            try:
                names = self.extractor.generate_feature_names_post_forward()
                if names and len(names) > 0:
                    return names
            except Exception:
                # Forward pass hasn't happened yet or other issue, continue to fallbacks
                pass

        # Try get_feature_metadata (provides detailed names)
        if hasattr(self.extractor, "get_feature_metadata"):
            metadata = self.extractor.get_feature_metadata()
            if "feature_names" in metadata and metadata["feature_names"]:
                return metadata["feature_names"]

        # Fall back to get_feature_names if it provides meaningful names
        if hasattr(self.extractor, "get_feature_names"):
            try:
                names = self.extractor.get_feature_names()
                # Only use if it's not the default ["feature"] and not placeholder names
                if (
                    names
                    and names != ["feature"]
                    and not all(
                        name.startswith(
                            (
                                "eigen_feature_",
                                "svd_feature_",
                                "ky_fan_feature_",
                                "head_align_feature_",
                            )
                        )
                        for name in names
                    )
                ):
                    return names
            except Exception:
                # get_feature_names failed (likely RuntimeError for missing tensor dimensions)
                pass

        # Fallback: generate generic names based on feature dimension
        if hasattr(self.extractor, "get_feature_dim"):
            dim = self.extractor.get_feature_dim()
            return [f"feature_{i}" for i in range(dim)]

        return ["feature"]  # Ultimate fallback

    def generate_feature_names_post_forward(self) -> list[str]:
        """Generate feature names after forward pass when tensor dimensions are available.

        This method specifically tries to get names after the wrapped extractor
        has gone through forward pass and stored tensor dimensions.

        Returns:
            List of proper feature names with layer/head structure
        """
        if hasattr(self.extractor, "generate_feature_names_post_forward"):
            return self.extractor.generate_feature_names_post_forward()
        else:
            # Fall back to regular get_feature_names
            return self.get_feature_names()

    @property
    def feature_names(self) -> list[str]:
        """Property access for feature names.

        Returns:
            List of feature names
        """
        return self.get_feature_names()
