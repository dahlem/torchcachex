"""Base implementations of feature sets."""

import logging
from typing import Any

import torch

from shade_io.interfaces.core import (
    AttentionData,
    FeatureKey,
    FeatureMetadata,
    FeatureResult,
    IFeatureSet,
    IFeatureStore,
)

logger = logging.getLogger(__name__)


class SimpleFeatureSet(IFeatureSet):
    """Basic feature set using feature extractors.

    This implementation supports both legacy attention processors and
    the new OutputProcessor architecture for different model outputs.
    """

    def __init__(
        self,
        name: str,
        extractors: list[Any],  # List of feature extractors
        attention_processor: Any | None = None,
        output_processor: Any | None = None,
        description: str = "",
    ):
        """Initialize simple feature set.

        Args:
            name: Name of this feature set
            extractors: List of feature extractors to use
            attention_processor: Optional legacy processor for attention matrices
            output_processor: Optional new OutputProcessor for any model outputs
            description: Human-readable description
        """
        self._name = name
        self.extractors = extractors
        self.attention_processor = attention_processor
        self.output_processor = output_processor
        self.description = description
        self._feature_names: list[str] | None = None
        self._feature_dim: int | None = None

        # Validate that we have at least one processor or neither (for raw data)
        if attention_processor is not None and output_processor is not None:
            raise ValueError(
                "Cannot specify both attention_processor and output_processor"
            )

        logger.debug(
            f"Initialized {name} with {'attention_processor' if attention_processor else 'output_processor' if output_processor else 'no processor'}"
        )

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self._name

    def compute_features(
        self, input_data: AttentionData | dict[str, Any]
    ) -> FeatureResult:
        """Compute features from input data (legacy AttentionData or new model outputs).

        Args:
            input_data: Either AttentionData (legacy) or dict of model outputs (new)

        Returns:
            Computed features with metadata
        """
        if isinstance(input_data, AttentionData):
            # Legacy path: AttentionData with attention_processor
            self.validate_input(input_data)

            if self.attention_processor:
                intermediate_results = self.attention_processor(
                    input_data.attention_matrices,
                    attention_mask=None,  # Will be generated internally for decoders
                    architecture=input_data.architecture,
                )
            else:
                # Pass attention directly as intermediate results
                intermediate_results = {
                    "attention_matrices": input_data.attention_matrices,
                    "architecture": input_data.architecture,
                }

            batch_size = input_data.batch_size
            model_name = input_data.model_name
            dataset_name = input_data.dataset_name

        else:
            # New path: model outputs dict with output_processor
            if self.output_processor:
                intermediate_results = self.output_processor.process(input_data)
            else:
                # Pass raw model outputs as intermediate results
                intermediate_results = input_data

            # Extract metadata from model outputs or intermediate results
            # Try to determine batch size from tensor shapes
            batch_size = self._infer_batch_size(intermediate_results)
            model_name = input_data.get("model_name", "unknown")
            dataset_name = input_data.get("dataset_name", "unknown")

        # Extract features using all extractors
        all_features = []
        all_feature_names = []

        for extractor in self.extractors:
            # Extract features
            if hasattr(extractor, "extract"):
                features = extractor.extract(intermediate_results)
            elif callable(extractor):
                features = extractor(intermediate_results)
            else:
                raise ValueError(f"Extractor {extractor} is not callable")

            # Handle batched vs single features
            if batch_size > 1 and features.dim() == 1:
                # Single feature for entire batch - expand
                features = features.unsqueeze(0).expand(batch_size, -1)

            all_features.append(features)

            # Get feature names if available
            if hasattr(extractor, "get_feature_names"):
                try:
                    names = extractor.get_feature_names()
                except RuntimeError:
                    # Extractor needs tensor dimensions - use placeholder names for now
                    n_features = features.shape[-1] if features.dim() > 0 else 1
                    names = [f"{self.name}_f{i}" for i in range(n_features)]
            elif hasattr(extractor, "feature_names"):
                names = extractor.feature_names
            else:
                # Generate generic names
                n_features = features.shape[-1] if features.dim() > 0 else 1
                names = [f"{self.name}_f{i}" for i in range(n_features)]

            all_feature_names.extend(names)

        # Concatenate all features
        if len(all_features) > 1:
            combined_features = torch.cat(all_features, dim=-1)
        else:
            combined_features = all_features[0] if all_features else torch.empty(0)

        # Cache feature names and dimension
        self._feature_names = all_feature_names
        self._feature_dim = len(all_feature_names)

        # Try to regenerate proper feature names after forward pass
        try:
            updated_names = self.regenerate_feature_names()
            if updated_names != all_feature_names:
                self._feature_names = updated_names
                all_feature_names = updated_names
        except Exception:
            # Keep the placeholder names if regeneration fails
            pass

        return FeatureResult(
            features=combined_features,
            feature_names=all_feature_names,
            metadata={
                "feature_set": self.name,
                "n_extractors": len(self.extractors),
                "model": model_name,
                "dataset": dataset_name,
            },
        )

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about this feature set."""
        if self._feature_names is None:
            # Haven't computed features yet, estimate
            self._estimate_metadata()

        return FeatureMetadata(
            name=self.name,
            feature_dim=self._feature_dim or 0,
            feature_names=self._feature_names or [],
            description=self.description,
            configuration={
                "n_extractors": len(self.extractors),
                "has_processor": self.attention_processor is not None,
            },
        )

    @property
    def feature_dim(self) -> int:
        """Get expected feature dimension."""
        if self._feature_dim is None:
            self._estimate_metadata()
        return self._feature_dim or 0

    def _estimate_metadata(self) -> None:
        """Estimate metadata without computing features."""
        all_names = []
        total_dim = 0

        for extractor in self.extractors:
            if hasattr(extractor, "get_feature_dim"):
                dim = extractor.get_feature_dim()
            elif hasattr(extractor, "feature_dim"):
                dim = extractor.feature_dim
            else:
                # Can't estimate
                logger.warning(f"Cannot estimate dimension for {extractor}")
                dim = 0

            if hasattr(extractor, "get_feature_names"):
                names = extractor.get_feature_names()
            else:
                names = [f"{self.name}_f{i}" for i in range(total_dim, total_dim + dim)]

            all_names.extend(names)
            total_dim += dim

        self._feature_names = all_names
        self._feature_dim = total_dim

    def regenerate_feature_names(self) -> list[str]:
        """Regenerate proper feature names after forward pass when tensor dimensions are available.

        This method attempts to get proper feature names from extractors that now have
        tensor dimensions stored from the forward pass.

        Returns:
            List of proper feature names
        """
        all_names = []

        for extractor in self.extractors:
            # Try to get updated feature names using the post-forward method
            # This works with both direct extractors and FeatureExtractorAdapter-wrapped ones
            if hasattr(extractor, "generate_feature_names_post_forward"):
                names = extractor.generate_feature_names_post_forward()
            elif hasattr(extractor, "get_feature_names"):
                try:
                    names = extractor.get_feature_names()
                except RuntimeError:
                    # Still no tensor dimensions - keep placeholder names
                    if hasattr(extractor, "get_feature_dim"):
                        dim = extractor.get_feature_dim()
                    else:
                        dim = 10  # Default fallback
                    names = [
                        f"{self.name}_f{i}"
                        for i in range(len(all_names), len(all_names) + dim)
                    ]
            elif hasattr(extractor, "feature_names"):
                names = extractor.feature_names
            else:
                # Generate generic names
                if hasattr(extractor, "get_feature_dim"):
                    dim = extractor.get_feature_dim()
                else:
                    dim = 10  # Default fallback
                names = [
                    f"{self.name}_f{i}"
                    for i in range(len(all_names), len(all_names) + dim)
                ]

            all_names.extend(names)

        return all_names

    def _infer_batch_size(self, intermediate_results: dict[str, Any]) -> int:
        """Infer batch size from intermediate results tensors.

        Args:
            intermediate_results: Dictionary containing tensors

        Returns:
            Inferred batch size
        """
        # Try common tensor keys to infer batch size
        for key in ["logits", "attention_matrices", "eigenvalues_tensor"]:
            if key in intermediate_results:
                tensor = intermediate_results[key]
                if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                    return tensor.shape[0]

        # Fallback: assume batch size 1
        return 1


class CompositeFeatureSet(IFeatureSet):
    """Feature set that combines multiple component sets.

    This allows building complex features by combining simpler ones,
    promoting reuse and modularity.
    """

    def __init__(
        self,
        name: str,
        components: list[str | IFeatureSet],
        lazy_loading: bool = False,
        feature_store: IFeatureStore | None = None,
        description: str = "",
    ):
        """Initialize composite feature set.

        Args:
            name: Name of this composite set
            components: List of component feature sets (names or instances)
            lazy_loading: If True, try to load components from cache first
            feature_store: Store for lazy loading (required if lazy_loading=True)
            description: Human-readable description
        """
        self._name = name
        self.component_specs = components
        self.lazy_loading = lazy_loading
        self.feature_store = feature_store
        self.description = description
        self._resolved_components: list[IFeatureSet] | None = None

        if lazy_loading and not feature_store:
            raise ValueError("feature_store required when lazy_loading=True")

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self._name

    def resolve_components(
        self, registry: dict[str, IFeatureSet] | None = None
    ) -> None:
        """Resolve component names to instances.

        Args:
            registry: Optional registry mapping names to feature sets
        """
        if self._resolved_components is not None:
            return  # Already resolved

        self._resolved_components = []
        for component in self.component_specs:
            if isinstance(component, str):
                # Look up by name
                if registry and component in registry:
                    self._resolved_components.append(registry[component])
                else:
                    raise ValueError(f"Component '{component}' not found in registry")
            elif isinstance(component, IFeatureSet):
                self._resolved_components.append(component)
            else:
                raise ValueError(f"Invalid component type: {type(component)}")

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features from all components.

        Args:
            attention_data: Input attention data

        Returns:
            Combined features from all components
        """
        if attention_data is not None:
            self.validate_input(attention_data)

        # Ensure components are resolved
        if self._resolved_components is None:
            self.resolve_components()

        # Check if all components are CachedFeatureSet instances
        all_cached = all(
            isinstance(comp, CachedFeatureSet)
            for comp in self._resolved_components or []
        )

        if all_cached:
            # Directly combine cached features without attention data
            logger.info(
                f"Combining {len(self._resolved_components)} cached component features"
            )

            all_features = []
            all_feature_names = []
            all_metadata = {}

            for i, component in enumerate(self._resolved_components):
                component_name = (
                    component.name if hasattr(component, "name") else f"component_{i}"
                )

                # Get cached features by calling compute_features to ensure loading
                result = component.compute_features(attention_data)

                if result is None or result.features is None:
                    raise ValueError(f"Component '{component_name}' returned None features")

                all_features.append(result.features)

                # Prefix feature names with component name
                prefixed_names = [
                    f"{component_name}.{name}"
                    for name in result.feature_names
                ]
                all_feature_names.extend(prefixed_names)

                # Collect metadata
                if hasattr(result, 'metadata') and result.metadata:
                    all_metadata[component_name] = result.metadata
                elif hasattr(component, '_metadata'):
                    all_metadata[component_name] = component._metadata
                else:
                    all_metadata[component_name] = {}

            # Concatenate horizontally
            combined_features = torch.cat(all_features, dim=-1)

            return FeatureResult(
                features=combined_features,
                feature_names=all_feature_names,
                metadata={
                    "feature_set": self.name,
                    "components": all_metadata,
                    "n_components": len(self._resolved_components),
                    "composite": True,
                    "from_cache": True,
                },
            )
        else:
            # Fall back to existing compute logic
            return self._compute_from_attention(attention_data)

    def _compute_from_attention(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features from attention data (original logic)."""
        all_features = []
        all_feature_names = []
        all_metadata = {}

        for i, component in enumerate(self._resolved_components or []):
            component_name = (
                component.name if hasattr(component, "name") else f"component_{i}"
            )

            # Try lazy loading if enabled
            if self.lazy_loading and self.feature_store:
                key = FeatureKey(
                    feature_set_name=component_name,
                    model_name=attention_data.model_name,
                    dataset_name=attention_data.dataset_name,
                )
                cached = self.feature_store.load(key)
                if cached:
                    logger.info(f"Loaded component '{component_name}' from cache")
                    result = cached
                else:
                    logger.info(f"Computing component '{component_name}'")
                    result = component.compute_features(attention_data)
                    # Cache for next time
                    self.feature_store.save(key, result)
            else:
                # Compute directly
                result = component.compute_features(attention_data)

            # Collect features
            all_features.append(result.features)

            # Prefix feature names with component name
            prefixed_names = [
                f"{component_name}.{name}" for name in result.feature_names
            ]
            all_feature_names.extend(prefixed_names)

            # Merge metadata
            all_metadata[component_name] = result.metadata

        # Concatenate all features
        if len(all_features) > 1:
            combined_features = torch.cat(all_features, dim=-1)
        else:
            combined_features = all_features[0] if all_features else torch.empty(0)

        return FeatureResult(
            features=combined_features,
            feature_names=all_feature_names,
            metadata={
                "feature_set": self.name,
                "components": all_metadata,
                "n_components": len(self._resolved_components or []),
                "model": attention_data.model_name if attention_data else None,
                "dataset": attention_data.dataset_name if attention_data else None,
            },
        )

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about this composite set."""
        # Ensure components are resolved
        if self._resolved_components is None:
            self.resolve_components()

        all_names = []
        total_dim = 0

        for component in self._resolved_components or []:
            comp_meta = component.get_metadata()
            # Prefix names
            prefixed = [f"{component.name}.{name}" for name in comp_meta.feature_names]
            all_names.extend(prefixed)
            total_dim += comp_meta.feature_dim

        return FeatureMetadata(
            name=self.name,
            feature_dim=total_dim,
            feature_names=all_names,
            description=self.description,
            configuration={
                "n_components": len(self._resolved_components or []),
                "lazy_loading": self.lazy_loading,
            },
        )

    @property
    def feature_dim(self) -> int:
        """Get expected feature dimension."""
        metadata = self.get_metadata()
        return metadata.feature_dim


class LazyCompositeFeatureSet(CompositeFeatureSet):
    """Composite set that prefers loading from cache over computation.

    This is a convenience class that defaults to lazy loading behavior.
    """

    def __init__(
        self,
        name: str,
        components: list[str | IFeatureSet],
        feature_store: IFeatureStore,
        description: str = "",
    ):
        """Initialize lazy composite feature set.

        Args:
            name: Name of this composite set
            components: List of component feature sets
            feature_store: Store for loading cached components
            description: Human-readable description
        """
        super().__init__(
            name=name,
            components=components,
            lazy_loading=True,
            feature_store=feature_store,
            description=description,
        )


class CachedFeatureSet(IFeatureSet):
    """Feature set wrapper for pre-computed cached features.

    This class provides a feature set interface for cached features,
    avoiding recomputation when features are already available in the cache.
    It maintains the same interface as other feature sets but returns
    cached data instead of computing features from attention matrices.
    """

    def __init__(
        self,
        name: str,
        features: torch.Tensor,
        feature_names: list[str],
        metadata: dict | None = None,
        description: str = "",
    ):
        """Initialize with cached data.

        Args:
            name: Name of the feature set
            features: Pre-computed feature tensor of shape (n_samples, n_features)
            feature_names: List of feature names matching features.shape[-1]
            metadata: Additional metadata about the cached features
            description: Human-readable description

        Raises:
            ValueError: If features and feature_names dimensions don't match
        """
        if len(feature_names) != features.shape[-1]:
            raise ValueError(
                f"Feature names length ({len(feature_names)}) must match "
                f"features last dimension ({features.shape[-1]})"
            )

        self._name = name
        self.description = description
        self._cached_features = features
        self._cached_feature_names = feature_names
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self._name

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return len(self._cached_feature_names)

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Return cached features instead of computing from attention data.

        Args:
            attention_data: Input attention data (used only for validation)

        Returns:
            FeatureResult with cached features

        Raises:
            ValueError: If attention_data batch size doesn't match cached features
        """
        # Validate that batch size matches if we have cached features for multiple samples
        if self._cached_features.dim() >= 2:
            expected_batch_size = self._cached_features.shape[0]
            if (
                hasattr(attention_data, "batch_size")
                and attention_data.batch_size != expected_batch_size
            ):
                logger.warning(
                    f"Batch size mismatch: cached features have {expected_batch_size} samples "
                    f"but attention_data has {attention_data.batch_size} samples. "
                    f"Using cached features as-is."
                )

        return FeatureResult(
            features=self._cached_features,
            feature_names=self._cached_feature_names,
            metadata={
                "cached": True,
                "feature_set": self.name,
                "cache_source": "precomputed",
                **self._metadata,
            },
        )

    def get_feature_names(self) -> list[str]:
        """Get cached feature names.

        Returns:
            List of feature names
        """
        return self._cached_feature_names

    def validate_input(self, attention_data: AttentionData) -> None:
        """Minimal validation for cached features.

        Args:
            attention_data: Input to validate (basic checks only)
        """
        if not isinstance(attention_data, AttentionData):
            raise TypeError(f"Expected AttentionData, got {type(attention_data)}")

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about this cached feature set.

        Returns:
            FeatureMetadata with cached feature information
        """
        return FeatureMetadata(
            name=self.name,
            feature_names=self._cached_feature_names,
            feature_dim=len(self._cached_feature_names),
            description=f"Cached {self.description}"
            if self.description
            else f"Cached features for {self.name}",
        )

    @property
    def feature_count(self) -> int:
        """Get number of features in cache.

        Returns:
            Number of features
        """
        return len(self._cached_feature_names)


class MultiCandidateCachedFeatureSet(CachedFeatureSet):
    """Cached feature set that can select from multiple candidates at runtime.

    This allows deferring the final selection until we know the actual
    batch size (n) at runtime, while pre-filtering by k at config time.

    The two-stage loading process:
    1. Config time: Filter candidates by k value
    2. Runtime: Select best match based on actual sample count (n)
    """

    def __init__(
        self,
        name: str,
        candidates: list[tuple[str, dict, int, int]],
        target_k: int | None = None,
        description: str = "",
    ):
        """Initialize multi-candidate cached feature set.

        Args:
            name: Feature set name
            candidates: List of (feature_id, entry, k, n) tuples
            target_k: Target k value for filtering
            description: Optional description
        """
        self._name = name
        self.candidates = candidates  # List of (feature_id, entry, k, n) tuples
        self.target_k = target_k
        self.selected_candidate = None
        self.description = description

        # Initialize with placeholder values - will be set when candidate is selected
        self._cached_features = None
        self._cached_feature_names = []
        self._metadata = {}

        logger.info(
            f"Created MultiCandidateCachedFeatureSet for '{name}' with "
            f"{len(candidates)} candidates (k={target_k})"
        )

    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features by selecting the best candidate at runtime.

        Args:
            attention_data: Input attention data (used for batch size determination)

        Returns:
            FeatureResult with features from selected candidate
        """
        # Determine expected sample count from attention data
        expected_n = None
        if hasattr(attention_data, "attention_matrices"):
            if torch.is_tensor(attention_data.attention_matrices):
                expected_n = attention_data.attention_matrices.shape[0]
            elif isinstance(attention_data.attention_matrices, (list, tuple)):
                expected_n = len(attention_data.attention_matrices)

        # Select best matching candidate
        best_match = self._select_best_candidate(expected_n)

        if not best_match:
            raise ValueError(f"No suitable cached features found for '{self._name}'")

        # Load features from selected candidate if not already loaded
        if self.selected_candidate != best_match:
            self._load_candidate(best_match)
            self.selected_candidate = best_match

        # Validate batch size compatibility
        feature_id, entry, k, n = best_match
        if expected_n and n != expected_n:
            logger.warning(
                f"Batch size mismatch for '{self._name}': "
                f"expected {expected_n} samples but cached features have {n}. "
                f"Using cached features as-is."
            )

        return FeatureResult(
            features=self._cached_features,
            feature_names=self._cached_feature_names,
            metadata={
                "cached": True,
                "feature_set": self._name,
                "cache_source": "multi_candidate",
                "selected_k": k,
                "selected_n": n,
                "selected_id": feature_id,
                "total_candidates": len(self.candidates),
                **self._metadata,
            },
        )

    def _select_best_candidate(
        self, expected_n: int | None
    ) -> tuple[str, dict, int, int] | None:
        """Select the best candidate based on expected sample count.

        Args:
            expected_n: Expected number of samples (None if unknown)

        Returns:
            Best matching candidate tuple or None
        """
        if not self.candidates:
            return None

        # First, try to find exact match
        if expected_n:
            for candidate in self.candidates:
                feature_id, entry, k, n = candidate
                if n == expected_n:
                    logger.info(f"Found exact match for '{self._name}': k={k}, n={n}")
                    return candidate

        # No exact match - use the largest available (already sorted)
        best_candidate = self.candidates[0]
        feature_id, entry, k, n = best_candidate

        if expected_n:
            logger.warning(
                f"No exact match for '{self._name}' with n={expected_n}. "
                f"Using cached version with k={k}, n={n}"
            )
        else:
            logger.info(f"Selected cached '{self._name}' with k={k}, n={n}")

        return best_candidate

    def _load_candidate(self, candidate: tuple[str, dict, int, int]) -> None:
        """Load features from a specific candidate.

        Args:
            candidate: (feature_id, entry, k, n) tuple to load
        """
        import numpy as np

        from shade_io.registry.unified_registry import UnifiedRegistry

        feature_id, entry, k, n = candidate

        try:
            # Load features using registry
            registry = UnifiedRegistry()
            features, labels, feature_names = registry.load_features_with_names(
                feature_id
            )

            # Convert to torch tensor if needed
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()

            # Store loaded features
            self._cached_features = features
            self._cached_feature_names = feature_names
            self._metadata = entry.get("metadata", {})

            logger.info(
                f"Loaded candidate '{self._name}' (ID: {feature_id[:8]}...) "
                f"with {features.shape[0]} samples, {features.shape[1]} features"
            )

        except Exception as e:
            logger.error(f"Failed to load candidate {feature_id}: {e}")
            raise

    @property
    def name(self) -> str:
        """Get feature set name."""
        return self._name

    @property
    def feature_dim(self) -> int:
        """Get feature dimension from best candidate or estimate."""
        if self._cached_features is not None:
            return self._cached_features.shape[-1]

        # Estimate from candidates (use first available)
        if self.candidates:
            # Try to get dimension from metadata or load first candidate
            feature_id, entry, k, n = self.candidates[0]
            metadata = entry.get("metadata", {})
            if "feature_dim" in metadata:
                return metadata["feature_dim"]

            # As last resort, load the candidate to get dimension
            try:
                self._load_candidate(self.candidates[0])
                self.selected_candidate = self.candidates[0]
                return self._cached_features.shape[-1]
            except Exception as e:
                logger.warning(f"Could not determine feature dimension: {e}")
                return 0

        return 0

    def get_feature_names(self) -> list[str]:
        """Get feature names from selected candidate."""
        if self._cached_feature_names:
            return self._cached_feature_names

        # Load first candidate to get names
        if self.candidates:
            try:
                self._load_candidate(self.candidates[0])
                self.selected_candidate = self.candidates[0]
                return self._cached_feature_names
            except Exception as e:
                logger.warning(f"Could not load feature names: {e}")
                return []

        return []

    def validate_input(self, attention_data: AttentionData) -> None:
        """Minimal validation for cached features."""
        if not isinstance(attention_data, AttentionData):
            raise TypeError(f"Expected AttentionData, got {type(attention_data)}")

    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about this multi-candidate feature set."""
        # Get feature names and dimension from best candidate
        feature_names = self.get_feature_names()
        feature_dim = self.feature_dim

        return FeatureMetadata(
            name=self._name,
            feature_names=feature_names,
            feature_dim=feature_dim,
            description=f"Multi-candidate cached features for {self._name}",
            metadata={
                "cached": True,
                "multi_candidate": True,
                "target_k": self.target_k,
                "candidates_count": len(self.candidates),
                "candidates_k_values": [c[2] for c in self.candidates],
                "candidates_n_values": [c[3] for c in self.candidates],
            },
        )
