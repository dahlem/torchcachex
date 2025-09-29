"""Core interfaces for the feature set architecture."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class AttentionData:
    """Container for attention matrices and related data.

    This encapsulates all the data needed to compute features from attention,
    keeping the interface clean and extensible.
    """

    attention_matrices: (
        torch.Tensor
    )  # (batch, layers, heads, seq_len, seq_len) or (layers, heads, seq_len, seq_len)
    model_name: str
    dataset_name: str
    architecture: str = "decoder"  # encoder, decoder, or encoder_decoder
    attention_type: str | None = None  # encoder_self, decoder_self, or cross
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Get batch size, handling both batched and single samples."""
        if self.attention_matrices.dim() == 5:
            return self.attention_matrices.shape[0]
        return 1

    @property
    def n_layers(self) -> int:
        """Number of attention layers."""
        if self.attention_matrices.dim() == 5:
            return self.attention_matrices.shape[1]
        return self.attention_matrices.shape[0]

    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        if self.attention_matrices.dim() == 5:
            return self.attention_matrices.shape[2]
        return self.attention_matrices.shape[1]

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self.attention_matrices.shape[-1]


@dataclass
class FeatureResult:
    """Container for computed features and metadata.

    This encapsulates the output of feature computation, including
    the features themselves and any associated metadata.
    """

    features: torch.Tensor  # (batch, feature_dim) or (feature_dim,)
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    labels: torch.Tensor | None = None  # Optional labels for supervised tasks
    splits: list[str] | None = None  # Optional train/val/test splits
    sample_hashes: list[str] | None = None  # SHA256 hashes for linking to original data
    original_indices: list[int] | None = (
        None  # Original dataset indices before filtering
    )

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        if self.features.dim() == 2:
            return self.features.shape[0]
        return 1

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        if self.features.dim() == 2:
            return self.features.shape[1]
        return self.features.shape[0]

    def __post_init__(self):
        """Validate feature result."""
        if len(self.feature_names) != self.feature_dim:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"doesn't match feature dimension ({self.feature_dim})"
            )


@dataclass
class FeatureMetadata:
    """Metadata about a feature set.

    This provides information about what features a set will produce,
    useful for validation and documentation.
    """

    name: str
    feature_dim: int
    feature_names: list[str]
    description: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    requires_computation: bool = True  # False for filtered/cached sets

    def __post_init__(self):
        """Validate metadata."""
        if len(self.feature_names) != self.feature_dim:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"doesn't match feature dimension ({self.feature_dim})"
            )


class IFeatureSet(ABC):
    """Interface for feature computation.

    This is the core interface that all feature sets must implement.
    It focuses solely on feature computation, with I/O handled separately.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this feature set."""
        pass

    @abstractmethod
    def compute_features(self, attention_data: AttentionData) -> FeatureResult:
        """Compute features from attention data.

        Args:
            attention_data: Container with attention matrices and metadata

        Returns:
            FeatureResult with computed features and metadata
        """
        pass

    @abstractmethod
    def get_metadata(self) -> FeatureMetadata:
        """Get metadata about this feature set.

        Returns:
            FeatureMetadata describing what features this set produces
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Get the expected feature dimension.

        Returns:
            Number of features this set will produce
        """
        pass

    def validate_input(self, attention_data: AttentionData) -> None:
        """Validate input data before computation.

        Args:
            attention_data: Data to validate

        Raises:
            ValueError: If input is invalid
        """
        if attention_data.attention_matrices is None:
            raise ValueError("Attention matrices cannot be None")
        if attention_data.attention_matrices.dim() not in [4, 5]:
            raise ValueError(
                f"Attention matrices must be 4D or 5D, got {attention_data.attention_matrices.dim()}D"
            )


class IFeatureFilter(ABC):
    """Interface for feature filters.

    Filters transform feature results, typically by removing
    or modifying certain features.
    """

    @abstractmethod
    def apply(self, features: FeatureResult) -> FeatureResult:
        """Apply filter to features.

        Args:
            features: Features to filter

        Returns:
            Filtered features with updated metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this filter."""
        pass

    def get_description(self) -> str:
        """Get a description of what this filter does."""
        return f"Filter: {self.name}"


@dataclass
class FeatureKey:
    """Key for identifying cached features.

    This provides a consistent way to identify feature sets
    across different models and datasets. Can use either a
    feature_set_name or constituent parts (attention_processor + extractors).
    """

    feature_set_name: str
    model_name: str
    dataset_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional constituent parts for more flexible matching
    attention_processor_name: str | None = None
    extractor_names: list[str] | None = None

    def to_string(self) -> str:
        """Convert to string for use as cache key."""
        # ALWAYS prefer constituent parts when available
        if self.attention_processor_name and self.extractor_names:
            # Create name from constituent parts
            extractor_part = "_".join(sorted(self.extractor_names))
            feature_name = f"{self.attention_processor_name}_{extractor_part}"
        else:
            # Fall back to feature_set_name for backward compatibility
            feature_name = self.feature_set_name

        parts = [
            feature_name,
            self.model_name,
            self.dataset_name,
        ]
        # Add any metadata that affects the cache key
        if "k" in self.metadata:
            parts.append(f"k{self.metadata['k']}")
        if "n_samples" in self.metadata:
            parts.append(f"n{self.metadata['n_samples']}")
        elif "sample_size" in self.metadata:
            parts.append(f"n{self.metadata['sample_size']}")
        return "_".join(parts)

    @classmethod
    def from_string(cls, key_str: str) -> "FeatureKey":
        """Parse key from string.

        Handles both old format (with version) and new format (without version).
        """
        parts = key_str.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid feature key: {key_str}")

        # Check if first part looks like a version (e.g., "v2")
        if parts[0].startswith("v") and parts[0][1:].isdigit():
            # Old format with version
            feature_set = parts[1]
            model = parts[2]
            dataset = parts[3] if len(parts) > 3 else ""
            metadata_parts = parts[4:] if len(parts) > 4 else []
        else:
            # New format without version
            feature_set = parts[0]
            model = parts[1]
            dataset = parts[2] if len(parts) > 2 else ""
            metadata_parts = parts[3:] if len(parts) > 3 else []

        # Parse metadata from remaining parts
        metadata = {}
        for part in metadata_parts:
            if part.startswith("k") and len(part) > 1 and part[1:].isdigit():
                metadata["k"] = int(part[1:])
            elif part.startswith("n") and len(part) > 1 and part[1:].isdigit():
                metadata["n_samples"] = int(part[1:])
            elif part.startswith("k") and len(part) > 1 and not part[1:].isdigit():
                logger.debug(
                    f"Skipping potential metadata field '{part}' - not numeric after 'k'"
                )
            elif part.startswith("n") and len(part) > 1 and not part[1:].isdigit():
                logger.debug(
                    f"Skipping potential metadata field '{part}' - not numeric after 'n'"
                )

        return cls(
            feature_set_name=feature_set,
            model_name=model,
            dataset_name=dataset,
            metadata=metadata,
        )

    def matches_constituent_parts(self, other: "FeatureKey") -> bool:
        """Check if this key matches another by constituent parts.

        This allows matching when both keys have attention_processor_name
        and extractor_names, even if feature_set_name differs.
        """
        if not (
            self.attention_processor_name
            and self.extractor_names
            and other.attention_processor_name
            and other.extractor_names
        ):
            return False

        return (
            self.attention_processor_name == other.attention_processor_name
            and self.model_name == other.model_name
            and self.dataset_name == other.dataset_name
            and set(self.extractor_names) == set(other.extractor_names)
        )


class IFeatureStore(ABC):
    """Interface for feature persistence.

    This abstracts away the details of how features are stored
    and loaded, allowing different implementations (file, database, etc).
    """

    @abstractmethod
    def load(self, key: FeatureKey) -> FeatureResult | None:
        """Load features from storage.

        Args:
            key: Key identifying the features

        Returns:
            FeatureResult if found, None otherwise
        """
        pass

    @abstractmethod
    def save(self, key: FeatureKey, features: FeatureResult) -> None:
        """Save features to storage.

        Args:
            key: Key identifying the features
            features: Features to save
        """
        pass

    @abstractmethod
    def exists(self, key: FeatureKey) -> bool:
        """Check if features exist in storage.

        Args:
            key: Key to check

        Returns:
            True if features exist, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: FeatureKey) -> bool:
        """Delete features from storage.

        Args:
            key: Key identifying features to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    def list_keys(self, pattern: str | None = None) -> list[FeatureKey]:
        """List available feature keys.

        Args:
            pattern: Optional pattern to filter keys

        Returns:
            List of available keys
        """
        return []
