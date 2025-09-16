"""Feature registry for tracking computed features and PCA models.

This registry manages feature sets, PCA models, and their relationships.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from shade_io.interfaces.core import FeatureKey

logger = logging.getLogger(__name__)


class FeatureInfo:
    """Information about registered features."""

    def __init__(
        self,
        feature_key: FeatureKey,
        file_path: Path,
        feature_dim: int,
        n_samples: int,
        metadata: dict[str, Any] | None = None,
    ):
        self.feature_key = feature_key
        self.file_path = Path(file_path)
        self.feature_dim = feature_dim
        self.n_samples = n_samples
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_key": self.feature_key.to_string(),
            "file_path": str(self.file_path),
            "feature_dim": self.feature_dim,
            "n_samples": self.n_samples,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureInfo":
        """Create from dictionary."""
        feature_key = FeatureKey.from_string(data["feature_key"])
        return cls(
            feature_key=feature_key,
            file_path=Path(data["file_path"]),
            feature_dim=data["feature_dim"],
            n_samples=data["n_samples"],
            metadata=data.get("metadata", {}),
        )


class PCAModelInfo:
    """Information about registered PCA models."""

    def __init__(
        self,
        model_id: str,
        file_path: Path,
        n_components: int,
        explained_variance: float,
        feature_key: FeatureKey,
        metadata: dict[str, Any] | None = None,
    ):
        self.model_id = model_id
        self.file_path = Path(file_path)
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.feature_key = feature_key
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "file_path": str(self.file_path),
            "n_components": self.n_components,
            "explained_variance": self.explained_variance,
            "feature_key": self.feature_key.to_string(),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PCAModelInfo":
        """Create from dictionary."""
        feature_key = FeatureKey.from_string(data["feature_key"])
        return cls(
            model_id=data["model_id"],
            file_path=Path(data["file_path"]),
            n_components=data["n_components"],
            explained_variance=data["explained_variance"],
            feature_key=feature_key,
            metadata=data.get("metadata", {}),
        )


class FeatureRegistry:
    """Registry for managing features and PCA models."""

    def __init__(self, registry_dir: Path | None = None):
        """Initialize registry.

        Args:
            registry_dir: Directory for registry files
        """
        if registry_dir is None:
            registry_dir = Path.home() / ".cache" / "shade_io" / "registry"

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.features_file = self.registry_dir / "features.json"
        self.pca_models_file = self.registry_dir / "pca_models.json"

        self.features = self._load_features()
        self.pca_models = self._load_pca_models()

    def _load_features(self) -> dict[str, FeatureInfo]:
        """Load features from registry file."""
        if not self.features_file.exists():
            return {}

        try:
            with open(self.features_file) as f:
                data = json.load(f)
            return {k: FeatureInfo.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load features registry: {e}")
            return {}

    def _load_pca_models(self) -> dict[str, PCAModelInfo]:
        """Load PCA models from registry file."""
        if not self.pca_models_file.exists():
            return {}

        try:
            with open(self.pca_models_file) as f:
                data = json.load(f)
            return {k: PCAModelInfo.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load PCA models registry: {e}")
            return {}

    def _save_features(self) -> None:
        """Save features to registry file."""
        data = {k: v.to_dict() for k, v in self.features.items()}
        with open(self.features_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_pca_models(self) -> None:
        """Save PCA models to registry file."""
        data = {k: v.to_dict() for k, v in self.pca_models.items()}
        with open(self.pca_models_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_features(
        self,
        feature_key: FeatureKey,
        file_path: Path,
        feature_dim: int,
        n_samples: int,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureInfo:
        """Register computed features.

        Args:
            feature_key: Key identifying the features
            file_path: Path to feature file
            feature_dim: Feature dimension
            n_samples: Number of samples
            metadata: Additional metadata

        Returns:
            FeatureInfo object
        """
        info = FeatureInfo(
            feature_key=feature_key,
            file_path=file_path,
            feature_dim=feature_dim,
            n_samples=n_samples,
            metadata=metadata,
        )

        key = feature_key.to_string()
        self.features[key] = info
        self._save_features()

        logger.info(f"Registered features: {key}")
        return info

    def register_pca_model(
        self,
        model_id: str,
        file_path: Path,
        n_components: int,
        explained_variance: float,
        feature_key: FeatureKey,
        metadata: dict[str, Any] | None = None,
    ) -> PCAModelInfo:
        """Register PCA model.

        Args:
            model_id: Unique model identifier
            file_path: Path to PCA model file
            n_components: Number of components
            explained_variance: Total explained variance
            feature_key: Key of features used for PCA
            metadata: Additional metadata

        Returns:
            PCAModelInfo object
        """
        info = PCAModelInfo(
            model_id=model_id,
            file_path=file_path,
            n_components=n_components,
            explained_variance=explained_variance,
            feature_key=feature_key,
            metadata=metadata,
        )

        self.pca_models[model_id] = info
        self._save_pca_models()

        logger.info(f"Registered PCA model: {model_id}")
        return info

    def get_features(self, feature_key: FeatureKey) -> FeatureInfo | None:
        """Get feature info by key.

        Args:
            feature_key: Feature key

        Returns:
            FeatureInfo if found, None otherwise
        """
        return self.features.get(feature_key.to_string())

    def get_pca_model(self, model_id: str) -> PCAModelInfo | None:
        """Get PCA model info by ID.

        Args:
            model_id: Model ID

        Returns:
            PCAModelInfo if found, None otherwise
        """
        return self.pca_models.get(model_id)

    def list_features(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        feature_set_name: str | None = None,
    ) -> list[FeatureInfo]:
        """List features with optional filtering.

        Args:
            model_name: Filter by model name
            dataset_name: Filter by dataset name
            feature_set_name: Filter by feature set name

        Returns:
            List of matching FeatureInfo objects
        """
        results = []

        for info in self.features.values():
            if model_name and info.feature_key.model_name != model_name:
                continue
            if dataset_name and info.feature_key.dataset_name != dataset_name:
                continue
            if (
                feature_set_name
                and info.feature_key.feature_set_name != feature_set_name
            ):
                continue
            results.append(info)

        return results

    def list_pca_models(
        self,
        feature_key: FeatureKey | None = None,
    ) -> list[PCAModelInfo]:
        """List PCA models with optional filtering.

        Args:
            feature_key: Filter by feature key

        Returns:
            List of matching PCAModelInfo objects
        """
        if feature_key is None:
            return list(self.pca_models.values())

        key_str = feature_key.to_string()
        return [
            info
            for info in self.pca_models.values()
            if info.feature_key.to_string() == key_str
        ]

    def clean_orphaned(self) -> int:
        """Remove entries for files that no longer exist.

        Returns:
            Number of entries removed
        """
        removed = 0

        # Clean features
        to_remove = []
        for key, info in self.features.items():
            if not info.file_path.exists():
                to_remove.append(key)

        for key in to_remove:
            del self.features[key]
            removed += 1

        if to_remove:
            self._save_features()

        # Clean PCA models
        to_remove = []
        for key, info in self.pca_models.items():
            if not info.file_path.exists():
                to_remove.append(key)

        for key in to_remove:
            del self.pca_models[key]
            removed += 1

        if to_remove:
            self._save_pca_models()

        if removed > 0:
            logger.info(f"Removed {removed} orphaned registry entries")

        return removed
