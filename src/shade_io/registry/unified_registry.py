"""Unified registry for all SHADE artifacts.

This registry consolidates management of:
- Computed features in multiple formats
- PCA models
- Trained models and checkpoints
- Enhanced datasets
- Semantic samples
- Generic artifacts

It provides a single, consistent API for all artifact types with support for:
- Auto-discovery of unregistered files
- Multiple storage formats (npz, arrow, torch, hdf5)
- Progressive/incremental caching
- Split-aware loading with alignment
- Fuzzy matching and compatibility checking
- Cross-artifact relationships
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Registry Entries
# ============================================================================


@dataclass
class FeatureMatch:
    """Result of a feature search in the registry."""

    feature_id: str
    config: dict[str, Any]
    file_path: str
    created_at: str
    match_score: float
    n_samples: int
    feature_dim: int
    metadata: dict[str, Any]


@dataclass
class PreprocessorMatch:
    """Result of a preprocessor search in the registry."""

    preprocessor_id: str
    config: dict[str, Any]
    file_path: str
    state_dict_path: str | None
    created_at: str
    match_score: float
    n_components: int | None
    explained_variance: float | None
    trained_on_features: str  # Link to feature_id


@dataclass
class ModelMatch:
    """Result of a model search in the registry."""

    model_id: str
    config: dict[str, Any]
    checkpoint_path: str
    created_at: str
    match_score: float
    metrics: dict[str, float]
    is_best: bool
    metadata: dict[str, Any]


@dataclass
class UnifiedMatch:
    """Result of a unified search across all registries."""

    artifact_type: str  # "feature", "pca", "model", "enhanced", "samples"
    artifact_id: str
    file_path: str
    created_at: str
    match_score: float
    metadata: dict[str, Any]

    # Type-specific attributes
    dataset_name: str | None = None
    model_name: str | None = None
    detector_name: str | None = None
    judge: str | None = None
    n_samples: int | None = None
    n_components: int | None = None
    feature_dim: int | None = None


# ============================================================================
# Storage Manager for Format Abstraction
# ============================================================================


class StorageManager:
    """Handles storage operations across different formats."""

    # Supported feature storage formats
    SUPPORTED_FORMATS = {
        "npz": ".npz",  # NumPy compressed format (default)
        "arrow": ".arrow",  # Apache Arrow format (memory-mapped)
        "torch": ".pt",  # PyTorch native format
        "hdf5": ".h5",  # HDF5 format (chunked access)
    }

    @staticmethod
    def save_features(
        features: np.ndarray,
        labels: np.ndarray,
        file_path: Path,
        format: str = "npz",
        metadata: dict | None = None,
        hash_ids: list[str] | None = None,
        splits: list[str] | None = None,
    ) -> None:
        """Save features in the specified format."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npz":
            np.savez_compressed(file_path, features=features, labels=labels)

        elif format == "arrow":
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Build table data
            data = {
                "features": features.tolist(),
                "label": labels.astype(int).tolist(),
            }

            if hash_ids:
                data["hash_id"] = hash_ids
            else:
                # Generate default hash IDs
                data["hash_id"] = [f"sample_{i}" for i in range(len(features))]

            if splits:
                data["split"] = splits
            else:
                data["split"] = ["unknown"] * len(features)

            # Create table and write
            table = pa.table(data)
            pq.write_table(table, str(file_path), compression="snappy")

        elif format == "torch":
            torch.save(
                {
                    "features": torch.from_numpy(features),
                    "labels": torch.from_numpy(labels),
                    "metadata": metadata or {},
                },
                file_path,
            )

        elif format == "hdf5":
            import h5py

            with h5py.File(file_path, "w") as f:
                f.create_dataset("features", data=features, compression="gzip")
                f.create_dataset("labels", data=labels, compression="gzip")
                if metadata:
                    f.attrs.update(metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def load_features(
        file_path: Path, format: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load features from file, auto-detecting format if needed."""
        file_path = Path(file_path)

        # Auto-detect format from extension
        if format is None:
            ext = file_path.suffix
            format = next(
                (
                    name
                    for name, fext in StorageManager.SUPPORTED_FORMATS.items()
                    if fext == ext
                ),
                "npz",
            )

        if format == "npz":
            data = np.load(file_path)
            return data["features"], data["labels"]

        elif format == "arrow":
            import pyarrow.parquet as pq

            table = pq.read_table(str(file_path))

            # Check if we have legacy format (single "features" column) or columnar format
            if "features" in table.schema.names:
                # Legacy format - single features column with arrays
                # Only convert to pandas for legacy format where we need tolist()
                df = table.to_pandas()
                features = np.array(df["features"].tolist())
                labels = np.array(df["label"]) if "label" in df.columns else None
            else:
                # Columnar format - more efficient direct conversion from Arrow!
                # Separate labels if present
                if "label" in table.schema.names:
                    labels = table.column("label").to_numpy()
                    # Drop label column for features
                    feature_columns = [
                        col for col in table.schema.names if col != "label"
                    ]
                    features_table = table.select(feature_columns)
                else:
                    labels = None
                    features_table = table

                # Convert to numpy array efficiently (direct from Arrow!)
                # Use Arrow's built-in to_numpy when possible, fallback to pandas
                try:
                    features = features_table.to_pandas(types_mapper=None).values
                except Exception:
                    # Fallback for edge cases
                    features = features_table.to_pandas().values

            return features, labels

        elif format == "torch":
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
            except Exception:
                # Fallback for files with non-weight data (PyTorch 2.6+ compatibility)
                data = torch.load(file_path, map_location="cpu", weights_only=False)
            features = (
                data["features"].numpy()
                if torch.is_tensor(data["features"])
                else data["features"]
            )
            labels = (
                data["labels"].numpy()
                if torch.is_tensor(data["labels"])
                else data["labels"]
            )
            return features, labels

        elif format == "hdf5":
            import h5py

            with h5py.File(file_path, "r") as f:
                features = f["features"][:]
                labels = f["labels"][:]
            return features, labels

        else:
            raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# Main Unified Registry Class
# ============================================================================


class UnifiedRegistry:
    """Unified registry for all SHADE cached artifacts.

    This registry consolidates management of features, PCA models, trained models,
    enhanced datasets, semantic samples, and other artifacts. It provides:

    - Consistent API across all artifact types
    - Auto-discovery of unregistered files
    - Multiple storage format support
    - Progressive caching capabilities
    - Smart matching and compatibility checking
    - Cross-artifact relationship tracking
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize unified registry.

        Args:
            cache_dir: Directory for cache and registry files (default: .cache/shade)
        """
        self.cache_dir = Path(cache_dir or ".cache/shade")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different artifacts
        self.features_dir = self.cache_dir / "shade_io"
        self.preprocessor_dir = self.cache_dir / "preprocessors"
        self.models_dir = self.cache_dir / "models"
        self.enhanced_dir = self.cache_dir / "enhanced"
        self.samples_dir = self.cache_dir / "semantic_samples"

        # Create directories
        for dir_path in [
            self.features_dir,
            self.preprocessor_dir,
            self.models_dir,
            self.enhanced_dir,
            self.samples_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        # Registry files
        self.registries_dir = self.cache_dir / "registries"
        self.registries_dir.mkdir(exist_ok=True)

        self.features_registry_file = self.registries_dir / "features_registry.json"
        self.preprocessor_registry_file = (
            self.registries_dir / "preprocessor_registry.json"
        )
        self.models_registry_file = self.registries_dir / "models_registry.json"
        self.enhanced_registry_file = self.registries_dir / "enhanced_registry.json"

        # Load all registries
        self.features_registry = self._load_json(
            self.features_registry_file, {"entries": {}}
        )
        self.preprocessor_registry = self._load_json(
            self.preprocessor_registry_file, {"entries": {}}
        )
        self.models_registry = self._load_json(
            self.models_registry_file, {"entries": {}, "best_models": {}}
        )
        self.enhanced_registry = self._load_json(
            self.enhanced_registry_file,
            {"enhanced_datasets": {}, "semantic_samples": {}},
        )

        # Storage manager
        self.storage_manager = StorageManager()

        # Progressive caching state
        self._progressive_caches = {}

        # Auto-discover unregistered files on initialization
        self._auto_discover_all()

    # ========================================================================
    # Core Utility Methods
    # ========================================================================

    def _load_json(self, file_path: Path, default: dict) -> dict[str, Any]:
        """Load JSON file or return default."""
        if file_path.exists():
            try:
                with open(file_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return default
        return default

    def _save_json(self, data: dict, file_path: Path) -> None:
        """Save data to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute deterministic hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _extract_config_key(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
    ) -> dict[str, Any]:
        """Extract key configuration elements for matching."""
        # Convert DictConfigs to dicts
        if isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.to_container(model_cfg)
        if isinstance(dataset_cfg, DictConfig):
            dataset_cfg = OmegaConf.to_container(dataset_cfg)
        if isinstance(detector_cfg, DictConfig):
            detector_cfg = OmegaConf.to_container(detector_cfg)

        # Model configuration
        model_key = {
            "name": model_cfg.get("model_name", model_cfg.get("name", "unknown")),
            "type": model_cfg.get("_target_", ""),
            "max_length": model_cfg.get("max_length", 1024),
        }

        # Dataset configuration
        dataset_key = {
            "type": dataset_cfg.get("_target_", ""),
            "name": dataset_cfg.get("dataset_name", dataset_cfg.get("name", "unknown")),
            "source": dataset_cfg.get("source", ""),
            "config_name": dataset_cfg.get("config_name", ""),
            "split": dataset_cfg.get("split", ""),
            "max_samples": dataset_cfg.get("max_samples"),
        }

        # Detector configuration
        detector_key = {
            "type": detector_cfg.get("_target_", ""),
            "name": detector_cfg.get(
                "detector_name", detector_cfg.get("name", "unknown")
            ),
        }

        # Add feature extractor config if present
        if "feature_extractor" in detector_cfg:
            fe_cfg = detector_cfg["feature_extractor"]
            detector_key["feature_extractor"] = {
                "type": fe_cfg.get("_target_", ""),
                "k_eigenvalues": fe_cfg.get("k_eigenvalues"),
                "normalize_attention": fe_cfg.get("normalize_attention", True),
            }

        return {"model": model_key, "dataset": dataset_key, "detector": detector_key}

    # ========================================================================
    # Auto-Discovery Methods
    # ========================================================================

    def _auto_discover_all(self) -> None:
        """Auto-discover all types of unregistered files."""
        discovered_count = 0

        # Discover features
        discovered_count += self._auto_discover_features()

        # Discover PCA models
        discovered_count += self._auto_discover_pca_models()

        # Discover trained models
        discovered_count += self._auto_discover_models()

        # Discover enhanced datasets
        discovered_count += self._auto_discover_enhanced()

        # Discover semantic samples
        discovered_count += self._auto_discover_samples()

        if discovered_count > 0:
            logger.info(f"Auto-discovered {discovered_count} unregistered artifacts")

    def _auto_discover_features(self) -> int:
        """Auto-discover feature files by syncing from shade-io registry."""
        discovered = 0

        # First, sync from shade-io registry if it exists
        shade_io_registry_path = (
            self.cache_dir / "shade_io" / "registry" / "features.json"
        )
        if shade_io_registry_path.exists():
            try:
                shade_io_data = self._load_json(shade_io_registry_path, {})
                for key, entry in shade_io_data.items():
                    # Check if this entry already exists in unified registry
                    if key not in self.features_registry["entries"]:
                        # Convert shade-io entry to unified registry format
                        file_path = entry.get("file_path", "")
                        if not file_path:
                            continue

                        # Convert to absolute path if relative, but avoid double cache_dir
                        if not Path(file_path).is_absolute():
                            # If file_path already starts with cache_dir components, use as-is
                            if file_path.startswith(".cache/shade/"):
                                # This is already a full relative path from project root
                                file_path = str(Path(file_path))
                            else:
                                # This is relative to cache_dir
                                file_path = str(self.cache_dir / file_path)

                        # Use the shade-io registry entry directly with correct metadata
                        self.features_registry["entries"][key] = {
                            "id": key,
                            "file": file_path,
                            "created_at": entry.get(
                                "created_at", datetime.now().isoformat()
                            ),
                            "config": {
                                "model": {
                                    "name": entry["metadata"].get("model", "unknown")
                                },
                                "dataset": {
                                    "name": entry["metadata"].get("dataset", "unknown")
                                },
                                "detector": {
                                    "name": entry["metadata"].get(
                                        "feature_set", "unknown"
                                    )
                                },
                                "format": Path(file_path).suffix[1:]
                                if Path(file_path).suffix
                                else "unknown",
                            },
                            "metadata": entry.get("metadata", {}),
                            "n_samples": entry.get("n_samples", 0),
                            "feature_dim": entry.get("feature_dim", 0),
                        }
                        discovered += 1
                        logger.info(
                            f"Synced feature entry from shade-io registry: {key}"
                        )
            except Exception as e:
                logger.warning(f"Failed to sync from shade-io registry: {e}")

        # Check for orphaned files (files that exist but aren't in any registry)
        for _format_name, extension in StorageManager.SUPPORTED_FORMATS.items():
            for file_path in self.features_dir.glob(f"*{extension}"):
                # Check if file is already registered (by checking file path)
                if not any(
                    str(file_path) == entry.get("file")
                    or str(file_path) == str(self.cache_dir / entry.get("file", ""))
                    for entry in self.features_registry["entries"].values()
                ):
                    logger.warning(
                        f"Orphaned feature file not in any registry: {file_path}"
                    )
                    # We don't auto-register it with bad metadata anymore

        if discovered > 0:
            self._save_json(self.features_registry, self.features_registry_file)

        return discovered

    def _register_discovered_feature_file(
        self, file_path: Path, format_name: str
    ) -> bool:
        """[DEPRECATED] Old method for registering discovered files - no longer used.

        Features should be registered through shade-io registry which has proper metadata.
        This method is kept for backwards compatibility but returns False.
        """
        logger.warning(
            f"Attempted to use deprecated _register_discovered_feature_file for {file_path}"
        )
        return False

    def _auto_discover_pca_models(self) -> int:
        """Auto-discover PCA model files by checking registry-first."""
        discovered = 0

        # Note: PCA models are now handled as part of the unified preprocessor system

        return discovered

    def _auto_discover_models(self) -> int:
        """Auto-discover trained model files by checking registry-first."""
        discovered = 0

        # First, try to sync from shade-train model registry if it exists
        # (This would require coordination with shade-train's ModelRegistry)

        # Check for orphaned model files (files that exist but aren't registered)
        for file_path in self.models_dir.glob("*.pt"):
            # Check if file is already registered
            if not any(
                str(file_path) == entry.get("checkpoint_path")
                or str(file_path)
                == str(self.cache_dir / entry.get("checkpoint_path", ""))
                for entry in self.models_registry["entries"].values()
            ):
                # This is an orphaned model file
                logger.warning(
                    f"Orphaned model checkpoint file not in registry: {file_path}"
                )
                # Note: We don't auto-register models with bad metadata anymore
                # Models should be registered through proper training workflows

        return discovered

    def _auto_discover_enhanced(self) -> int:
        """Auto-discover enhanced dataset directories."""
        discovered = 0

        if self.enhanced_dir.exists():
            for dataset_dir in self.enhanced_dir.glob("*/*"):
                if dataset_dir.is_dir() and (dataset_dir / "metadata.json").exists():
                    if self._register_discovered_enhanced(dataset_dir):
                        discovered += 1

        if discovered > 0:
            self._save_json(self.enhanced_registry, self.enhanced_registry_file)

        return discovered

    def _register_discovered_enhanced(self, dataset_dir: Path) -> bool:
        """Register a discovered enhanced dataset."""
        try:
            dataset_id = self._compute_config_hash({"path": str(dataset_dir)})

            # Skip if already registered
            if dataset_id in self.enhanced_registry.get("enhanced_datasets", {}):
                return False

            # Load metadata
            with open(dataset_dir / "metadata.json") as f:
                metadata = json.load(f)

            # Extract dataset name from path
            dataset_name = dataset_dir.parent.name

            # Register
            self.enhanced_registry.setdefault("enhanced_datasets", {})[dataset_id] = {
                "dataset_name": dataset_name,
                "judge": metadata.get("judge_model"),
                "enhancement_type": "judge_annotation",
                "file_path": str(dataset_dir),
                "created_at": metadata.get(
                    "enhancement_date", datetime.now().isoformat()
                ),
                "n_examples": metadata.get("total_examples", 0),
                "format": metadata.get("format", "unknown"),
                "metadata": metadata,
            }

            logger.info(f"Auto-registered enhanced dataset: {dataset_dir.name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to register enhanced dataset {dataset_dir}: {e}")
            return False

    def _auto_discover_samples(self) -> int:
        """Auto-discover semantic sample files."""
        discovered = 0

        if self.samples_dir.exists():
            for sample_file in self.samples_dir.glob("*/*.json"):
                if self._register_discovered_samples(sample_file):
                    discovered += 1

        if discovered > 0:
            self._save_json(self.enhanced_registry, self.enhanced_registry_file)

        return discovered

    def _register_discovered_samples(self, sample_file: Path) -> bool:
        """Register discovered semantic samples."""
        try:
            sample_id = self._compute_config_hash({"path": str(sample_file)})

            # Skip if already registered
            if sample_id in self.enhanced_registry.get("semantic_samples", {}):
                return False

            # Load to get metadata
            with open(sample_file) as f:
                data = json.load(f)

            if not data or not isinstance(data, list):
                return False

            # Extract dataset name from path
            dataset_name = sample_file.parent.name

            # Count samples
            unique_prompts = set()
            total_samples = 0
            for item in data:
                if "prompt" in item:
                    unique_prompts.add(item["prompt"])
                if "generated_samples" in item:
                    total_samples += len(item["generated_samples"])

            avg_samples = total_samples / len(unique_prompts) if unique_prompts else 0

            # Register
            self.enhanced_registry.setdefault("semantic_samples", {})[sample_id] = {
                "dataset_name": dataset_name,
                "file_path": str(sample_file),
                "created_at": datetime.fromtimestamp(
                    sample_file.stat().st_mtime
                ).isoformat(),
                "n_prompts": len(unique_prompts),
                "n_samples_per_prompt": avg_samples,
                "total_samples": total_samples,
            }

            logger.info(f"Auto-registered semantic samples: {sample_file.name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to register samples {sample_file}: {e}")
            return False

    # ========================================================================
    # Feature Management Methods
    # ========================================================================

    def register_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        metadata: dict[str, Any] | None = None,
        format: str = "npz",
        hash_ids: list[str] | None = None,
        splits: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> str:
        """Register precomputed features.

        Args:
            features: Feature array
            labels: Label array
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            metadata: Additional metadata to store
            format: Storage format (npz, arrow, torch, hdf5)
            hash_ids: Hash IDs for each sample (for split tracking)
            splits: Split assignments for each sample
            feature_names: Names of features (meaningful names like L0_H0_λ_1)

        Returns:
            Feature ID for retrieval
        """
        # Extract configuration
        config_key = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)

        # Add data statistics
        config_key["data"] = {
            "n_samples": len(features),
            "feature_dim": features.shape[1],
            "label_distribution": {
                str(int(label)): int(np.sum(labels == label))
                for label in np.unique(labels)
            },
        }
        config_key["format"] = format

        # Generate ID
        feature_id = self._compute_config_hash(config_key)

        # Save features
        feature_file = (
            self.features_dir
            / f"features_{feature_id}{StorageManager.SUPPORTED_FORMATS[format]}"
        )

        # Include feature names in metadata for storage
        combined_metadata = metadata.copy() if metadata else {}
        if feature_names is not None:
            combined_metadata["feature_names"] = feature_names
            logger.info(f"Storing {len(feature_names)} feature names with features")

        self.storage_manager.save_features(
            features, labels, feature_file, format, combined_metadata, hash_ids, splits
        )

        # Update registry
        registry_metadata = combined_metadata.copy()
        if feature_names is not None:
            registry_metadata["feature_names"] = feature_names

        self.features_registry["entries"][feature_id] = {
            "id": feature_id,
            "config": config_key,
            "file": str(feature_file),
            "created_at": datetime.now().isoformat(),
            "metadata": registry_metadata,
        }

        # Save registry
        self._save_json(self.features_registry, self.features_registry_file)

        logger.info(
            f"Registered features with ID: {feature_id} "
            f"({config_key['data']['n_samples']} samples, "
            f"{config_key['data']['feature_dim']} dims, {format} format)"
        )

        return feature_id

    def find_features(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        min_samples: int | None = None,
        match_mode: str = "best",
    ) -> FeatureMatch | None:
        """Find precomputed features matching configuration.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            min_samples: Minimum required samples
            match_mode: "exact", "compatible", or "best"

        Returns:
            Best matching features or None
        """
        target_config = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)
        candidates = []

        for feature_id, entry in self.features_registry["entries"].items():
            saved_config = entry["config"]

            # Calculate match score
            score = self._calculate_match_score(target_config, saved_config, match_mode)

            if score > 0:
                # Check minimum samples requirement
                if (
                    min_samples
                    and saved_config.get("data", {}).get("n_samples", 0) < min_samples
                ):
                    continue

                candidates.append((score, feature_id, entry))

        if not candidates:
            return None

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, feature_id, entry = candidates[0]

        data_config = entry["config"].get("data", {})

        return FeatureMatch(
            feature_id=feature_id,
            config=entry["config"],
            file_path=entry["file"],
            created_at=entry["created_at"],
            match_score=best_score,
            n_samples=data_config.get("n_samples", 0),
            feature_dim=data_config.get("feature_dim", 0),
            metadata=entry.get("metadata", {}),
        )

    def list_features(self) -> list[FeatureMatch]:
        """List all available features in the registry.

        Returns:
            List of FeatureMatch objects for all registered features
        """
        features = []

        for feature_id, entry in self.features_registry["entries"].items():
            # Skip entries with incomplete metadata (bad auto-discovery)
            config = entry.get("config", {})
            metadata = entry.get("metadata", {})

            # Check if we have proper metadata
            model_name = metadata.get("model") or config.get("model", {}).get("name")
            dataset_name = metadata.get("dataset") or config.get("dataset", {}).get(
                "name"
            )
            feature_set_name = metadata.get("feature_set") or config.get(
                "detector", {}
            ).get("name")

            if (
                not all([model_name, dataset_name, feature_set_name])
                or model_name == "unknown"
                or dataset_name == "unknown"
                or feature_set_name == "unknown"
            ):
                # Skip entries with missing or bad metadata
                continue

            # Get data dimensions
            n_samples = entry.get(
                "n_samples", config.get("data", {}).get("n_samples", 0)
            )
            feature_dim = entry.get(
                "feature_dim", config.get("data", {}).get("feature_dim", 0)
            )

            features.append(
                FeatureMatch(
                    feature_id=feature_id,
                    config=config,
                    file_path=entry["file"],
                    created_at=entry["created_at"],
                    match_score=1.0,  # All listed features are valid
                    n_samples=n_samples,
                    feature_dim=feature_dim,
                    metadata=metadata,
                )
            )

        # Sort by creation time (newest first)
        features.sort(key=lambda x: x.created_at, reverse=True)
        return features

    def load_features(self, feature_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load features by ID."""
        if feature_id not in self.features_registry["entries"]:
            raise KeyError(f"Feature ID not found: {feature_id}")

        entry = self.features_registry["entries"][feature_id]
        file_path = Path(entry["file"])
        format = entry.get("config", {}).get("format")

        return self.storage_manager.load_features(file_path, format)

    def load_features_with_names(
        self, feature_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load features by ID, returning features, labels, and feature names.

        Returns:
            Tuple of (features, labels, feature_names)
        """
        if feature_id not in self.features_registry["entries"]:
            raise KeyError(f"Feature ID not found: {feature_id}")

        entry = self.features_registry["entries"][feature_id]
        file_path = Path(entry["file"])
        format = entry.get("config", {}).get("format", "arrow")

        # Efficient loading with feature names (single file read for Arrow!)
        if format == "arrow":
            import pyarrow.parquet as pq

            table = pq.read_table(str(file_path))

            # Load features and labels efficiently
            if "features" in table.schema.names:
                # Legacy format - single features column with arrays
                df = table.to_pandas()
                features = np.array(df["features"].tolist())
                labels = np.array(df["label"]) if "label" in df.columns else None

                # Try to get feature names from metadata
                if table.schema.metadata and b"feature_names" in table.schema.metadata:
                    import json

                    feature_names = json.loads(
                        table.schema.metadata[b"feature_names"].decode()
                    )
                else:
                    feature_names = [f"feature_{i}" for i in range(features.shape[-1])]
            else:
                # Columnar format - efficient direct conversion
                if "label" in table.schema.names:
                    labels = table.column("label").to_numpy()
                    feature_columns = [
                        col for col in table.schema.names if col != "label"
                    ]
                    features_table = table.select(feature_columns)
                else:
                    labels = None
                    features_table = table
                    feature_columns = table.schema.names

                # Feature names from column names (most efficient!)
                feature_names = feature_columns

                # Convert to numpy
                try:
                    features = features_table.to_pandas(types_mapper=None).values
                except Exception:
                    features = features_table.to_pandas().values
        else:
            # For other formats, use storage manager and generate names
            features, labels = self.storage_manager.load_features(file_path, format)
            feature_names = [f"feature_{i}" for i in range(features.shape[-1])]

        return features, labels, feature_names

    def load_features_with_metadata(
        self, feature_id: str
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load features by ID, returning features, labels, and metadata including splits.

        This method returns the full metadata from the cached features, including
        splits, feature names, and other information stored during precomputation.

        Args:
            feature_id: Feature ID to load

        Returns:
            Tuple of (features, labels, metadata) where metadata contains:
            - splits: list of train/val/test split assignments
            - feature_names: list of feature names
            - Other precomputation metadata
        """
        if feature_id not in self.features_registry["entries"]:
            raise KeyError(f"Feature ID not found: {feature_id}")

        entry = self.features_registry["entries"][feature_id]
        file_path = Path(entry["file"])
        format = entry.get("config", {}).get("format", "arrow")

        # Use the FileFeatureStore to load the complete FeatureResult
        from shade_io import FeatureKey
        from shade_io.stores.file import FileFeatureStore

        # Create a dummy key to use the file store's load method
        # We'll construct it from the registry entry metadata
        registry_metadata = entry.get("metadata", {})
        dummy_key = FeatureKey(
            feature_set_name=registry_metadata.get("feature_set", "unknown"),
            model_name=registry_metadata.get("model", "unknown"),
            dataset_name=registry_metadata.get("dataset", "unknown"),
            metadata={},
        )

        # Create store with same format as the file
        store = FileFeatureStore(
            cache_dir=file_path.parent,
            format=format,
        )

        # Load using the actual file path by temporarily setting the key
        original_get_path = store._get_path
        store._get_path = lambda key: file_path

        try:
            result = store.load(dummy_key)
            if result is None:
                raise ValueError(f"Failed to load features from {file_path}")

            # Combine registry metadata with file metadata
            combined_metadata = {
                **registry_metadata,  # Registry metadata (from precompute_features registration)
                **result.metadata,  # File metadata (from FeatureResult)
            }

            # Ensure splits and feature names are available
            if result.splits is not None:
                combined_metadata["splits"] = result.splits
            if result.feature_names is not None:
                combined_metadata["feature_names"] = result.feature_names

            return (
                result.features.numpy(),
                result.labels.numpy() if result.labels is not None else None,
                combined_metadata,
            )

        finally:
            # Restore original method
            store._get_path = original_get_path

    def load_features_with_names(
        self, feature_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load features by ID, returning features, labels, and feature names.

        Args:
            feature_id: Feature ID to load

        Returns:
            Tuple of (features, labels, feature_names)
        """
        features, labels, metadata = self.load_features_with_metadata(feature_id)
        feature_names = metadata.get("feature_names", [])

        if not feature_names:
            logger.warning(
                f"No feature names found for feature_id {feature_id}, generating generic names"
            )
            # Generate generic names based on feature dimensionality
            n_features = features.shape[1] if features.ndim > 1 else len(features)
            feature_names = [f"feature_{i}" for i in range(n_features)]

        return features, labels, feature_names

    def load_features_for_split(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        split: str = "train",
        split_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load features for a specific data split with proper alignment.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            split: Which split to load ('train', 'val', or 'test')
            split_config: Data splitting configuration

        Returns:
            Tuple of (features, labels) for the requested split, or None
        """
        # Find matching features
        match = self.find_features(
            model_cfg, dataset_cfg, detector_cfg, match_mode="best"
        )

        if not match:
            return None

        logger.info(f"Loading features from {match.file_path} (ID: {match.feature_id})")

        # If no split config, load all features
        if not split_config:
            logger.warning("No split config provided, returning all features")
            return self.load_features(match.feature_id)

        # Check if this is an Arrow file with split support
        file_path = Path(match.file_path)
        if file_path.suffix == ".arrow":
            return self._load_arrow_split(file_path, split, split_config)
        else:
            # For other formats, need to load all and split
            logger.warning(
                f"Split filtering for {file_path.suffix} requires loading all data"
            )
            features, labels = self.load_features(match.feature_id)
            # TODO: Implement proper split filtering for non-Arrow formats
            return features, labels

    def _load_arrow_split(
        self, file_path: Path, split: str, split_config: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a specific split from an Arrow file."""
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        # Load table
        table = pq.read_table(str(file_path))

        # Check for split column
        if "split" in table.column_names:
            # Use split column for filtering
            mask = pc.equal(table["split"], split)
            filtered_table = table.filter(mask)
        else:
            # Use hash-based splitting
            logger.info("Using hash-based splitting for backward compatibility")
            hash_ids = table["hash_id"].to_pylist()

            # Determine split assignment
            train_ratio = split_config.get("train_ratio", 0.7)
            val_ratio = split_config.get("val_ratio", 0.15)

            split_hash_ids = []
            for hash_id in hash_ids:
                hash_int = int(hash_id[:16], 16)
                prob = (hash_int % (2**64)) / (2**64)

                if (
                    split == "train"
                    and prob < train_ratio
                    or split == "val"
                    and train_ratio <= prob < train_ratio + val_ratio
                    or split == "test"
                    and prob >= train_ratio + val_ratio
                ):
                    split_hash_ids.append(hash_id)

            # Filter table
            mask = pc.is_in(table["hash_id"], pa.array(split_hash_ids))
            filtered_table = table.filter(mask)

        # Convert to numpy
        df = filtered_table.to_pandas()
        features = np.array(df["features"].tolist())
        labels = np.array(df["label"])

        logger.info(f"Loaded {split} split: {len(features)} samples")
        return features, labels

    # ========================================================================
    # Preprocessor Management Methods
    # ========================================================================

    def register_preprocessor(
        self,
        preprocessor: Any,  # PreprocessingPipeline instance
        feature_id: str,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        preprocessor_config: dict[str, Any],
        state_dict_path: str | None = None,
    ) -> str:
        """Register fitted preprocessor.

        Args:
            preprocessor: Fitted preprocessor instance
            feature_id: ID of features used for training
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            preprocessor_config: Preprocessor-specific configuration
            state_dict_path: Optional path to PyTorch state dict file

        Returns:
            Preprocessor ID for retrieval
        """
        # Build config including preprocessor settings
        config_key = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)

        # Extract preprocessor info
        n_components = None
        explained_variance = None

        # Check if preprocessor has PCA component
        if hasattr(preprocessor, "preprocessors"):
            # Pipeline with multiple preprocessors
            for p in preprocessor.preprocessors:
                if hasattr(p, "pca"):
                    n_components = p.n_components
                    if (
                        hasattr(p.pca, "explained_variance_ratio_")
                        and p.pca.explained_variance_ratio_ is not None
                    ):
                        if hasattr(p.pca.explained_variance_ratio_, "sum"):
                            explained_variance = float(
                                p.pca.explained_variance_ratio_.sum().item()
                            )
                        else:
                            explained_variance = float(
                                sum(p.pca.explained_variance_ratio_)
                            )
                    break
        elif hasattr(preprocessor, "pca"):
            # Direct PCA preprocessor
            n_components = preprocessor.n_components
            if (
                hasattr(preprocessor.pca, "explained_variance_ratio_")
                and preprocessor.pca.explained_variance_ratio_ is not None
            ):
                if hasattr(preprocessor.pca.explained_variance_ratio_, "sum"):
                    explained_variance = float(
                        preprocessor.pca.explained_variance_ratio_.sum().item()
                    )
                else:
                    explained_variance = float(
                        sum(preprocessor.pca.explained_variance_ratio_)
                    )

        config_key["preprocessor"] = {
            "n_components": n_components,
            "explained_variance": explained_variance,
            "fitted": getattr(preprocessor, "fitted", True),
            "input_dim": getattr(preprocessor, "_input_dim", None),
            "output_dim": getattr(preprocessor, "_output_dim", None),
        }
        config_key["trained_on_features"] = feature_id

        # Generate ID
        preprocessor_id = self._compute_config_hash(config_key)

        # Save preprocessor (PyTorch state dict as primary, joblib as backup)
        preprocessor_file = self.preprocessor_dir / f"preprocessor_{preprocessor_id}.pt"
        joblib_file = (
            self.preprocessor_dir / f"preprocessor_{preprocessor_id}_joblib.pkl"
        )

        # Save PyTorch state dict (primary format)
        if not state_dict_path:
            # Generate state dict if not provided
            import torch

            preprocessor.eval()
            torch.save(
                {
                    "state_dict": preprocessor.state_dict(),
                    "class_name": type(preprocessor).__name__,
                    "module_path": type(preprocessor).__module__,
                    "config": {
                        "fitted": getattr(preprocessor, "fitted", True),
                        "input_dim": getattr(preprocessor, "_input_dim", None),
                        "output_dim": getattr(preprocessor, "_output_dim", None),
                    },
                },
                preprocessor_file,
            )
            state_dict_path = str(preprocessor_file)

        # Save joblib backup
        joblib.dump(preprocessor, joblib_file)

        # Update registry
        self.preprocessor_registry["entries"][preprocessor_id] = {
            "id": preprocessor_id,
            "config": config_key,
            "file": state_dict_path,  # Primary format (state dict)
            "joblib_file": str(joblib_file),  # Backup format
            "state_dict_file": state_dict_path,  # For compatibility
            "created_at": datetime.now().isoformat(),
            "feature_id": feature_id,
        }

        # Save registry
        self._save_json(self.preprocessor_registry, self.preprocessor_registry_file)

        logger.info(
            f"Registered preprocessor with ID: {preprocessor_id} "
            f"({n_components} components, {explained_variance:.1%} variance)"
            if n_components and explained_variance
            else f"Registered preprocessor with ID: {preprocessor_id}"
        )

        return preprocessor_id

    def find_preprocessor(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        feature_id: str | None = None,
        n_samples_used: int | None = None,
        min_seq_length: int | None = None,
    ) -> PreprocessorMatch | None:
        """Find preprocessor matching configuration.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            feature_id: Optionally require specific feature ID
            n_samples_used: Number of samples used to fit preprocessor
            min_seq_length: Minimum sequence length (k parameter)

        Returns:
            Best matching preprocessor or None
        """
        target_config = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)
        candidates = []

        for preprocessor_id, entry in self.preprocessor_registry["entries"].items():
            saved_config = entry["config"]

            # If feature_id specified, must match
            if feature_id and entry.get("feature_id") != feature_id:
                continue

            # Calculate match score (without preprocessor config itself)
            config_without_preprocessor = {
                k: v for k, v in saved_config.items() if k != "preprocessor"
            }
            target_without_preprocessor = {
                k: v for k, v in target_config.items() if k != "preprocessor"
            }

            score = self._calculate_match_score(
                target_without_preprocessor, config_without_preprocessor, "compatible"
            )

            if score > 0:
                # Check preprocessor-specific dimensions if provided
                preprocessor_config = saved_config.get("preprocessor", {})

                # Check sample size match
                if n_samples_used is not None:
                    saved_n_samples = preprocessor_config.get("n_samples_used")
                    if saved_n_samples is not None and saved_n_samples != n_samples_used:
                        continue  # Skip this candidate

                # Check min sequence length match
                if min_seq_length is not None:
                    saved_min_seq_length = preprocessor_config.get("min_seq_length")
                    if saved_min_seq_length is not None and saved_min_seq_length != min_seq_length:
                        continue  # Skip this candidate

                candidates.append((score, preprocessor_id, entry))

        if not candidates:
            return None

        # Return best match
        best_score, best_id, best_entry = max(candidates, key=lambda x: x[0])
        preprocessor_config = best_entry["config"].get("preprocessor", {})

        return PreprocessorMatch(
            preprocessor_id=best_id,
            config=best_entry["config"],
            file_path=best_entry["file"],  # Primary format (state dict)
            state_dict_path=best_entry.get("state_dict_file") or best_entry.get("file"),
            created_at=best_entry["created_at"],
            match_score=best_score,
            n_components=preprocessor_config.get("n_components"),
            explained_variance=preprocessor_config.get("explained_variance"),
            trained_on_features=best_entry.get("feature_id", ""),
        )

    def load_preprocessor(
        self, preprocessor_id: str, use_state_dict: bool = True
    ) -> Any:
        """Load preprocessor by ID.

        Args:
            preprocessor_id: Preprocessor ID
            use_state_dict: Whether to load from PyTorch state dict (if available)

        Returns:
            Loaded preprocessor instance
        """
        if preprocessor_id not in self.preprocessor_registry["entries"]:
            raise KeyError(f"Preprocessor ID not found: {preprocessor_id}")

        entry = self.preprocessor_registry["entries"][preprocessor_id]

        if use_state_dict:
            # Try primary state dict file first
            state_dict_path = entry.get("file") or entry.get("state_dict_file")
            if state_dict_path and Path(state_dict_path).exists():
                # Load using PyTorch state dict
                from importlib import import_module

                import torch

                # Load state dict
                try:
                    try:
                        checkpoint = torch.load(
                            state_dict_path, map_location="cpu", weights_only=True
                        )
                    except Exception:
                        # Fallback for checkpoints with non-weight data (PyTorch 2.6+ compatibility)
                        logger.debug(
                            "Falling back to weights_only=False for state dict loading"
                        )
                        checkpoint = torch.load(
                            state_dict_path, map_location="cpu", weights_only=False
                        )
                    class_name = checkpoint["class_name"]
                    module_path = checkpoint["module_path"]
                    state_dict = checkpoint["state_dict"]

                    # Dynamically import and recreate the class
                    module = import_module(module_path)
                    preprocessor_class = getattr(module, class_name)

                    # Use custom loading methods that handle buffer shape mismatches
                    if (
                        class_name == "PreprocessingPipeline"
                        and "pipeline_info" in checkpoint
                    ):
                        # Use PreprocessingPipeline's custom loading method
                        pipeline_info = checkpoint["pipeline_info"]
                        preprocessor = preprocessor_class.load_from_state_dict(
                            state_dict, pipeline_info
                        )
                    elif hasattr(preprocessor_class, "load_from_state_dict"):
                        # Use custom loading method for individual preprocessors
                        constructor_args = checkpoint.get("config", {}).get(
                            "constructor_args", {}
                        )
                        preprocessor = preprocessor_class.load_from_state_dict(
                            state_dict, constructor_args
                        )
                    else:
                        # Fallback to standard loading for preprocessors without custom methods
                        preprocessor = preprocessor_class()
                        try:
                            missing_keys, unexpected_keys = (
                                preprocessor.load_state_dict(state_dict, strict=False)
                            )
                            if missing_keys:
                                logger.debug(
                                    f"Missing keys when loading state dict: {missing_keys}"
                                )
                            if unexpected_keys:
                                logger.debug(
                                    f"Unexpected keys when loading state dict: {unexpected_keys}"
                                )
                        except Exception as e:
                            logger.warning(f"Standard loading failed: {e}")
                            raise

                    preprocessor.eval()  # Set to eval mode

                    logger.info(
                        f"Loaded preprocessor from state dict: {state_dict_path}"
                    )
                    return preprocessor
                except Exception as e:
                    logger.warning(
                        f"Failed to load from state dict: {e}, falling back to joblib"
                    )
            else:
                logger.debug(
                    f"State dict file not found: {state_dict_path}, falling back to joblib"
                )

        # Fallback to joblib
        joblib_path = entry.get("joblib_file") or entry.get("file")
        if joblib_path and Path(joblib_path).exists():
            logger.info(f"Loading preprocessor from joblib: {joblib_path}")
            return joblib.load(joblib_path)
        else:
            raise FileNotFoundError(
                f"No valid preprocessor file found for ID: {preprocessor_id}"
            )

    # ========================================================================
    # Model Management Methods
    # ========================================================================

    def register_model(
        self,
        checkpoint_path: str | Path,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        metrics: dict[str, float],
        training_cfg: DictConfig | dict | None = None,
        is_cv_ensemble: bool = False,
        fold_checkpoints: list[str] | None = None,
        force_update: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a trained model.

        Args:
            checkpoint_path: Path to model checkpoint
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            metrics: Performance metrics
            training_cfg: Training configuration
            is_cv_ensemble: Whether this is a CV ensemble model
            fold_checkpoints: List of fold checkpoint paths (for CV)
            force_update: Whether to update if better model exists
            metadata: Additional metadata to store

        Returns:
            Model ID
        """
        # Ensure checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Extract config key
        config_dict = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)
        config_key = f"{config_dict['model']['name']}_{config_dict['dataset']['name']}_{config_dict['detector']['name']}"

        # Generate model ID
        timestamp = datetime.now().isoformat()
        model_id = hashlib.md5(f"{config_key}_{timestamp}".encode()).hexdigest()[:16]

        # Prepare config
        config = {
            "model": OmegaConf.to_container(model_cfg)
            if isinstance(model_cfg, DictConfig)
            else model_cfg,
            "dataset": OmegaConf.to_container(dataset_cfg)
            if isinstance(dataset_cfg, DictConfig)
            else dataset_cfg,
            "detector": OmegaConf.to_container(detector_cfg)
            if isinstance(detector_cfg, DictConfig)
            else detector_cfg,
        }

        if training_cfg:
            config["training"] = (
                OmegaConf.to_container(training_cfg)
                if isinstance(training_cfg, DictConfig)
                else training_cfg
            )

        # Prepare metadata
        base_metadata = {
            "is_cv_ensemble": is_cv_ensemble,
            "config_key": config_key,
        }

        if is_cv_ensemble and fold_checkpoints:
            base_metadata["fold_checkpoints"] = [str(p) for p in fold_checkpoints]
            base_metadata["n_folds"] = len(fold_checkpoints)

        if metadata:
            base_metadata.update(metadata)

        # Check if we should update best model
        current_best_id = self.models_registry.get("best_models", {}).get(config_key)
        should_update_best = True

        if current_best_id and not force_update:
            current_best = self.models_registry["entries"].get(current_best_id)
            if current_best:
                # Compare primary metric (prefer val_auroc)
                metric_name = (
                    "val_auroc" if "val_auroc" in metrics else next(iter(metrics))
                )
                current_metric = current_best.get("metrics", {}).get(
                    metric_name, -float("inf")
                )
                new_metric = metrics.get(metric_name, -float("inf"))
                should_update_best = new_metric > current_metric

        # Register model
        self.models_registry["entries"][model_id] = {
            "model_id": model_id,
            "checkpoint_path": str(checkpoint_path),
            "config": config,
            "metrics": metrics,
            "metadata": base_metadata,
            "created_at": timestamp,
        }

        # Update best model if appropriate
        if should_update_best:
            self.models_registry.setdefault("best_models", {})[config_key] = model_id
            self.models_registry["entries"][model_id]["metadata"]["is_best"] = True

            # Unmark previous best
            if current_best_id and current_best_id != model_id:
                if current_best_id in self.models_registry["entries"]:
                    self.models_registry["entries"][current_best_id]["metadata"][
                        "is_best"
                    ] = False

            logger.info(f"Registered new best model for {config_key}: {model_id}")
        else:
            self.models_registry["entries"][model_id]["metadata"]["is_best"] = False
            logger.info(f"Registered model for {config_key}: {model_id} (not best)")

        # Save registry
        self._save_json(self.models_registry, self.models_registry_file)

        return model_id

    def find_best_model(
        self,
        model_cfg: DictConfig | dict | None = None,
        dataset_cfg: DictConfig | dict | None = None,
        detector_cfg: DictConfig | dict | None = None,
        config_key: str | None = None,
        metric: str = "val_auroc",
    ) -> ModelMatch | None:
        """Find the best model for a configuration.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            config_key: Pre-computed config key (alternative to configs)
            metric: Metric to use for selecting best model

        Returns:
            Best model info or None if not found
        """
        # Get config key
        if config_key is None:
            if not all([model_cfg, dataset_cfg, detector_cfg]):
                raise ValueError("Must provide either config_key or all configs")
            config_dict = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)
            config_key = f"{config_dict['model']['name']}_{config_dict['dataset']['name']}_{config_dict['detector']['name']}"

        # First check if we have a tracked best model
        best_model_id = self.models_registry.get("best_models", {}).get(config_key)
        if best_model_id:
            model_data = self.models_registry["entries"].get(best_model_id)
            if model_data:
                return ModelMatch(
                    model_id=model_data["model_id"],
                    config=model_data["config"],
                    checkpoint_path=model_data["checkpoint_path"],
                    created_at=model_data["created_at"],
                    match_score=1.0,
                    metrics=model_data["metrics"],
                    is_best=model_data.get("metadata", {}).get("is_best", False),
                    metadata=model_data.get("metadata", {}),
                )

        # Otherwise find best based on metric
        all_models = self.list_models(config_key=config_key)
        if not all_models:
            return None

        # Select best model based on metric
        best_model = None
        best_metric_value = -float("inf")

        for model in all_models:
            metric_value = model.metrics.get(metric, -float("inf"))
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model = model

        if best_model:
            # Update the registry to track this as best
            self.models_registry.setdefault("best_models", {})[config_key] = (
                best_model.model_id
            )
            self._save_json(self.models_registry, self.models_registry_file)

        return best_model

    def list_models(
        self,
        config_key: str | None = None,
        is_cv_ensemble: bool | None = None,
    ) -> list[ModelMatch]:
        """List registered models.

        Args:
            config_key: Filter by config key
            is_cv_ensemble: Filter by CV ensemble status

        Returns:
            List of model matches
        """
        models = []

        for _model_id, model_data in self.models_registry["entries"].items():
            # Apply filters
            if (
                config_key
                and model_data.get("metadata", {}).get("config_key") != config_key
            ):
                continue

            if is_cv_ensemble is not None:
                if (
                    model_data.get("metadata", {}).get("is_cv_ensemble", False)
                    != is_cv_ensemble
                ):
                    continue

            models.append(
                ModelMatch(
                    model_id=model_data["model_id"],
                    config=model_data.get("config", {}),
                    checkpoint_path=model_data["checkpoint_path"],
                    created_at=model_data.get("created_at", ""),
                    match_score=1.0,
                    metrics=model_data.get("metrics", {}),
                    is_best=model_data.get("metadata", {}).get("is_best", False),
                    metadata=model_data.get("metadata", {}),
                )
            )

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    # ========================================================================
    # Unified Search Methods
    # ========================================================================

    def find_artifact(
        self,
        artifact_type: str | None = None,
        dataset_name: str | None = None,
        model_name: str | None = None,
        detector_name: str | None = None,
        judge: str | None = None,
    ) -> list[UnifiedMatch]:
        """Find artifacts matching criteria across all registries.

        Args:
            artifact_type: Type of artifact ("feature", "pca", "model", "enhanced", "samples")
            dataset_name: Dataset name to match
            model_name: Model name to match
            detector_name: Detector name to match
            judge: Judge name to match (for enhanced datasets)

        Returns:
            List of matching artifacts sorted by relevance
        """
        matches = []

        # Search features
        if not artifact_type or artifact_type == "feature":
            for feature_id, entry in self.features_registry["entries"].items():
                score = self._calculate_artifact_match_score(
                    entry, dataset_name, model_name, detector_name
                )
                if score > 0:
                    data_config = entry.get("config", {}).get("data", {})
                    matches.append(
                        UnifiedMatch(
                            artifact_type="feature",
                            artifact_id=feature_id,
                            file_path=entry["file"],
                            created_at=entry["created_at"],
                            match_score=score,
                            metadata=entry.get("metadata", {}),
                            dataset_name=entry.get("config", {})
                            .get("dataset", {})
                            .get("name"),
                            model_name=entry.get("config", {})
                            .get("model", {})
                            .get("name"),
                            detector_name=entry.get("config", {})
                            .get("detector", {})
                            .get("name"),
                            n_samples=data_config.get("n_samples"),
                            feature_dim=data_config.get("feature_dim"),
                        )
                    )

        # Note: PCA models are now part of preprocessor registry

        # Search models
        if not artifact_type or artifact_type == "model":
            for model_id, entry in self.models_registry["entries"].items():
                score = self._calculate_artifact_match_score(
                    entry.get("metadata", {}), dataset_name, model_name, detector_name
                )
                if score > 0:
                    matches.append(
                        UnifiedMatch(
                            artifact_type="model",
                            artifact_id=model_id,
                            file_path=entry["checkpoint_path"],
                            created_at=entry.get("created_at", "unknown"),
                            match_score=score,
                            metadata=entry.get("metadata", {}),
                            model_name=entry.get("metadata", {}).get("model_name"),
                        )
                    )

        # Search enhanced datasets
        if not artifact_type or artifact_type == "enhanced":
            for dataset_id, entry in self.enhanced_registry.get(
                "enhanced_datasets", {}
            ).items():
                score = 0.0
                if dataset_name and entry.get("dataset_name") == dataset_name:
                    score += 1.0
                if judge and entry.get("judge") == judge:
                    score += 0.5

                if score > 0:
                    matches.append(
                        UnifiedMatch(
                            artifact_type="enhanced",
                            artifact_id=dataset_id,
                            file_path=entry["file_path"],
                            created_at=entry["created_at"],
                            match_score=score,
                            metadata=entry.get("metadata", {}),
                            dataset_name=entry["dataset_name"],
                            judge=entry.get("judge"),
                            n_samples=entry.get("n_examples"),
                        )
                    )

        # Search semantic samples
        if not artifact_type or artifact_type == "samples":
            for sample_id, entry in self.enhanced_registry.get(
                "semantic_samples", {}
            ).items():
                if dataset_name and entry.get("dataset_name") == dataset_name:
                    matches.append(
                        UnifiedMatch(
                            artifact_type="samples",
                            artifact_id=sample_id,
                            file_path=entry["file_path"],
                            created_at=entry["created_at"],
                            match_score=1.0,
                            metadata={},
                            dataset_name=entry["dataset_name"],
                            n_samples=entry.get("total_samples"),
                        )
                    )

        # Sort by score and recency
        matches.sort(key=lambda m: (m.match_score, m.created_at), reverse=True)
        return matches

    def _calculate_artifact_match_score(
        self,
        entry: dict[str, Any],
        dataset_name: str | None,
        model_name: str | None,
        detector_name: str | None,
    ) -> float:
        """Calculate match score for an artifact entry."""
        score = 0.0

        # Check dataset match
        if dataset_name:
            entry_dataset = str(
                entry.get("dataset", entry.get("dataset_name", ""))
            ).lower()
            if entry_dataset and (
                dataset_name.lower() in entry_dataset
                or entry_dataset in dataset_name.lower()
            ):
                score += 1.0

        # Check model match
        if model_name:
            entry_model = str(entry.get("model", entry.get("model_name", ""))).lower()
            if entry_model and (
                model_name.lower() in entry_model or entry_model in model_name.lower()
            ):
                score += 0.5

        # Check detector match
        if detector_name:
            entry_detector = str(
                entry.get("detector", entry.get("detector_name", ""))
            ).lower()
            if entry_detector and (
                detector_name.lower() in entry_detector
                or entry_detector in detector_name.lower()
            ):
                score += 0.5

        return score

    def _calculate_match_score(
        self, target: dict[str, Any], saved: dict[str, Any], mode: str
    ) -> float:
        """Calculate match score between configurations."""
        score = 0.0

        # Model matching (must be exact)
        if target["model"]["name"] != saved["model"]["name"]:
            return 0.0
        score += 10.0

        # Dataset matching
        target_dataset_type = target["dataset"].get("type", "")
        saved_dataset_type = saved["dataset"].get("type", "")

        # Handle auto-discovered files
        if saved.get("auto_discovered", False):
            target_name = target["dataset"].get("name", "").lower()
            saved_name = saved["dataset"].get("name", "").lower()
            if target_name and saved_name and target_name in saved_name:
                score += 10.0
            elif saved_name == "haluevaldataset" and target_name == "halueval":
                score += 10.0  # Common name variation
        else:
            if target_dataset_type != saved_dataset_type:
                return 0.0
            score += 10.0

        # Dataset details
        if target["dataset"].get("config_name") == saved["dataset"].get("config_name"):
            score += 5.0

        # Detector matching
        target_detector_name = target["detector"].get("name", "").lower()
        saved_detector_name = saved["detector"].get("name", "").lower()

        if target_detector_name != saved_detector_name:
            return 0.0
        score += 10.0

        # Feature extractor matching (if present)
        if "feature_extractor" in target["detector"]:
            if "feature_extractor" in saved["detector"]:
                target_fe = target["detector"]["feature_extractor"]
                saved_fe = saved["detector"]["feature_extractor"]

                # Check type and parameters
                if target_fe.get("type") == saved_fe.get("type"):
                    score += 5.0
                elif mode == "exact":
                    return 0.0

                if target_fe.get("k_eigenvalues") == saved_fe.get("k_eigenvalues"):
                    score += 3.0

        # Prefer more samples
        n_samples = saved.get("data", {}).get("n_samples", 0)
        score += min(n_samples / 1000, 10.0)  # Up to 10 points for samples

        # Prefer newer
        try:
            created_at = datetime.fromisoformat(saved.get("created_at", ""))
            age_days = (datetime.now() - created_at).days
            score -= age_days * 0.1  # Penalty for age
        except:
            pass

        return score

    # ========================================================================
    # Statistics and Maintenance Methods
    # ========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics across all registries."""
        stats = {
            "features": {
                "count": len(self.features_registry["entries"]),
                "total_size_mb": sum(
                    Path(e["file"]).stat().st_size / 1024 / 1024
                    for e in self.features_registry["entries"].values()
                    if Path(e["file"]).exists()
                ),
            },
            "preprocessors": {
                "count": len(self.preprocessor_registry["entries"]),
            },
            "trained_models": {
                "count": len(self.models_registry["entries"]),
                "best_models": len(self.models_registry.get("best_models", {})),
            },
            "enhanced_datasets": {
                "count": len(self.enhanced_registry.get("enhanced_datasets", {})),
                "by_dataset": {},
            },
            "semantic_samples": {
                "count": len(self.enhanced_registry.get("semantic_samples", {})),
                "total_samples": sum(
                    e.get("total_samples", 0)
                    for e in self.enhanced_registry.get("semantic_samples", {}).values()
                ),
            },
            "total_artifacts": 0,
            "cache_dir": str(self.cache_dir),
        }

        # Count by dataset for enhanced
        for entry in self.enhanced_registry.get("enhanced_datasets", {}).values():
            dataset = entry.get("dataset_name", "unknown")
            stats["enhanced_datasets"]["by_dataset"][dataset] = (
                stats["enhanced_datasets"]["by_dataset"].get(dataset, 0) + 1
            )

        # Total count
        stats["total_artifacts"] = sum(
            [
                stats["features"]["count"],
                stats["preprocessors"]["count"],
                stats["trained_models"]["count"],
                stats["enhanced_datasets"]["count"],
                stats["semantic_samples"]["count"],
            ]
        )

        return stats

    def cleanup_orphaned_files(self, dry_run: bool = True) -> list[Path]:
        """Find and optionally remove orphaned cache files not in any registry.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            List of orphaned files
        """
        orphaned = []

        # Collect all registered file paths
        registered_paths = set()

        # From features registry
        for entry in self.features_registry["entries"].values():
            registered_paths.add(Path(entry["file"]).resolve())

        # From preprocessor registry
        for entry in self.preprocessor_registry["entries"].values():
            if "file" in entry:
                registered_paths.add(Path(entry["file"]).resolve())
            if "joblib_file" in entry:
                registered_paths.add(Path(entry["joblib_file"]).resolve())

        # From models registry
        for entry in self.models_registry["entries"].values():
            registered_paths.add(Path(entry["checkpoint_path"]).resolve())

        # From enhanced registry
        for entry in self.enhanced_registry.get("enhanced_datasets", {}).values():
            registered_paths.add(Path(entry["file_path"]).resolve())
        for entry in self.enhanced_registry.get("semantic_samples", {}).values():
            registered_paths.add(Path(entry["file_path"]).resolve())

        # Check all files in cache directories
        for cache_subdir in [
            self.features_dir,
            self.preprocessor_dir,
            self.models_dir,
            self.enhanced_dir,
            self.samples_dir,
        ]:
            if cache_subdir.exists():
                for file_path in cache_subdir.rglob("*"):
                    if (
                        file_path.is_file()
                        and file_path.resolve() not in registered_paths
                    ):
                        orphaned.append(file_path)
                        if not dry_run:
                            file_path.unlink()
                            logger.info(f"Removed orphaned file: {file_path}")

        return orphaned

    def get_linked_artifacts(self, feature_id: str) -> dict[str, list[str]]:
        """Get all artifacts linked to a feature set.

        Args:
            feature_id: Feature ID to check

        Returns:
            Dictionary mapping artifact types to linked IDs
        """
        linked = {"preprocessors": []}

        # Find preprocessors trained on these features
        for preprocessor_id, entry in self.preprocessor_registry["entries"].items():
            if entry.get("feature_id") == feature_id:
                linked["preprocessors"].append(preprocessor_id)

        return linked

    # ========================================================================
    # Progressive Caching Methods
    # ========================================================================

    def initialize_progressive_cache(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
        total_samples: int,
        feature_dim: int,
        cache_format: str = "npz",
    ) -> str:
        """Initialize a progressive cache for incremental feature storage.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            total_samples: Total number of samples to cache
            feature_dim: Feature dimension
            cache_format: Storage format

        Returns:
            Cache ID for tracking
        """
        # Generate cache ID
        config_key = self._extract_config_key(model_cfg, dataset_cfg, detector_cfg)
        cache_id = self._compute_config_hash(config_key)

        # Create temporary cache file
        temp_file = self.features_dir / f"features_{cache_id}_partial.{cache_format}"

        # Initialize based on format
        if cache_format == "npz":
            # For NPZ, accumulate in memory
            self._progressive_caches[cache_id] = {
                "features": [],
                "labels": [],
                "indices": [],
                "total_samples": total_samples,
                "feature_dim": feature_dim,
                "format": cache_format,
                "temp_file": temp_file,
                "config": config_key,
            }
        elif cache_format == "arrow":
            # For Arrow, we can write incrementally
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pa.schema(
                [
                    ("hash_id", pa.string()),
                    ("idx", pa.int64()),
                    ("features", pa.list_(pa.float32())),
                    ("label", pa.int64()),
                    ("split", pa.string()),
                ]
            )

            self._progressive_caches[cache_id] = {
                "writer": pq.ParquetWriter(str(temp_file), schema),
                "total_samples": total_samples,
                "cached_count": 0,
                "format": cache_format,
                "temp_file": temp_file,
                "config": config_key,
            }
        else:
            raise ValueError(
                f"Progressive caching not yet supported for format: {cache_format}"
            )

        logger.info(f"Initialized progressive cache (ID: {cache_id})")
        return cache_id

    def cache_batch(
        self,
        cache_id: str,
        indices: list[int],
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        hash_ids: list[str] | None = None,
        splits: list[str] | None = None,
    ) -> None:
        """Cache a batch of features progressively.

        Args:
            cache_id: Cache ID from initialize_progressive_cache
            indices: Sample indices
            features: Feature array/tensor
            labels: Label array/tensor
            hash_ids: Hash IDs for each sample
            splits: Split assignments for each sample
        """
        if cache_id not in self._progressive_caches:
            raise KeyError(f"Progressive cache not initialized: {cache_id}")

        cache = self._progressive_caches[cache_id]

        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        if cache["format"] == "npz":
            # Accumulate in memory
            cache["features"].extend(features)
            cache["labels"].extend(labels)
            cache["indices"].extend(indices)
        elif cache["format"] == "arrow":
            # Write to Arrow file
            import pyarrow as pa

            if hash_ids is None:
                hash_ids = [f"sample_{idx}" for idx in indices]
            if splits is None:
                splits = ["unknown"] * len(indices)

            # Create table from batch
            data = []
            for _i, (hash_id, idx, feat, label, split) in enumerate(
                zip(hash_ids, indices, features, labels, splits, strict=False)
            ):
                data.append(
                    {
                        "hash_id": hash_id,
                        "idx": idx,
                        "features": feat.astype(np.float32).tolist(),
                        "label": int(label),
                        "split": split,
                    }
                )

            table = pa.Table.from_pylist(data)
            cache["writer"].write_table(table)
            cache["cached_count"] += len(indices)

        # Check if cache is complete
        if (
            cache["format"] == "npz"
            and len(cache["indices"]) >= cache["total_samples"]
            or cache["format"] == "arrow"
            and cache["cached_count"] >= cache["total_samples"]
        ):
            self.finalize_progressive_cache(cache_id)

    def finalize_progressive_cache(self, cache_id: str) -> str:
        """Finalize a progressive cache and register it.

        Args:
            cache_id: Cache ID

        Returns:
            Registered feature ID
        """
        if cache_id not in self._progressive_caches:
            # Check if already finalized
            if cache_id in self.features_registry["entries"]:
                logger.info(f"Progressive cache {cache_id} already finalized")
                return cache_id
            else:
                raise ValueError(f"Progressive cache {cache_id} not found")

        cache = self._progressive_caches[cache_id]

        if cache["format"] == "npz":
            # Sort by indices to maintain order
            indices = np.array(cache["indices"])
            features = np.array(cache["features"])
            labels = np.array(cache["labels"])

            sort_idx = np.argsort(indices)
            features = features[sort_idx]
            labels = labels[sort_idx]

            # Save final file
            final_file = self.features_dir / f"features_{cache_id}.npz"
            np.savez_compressed(final_file, features=features, labels=labels)

            # Remove temp file
            if cache["temp_file"].exists():
                cache["temp_file"].unlink()

        elif cache["format"] == "arrow":
            # Close writer
            cache["writer"].close()

            # Move temp file to final location
            final_file = self.features_dir / f"features_{cache_id}.arrow"
            cache["temp_file"].rename(final_file)

        # Register in the registry
        config = cache["config"]
        config["data"] = {
            "n_samples": cache.get("total_samples", len(cache.get("features", []))),
            "feature_dim": cache.get("feature_dim", 0),
        }
        config["format"] = cache["format"]

        self.features_registry["entries"][cache_id] = {
            "id": cache_id,
            "config": config,
            "file": str(final_file),
            "created_at": datetime.now().isoformat(),
            "metadata": {"progressive_cache": True},
        }

        # Save registry
        self._save_json(self.features_registry, self.features_registry_file)

        # Clean up progressive cache
        del self._progressive_caches[cache_id]

        logger.info(f"Finalized progressive cache: {cache_id}")
        return cache_id
