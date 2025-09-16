"""Model registry for managing trained models and checkpoints.

This registry tracks the best performing models for each configuration,
handles model versioning, and supports both standard and CV ensemble models.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# Utility functions for extracting names from configs
def get_model_name(cfg: DictConfig) -> str:
    """Extract model name from config."""
    if hasattr(cfg, "model_name"):
        return cfg.model_name
    elif hasattr(cfg, "name"):
        return cfg.name
    elif hasattr(cfg, "_target_"):
        return cfg._target_.split(".")[-1]
    return "unknown"


def get_dataset_name(cfg: DictConfig) -> str:
    """Extract dataset name from config."""
    if hasattr(cfg, "dataset_name"):
        return cfg.dataset_name
    elif hasattr(cfg, "name"):
        return cfg.name
    elif hasattr(cfg, "_target_"):
        name = cfg._target_.split(".")[-1]
        return name.replace("Dataset", "")
    return "unknown"


def get_detector_name(cfg: DictConfig) -> str:
    """Extract detector name from config."""
    if hasattr(cfg, "detector_name"):
        return cfg.detector_name
    elif hasattr(cfg, "name"):
        return cfg.name
    elif hasattr(cfg, "_target_"):
        return cfg._target_.split(".")[-1]
    return "unknown"


logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about a registered model."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ):
        self.model_id = model_id
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.metrics = metrics
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "checkpoint_path": str(self.checkpoint_path),
            "config": self.config,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        info = cls(
            model_id=data["model_id"],
            checkpoint_path=data["checkpoint_path"],
            config=data["config"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )
        info.created_at = data.get("created_at", datetime.now().isoformat())
        return info


class ModelRegistry:
    """Registry for managing trained models and checkpoints."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the model registry.

        Args:
            cache_dir: Directory for registry files (default: .cache/shade)
        """
        self.cache_dir = cache_dir or Path(".cache/shade")
        self.models_dir = self.cache_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Registry file
        self.registry_file = self.cache_dir / "model_registry.json"

        # Load registry
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                return {"entries": {}, "best_models": {}}
        return {"entries": {}, "best_models": {}}

    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _compute_config_key(
        self,
        model_cfg: DictConfig | dict,
        dataset_cfg: DictConfig | dict,
        detector_cfg: DictConfig | dict,
    ) -> str:
        """Compute configuration key for lookup.

        Creates a standardized key from the configuration that uniquely
        identifies a model/dataset/detector combination.
        """
        # Convert to plain dicts if needed
        if isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.to_container(model_cfg)
        if isinstance(dataset_cfg, DictConfig):
            dataset_cfg = OmegaConf.to_container(dataset_cfg)
        if isinstance(detector_cfg, DictConfig):
            detector_cfg = OmegaConf.to_container(detector_cfg)

        # Extract key components using naming utilities
        model_name = get_model_name(model_cfg)
        dataset_name = get_dataset_name(dataset_cfg)
        detector_name = get_detector_name(detector_cfg)

        # Create standardized key
        return f"{model_name}_{dataset_name}_{detector_name}"

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
            metrics: Performance metrics (e.g., val_auroc, test_auroc)
            training_cfg: Training configuration
            is_cv_ensemble: Whether this is a CV ensemble model
            fold_checkpoints: List of fold checkpoint paths (for CV)
            force_update: Whether to update if better model exists
            metadata: Additional metadata to store with the model

        Returns:
            Model ID
        """
        # Ensure checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Compute config key
        config_key = self._compute_config_key(model_cfg, dataset_cfg, detector_cfg)

        # Generate model ID
        import hashlib

        timestamp = datetime.now().isoformat()
        model_id = hashlib.md5(f"{config_key}_{timestamp}".encode()).hexdigest()[:16]

        # Prepare config dict
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

        # Merge with provided metadata
        if metadata:
            base_metadata.update(metadata)

        final_metadata = base_metadata

        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            checkpoint_path=str(checkpoint_path),
            config=config,
            metrics=metrics,
            metadata=final_metadata,
        )

        # Check if we should update best model
        current_best_id = self.registry["best_models"].get(config_key)
        should_update_best = True

        if current_best_id and not force_update:
            current_best = self.registry["entries"].get(current_best_id)
            if current_best:
                # Compare primary metric (prefer val_auroc)
                metric_name = (
                    "val_auroc" if "val_auroc" in metrics else next(iter(metrics))
                )
                current_metric = current_best["metrics"].get(metric_name, -float("inf"))
                new_metric = metrics.get(metric_name, -float("inf"))
                should_update_best = new_metric > current_metric

                if should_update_best:
                    logger.info(
                        f"New model has better {metric_name}: {new_metric:.4f} > {current_metric:.4f}"
                    )

        # Register model
        self.registry["entries"][model_id] = model_info.to_dict()

        # Update best model if appropriate
        if should_update_best:
            self.registry["best_models"][config_key] = model_id
            logger.info(f"Registered new best model for {config_key}: {model_id}")

            # Mark this as the best model in metadata
            self.registry["entries"][model_id]["metadata"]["is_best"] = True

            # Unmark previous best if exists
            if current_best_id and current_best_id != model_id:
                if current_best_id in self.registry["entries"]:
                    self.registry["entries"][current_best_id]["metadata"]["is_best"] = (
                        False
                    )
        else:
            logger.info(f"Registered model for {config_key}: {model_id} (not best)")
            self.registry["entries"][model_id]["metadata"]["is_best"] = False

        # Save registry
        self._save_registry()

        return model_id

    def find_best_model(
        self,
        model_cfg: DictConfig | dict | None = None,
        dataset_cfg: DictConfig | dict | None = None,
        detector_cfg: DictConfig | dict | None = None,
        config_key: str | None = None,
        metric: str = "val_auroc",
    ) -> ModelInfo | None:
        """Find the best model for a configuration.

        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            detector_cfg: Detector configuration
            config_key: Pre-computed config key (alternative to configs)
            metric: Metric to use for selecting best model (default: val_auroc)

        Returns:
            Best model info or None if not found
        """
        # Get config key
        if config_key is None:
            if not all([model_cfg, dataset_cfg, detector_cfg]):
                raise ValueError("Must provide either config_key or all configs")
            config_key = self._compute_config_key(model_cfg, dataset_cfg, detector_cfg)

        # First check if we have a tracked best model
        best_model_id = self.registry["best_models"].get(config_key)
        if best_model_id:
            model_data = self.registry["entries"].get(best_model_id)
            if model_data:
                return ModelInfo.from_dict(model_data)

        # If no tracked best model, find all models for this config and select best
        all_models = self.list_models(config_key=config_key)
        if not all_models:
            logger.warning(f"No models found for {config_key}")
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
            logger.info(
                f"Selected model {best_model.model_id} with {metric}={best_metric_value:.4f}"
            )
            # Update the registry to track this as best
            self.registry["best_models"][config_key] = best_model.model_id
            self._save_registry()

        return best_model

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get information about a specific model.

        Args:
            model_id: Model ID

        Returns:
            Model info or None if not found
        """
        model_data = self.registry["entries"].get(model_id)
        if not model_data:
            return None
        return ModelInfo.from_dict(model_data)

    def list_models(
        self,
        config_key: str | None = None,
        is_cv_ensemble: bool | None = None,
    ) -> list[ModelInfo]:
        """List registered models.

        Args:
            config_key: Filter by config key
            is_cv_ensemble: Filter by CV ensemble status

        Returns:
            List of model infos
        """
        models = []

        for _model_id, model_data in self.registry["entries"].items():
            # Apply filters
            if config_key and model_data["metadata"].get("config_key") != config_key:
                continue

            if is_cv_ensemble is not None and (
                model_data["metadata"].get("is_cv_ensemble", False) != is_cv_ensemble
            ):
                continue

            models.append(ModelInfo.from_dict(model_data))

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def cleanup_old_models(
        self,
        keep_best: bool = True,
        keep_recent: int = 5,
        max_age_days: int = 30,
    ) -> int:
        """Clean up old model entries.

        Args:
            keep_best: Keep best models regardless of age
            keep_recent: Number of recent models to keep per config
            max_age_days: Maximum age in days

        Returns:
            Number of models removed
        """
        removed = 0
        now = datetime.now()

        # Group models by config key
        models_by_config = {}
        for model_id, model_data in list(self.registry["entries"].items()):
            config_key = model_data["metadata"].get("config_key", "unknown")
            if config_key not in models_by_config:
                models_by_config[config_key] = []
            models_by_config[config_key].append((model_id, model_data))

        # Process each config
        for config_key, models in models_by_config.items():
            # Sort by creation time (newest first)
            models.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)

            # Determine which to keep
            best_model_id = self.registry["best_models"].get(config_key)

            for i, (model_id, model_data) in enumerate(models):
                # Keep best model
                if keep_best and model_id == best_model_id:
                    continue

                # Keep recent models
                if i < keep_recent:
                    continue

                # Check age
                created_at = datetime.fromisoformat(
                    model_data.get("created_at", now.isoformat())
                )
                age_days = (now - created_at).days

                if age_days > max_age_days:
                    # Remove model
                    del self.registry["entries"][model_id]
                    removed += 1
                    logger.info(f"Removed old model {model_id} (age: {age_days} days)")

        if removed > 0:
            self._save_registry()

        return removed

    def export_best_models(self, output_dir: Path) -> dict[str, str]:
        """Export best models to a directory.

        Args:
            output_dir: Directory to export models to

        Returns:
            Mapping of config keys to exported paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = {}

        for config_key, model_id in self.registry["best_models"].items():
            model_data = self.registry["entries"].get(model_id)
            if not model_data:
                logger.warning(f"Best model {model_id} for {config_key} not found")
                continue

            # Copy checkpoint
            src_path = Path(model_data["checkpoint_path"])
            if not src_path.exists():
                logger.warning(f"Checkpoint {src_path} not found")
                continue

            dst_path = output_dir / f"{config_key}_best.ckpt"

            # Copy file
            import shutil

            shutil.copy2(src_path, dst_path)

            # Save metadata
            meta_path = output_dir / f"{config_key}_best.json"
            with open(meta_path, "w") as f:
                json.dump(model_data, f, indent=2)

            exported[config_key] = str(dst_path)
            logger.info(f"Exported best model for {config_key} to {dst_path}")

        return exported
