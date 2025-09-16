"""Migration utilities for transitioning to UnifiedRegistry.

This module provides utilities to migrate existing registries from:
- shade-train's FeatureRegistry and UnifiedRegistry
- shade's model registry
- shade-apps' various caches

To the new consolidated UnifiedRegistry in shade-io.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import logging
import shutil
from pathlib import Path

from shade_io.registry.unified_registry import UnifiedRegistry

logger = logging.getLogger(__name__)


class RegistryMigrator:
    """Migrates existing registries to the new UnifiedRegistry."""

    def __init__(
        self,
        source_cache_dir: Path,
        target_cache_dir: Path | None = None,
        dry_run: bool = False,
    ):
        """Initialize the registry migrator.

        Args:
            source_cache_dir: Source directory containing old registries
            target_cache_dir: Target directory for new registry (defaults to source)
            dry_run: If True, only report what would be done
        """
        self.source_cache_dir = Path(source_cache_dir)
        self.target_cache_dir = Path(target_cache_dir or source_cache_dir)
        self.dry_run = dry_run

        # Initialize new unified registry
        if not dry_run:
            self.unified_registry = UnifiedRegistry(self.target_cache_dir)
        else:
            self.unified_registry = None

        self.stats = {
            "features_migrated": 0,
            "pca_migrated": 0,
            "models_migrated": 0,
            "enhanced_migrated": 0,
            "samples_migrated": 0,
            "errors": [],
        }

    def migrate_all(self) -> dict:
        """Migrate all registries.

        Returns:
            Migration statistics
        """
        logger.info(f"Starting migration from {self.source_cache_dir}")

        # Migrate shade-train registries
        self._migrate_shade_train_features()
        self._migrate_shade_train_pca()
        self._migrate_shade_train_unified()

        # Migrate shade model registry
        self._migrate_shade_models()

        # Auto-discovery will handle unregistered files
        if not self.dry_run:
            self.unified_registry._auto_discover_all()

        logger.info(f"Migration complete: {self.stats}")
        return self.stats

    def _migrate_shade_train_features(self):
        """Migrate shade-train's features_registry.json."""
        source_file = self.source_cache_dir / "features_registry.json"
        if not source_file.exists():
            logger.info("No shade-train features registry found")
            return

        logger.info(f"Migrating features from {source_file}")

        try:
            with open(source_file) as f:
                old_registry = json.load(f)

            for feature_id, entry in old_registry.get("entries", {}).items():
                # Check if file still exists
                file_path = Path(entry["file"])
                if not file_path.exists():
                    logger.warning(f"Feature file not found: {file_path}")
                    self.stats["errors"].append(f"Missing feature file: {file_path}")
                    continue

                if self.dry_run:
                    logger.info(f"Would migrate feature {feature_id}")
                else:
                    # Add to new registry
                    self.unified_registry.features_registry["entries"][feature_id] = (
                        entry
                    )

                self.stats["features_migrated"] += 1

            # Save if not dry run
            if not self.dry_run and self.stats["features_migrated"] > 0:
                self.unified_registry._save_json(
                    self.unified_registry.features_registry,
                    self.unified_registry.features_registry_file,
                )

        except Exception as e:
            logger.error(f"Error migrating features registry: {e}")
            self.stats["errors"].append(f"Features migration error: {e}")

    def _migrate_shade_train_pca(self):
        """Migrate shade-train's pca_registry.json."""
        source_file = self.source_cache_dir / "pca_registry.json"
        if not source_file.exists():
            logger.info("No shade-train PCA registry found")
            return

        logger.info(f"Migrating PCA models from {source_file}")

        try:
            with open(source_file) as f:
                old_registry = json.load(f)

            for pca_id, entry in old_registry.get("entries", {}).items():
                # Check if file still exists
                file_path = Path(entry["file"])
                if not file_path.exists():
                    logger.warning(f"PCA file not found: {file_path}")
                    self.stats["errors"].append(f"Missing PCA file: {file_path}")
                    continue

                if self.dry_run:
                    logger.info(f"Would migrate PCA model {pca_id}")
                else:
                    # Add to new registry
                    self.unified_registry.pca_registry["entries"][pca_id] = entry

                self.stats["pca_migrated"] += 1

            # Save if not dry run
            if not self.dry_run and self.stats["pca_migrated"] > 0:
                self.unified_registry._save_json(
                    self.unified_registry.pca_registry,
                    self.unified_registry.pca_registry_file,
                )

        except Exception as e:
            logger.error(f"Error migrating PCA registry: {e}")
            self.stats["errors"].append(f"PCA migration error: {e}")

    def _migrate_shade_train_unified(self):
        """Migrate shade-train's unified registry (enhanced datasets, samples)."""
        # Check for enhanced registry
        enhanced_file = self.source_cache_dir / "registries" / "enhanced_registry.json"
        if enhanced_file.exists():
            logger.info(f"Migrating enhanced datasets from {enhanced_file}")

            try:
                with open(enhanced_file) as f:
                    old_registry = json.load(f)

                # Migrate enhanced datasets
                for dataset_id, entry in old_registry.get(
                    "enhanced_datasets", {}
                ).items():
                    if self.dry_run:
                        logger.info(f"Would migrate enhanced dataset {dataset_id}")
                    else:
                        self.unified_registry.enhanced_registry.setdefault(
                            "enhanced_datasets", {}
                        )[dataset_id] = entry
                    self.stats["enhanced_migrated"] += 1

                # Migrate semantic samples
                for sample_id, entry in old_registry.get(
                    "semantic_samples", {}
                ).items():
                    if self.dry_run:
                        logger.info(f"Would migrate semantic samples {sample_id}")
                    else:
                        self.unified_registry.enhanced_registry.setdefault(
                            "semantic_samples", {}
                        )[sample_id] = entry
                    self.stats["samples_migrated"] += 1

                # Save if not dry run
                if not self.dry_run and (
                    self.stats["enhanced_migrated"] > 0
                    or self.stats["samples_migrated"] > 0
                ):
                    self.unified_registry._save_json(
                        self.unified_registry.enhanced_registry,
                        self.unified_registry.enhanced_registry_file,
                    )

            except Exception as e:
                logger.error(f"Error migrating enhanced registry: {e}")
                self.stats["errors"].append(f"Enhanced migration error: {e}")

    def _migrate_shade_models(self):
        """Migrate shade's model_registry.json."""
        source_file = self.source_cache_dir / "model_registry.json"
        if not source_file.exists():
            logger.info("No shade model registry found")
            return

        logger.info(f"Migrating models from {source_file}")

        try:
            with open(source_file) as f:
                old_registry = json.load(f)

            # Migrate model entries
            for model_id, entry in old_registry.get("entries", {}).items():
                # Check if checkpoint still exists
                checkpoint_path = Path(
                    entry.get("checkpoint_path", entry.get("file", ""))
                )
                if checkpoint_path and not checkpoint_path.exists():
                    logger.warning(f"Model checkpoint not found: {checkpoint_path}")
                    self.stats["errors"].append(
                        f"Missing checkpoint: {checkpoint_path}"
                    )
                    continue

                if self.dry_run:
                    logger.info(f"Would migrate model {model_id}")
                else:
                    # Ensure proper structure
                    if "checkpoint_path" not in entry and "file" in entry:
                        entry["checkpoint_path"] = entry["file"]

                    self.unified_registry.models_registry["entries"][model_id] = entry

                self.stats["models_migrated"] += 1

            # Migrate best models mapping
            if "best_models" in old_registry:
                if not self.dry_run:
                    self.unified_registry.models_registry["best_models"] = old_registry[
                        "best_models"
                    ]

            # Save if not dry run
            if not self.dry_run and self.stats["models_migrated"] > 0:
                self.unified_registry._save_json(
                    self.unified_registry.models_registry,
                    self.unified_registry.models_registry_file,
                )

        except Exception as e:
            logger.error(f"Error migrating model registry: {e}")
            self.stats["errors"].append(f"Model migration error: {e}")

    def copy_cache_files(self) -> None:
        """Copy actual cache files to new location if different."""
        if self.source_cache_dir == self.target_cache_dir:
            logger.info("Source and target are the same, no files to copy")
            return

        if self.dry_run:
            logger.info(
                f"Would copy cache files from {self.source_cache_dir} to {self.target_cache_dir}"
            )
            return

        logger.info(
            f"Copying cache files from {self.source_cache_dir} to {self.target_cache_dir}"
        )

        # Define subdirectories to copy
        subdirs = ["features", "pca", "models", "enhanced", "semantic_samples"]

        for subdir in subdirs:
            source_dir = self.source_cache_dir / subdir
            if source_dir.exists():
                target_dir = self.target_cache_dir / subdir
                target_dir.mkdir(parents=True, exist_ok=True)

                for file_path in source_dir.glob("*"):
                    if file_path.is_file():
                        target_path = target_dir / file_path.name
                        if not target_path.exists():
                            shutil.copy2(file_path, target_path)
                            logger.debug(f"Copied {file_path} to {target_path}")

    def validate_migration(self) -> dict:
        """Validate the migrated registry.

        Returns:
            Validation results
        """
        if self.dry_run:
            logger.info("Skipping validation in dry-run mode")
            return {"status": "skipped", "reason": "dry_run"}

        logger.info("Validating migrated registry")

        results = {
            "status": "success",
            "issues": [],
        }

        # Check that all registered files exist
        for entry in self.unified_registry.features_registry["entries"].values():
            if not Path(entry["file"]).exists():
                results["issues"].append(f"Missing feature file: {entry['file']}")

        for entry in self.unified_registry.pca_registry["entries"].values():
            if not Path(entry["file"]).exists():
                results["issues"].append(f"Missing PCA file: {entry['file']}")

        for entry in self.unified_registry.models_registry["entries"].values():
            checkpoint_path = entry.get("checkpoint_path", entry.get("file"))
            if checkpoint_path and not Path(checkpoint_path).exists():
                results["issues"].append(f"Missing checkpoint: {checkpoint_path}")

        if results["issues"]:
            results["status"] = "warning"
            logger.warning(f"Validation found {len(results['issues'])} issues")
        else:
            logger.info("Validation successful - all files accounted for")

        return results


def migrate_to_unified_registry(
    source_dir: Path,
    target_dir: Path | None = None,
    dry_run: bool = False,
    validate: bool = True,
) -> dict:
    """Convenience function to migrate registries to UnifiedRegistry.

    Args:
        source_dir: Source cache directory
        target_dir: Target cache directory (defaults to source)
        dry_run: If True, only report what would be done
        validate: Whether to validate after migration

    Returns:
        Migration results including statistics and validation
    """
    migrator = RegistryMigrator(source_dir, target_dir, dry_run)

    # Run migration
    stats = migrator.migrate_all()

    # Copy files if needed
    if target_dir and target_dir != source_dir:
        migrator.copy_cache_files()

    # Validate if requested
    validation = {}
    if validate and not dry_run:
        validation = migrator.validate_migration()

    return {
        "stats": stats,
        "validation": validation,
    }


if __name__ == "__main__":
    # Example usage for testing
    import argparse

    parser = argparse.ArgumentParser(description="Migrate to UnifiedRegistry")
    parser.add_argument("source", type=Path, help="Source cache directory")
    parser.add_argument("--target", type=Path, help="Target cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run migration
    results = migrate_to_unified_registry(
        args.source, args.target, args.dry_run, not args.no_validate
    )

    # Print results
    print(f"\nMigration Results: {json.dumps(results, indent=2)}")
