"""File-based feature storage implementation."""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from shade_io.interfaces.core import FeatureKey, FeatureResult, IFeatureStore

logger = logging.getLogger(__name__)


class FileFeatureStore(IFeatureStore):
    """File-based feature store for persistent storage.

    Supports multiple formats:
    - pt: PyTorch native format (no compression)
    - npz: NumPy format (optional zlib compression)
    - pkl: Python pickle format (no compression)
    - arrow: Apache Arrow/Parquet format (multiple compression options)
    """

    # Valid compression options per format
    ARROW_COMPRESSION_TYPES = {"snappy", "gzip", "brotli", "lz4", "zstd", None}

    def __init__(
        self,
        cache_dir: Path,
        format: str = "npz",
        compression: str | None = None,
    ):
        """Initialize file store.

        Args:
            cache_dir: Directory for cache files
            format: File format (pt, npz, pkl, arrow)
            compression: Compression setting:
                - For npz: Any non-null value enables zlib compression
                - For arrow: snappy, gzip, brotli, lz4, zstd, or null
                - For pt/pkl: Ignored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

        # Validate compression settings
        if format == "arrow" and compression not in self.ARROW_COMPRESSION_TYPES:
            logger.warning(
                f"Invalid compression '{compression}' for arrow format. "
                f"Valid options: {self.ARROW_COMPRESSION_TYPES}. Using default."
            )
            compression = None
        elif format in ("pt", "pkl") and compression is not None:
            logger.info(f"Compression not supported for {format} format, ignoring.")
            compression = None

        self.compression = compression

    def _get_path(self, key: FeatureKey) -> Path:
        """Get file path for a feature key.

        Args:
            key: Feature key

        Returns:
            Path to file
        """
        filename = f"{key.to_string()}.{self.format}"
        # Only add compression extension for formats that support it
        if self.compression and self.format in ("npz", "arrow"):
            if self.format == "npz":
                # NPZ uses generic 'compressed' suffix
                filename += ".compressed"
            elif self.format == "arrow" and self.compression:
                # Arrow uses specific compression type
                filename += f".{self.compression}"
        return self.cache_dir / filename

    def _detect_format_from_path(self, path: Path) -> str:
        """Detect file format from the path.
        
        Args:
            path: File path to analyze
            
        Returns:
            Detected format string
        """
        path_str = str(path).lower()

        # Handle compressed Arrow files
        if ".arrow." in path_str or path_str.endswith(".arrow"):
            return "arrow"
        elif path_str.endswith(".npz"):
            return "npz"
        elif path_str.endswith(".pt"):
            return "pt"
        elif path_str.endswith(".pkl"):
            return "pkl"
        else:
            # Fall back to the configured format
            return self.format

    def load(self, key: FeatureKey) -> FeatureResult | None:
        """Load features from file.

        Args:
            key: Key identifying features

        Returns:
            Features if found, None otherwise
        """
        path = self._get_path(key)

        if not path.exists():
            return None

        # Detect format from actual file path if needed
        detected_format = self._detect_format_from_path(path)

        try:
            if detected_format == "pt":
                data = torch.load(path, map_location="cpu")
                return FeatureResult(
                    features=data["features"],
                    feature_names=data["feature_names"],
                    metadata=data.get("metadata", {}),
                    labels=data.get("labels"),
                    splits=data.get("splits"),
                    sample_hashes=data.get("sample_hashes"),
                    original_indices=data.get("original_indices"),
                )

            elif detected_format == "npz":
                with np.load(path, allow_pickle=True) as data:
                    return FeatureResult(
                        features=torch.from_numpy(data["features"]),
                        feature_names=data["feature_names"].tolist(),
                        metadata=data.get("metadata", {}).item()
                        if "metadata" in data
                        else {},
                        labels=torch.from_numpy(data["labels"])
                        if "labels" in data
                        else None,
                        splits=data.get("splits", None).tolist()
                        if "splits" in data
                        else None,
                        sample_hashes=data.get("sample_hashes", None).tolist()
                        if "sample_hashes" in data
                        else None,
                        original_indices=data.get("original_indices", None).tolist()
                        if "original_indices" in data
                        else None,
                    )

            elif detected_format == "pkl":
                with open(path, "rb") as f:
                    data = pickle.load(f)
                return FeatureResult(**data)

            elif detected_format == "arrow":
                # Optional Arrow support
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(path)

                    # Check for special columns and extract them separately
                    splits = None
                    sample_hashes = None
                    original_indices = None
                    labels = None

                    special_columns = [
                        "split",
                        "sample_hash",
                        "original_index",
                        "label",
                    ]

                    if "split" in table.column_names:
                        splits = table["split"].to_pylist()

                    if "sample_hash" in table.column_names:
                        sample_hashes = table["sample_hash"].to_pylist()

                    if "original_index" in table.column_names:
                        original_indices = table["original_index"].to_pylist()

                    if "label" in table.column_names:
                        labels = torch.tensor(table["label"].to_pylist())

                    # Remove special columns from feature columns
                    feature_columns = [
                        col for col in table.column_names if col not in special_columns
                    ]
                    table = table.select(feature_columns)

                    # Convert Arrow table to FeatureResult
                    # Check if we have multiple feature columns or a single "features" column
                    if "features" in table.column_names:
                        # Single column with all features
                        features = torch.from_numpy(table["features"].to_numpy())
                    else:
                        # Multiple columns, one per feature - reconstruct 2D array
                        arrays = [table[col].to_numpy() for col in table.column_names]
                        features = torch.from_numpy(np.column_stack(arrays))

                    # Get metadata
                    if table.schema.metadata:
                        feature_names = json.loads(
                            table.schema.metadata.get(b"feature_names", b"[]")
                        )
                        metadata = json.loads(
                            table.schema.metadata.get(b"metadata", b"{}")
                        )
                    else:
                        # If no metadata, use column names as feature names
                        feature_names = table.column_names
                        metadata = {}

                    # If labels/splits not found in columns, try to get them from metadata
                    if labels is None and "labels" in metadata:
                        labels = torch.tensor(metadata["labels"])

                    if splits is None and "splits" in metadata:
                        splits = metadata["splits"]

                    return FeatureResult(
                        features=features,
                        feature_names=feature_names,
                        metadata=metadata,
                        labels=labels,
                        splits=splits,
                        sample_hashes=sample_hashes,
                        original_indices=original_indices,
                    )
                except ImportError:
                    logger.warning("Arrow format requested but pyarrow not available")
                    return None

            else:
                logger.error(f"Unsupported format: {detected_format}")
                return None

        except Exception as e:
            logger.error(f"Failed to load features from {path}: {e}")
            return None

    def save(self, key: FeatureKey, features: FeatureResult) -> None:
        """Save features to file.

        Args:
            key: Key identifying features
            features: Features to save
        """
        path = self._get_path(key)

        try:
            if self.format == "pt":
                torch.save(
                    {
                        "features": features.features,
                        "feature_names": features.feature_names,
                        "metadata": features.metadata,
                        "labels": features.labels,
                        "splits": features.splits,
                        "sample_hashes": features.sample_hashes,
                        "original_indices": features.original_indices,
                    },
                    path,
                )

            elif self.format == "npz":
                # Handle device transfer for MPS/CUDA tensors
                feature_array = (
                    features.features.cpu().numpy()
                    if features.features.is_cuda or features.features.is_mps
                    else features.features.numpy()
                )
                save_dict = {
                    "features": feature_array,
                    "feature_names": np.array(features.feature_names),
                    "metadata": features.metadata,
                }
                if features.labels is not None:
                    labels_array = (
                        features.labels.cpu().numpy()
                        if features.labels.is_cuda or features.labels.is_mps
                        else features.labels.numpy()
                    )
                    save_dict["labels"] = labels_array
                if features.splits is not None:
                    save_dict["splits"] = np.array(features.splits)
                if features.sample_hashes is not None:
                    save_dict["sample_hashes"] = np.array(features.sample_hashes)
                if features.original_indices is not None:
                    save_dict["original_indices"] = np.array(features.original_indices)

                # NPZ compression is binary - any non-null value means compressed
                if self.compression:
                    np.savez_compressed(path, **save_dict)  # Uses zlib internally
                else:
                    np.savez(path, **save_dict)

            elif self.format == "pkl":
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "features": features.features,
                            "feature_names": features.feature_names,
                            "metadata": features.metadata,
                            "labels": features.labels,
                            "splits": features.splits,
                            "sample_hashes": features.sample_hashes,
                            "original_indices": features.original_indices,
                        },
                        f,
                    )

            elif self.format == "arrow":
                # Optional Arrow support
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    # Create Arrow table - handle device transfer for MPS/CUDA tensors
                    feature_array = (
                        features.features.cpu().numpy()
                        if features.features.is_cuda or features.features.is_mps
                        else features.features.numpy()
                    )

                    # Handle 2D arrays by creating one column per feature
                    if feature_array.ndim == 2:
                        # Each row is a sample, each column is a feature
                        arrays = [
                            pa.array(feature_array[:, i])
                            for i in range(feature_array.shape[1])
                        ]
                        # Use feature names if available, otherwise generic names
                        if (
                            features.feature_names
                            and len(features.feature_names) == feature_array.shape[1]
                        ):
                            names = features.feature_names.copy()
                        else:
                            names = [
                                f"feature_{i}" for i in range(feature_array.shape[1])
                            ]

                        # Add splits column if available
                        if features.splits is not None:
                            arrays.append(pa.array(features.splits))
                            names.append("split")

                        # Add sample_hashes column if available
                        if features.sample_hashes is not None:
                            arrays.append(pa.array(features.sample_hashes))
                            names.append("sample_hash")

                        # Add original_indices column if available
                        if features.original_indices is not None:
                            arrays.append(pa.array(features.original_indices))
                            names.append("original_index")

                        # Add labels column if available
                        if features.labels is not None:
                            labels_array = (
                                features.labels.cpu().numpy()
                                if features.labels.is_cuda or features.labels.is_mps
                                else features.labels.numpy()
                            )
                            arrays.append(pa.array(labels_array))
                            names.append("label")

                        table = pa.Table.from_arrays(arrays, names=names)
                    else:
                        # 1D array - single feature
                        arrays = [pa.array(feature_array)]
                        names = ["features"]

                        # Add splits column if available
                        if features.splits is not None:
                            arrays.append(pa.array(features.splits))
                            names.append("split")

                        # Add sample_hashes column if available
                        if features.sample_hashes is not None:
                            arrays.append(pa.array(features.sample_hashes))
                            names.append("sample_hash")

                        # Add original_indices column if available
                        if features.original_indices is not None:
                            arrays.append(pa.array(features.original_indices))
                            names.append("original_index")

                        # Add labels column if available
                        if features.labels is not None:
                            labels_array = (
                                features.labels.cpu().numpy()
                                if features.labels.is_cuda or features.labels.is_mps
                                else features.labels.numpy()
                            )
                            arrays.append(pa.array(labels_array))
                            names.append("label")

                        table = pa.Table.from_arrays(arrays, names=names)

                    # Add metadata
                    metadata = {
                        b"feature_names": json.dumps(features.feature_names).encode(),
                        b"metadata": json.dumps(features.metadata).encode(),
                    }
                    table = table.replace_schema_metadata(metadata)

                    # Write to file with validated compression
                    # Parquet defaults to snappy if None is passed
                    pq.write_table(
                        table, path, compression=self.compression or "snappy"
                    )

                except ImportError:
                    logger.error("Arrow format requested but pyarrow not available")
                    raise

            else:
                raise ValueError(f"Unsupported format: {self.format}")

            logger.debug(f"Saved features to {path}")

        except Exception as e:
            logger.error(f"Failed to save features to {path}: {e}")
            raise

    def exists(self, key: FeatureKey) -> bool:
        """Check if features exist.

        Args:
            key: Key to check

        Returns:
            True if exists, False otherwise
        """
        return self._get_path(key).exists()

    def delete(self, key: FeatureKey) -> bool:
        """Delete features from file system.

        Args:
            key: Key identifying features

        Returns:
            True if deleted, False if not found
        """
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self, pattern: str | None = None) -> list[FeatureKey]:
        """List available feature keys.

        Args:
            pattern: Optional pattern to filter keys

        Returns:
            List of available keys
        """
        keys = []

        # Get all files matching format
        glob_pattern = f"*.{self.format}"
        if self.compression and self.format in ("npz", "arrow"):
            if self.format == "npz":
                glob_pattern += ".compressed"
            elif self.format == "arrow":
                glob_pattern += f".{self.compression}"

        for path in self.cache_dir.glob(glob_pattern):
            # Extract key from filename
            filename = path.stem
            if self.compression:
                filename = Path(filename).stem

            try:
                key = FeatureKey.from_string(filename)
                if pattern is None or pattern in filename:
                    keys.append(key)
            except ValueError:
                # Skip invalid filenames
                continue

        return keys

    def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dict with storage stats
        """
        total_size = 0
        n_files = 0

        for path in self.cache_dir.glob(f"*.{self.format}*"):
            total_size += path.stat().st_size
            n_files += 1

        return {
            "n_files": n_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "format": self.format,
            "compression": self.compression,
        }

    def create_streaming_writer(
        self,
        key: FeatureKey,
        feature_names: list[str],
        buffer_size: int = 1000,
        enable_async: bool = True,
    ) -> "StreamingArrowWriter":
        """Create streaming writer for incremental Arrow writes.

        Args:
            key: Feature key identifying the file
            feature_names: Names of features to write
            buffer_size: Number of samples to buffer before writing
            enable_async: Whether to enable async writes

        Returns:
            StreamingArrowWriter instance

        Raises:
            ValueError: If format is not 'arrow'
            ImportError: If pyarrow is not available
        """
        if self.format != "arrow":
            raise ValueError(
                f"Streaming writer only supports arrow format, got {self.format}"
            )

        # Import here to avoid dependency issues
        from .streaming_arrow import StreamingArrowWriter

        path = self._get_path(key)

        return StreamingArrowWriter(
            path=path,
            feature_names=feature_names,
            compression=self.compression,
            buffer_size=buffer_size,
            enable_async=enable_async,
        )

    def supports_streaming(self) -> bool:
        """Check if this store supports streaming writes.

        Returns:
            True if streaming is supported
        """
        return self.format == "arrow"
