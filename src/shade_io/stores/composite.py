"""Composite store for multi-level caching."""

import logging

from shade_io.interfaces.core import FeatureKey, FeatureResult, IFeatureStore

logger = logging.getLogger(__name__)


class CompositeStore(IFeatureStore):
    """Multi-level cache with multiple stores.

    This implements a cache hierarchy where faster stores
    are checked first, and data is promoted/demoted between levels.
    """

    def __init__(self, stores: list[IFeatureStore], write_through: bool = True):
        """Initialize composite store.

        Args:
            stores: List of stores in priority order (L1, L2, ...)
            write_through: If True, write to all levels
        """
        self.stores = stores
        self.write_through = write_through

    def load(self, key: FeatureKey) -> FeatureResult | None:
        """Load features from first available store.

        Args:
            key: Key identifying features

        Returns:
            Features if found, None otherwise
        """
        for i, store in enumerate(self.stores):
            result = store.load(key)
            if result is not None:
                # Promote to higher cache levels
                if i > 0:
                    for j in range(i):
                        try:
                            self.stores[j].save(key, result)
                        except Exception as e:
                            logger.warning(f"Failed to promote to L{j} cache: {e}")

                logger.debug(f"Loaded from L{i} cache: {key.to_string()}")
                return result

        return None

    def save(self, key: FeatureKey, features: FeatureResult) -> None:
        """Save features to stores.

        Args:
            key: Key identifying features
            features: Features to save
        """
        if self.write_through:
            # Write to all levels
            for i, store in enumerate(self.stores):
                try:
                    store.save(key, features)
                except Exception as e:
                    logger.warning(f"Failed to save to L{i} cache: {e}")
        else:
            # Write only to first level
            self.stores[0].save(key, features)

    def exists(self, key: FeatureKey) -> bool:
        """Check if features exist in any store.

        Args:
            key: Key to check

        Returns:
            True if exists in any store, False otherwise
        """
        return any(store.exists(key) for store in self.stores)

    def delete(self, key: FeatureKey) -> bool:
        """Delete features from all stores.

        Args:
            key: Key identifying features

        Returns:
            True if deleted from any store
        """
        deleted = False
        for store in self.stores:
            if store.delete(key):
                deleted = True
        return deleted

    def list_keys(self, pattern: str | None = None) -> list[FeatureKey]:
        """List unique keys across all stores.

        Args:
            pattern: Optional pattern to filter keys

        Returns:
            List of unique keys
        """
        all_keys = set()
        for store in self.stores:
            keys = store.list_keys(pattern)
            all_keys.update(key.to_string() for key in keys)

        # Convert back to FeatureKey objects
        return [FeatureKey.from_string(key_str) for key_str in all_keys]

    def get_stats(self) -> dict:
        """Get statistics from all stores.

        Returns:
            Combined statistics
        """
        stats = {
            "n_stores": len(self.stores),
            "write_through": self.write_through,
            "store_stats": [],
        }

        for i, store in enumerate(self.stores):
            store_stats = {"level": f"L{i}"}
            if hasattr(store, "get_stats"):
                store_stats.update(store.get_stats())
            else:
                store_stats["type"] = type(store).__name__
            stats["store_stats"].append(store_stats)

        return stats
