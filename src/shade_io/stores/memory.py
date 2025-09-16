"""In-memory feature storage implementation."""

import logging

from shade_io.interfaces.core import FeatureKey, FeatureResult, IFeatureStore

logger = logging.getLogger(__name__)


class MemoryFeatureStore(IFeatureStore):
    """In-memory feature store for testing and caching.

    This is useful for testing and as an L1 cache in front
    of a slower persistent store.
    """

    def __init__(self, max_size_mb: float = 100):
        """Initialize memory store.

        Args:
            max_size_mb: Maximum size in megabytes
        """
        self.cache: dict[str, FeatureResult] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0

    def load(self, key: FeatureKey) -> FeatureResult | None:
        """Load features from memory.

        Args:
            key: Key identifying features

        Returns:
            Features if found, None otherwise
        """
        key_str = key.to_string()
        return self.cache.get(key_str)

    def save(self, key: FeatureKey, features: FeatureResult) -> None:
        """Save features to memory.

        Args:
            key: Key identifying features
            features: Features to save
        """
        key_str = key.to_string()

        # Estimate size (rough approximation)
        size_bytes = (
            features.features.element_size() * features.features.nelement()
            + len(str(features.feature_names)) * 2  # Unicode chars
            + len(str(features.metadata)) * 2
        )

        # Check if we need to evict
        if self.current_size_bytes + size_bytes > self.max_size_bytes:
            self._evict_lru()

        self.cache[key_str] = features
        self.current_size_bytes += size_bytes

    def exists(self, key: FeatureKey) -> bool:
        """Check if features exist.

        Args:
            key: Key to check

        Returns:
            True if exists, False otherwise
        """
        return key.to_string() in self.cache

    def delete(self, key: FeatureKey) -> bool:
        """Delete features from memory.

        Args:
            key: Key identifying features

        Returns:
            True if deleted, False if not found
        """
        key_str = key.to_string()
        if key_str in self.cache:
            del self.cache[key_str]
            # Update size (rough approximation)
            self.current_size_bytes *= 0.9  # Approximate
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict least recently used items.

        Simple implementation: remove first 25% of items.
        """
        if not self.cache:
            return

        n_to_evict = max(1, len(self.cache) // 4)
        for _ in range(n_to_evict):
            # Remove first item (dict maintains insertion order in Python 3.7+)
            key = next(iter(self.cache))
            del self.cache[key]

        # Update size estimate
        self.current_size_bytes *= 0.75

        logger.info(f"Evicted {n_to_evict} items from memory cache")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return {
            "n_items": len(self.cache),
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes,
        }
