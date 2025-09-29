#!/usr/bin/env python3
"""Simple test to verify shade-io imports work correctly."""


def test_imports():
    """Test that all main components can be imported."""
    try:
        # Test interfaces
        from shade_io.interfaces.core import (  # noqa: F401
            AttentionData,
            FeatureResult,
            IFeatureSet,
            IFeatureStore,
        )

        print("✓ Interfaces imported successfully")

        # Test feature sets
        from shade_io.feature_sets.base import SimpleFeatureSet  # noqa: F401
        from shade_io.feature_sets.decorators import FilteredFeatureSet  # noqa: F401

        print("✓ Feature sets imported successfully")

        # Test stores
        from shade_io.stores.file import FileFeatureStore  # noqa: F401
        from shade_io.stores.memory import MemoryFeatureStore  # noqa: F401

        print("✓ Stores imported successfully")

        # Test processor
        from shade_io.processor.feature_processor import FeatureProcessor  # noqa: F401

        print("✓ Processor imported successfully")

        # Test adapters
        from shade_io.adapters.v1_adapter import V1FeatureSetAdapter  # noqa: F401

        print("✓ Adapters imported successfully")

        print("\n✅ All imports successful! shade-io is ready to use.")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
