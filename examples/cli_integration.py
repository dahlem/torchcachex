"""Example of integrating shade-io with SHADE CLI commands.

This shows how to use shade-io components in a SHADE command.
"""

import logging
from pathlib import Path

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.console import Console
from shade_io.feature_sets.filters import RemoveConstantFeaturesFilter
from sklearn.decomposition import PCA

# Import shade-io components
from shade_io import (
    AttentionData,
    FeatureProcessor,
    FileFeatureStore,
    FilteredFeatureSet,
    SimpleFeatureSet,
)

logger = logging.getLogger(__name__)
console = Console()


# Example feature extractor
class SimpleLaplacianExtractor:
    """Simple Laplacian eigenvalue extractor."""

    def __init__(self, num_eigenvalues: int = 50):
        self.num_eigenvalues = num_eigenvalues
        self.name = "laplacian_eigenvalues"

    def extract(
        self, attention_matrices: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """Extract Laplacian eigenvalues from attention matrices."""
        layers, heads, seq_len, _ = attention_matrices.shape
        features = []

        for layer in range(layers):
            for head in range(heads):
                attn = attention_matrices[layer, head]

                # Compute degree matrix
                degree = torch.diag(attn.sum(dim=1))

                # Compute Laplacian
                laplacian = degree - attn

                # Get eigenvalues
                try:
                    eigenvalues = torch.linalg.eigvalsh(laplacian)
                    # Take top k eigenvalues
                    top_k = min(self.num_eigenvalues, len(eigenvalues))
                    features.extend(eigenvalues[:top_k].tolist())
                except Exception:
                    # Fallback if eigendecomposition fails
                    features.extend([0.0] * self.num_eigenvalues)

        features_tensor = torch.tensor(
            features[: layers * heads * self.num_eigenvalues]
        )

        # Generate feature names
        feature_names = [
            f"L{layer}_H{h}_eig{i}"
            for layer in range(layers)
            for h in range(heads)
            for i in range(self.num_eigenvalues)
        ][: len(features_tensor)]

        return features_tensor, feature_names


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def precompute_with_shade_io(cfg: DictConfig) -> None:
    """Example command showing shade-io integration.

    This demonstrates:
    1. Creating feature sets with shade-io
    2. Using filtering decorators
    3. File-based caching
    4. PCA fitting on computed features
    """

    console.print("[bold cyan]SHADE CLI with shade-io Integration[/bold cyan]")

    # Setup paths
    cache_dir = Path(cfg.get("paths", {}).get("cache_dir", ".cache/shade"))
    shade_io_cache = cache_dir / "shade_io_example"
    shade_io_cache.mkdir(parents=True, exist_ok=True)

    # Load model and dataset using SHADE's existing infrastructure
    console.print("\n[cyan]Loading model and dataset...[/cyan]")
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)

    # Get names for caching
    model_name = getattr(model, "model_name", "unknown")
    dataset_name = cfg.dataset._target_.split(".")[-1].replace("Dataset", "")

    console.print(f"  Model: {model_name}")
    console.print(f"  Dataset: {dataset_name}")

    # Create shade-io feature set
    console.print("\n[cyan]Creating shade-io feature set...[/cyan]")

    # Base feature set
    base_feature_set = SimpleFeatureSet(
        name="laplacian_features",
        extractors=[SimpleLaplacianExtractor(num_eigenvalues=30)],
        description="Laplacian eigenvalue features",
    )

    # Add filtering decorator
    filtered_feature_set = FilteredFeatureSet(
        base=base_feature_set,
        filters=[RemoveConstantFeaturesFilter(threshold=1e-10)],
        name="filtered_laplacian",
    )

    # Create file store
    store = FileFeatureStore(cache_dir=shade_io_cache, format="npz", compression=None)

    # Create processor
    processor = FeatureProcessor(
        feature_set=filtered_feature_set,
        store=store,
        batch_size=4,
        device="cpu",
    )

    # Process samples
    console.print("\n[cyan]Computing features with shade-io...[/cyan]")

    max_samples = min(20, len(dataset))  # Limit for example
    samples = list(dataset)[:max_samples]

    all_features = []
    all_labels = []

    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", sample.get("question", ""))
        response = sample.get("response", sample.get("answer", ""))

        if not prompt or not response:
            continue

        try:
            # Get attention matrices from model
            attention_matrices = model.get_attention_matrices(prompt, response)

            # Create AttentionData for shade-io
            attention_data = AttentionData(
                attention_matrices=attention_matrices,
                model_name=model_name,
                dataset_name=dataset_name,
                metadata={"sample_idx": i},
            )

            # Process with shade-io
            result = processor.process_single(attention_data)

            all_features.append(result.features.numpy())
            all_labels.append(sample.get("label", 0))

            if i == 0:
                console.print(f"  Feature dimension: {result.feature_dim}")
                console.print(f"  Sample features: {result.features[:5].tolist()}")

        except Exception as e:
            logger.warning(f"Skipped sample {i}: {e}")
            continue

    if not all_features:
        console.print("[red]No features extracted![/red]")
        return

    # Stack features
    X = np.vstack(all_features)
    y = np.array(all_labels)

    console.print("\n[green]✓ Computed features:[/green]")
    console.print(f"  Shape: {X.shape}")
    console.print(f"  Labels: {np.unique(y, return_counts=True)}")

    # Fit PCA
    console.print("\n[cyan]Fitting PCA on shade-io features...[/cyan]")

    n_components = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)

    pca.fit_transform(X)

    console.print(f"  Components: {n_components}")
    console.print(f"  Explained variance: {pca.explained_variance_ratio_[:3].tolist()}")
    console.print(f"  Total variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Save PCA model
    pca_path = shade_io_cache / f"pca_{model_name}_{dataset_name}.pkl"
    joblib.dump(pca, pca_path)
    console.print(f"\n[green]✓ PCA model saved to:[/green] {pca_path}")

    # Demonstrate cache usage
    console.print("\n[cyan]Testing cache...[/cyan]")

    # Process first sample again - should load from cache
    if samples:
        attention_matrices = model.get_attention_matrices(
            samples[0]["prompt"], samples[0]["response"]
        )
        attention_data = AttentionData(
            attention_matrices=attention_matrices,
            model_name=model_name,
            dataset_name=dataset_name,
            metadata={"sample_idx": 0},
        )

        # This should be loaded from cache
        cached_result = processor.process_single(attention_data)
        console.print(
            f"  Loaded from cache: {torch.allclose(cached_result.features, torch.from_numpy(all_features[0]))}"
        )

    console.print("\n[green]✅ shade-io integration example complete![/green]")
    console.print("\nThis example demonstrated:")
    console.print("  • Creating feature sets with shade-io")
    console.print("  • Using filtering decorators")
    console.print("  • File-based caching")
    console.print("  • Integration with existing SHADE models/datasets")


if __name__ == "__main__":
    precompute_with_shade_io()
