"""Feature processor that orchestrates computation and storage.

This class ties together feature sets and stores, providing
a high-level interface for feature computation with caching.
"""

import logging
from typing import Any

import torch
from tqdm import tqdm

from shade_io.interfaces.core import (
    AttentionData,
    FeatureKey,
    FeatureResult,
    IFeatureSet,
    IFeatureStore,
)

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Orchestrates feature computation and caching.

    This class provides a high-level interface for computing features,
    handling caching, batching, and other cross-cutting concerns.
    """

    def __init__(
        self,
        feature_set: IFeatureSet,
        store: IFeatureStore | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        force_recompute: bool = False,
        show_progress: bool = True,
        max_samples_in_memory: int | None = None,
        enable_streaming: bool = True,
    ):
        """Initialize feature processor.

        Args:
            feature_set: Feature set to compute
            store: Optional storage backend
            batch_size: Batch size for processing
            device: Device for computation
            force_recompute: If True, always recompute
            show_progress: If True, show progress bar
            max_samples_in_memory: If set, process in chunks to limit memory usage
            enable_streaming: If True, use streaming writes when possible
        """
        self.feature_set = feature_set
        self.store = store
        self.batch_size = batch_size
        self.device = device
        self.force_recompute = force_recompute
        self.show_progress = show_progress
        self.max_samples_in_memory = max_samples_in_memory
        self.enable_streaming = enable_streaming

    def get_features(
        self,
        model: Any,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Get features for a model and dataset.

        This is the main entry point, handling caching and computation.

        Args:
            model: Model to extract attention from
            dataset: Dataset to process
            split: Optional split (train/val/test)
            max_samples: Optional limit on samples

        Returns:
            Computed or cached features
        """
        # Build cache key
        model_name = self._get_model_name(model)
        dataset_name = self._get_dataset_name(dataset)

        key = FeatureKey(
            feature_set_name=self.feature_set.name,
            model_name=model_name,
            dataset_name=dataset_name,
            metadata={
                "split": split,
                "max_samples": max_samples,
            },
        )

        # Check cache first
        if self.store and not self.force_recompute:
            cached = self.store.load(key)
            if cached:
                logger.info(f"Loaded cached features: {key.to_string()}")
                return cached

        # Compute features
        logger.info(f"Computing features: {key.to_string()}")
        features = self.compute_features(model, dataset, split, max_samples)

        # Save to cache
        if self.store:
            self.store.save(key, features)
            logger.info(f"Saved features to cache: {key.to_string()}")

        return features

    def compute_features(
        self,
        model: Any,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Compute features from scratch.

        Args:
            model: Model to extract attention from
            dataset: Dataset to process
            split: Optional split to use
            max_samples: Optional limit on samples

        Returns:
            Computed features
        """
        # Get samples from dataset
        samples = self._get_samples(dataset, split, max_samples)

        # Check if we should use streaming (preferred for large datasets)
        if (
            self.enable_streaming
            and self.store
            and hasattr(self.store, "supports_streaming")
            and self.store.supports_streaming()
            and len(samples) > 1000
        ):  # Use streaming for datasets > 1000 samples
            logger.info(f"Using streaming Arrow processing for {len(samples)} samples")
            return self._compute_features_streaming(model, samples, split, max_samples)

        # Check if we should use chunked processing for very large datasets
        if self.max_samples_in_memory and len(samples) > self.max_samples_in_memory:
            logger.info(
                f"Using chunked processing: {len(samples)} samples in chunks of {self.max_samples_in_memory}"
            )
            return self._compute_features_chunked(model, samples, split)

        # Process in batches with memory optimization
        all_features = []
        all_labels = []
        all_splits = []

        iterator = range(0, len(samples), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Processing batches")

        # Import garbage collection for memory optimization
        import gc

        for batch_idx, i in enumerate(iterator):
            batch = samples[i : i + self.batch_size]

            # Determine which extraction method to use based on feature set configuration
            if (
                hasattr(self.feature_set, "output_processor")
                and self.feature_set.output_processor is not None
            ):
                # New path: extract all model outputs for OutputProcessor
                model_outputs = self._extract_model_outputs_batch(model, batch)
                model_outputs["model_name"] = self._get_model_name(model)
                model_outputs["dataset_name"] = self._get_dataset_name(dataset)
                model_outputs["architecture"] = self._get_model_architecture_type(model)

                # Compute features using new interface
                batch_result = self.feature_set.compute_features(model_outputs)

                # Clear intermediate variables to free memory
                del model_outputs
            else:
                # Legacy path: extract attention for attention processors
                attention_matrices, attention_type = self._extract_attention_batch(model, batch)

                # Create AttentionData
                attention_data = AttentionData(
                    attention_matrices=attention_matrices,
                    model_name=self._get_model_name(model),
                    dataset_name=self._get_dataset_name(dataset),
                    architecture=self._get_model_architecture_type(model),
                    attention_type=attention_type,
                    metadata={
                        "batch_size": len(batch),
                        "batch_idx": i // self.batch_size,
                    },
                )

                # Compute features using legacy interface
                batch_result = self.feature_set.compute_features(attention_data)

                # Clear intermediate variables to free memory
                del attention_matrices, attention_data

            # Collect results - move to CPU to save GPU memory
            batch_features = batch_result.features
            if batch_features.is_cuda or batch_features.device.type == "mps":
                batch_features = batch_features.cpu()
            all_features.append(batch_features)

            # Collect labels if available
            if hasattr(batch[0], "label"):
                batch_labels = torch.tensor([s.label for s in batch])
                all_labels.append(batch_labels)

            # Collect splits if available
            if hasattr(batch[0], "split"):
                batch_splits = [s.split for s in batch]
                all_splits.extend(batch_splits)

            # Clear batch result to free memory
            del batch_result

            # Force garbage collection every 10 batches to prevent memory buildup
            if (batch_idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Combine all batches
        if all_features:
            combined_features = torch.cat(all_features, dim=0)
            # Clear the list to free memory before returning
            del all_features
        else:
            combined_features = torch.empty(0, self.feature_set.feature_dim)

        if all_labels:
            combined_labels = torch.cat(all_labels, dim=0)
            # Clear the list to free memory
            del all_labels
        else:
            combined_labels = None

        if not all_splits:
            all_splits = None

        # Get feature names from metadata
        metadata = self.feature_set.get_metadata()

        return FeatureResult(
            features=combined_features,
            feature_names=metadata.feature_names,
            metadata={
                "feature_set": self.feature_set.name,
                "model": self._get_model_name(model),
                "dataset": self._get_dataset_name(dataset),
                "n_samples": len(combined_features),
                "split": split,
            },
            labels=combined_labels,
            splits=all_splits if all_splits else None,
        )

    def _extract_attention_batch(self, model: Any, batch: list) -> tuple[torch.Tensor, str | None]:
        """Extract attention matrices for a batch.

        Args:
            model: Model to use
            batch: List of samples

        Returns:
            Tuple of (attention tensor, attention_type)
        """
        # Check if feature set has attention processor with attention type
        attention_type = None
        if hasattr(self.feature_set, "attention_processor") and self.feature_set.attention_processor:
            if hasattr(self.feature_set.attention_processor, "attention_type"):
                attention_type = self.feature_set.attention_processor.attention_type

        # Extract prompts and responses
        prompts = [s.prompt for s in batch if hasattr(s, "prompt")]
        responses = [s.response for s in batch if hasattr(s, "response")]

        if not prompts or not responses:
            raise ValueError("Batch samples missing prompt or response")

        # Use attention type specific method if available
        if attention_type and hasattr(model, "get_attention_by_type"):
            logger.debug(f"Extracting {attention_type} attention from model")
            # Individual processing with attention type
            attention_list = []
            for prompt, response in zip(prompts, responses, strict=False):
                attention = model.get_attention_by_type(prompt, response, attention_type)
                if attention is not None:
                    attention_list.append(attention)

            if attention_list:
                # Check if all have same shape
                shapes = [a.shape for a in attention_list]
                if len(set(shapes)) == 1:
                    return torch.stack(attention_list), attention_type
                else:
                    # Pad to common shape instead of using first only
                    logger.debug(
                        f"Batch contains different sequence lengths: {shapes}, padding to common size"
                    )
                    padded_attention_list, attention_masks = self._pad_attention_matrices(attention_list)
                    return torch.stack(padded_attention_list), attention_type

        # Fallback to default attention extraction
        if hasattr(model, "get_attention_matrices_batch"):
            # Model supports batch processing
            if prompts and responses:
                attention_matrices = model.get_attention_matrices_batch(prompts, responses)
                return attention_matrices, attention_type

        # Fallback to individual processing
        attention_list = []
        for sample in batch:
            if hasattr(model, "get_attention_matrices"):
                if hasattr(sample, "prompt") and hasattr(sample, "response"):
                    attention = model.get_attention_matrices(
                        sample.prompt, sample.response
                    )
                    attention_list.append(attention)

        if attention_list:
            # Check if all have same shape
            shapes = [a.shape for a in attention_list]
            if len(set(shapes)) == 1:
                # Same shape, can stack
                return torch.stack(attention_list), attention_type
            else:
                # Pad to common shape instead of using first only
                logger.debug(
                    f"Batch contains different sequence lengths: {shapes}, padding to common size"
                )
                padded_attention_list, attention_masks = self._pad_attention_matrices(attention_list)
                return torch.stack(padded_attention_list), attention_type

        raise ValueError("Could not extract attention from batch")

    def _pad_attention_matrices(
        self, attention_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Pad attention matrices to common size within batch.

        Args:
            attention_list: List of attention tensors with potentially different shapes

        Returns:
            Tuple of (padded_attention_list, attention_masks)
            - padded_attention_list: All tensors padded to max dimensions
            - attention_masks: Boolean masks indicating valid positions
        """
        if not attention_list:
            return [], []

        # Determine the structure of attention tensors
        # They could be 3D (layers, heads, seq_len, seq_len) or 4D (layers, heads, seq1_len, seq2_len)
        first_tensor = attention_list[0]

        if first_tensor.dim() == 3:
            # Format: (layers, seq_len, seq_len) - square attention
            max_seq_len = max(tensor.shape[-1] for tensor in attention_list)

            padded_list = []
            mask_list = []

            for tensor in attention_list:
                layers, seq_len, _ = tensor.shape

                # Create padding
                pad_size = max_seq_len - seq_len
                if pad_size > 0:
                    # Pad last two dimensions: (layers, seq_len, seq_len) -> (layers, max_seq_len, max_seq_len)
                    padded = torch.nn.functional.pad(
                        tensor, (0, pad_size, 0, pad_size), mode='constant', value=0.0
                    )
                else:
                    padded = tensor

                # Create attention mask: True for valid positions, False for padding
                mask = torch.ones(layers, max_seq_len, max_seq_len, dtype=torch.bool, device=tensor.device)
                if pad_size > 0:
                    mask[:, seq_len:, :] = False  # Padded rows
                    mask[:, :, seq_len:] = False  # Padded columns

                padded_list.append(padded)
                mask_list.append(mask)

        elif first_tensor.dim() == 4:
            # Format: (layers, heads, seq1_len, seq2_len) - rectangular cross-attention
            max_seq1_len = max(tensor.shape[-2] for tensor in attention_list)
            max_seq2_len = max(tensor.shape[-1] for tensor in attention_list)

            padded_list = []
            mask_list = []

            for tensor in attention_list:
                layers, heads, seq1_len, seq2_len = tensor.shape

                # Calculate padding for both dimensions
                pad_seq1 = max_seq1_len - seq1_len
                pad_seq2 = max_seq2_len - seq2_len

                if pad_seq1 > 0 or pad_seq2 > 0:
                    # Pad last two dimensions: (layers, heads, seq1, seq2) -> (layers, heads, max_seq1, max_seq2)
                    padded = torch.nn.functional.pad(
                        tensor, (0, pad_seq2, 0, pad_seq1), mode='constant', value=0.0
                    )
                else:
                    padded = tensor

                # Create attention mask: True for valid positions, False for padding
                mask = torch.ones(layers, heads, max_seq1_len, max_seq2_len, dtype=torch.bool, device=tensor.device)
                if pad_seq1 > 0:
                    mask[:, :, seq1_len:, :] = False  # Padded rows (decoder positions)
                if pad_seq2 > 0:
                    mask[:, :, :, seq2_len:] = False  # Padded columns (encoder positions)

                padded_list.append(padded)
                mask_list.append(mask)

        else:
            raise ValueError(f"Unsupported attention tensor dimension: {first_tensor.dim()}D")

        return padded_list, mask_list

    def _get_samples(
        self,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> list:
        """Get samples from dataset.

        Args:
            dataset: Dataset to sample from
            split: Optional split to use
            max_samples: Optional limit

        Returns:
            List of samples
        """
        # Handle different dataset types
        if hasattr(dataset, "get_split"):
            samples = dataset.get_split(split) if split else list(dataset)
        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            samples = [dataset[i] for i in range(len(dataset))]
        elif hasattr(dataset, "__iter__"):
            samples = list(dataset)
        else:
            raise ValueError(f"Don't know how to get samples from {type(dataset)}")

        # Apply limit
        if max_samples:
            samples = samples[:max_samples]

        return samples

    def _get_model_name(self, model: Any) -> str:
        """Extract model name.

        Args:
            model: Model instance

        Returns:
            Model name string
        """
        if hasattr(model, "model_name"):
            return model.model_name
        elif hasattr(model, "name"):
            return model.name
        elif hasattr(model, "__class__"):
            return model.__class__.__name__
        else:
            return "unknown"

    def _get_dataset_name(self, dataset: Any) -> str:
        """Extract dataset name.

        Args:
            dataset: Dataset instance

        Returns:
            Dataset name string
        """
        if hasattr(dataset, "dataset_name"):
            return dataset.dataset_name
        elif hasattr(dataset, "name"):
            return dataset.name
        elif hasattr(dataset, "__class__"):
            return dataset.__class__.__name__
        else:
            return "unknown"

    def _get_model_architecture_type(self, model: Any) -> str:
        """Determine model architecture type (encoder/decoder/encoder_decoder).

        Args:
            model: Model instance

        Returns:
            Architecture type string
        """
        # Check if model has architecture info
        if hasattr(model, "get_architecture"):
            arch = model.get_architecture()
            if arch and hasattr(arch, "model_type"):
                model_type = arch.model_type.lower()
                # Map model types to architecture types
                if model_type in ["bert", "roberta", "electra", "albert"]:
                    return "encoder"
                elif model_type in ["gpt2", "gpt", "llama", "mistral", "falcon", "phi"]:
                    return "decoder"
                elif model_type in ["t5", "bart", "marian"]:
                    return "encoder_decoder"

        # Default to decoder for autoregressive models
        return "decoder"

    def _extract_model_outputs_batch(self, model: Any, batch: list) -> dict[str, Any]:
        """Extract all model outputs for a batch using forward_with_outputs.

        Args:
            model: Model instance with forward_with_outputs method
            batch: List of samples with prompt and response

        Returns:
            Dictionary containing all model outputs (logits, attention, etc.)
        """
        if not hasattr(model, "forward_with_outputs"):
            raise ValueError(
                f"Model {type(model)} does not support forward_with_outputs method "
                "required for OutputProcessor architecture"
            )

        # Check for batch processing capability
        if hasattr(model, "forward_with_outputs_batch") and len(batch) > 1:
            # Use batch processing
            prompts = []
            responses = []

            for sample in batch:
                if hasattr(sample, "prompt") and hasattr(sample, "response"):
                    prompts.append(sample.prompt)
                    responses.append(sample.response)
                else:
                    raise ValueError(
                        f"Sample {type(sample)} must have 'prompt' and 'response' attributes "
                        "for OutputProcessor architecture"
                    )

            # Process entire batch at once
            model_outputs = model.forward_with_outputs_batch(prompts, responses)
            logger.debug(f"Processed batch of {len(batch)} samples")

        else:
            # Fall back to individual processing
            if len(batch) > 1:
                logger.info(
                    f"Processing {len(batch)} samples individually (no batch method available)"
                )

            outputs_list = []
            for _i, sample in enumerate(batch):
                if hasattr(sample, "prompt") and hasattr(sample, "response"):
                    prompt = sample.prompt
                    response = sample.response
                else:
                    raise ValueError(
                        f"Sample {type(sample)} must have 'prompt' and 'response' attributes "
                        "for OutputProcessor architecture"
                    )

                # Get single sample outputs
                single_outputs = model.forward_with_outputs(prompt, response)
                outputs_list.append(single_outputs)

            # Combine outputs from all samples
            model_outputs = self._combine_individual_outputs(outputs_list)
            logger.debug(f"Combined outputs from {len(batch)} individual samples")

        # Move tensors to target device if needed
        if self.device != "cpu" and self.device != "auto":
            for key, value in model_outputs.items():
                if hasattr(value, "to"):
                    model_outputs[key] = value.to(self.device)

        logger.debug(f"Extracted model outputs with keys: {list(model_outputs.keys())}")
        return model_outputs

    def _combine_individual_outputs(
        self, outputs_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Combine outputs from individual samples into batch format.

        Args:
            outputs_list: List of individual sample outputs

        Returns:
            Combined batch outputs
        """
        if not outputs_list:
            return {}

        combined = {}
        for key in outputs_list[0].keys():
            if key == "attention_weights":
                # Stack attention weights (tuple of tensors per layer)
                # Each element is (batch, heads, seq_len, seq_len)
                attention_layers = []
                for layer_idx in range(len(outputs_list[0][key])):
                    layer_attentions = [
                        outputs[key][layer_idx] for outputs in outputs_list
                    ]
                    # Stack along batch dimension
                    combined_layer = torch.cat(layer_attentions, dim=0)
                    attention_layers.append(combined_layer)
                combined[key] = tuple(attention_layers)

            elif key == "logits":
                # Stack logits: (batch, seq_len, vocab_size)
                logits_list = [outputs[key] for outputs in outputs_list]
                combined[key] = torch.cat(logits_list, dim=0)

            elif key == "input_ids":
                # Stack input_ids: (batch, seq_len)
                ids_list = [outputs[key] for outputs in outputs_list]
                combined[key] = torch.cat(ids_list, dim=0)

            elif key == "input_length":
                # Combine input lengths into tensor
                lengths = [outputs[key] for outputs in outputs_list]
                if isinstance(lengths[0], int):
                    combined[key] = torch.tensor(lengths)
                else:
                    combined[key] = torch.stack(lengths)
            else:
                # For other keys, try to stack if they're tensors
                try:
                    values = [outputs[key] for outputs in outputs_list]
                    if hasattr(values[0], "shape"):
                        combined[key] = torch.stack(values)
                    else:
                        combined[key] = values
                except Exception:
                    # Keep as list if stacking fails
                    combined[key] = [outputs[key] for outputs in outputs_list]

        return combined

    def validate_setup(self) -> bool:
        """Validate that the processor is properly configured.

        Returns:
            True if valid, raises otherwise
        """
        # Check feature set
        if not self.feature_set:
            raise ValueError("No feature set configured")

        # Check feature set has required methods
        if not hasattr(self.feature_set, "compute_features"):
            raise ValueError("Feature set missing compute_features method")

        # Check store if configured
        if self.store:
            if not hasattr(self.store, "load") or not hasattr(self.store, "save"):
                raise ValueError("Store missing required methods")

        # Check device
        if self.device not in ["cpu", "cuda", "mps"]:
            if not self.device.startswith("cuda:"):
                raise ValueError(f"Invalid device: {self.device}")

        return True

    def _compute_features_chunked(
        self,
        model: Any,
        samples: list,
        split: str | None = None,
    ) -> FeatureResult:
        """Process features in chunks to limit memory usage.

        This method processes large datasets in chunks, writing each chunk
        to disk and then combining them to avoid memory accumulation.

        Args:
            model: Model to use
            samples: List of samples to process
            split: Split name for metadata

        Returns:
            Combined features from all chunks
        """
        import gc
        import pickle
        import tempfile

        chunk_size = self.max_samples_in_memory
        n_chunks = (len(samples) + chunk_size - 1) // chunk_size

        logger.info(
            f"Processing {len(samples)} samples in {n_chunks} chunks of max {chunk_size}"
        )

        # Store chunk results in temporary files to avoid memory accumulation
        chunk_files = []
        combined_features = []
        combined_labels = []
        combined_splits = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(samples))
                chunk_samples = samples[start_idx:end_idx]

                logger.info(
                    f"Processing chunk {chunk_idx + 1}/{n_chunks}: samples {start_idx}-{end_idx}"
                )

                # Process this chunk normally
                chunk_result = self._process_chunk(model, chunk_samples, split)

                # Save chunk to temporary file instead of keeping in memory
                chunk_file = f"{temp_dir}/chunk_{chunk_idx}.pkl"
                with open(chunk_file, "wb") as f:
                    pickle.dump(
                        {
                            "features": chunk_result.features.cpu(),  # Move to CPU
                            "labels": chunk_result.labels,
                            "splits": chunk_result.splits,
                        },
                        f,
                    )
                chunk_files.append(chunk_file)

                # Clear memory
                del chunk_result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Now combine all chunks from disk
            logger.info("Combining chunks...")
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)
                    combined_features.append(chunk_data["features"])
                    if chunk_data["labels"] is not None:
                        combined_labels.append(chunk_data["labels"])
                    if chunk_data["splits"]:
                        combined_splits.extend(chunk_data["splits"])

        # Final combination
        if combined_features:
            final_features = torch.cat(combined_features, dim=0)
        else:
            final_features = torch.empty(0, self.feature_set.feature_dim)

        if combined_labels:
            final_labels = torch.cat(combined_labels, dim=0)
        else:
            final_labels = None

        if not combined_splits:
            combined_splits = None

        # Get feature names from metadata
        metadata = self.feature_set.get_metadata()

        logger.info(
            f"Chunked processing complete: {final_features.shape[0]} samples, {final_features.shape[1]} features"
        )

        return FeatureResult(
            features=final_features,
            feature_names=metadata.feature_names,
            metadata={
                "feature_set": self.feature_set.name,
                "model": self._get_model_name(model),
                "dataset": f"chunked_{len(samples)}_samples",
                "n_samples": len(final_features),
                "split": split,
                "chunked_processing": True,
                "n_chunks": n_chunks,
            },
            labels=final_labels,
            splits=combined_splits,
        )

    def _process_chunk(
        self, model: Any, chunk_samples: list, split: str | None
    ) -> FeatureResult:
        """Process a single chunk of samples.

        This uses the same logic as the main compute_features method
        but only for a limited number of samples.
        """
        # Process in batches with memory optimization (same as main method)
        all_features = []
        all_labels = []
        all_splits = []

        iterator = range(0, len(chunk_samples), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Processing chunk batches", leave=False)

        # Import garbage collection for memory optimization
        import gc

        for batch_idx, i in enumerate(iterator):
            batch = chunk_samples[i : i + self.batch_size]

            # Determine which extraction method to use based on feature set configuration
            if (
                hasattr(self.feature_set, "output_processor")
                and self.feature_set.output_processor is not None
            ):
                # New path: extract all model outputs for OutputProcessor
                model_outputs = self._extract_model_outputs_batch(model, batch)
                model_outputs["model_name"] = self._get_model_name(model)
                model_outputs["dataset_name"] = f"chunk_{len(chunk_samples)}"
                model_outputs["architecture"] = self._get_model_architecture_type(model)

                # Compute features using new interface
                batch_result = self.feature_set.compute_features(model_outputs)

                # Clear intermediate variables to free memory
                del model_outputs
            else:
                # Legacy path: extract attention for attention processors
                attention_matrices, attention_type = self._extract_attention_batch(model, batch)

                # Create AttentionData
                attention_data = AttentionData(
                    attention_matrices=attention_matrices,
                    model_name=self._get_model_name(model),
                    dataset_name=f"chunk_{len(chunk_samples)}",
                    architecture=self._get_model_architecture_type(model),
                    attention_type=attention_type,
                    metadata={
                        "batch_size": len(batch),
                        "batch_idx": i // self.batch_size,
                    },
                )

                # Compute features using legacy interface
                batch_result = self.feature_set.compute_features(attention_data)

                # Clear intermediate variables to free memory
                del attention_matrices, attention_data

            # Collect results - move to CPU to save GPU memory
            batch_features = batch_result.features
            if batch_features.is_cuda or batch_features.device.type == "mps":
                batch_features = batch_features.cpu()
            all_features.append(batch_features)

            # Collect labels if available
            if hasattr(batch[0], "label"):
                batch_labels = torch.tensor([s.label for s in batch])
                all_labels.append(batch_labels)

            # Collect splits if available
            if hasattr(batch[0], "split"):
                batch_splits = [s.split for s in batch]
                all_splits.extend(batch_splits)

            # Clear batch result to free memory
            del batch_result

            # Force garbage collection more frequently in chunks
            if (batch_idx + 1) % 5 == 0:
                gc.collect()

        # Combine chunk results
        if all_features:
            chunk_features = torch.cat(all_features, dim=0)
            # Clear the list to free memory before returning
            del all_features
        else:
            chunk_features = torch.empty(0, self.feature_set.feature_dim)

        if all_labels:
            chunk_labels = torch.cat(all_labels, dim=0)
            # Clear the list to free memory
            del all_labels
        else:
            chunk_labels = None

        if not all_splits:
            all_splits = None

        # Get feature names from metadata
        metadata = self.feature_set.get_metadata()

        return FeatureResult(
            features=chunk_features,
            feature_names=metadata.feature_names,
            labels=chunk_labels,
            splits=all_splits,
        )

    def _compute_features_streaming(
        self,
        model: Any,
        samples: list,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Compute features using streaming Arrow writes.

        This method processes samples in batches and streams them directly
        to disk using Arrow format, avoiding memory accumulation entirely.

        Args:
            model: Model to use for feature extraction
            samples: List of samples to process
            split: Split name for metadata
            max_samples: Maximum samples (used for metadata only)

        Returns:
            FeatureResult with summary (features loaded from disk)
        """
        logger.info(f"Starting streaming processing of {len(samples)} samples")

        # Phase 1: Schema Discovery - Process first batch to determine actual feature dimensions
        if len(samples) == 0:
            raise ValueError("Cannot process empty sample list")

        first_batch_size = min(self.batch_size, len(samples))
        first_batch = samples[0:first_batch_size]

        logger.info(f"Phase 1: Processing first batch of {len(first_batch)} samples for schema discovery")
        first_batch_result = self._process_single_batch(model, first_batch)

        # Now we know actual dimensions - get proper feature names
        metadata = self.feature_set.get_metadata()
        actual_feature_names = metadata.feature_names

        logger.info(f"Discovered {len(actual_feature_names)} features: {actual_feature_names[:5]}..." if len(actual_feature_names) > 5 else f"Discovered features: {actual_feature_names}")

        # Create feature key for this dataset
        model_name = self._get_model_name(model)
        dataset_name = (
            self._get_dataset_name(samples)
            if hasattr(self, "_get_dataset_name")
            else "streaming_dataset"
        )

        key = FeatureKey(
            feature_set_name=self.feature_set.name,
            model_name=model_name,
            dataset_name=dataset_name,
            metadata={
                "split": split,
                "max_samples": max_samples,
                "streaming": True,
            },
        )

        # Phase 2: Create streaming writer with correct schema
        streaming_writer = self.store.create_streaming_writer(
            key=key,
            feature_names=actual_feature_names,
            buffer_size=5,  # Buffer 5 batches
            enable_async=True,
        )

        # Track processing statistics
        total_processed = 0

        try:
            with streaming_writer:
                # Write first batch that we already processed
                first_batch_labels = None
                first_batch_splits = None
                first_batch_hashes = []
                first_batch_indices = []

                if hasattr(first_batch[0], "label"):
                    first_batch_labels = torch.tensor([s.label for s in first_batch])

                if hasattr(first_batch[0], "split"):
                    first_batch_splits = [s.split for s in first_batch]

                # Generate sample hashes and indices for first batch
                for idx, sample in enumerate(first_batch):
                    if hasattr(sample, "prompt") and hasattr(sample, "response"):
                        import hashlib
                        content = f"{sample.prompt}|{sample.response}"
                        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
                    else:
                        hash_val = f"sample_{idx}"
                    first_batch_hashes.append(hash_val)
                    first_batch_indices.append(idx)

                # Write first batch
                streaming_writer.write_batch(
                    features=first_batch_result.features.detach().cpu(),
                    labels=first_batch_labels,
                    splits=first_batch_splits,
                    sample_hashes=first_batch_hashes,
                    original_indices=first_batch_indices,
                )

                total_processed = len(first_batch)
                logger.info(f"Wrote first batch ({total_processed} samples)")

                # Clean up first batch
                del first_batch_result

                # Process remaining batches
                remaining_samples = samples[first_batch_size:]
                if remaining_samples:
                    iterator = range(0, len(remaining_samples), self.batch_size)
                    if self.show_progress:
                        iterator = tqdm(iterator, desc="Streaming remaining batches", initial=1, total=len(iterator)+1)

                    for i in iterator:
                        batch = remaining_samples[i : i + self.batch_size]

                        # Process batch
                        batch_result = self._process_single_batch(model, batch)

                        # Extract batch data for streaming
                        batch_labels = None
                        batch_splits = None
                        batch_hashes = []
                        batch_indices = []

                        if hasattr(batch[0], "label"):
                            batch_labels = torch.tensor([s.label for s in batch])

                        if hasattr(batch[0], "split"):
                            batch_splits = [s.split for s in batch]

                        # Generate sample hashes and indices
                        for idx, sample in enumerate(batch):
                            if hasattr(sample, "prompt") and hasattr(sample, "response"):
                                import hashlib
                                content = f"{sample.prompt}|{sample.response}"
                                hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
                            else:
                                hash_val = f"sample_{total_processed + idx}"
                            batch_hashes.append(hash_val)
                            batch_indices.append(total_processed + idx)

                        # Stream batch to disk
                        streaming_writer.write_batch(
                            features=batch_result.features.detach().cpu(),
                            labels=batch_labels,
                            splits=batch_splits,
                            sample_hashes=batch_hashes,
                            original_indices=batch_indices,
                        )

                        total_processed += len(batch)

                        # Clean up batch to save memory
                        del batch_result

                        # More frequent garbage collection to prevent memory accumulation
                        if total_processed % (self.batch_size * 10) == 0:
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                else:
                    logger.info("Only processing first batch (single batch dataset)")

        except Exception as e:
            logger.error(f"Error during streaming processing: {e}")
            raise

        logger.info(
            f"Streaming processing complete: {total_processed} samples written to {streaming_writer.path}"
        )

        # Now load the streamed data back for return (this is a summary only)
        # In practice, consumers would load directly from the Arrow file
        cached_result = self.store.load(key)
        if cached_result:
            logger.info("Successfully loaded streamed features for verification")
            return cached_result
        else:
            # Create a minimal result for compatibility
            logger.warning(
                "Could not load streamed features back - creating minimal result"
            )
            return FeatureResult(
                features=torch.empty(0, len(actual_feature_names)),
                feature_names=actual_feature_names,
                metadata={
                    "streaming": True,
                    "total_samples": total_processed,
                    "file_path": str(streaming_writer.path),
                    "feature_set": self.feature_set.name,
                    "model": model_name,
                    "dataset": dataset_name,
                },
                labels=None,
                splits=None,
            )

    @torch.no_grad()
    def _process_single_batch(self, model: Any, batch: list) -> FeatureResult:
        """Process a single batch and return features.

        This is extracted from the main compute_features method to be reusable
        for streaming processing.
        """
        # Determine which extraction method to use based on feature set configuration
        if (
            hasattr(self.feature_set, "output_processor")
            and self.feature_set.output_processor is not None
        ):
            # New path: extract all model outputs for OutputProcessor
            model_outputs = self._extract_model_outputs_batch(model, batch)
            model_outputs["model_name"] = self._get_model_name(model)
            model_outputs["dataset_name"] = "batch"
            model_outputs["architecture"] = self._get_model_architecture_type(model)

            # Compute features using new interface
            batch_result = self.feature_set.compute_features(model_outputs)

        else:
            # Legacy path: extract attention for attention processors
            attention_matrices, attention_type = self._extract_attention_batch(model, batch)

            # Create AttentionData
            attention_data = AttentionData(
                attention_matrices=attention_matrices,
                model_name=self._get_model_name(model),
                dataset_name="batch",
                architecture=self._get_model_architecture_type(model),
                attention_type=attention_type,
                metadata={
                    "batch_size": len(batch),
                    "streaming": True,
                },
            )

            # Compute features using legacy interface
            batch_result = self.feature_set.compute_features(attention_data)

        return batch_result


class ParallelFeatureProcessor(FeatureProcessor):
    """Feature processor with parallel computation support.

    This extends the basic processor with support for parallel
    computation across multiple devices or processes.
    """

    def __init__(
        self,
        feature_set: IFeatureSet,
        store: IFeatureStore | None = None,
        batch_size: int = 32,
        device: str = "cpu",
        force_recompute: bool = False,
        show_progress: bool = True,
        n_workers: int = 4,
        distributed: bool = False,
    ):
        """Initialize parallel processor.

        Args:
            feature_set: Feature set to compute
            store: Optional storage backend
            batch_size: Batch size per worker
            device: Device for computation
            force_recompute: If True, always recompute
            show_progress: If True, show progress
            n_workers: Number of parallel workers
            distributed: If True, use distributed processing
        """
        super().__init__(
            feature_set, store, batch_size, device, force_recompute, show_progress
        )
        self.n_workers = n_workers
        self.distributed = distributed

    def compute_features(
        self,
        model: Any,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Compute features with parallel processing.

        Args:
            model: Model to use
            dataset: Dataset to process
            split: Optional split
            max_samples: Optional limit

        Returns:
            Computed features
        """
        if self.distributed:
            # Use distributed processing
            return self._compute_distributed(model, dataset, split, max_samples)
        elif self.n_workers > 1:
            # Use multiprocessing
            return self._compute_multiprocess(model, dataset, split, max_samples)
        else:
            # Fall back to sequential
            return super().compute_features(model, dataset, split, max_samples)

    def _compute_multiprocess(
        self,
        model: Any,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Compute using multiprocessing.

        This is a placeholder - real implementation would use
        multiprocessing or concurrent.futures.
        """
        logger.info(f"Computing with {self.n_workers} workers")
        # Simplified - real implementation would parallelize
        return super().compute_features(model, dataset, split, max_samples)

    def _compute_distributed(
        self,
        model: Any,
        dataset: Any,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> FeatureResult:
        """Compute using distributed processing.

        This is a placeholder - real implementation would use
        torch.distributed or similar.
        """
        logger.info("Computing with distributed processing")
        # Simplified - real implementation would distribute
        return super().compute_features(model, dataset, split, max_samples)
