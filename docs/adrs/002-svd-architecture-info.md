# ADR-002: SVD Feature Dimension Calculation with Architecture Info

## Status
Accepted

## Context
SVD feature extractors using `aggregation="concat"` mode were failing with dimension mismatches due to inability to calculate correct feature dimensions before the forward pass. This prevented proper feature name generation and caused streaming operations to fail.

### Problem Statement
- SVD extractor returned 0 dimensions for concat mode when `_last_tensor_dims` was None
- Feature names generation failed (40 names vs 5760 expected for GPT-2)
- Streaming operations failed due to schema mismatch
- No architecture information available at initialization time

### Specific Error
```
ValueError: Number of feature names (40) doesn't match feature dimension (5760)
```

For GPT-2: 40 base features × 12 layers × 12 heads = 5760 total features

## Decision
Provide model architecture information (num_layers, num_heads) to feature extractors at initialization time through configuration, enabling correct dimension calculation before the forward pass.

### Architecture Components

#### 1. Model Configuration Enhancement
```yaml
# conf/model/gpt2.yaml
_target_: shade.models.huggingface.adapter.HuggingFaceModelAdapter
model_name: gpt2
device_map: auto
max_length: 1024

# Architecture details for feature extractors
architecture:
  num_layers: 12
  num_heads: 12
```

**Added to models:**
- GPT-2: 12 layers, 12 heads  
- DistilGPT-2: 6 layers, 12 heads
- BERT-base: 12 layers, 12 heads
- Llama-3.2-3B: 28 layers, 16 heads

#### 2. SVD Extractor Enhancement
```python
class ComprehensiveSVDExtractor(FeatureExtractor):
    def __init__(self, ..., n_layers=None, n_heads=None, **kwargs):
        # Store provided architecture info for dimension calculation
        self.n_layers = n_layers
        self.n_heads = n_heads
        
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        if self.aggregation == "concat":
            if self._last_tensor_dims is not None:
                # Use actual dimensions from runtime
                n_matrices = self._last_tensor_dims["n_matrices"]
            elif self.n_layers is not None and self.n_heads is not None:
                # Use provided architecture info
                n_matrices = self.n_layers * self.n_heads
            else:
                return 0  # Cannot determine dimension
```

#### 3. Configuration Integration
```yaml
# conf/components/extractors/svd_comprehensive.yaml
- _target_: shade.detectors.feature_extractors.svd.ComprehensiveSVDExtractor
  name: svd_comprehensive
  aggregation: concat
  n_layers: ${oc.select:model.architecture.num_layers,null}
  n_heads: ${oc.select:model.architecture.num_heads,null}
```

**Hydra Integration:**
- Uses `oc.select` for optional architecture info
- Graceful fallback to null when architecture not available
- Automatic resolution from model configuration

### Technical Implementation

#### Dimension Calculation Strategy
1. **Runtime dimensions** (highest priority): Use actual tensor dimensions from forward pass
2. **Architecture info** (fallback): Use provided num_layers × num_heads
3. **Unknown** (error case): Return 0 and require forward pass

#### Feature Name Generation
```python
def get_feature_names(self, config=None) -> list[str]:
    if self.aggregation != "concat":
        return per_matrix_features
        
    if self._last_tensor_dims is not None:
        # Use runtime dimensions
        n_layers = self._last_tensor_dims["n_layers"]
        n_heads = self._last_tensor_dims["n_heads"]
    elif self.n_layers is not None and self.n_heads is not None:
        # Use provided architecture info
        return self._generate_feature_names_with_dims(
            self.n_layers, self.n_heads, per_matrix_features
        )
```

**Feature naming pattern:**
- `L0_H0_σ_1`, `L0_H0_nuclear_norm`, etc.
- Systematic layer/head prefixes for all 144 matrices (GPT-2)

## Alternatives Considered

### 1. Change Aggregation Mode to "mean"
**Rejected**: User explicitly requested concat mode for full matrix information

### 2. Delay Schema Creation Until Forward Pass
**Rejected**: Streaming requires schema upfront for Arrow file creation

### 3. Dynamic Architecture Detection
**Rejected**: Requires model instantiation, adds complexity and latency

### 4. Runtime Feature Name Updates
**Rejected**: Breaks compatibility with streaming and caching systems

## Consequences

### Positive
- ✅ **Correct dimensions**: Proper feature dimensions calculated at initialization
- ✅ **Streaming compatibility**: Schema created correctly before processing
- ✅ **Architecture awareness**: Feature extractors understand model structure  
- ✅ **Backward compatibility**: Runtime detection still works as fallback
- ✅ **Scalability**: Works with any model architecture

### Negative
- ❌ **Configuration overhead**: Requires architecture info in model configs
- ❌ **Maintenance**: Must update configs when adding new models

### Neutral
- 🔄 **Validation**: Architecture info can be validated against runtime dimensions
- 🔄 **Flexibility**: Supports both config-based and runtime-based dimension calculation

## Implementation Results

### Before Fix
```
Feature extraction failed: Number of feature names (40) doesn't match feature dimension (5760)
```

### After Fix
```
✅ Features precomputed successfully using shade-io!
Features: 5760
Feature names extracted: 5760 names
```

### Performance Impact
- **Memory**: No additional memory overhead
- **Computation**: Negligible overhead from architecture info storage
- **Initialization**: Slightly faster due to upfront dimension calculation

## Integration Points

### Configuration System
- **Hydra interpolation**: `${oc.select:model.architecture.num_layers,null}`
- **Optional parameters**: Graceful handling when architecture not available
- **Type safety**: Proper null handling and validation

### Feature Processing Pipeline
- **Schema creation**: Correct Arrow schema with proper field count
- **Streaming**: Compatible with StreamingArrowWriter requirements
- **Caching**: Feature metadata includes correct dimensions

## Decision Rationale
This approach provides the necessary architecture information while maintaining flexibility and backward compatibility. It solves the immediate dimension calculation problem and enables proper streaming support for concat-mode SVD features.

## Related Decisions
- ADR-001: Streaming Arrow Writer (requires correct feature dimensions)
- Future ADR: Model Architecture Registry (potential centralized architecture management)

---

**Author**: Claude Code Assistant  
**Date**: 2025-09-13  
**Dependencies**: ADR-001 (streaming requires correct dimensions)  
**Reviewers**: To be assigned