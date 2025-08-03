# MLX HRM Implementation

Hierarchical Reasoning Model (HRM) implementation in MLX for Apple Silicon.

## Overview

This is a complete port of the Hierarchical Reasoning Model (HRM) from PyTorch to MLX, maintaining exact behavioral match with the original implementation while optimizing for Apple Silicon performance. The model features a two-level hierarchical structure with Adaptive Computation Time (ACT) for dynamic reasoning.

## Installation

```bash
pip install -e .
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Current Implementation Status

### Completed Phases ✅

#### Phase 1: Foundation Components ✅
- [x] Custom Weight Initialization (truncated normal with variance correction)
- [x] RMSNorm Layer (functional and module variants)

#### Phase 2: Core Components ✅
- [x] SwiGLU Activation (with fused projections)
- [x] Sparse Embeddings (with SignSGD optimizer)
- [x] Rotary Position Embeddings (RoPE) with caching

#### Phase 3: Complex Modules ✅
- [x] Multi-Head Attention with GQA support

#### Phase 4: Advanced Features ✅
- [x] Adaptive Computation Time (ACT) mechanism
- [x] HRM Inner model (H-level + L-level modules)
- [x] Complete HRM_ACT wrapper

#### Phase 5: Model Integration ✅
- [x] Complete HRM wrapper with user-friendly API
- [x] Model presets (tiny, small, base, large)
- [x] Factory functions for model creation
- [x] Checkpoint save/load functionality
- [x] Integration tests

### Next Steps
- [ ] Phase 6: Loss Functions & Metrics
- [ ] Phase 7: Training Infrastructure
- [ ] Phase 8: Validation & Testing
- [ ] Phase 9: Documentation & Examples

## Quick Start

```python
from mlx_hrm import create_hrm

# Create model using preset
model = create_hrm('small')  # 27M parameter model from paper

# Forward pass
batch = {'input_ids': mx.array([[1, 2, 3, 4, 5]])}
carry, outputs = model(batch)
logits = outputs['logits']

# Generation
prompt = mx.array([1, 2, 3])
generated = model.generate(prompt, max_length=20, temperature=0.8)

# Save/load checkpoint
from mlx_hrm.utils import save_checkpoint
save_checkpoint(model, 'checkpoints/my_model')
```

## Model Presets

| Preset | Parameters | Hidden Size | Layers | Use Case |
|--------|------------|-------------|---------|----------|
| tiny   | ~7M        | 256         | 2+2     | Testing/Development |
| small  | ~27M       | 512         | 4+4     | Paper configuration |
| base   | ~100M      | 768         | 8+8     | Scaling experiments |
| large  | ~200M      | 1024        | 12+12   | Large-scale tasks |

## Architecture Features

- **Hierarchical Structure**: H-level (planning) and L-level (computation) modules
- **Adaptive Computation Time**: Dynamic reasoning steps up to 64 (configurable)
- **Grouped Query Attention**: Efficient attention with KV head sharing
- **Sparse Embeddings**: Memory-efficient puzzle-specific embeddings
- **SwiGLU Activation**: Gated linear units for improved performance

## Testing

Run unit tests:
```bash
python -m pytest tests/unit/
```

Run integration tests:
```bash
python -m pytest tests/integration/
```

## Examples

See the `examples/` directory for detailed usage examples:
- `basic_usage.py` - Complete walkthrough of all features
- Model creation with different configurations
- Forward pass and inference
- Text generation with various sampling strategies
- Checkpoint management
- Training setup

## Performance

Optimized for Apple Silicon with:
- Efficient memory usage through MLX's lazy evaluation
- Mixed precision support (bfloat16/float32)
- Optimized attention kernels
- Memory-efficient sparse embeddings

## License

[License information here]