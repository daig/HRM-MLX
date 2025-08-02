# MLX HRM Implementation

Hierarchical Reasoning Model (HRM) implementation in MLX for Apple Silicon.

## Installation

```bash
pip install -e .
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Testing

Run tests:
```bash
pytest tests/
```

## Current Implementation Status

### Phase 1: Foundation Components ✅
- [x] Custom Weight Initialization (truncated normal with variance correction)
- [x] Linear layer with truncated normal initialization
- [x] Embedding layer with truncated normal initialization
- [x] Comprehensive unit tests
- [x] Project setup (pyproject.toml, requirements.txt)

### Next Steps
- [ ] RMSNorm Layer
- [ ] SwiGLU Activation
- [ ] Sparse Embeddings
- [ ] Rotary Position Embeddings (RoPE)
- [ ] Multi-Head Attention
- [ ] Adaptive Computation Time (ACT)
- [ ] Complete HRM Model

## Usage Example

```python
import mlx.core as mx
from mlx_hrm.layers import truncated_normal, LinearTruncNormal, EmbeddingTruncNormal

# Basic truncated normal initialization
weights = truncated_normal((256, 128), std=0.02)

# Custom layers with truncated normal init
linear = LinearTruncNormal(784, 256, std=0.02)
embedding = EmbeddingTruncNormal(10000, 512)

# Forward pass
x = mx.random.normal((32, 784))
y = linear(x)
```