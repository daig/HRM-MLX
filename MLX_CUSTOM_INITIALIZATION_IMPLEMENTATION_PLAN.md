# MLX Custom Initialization Implementation Plan

## Executive Summary

This document outlines the implementation plan for porting the HRM (Hierarchical Recurrent Model) custom initialization from PyTorch to MLX. The primary goal is to implement a mathematically correct truncated normal initialization that matches JAX's behavior, ensuring deterministic and numerically stable parameter initialization for neural networks.

### Key Requirements
- Port `trunc_normal_init_` function with JAX-compatible behavior
- Support configurable standard deviation and truncation bounds
- Ensure deterministic initialization with proper seeding
- Maintain numerical properties of the original implementation
- Integrate seamlessly with MLX's parameter initialization system

## 1. Mathematical Formulation

### Truncated Normal Distribution

The truncated normal distribution is a normal distribution bounded within a specific range [a, b]. For neural network initialization, we typically use [-2σ, 2σ] bounds.

#### Probability Density Function
For a truncated normal distribution with parameters μ (mean), σ (standard deviation), and bounds [a, b]:

```
f(x; μ, σ, a, b) = φ(x; μ, σ) / [Φ(b; μ, σ) - Φ(a; μ, σ)]  for x ∈ [a, b]
```

Where:
- φ(x; μ, σ) is the PDF of the standard normal distribution
- Φ(x; μ, σ) is the CDF of the standard normal distribution

#### Variance Correction
The key insight is that truncation changes the variance of the distribution. To maintain the desired variance σ², we need to apply a correction factor:

```
σ_corrected = σ / sqrt(correction_factor)
```

Where the correction factor accounts for the mass within the truncation bounds.

## 2. Implementation Architecture

### Core Components

```python
# mlx_hrm/initializers.py

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math

def truncated_normal(
    shape: Tuple[int, ...],
    dtype: mx.Dtype = mx.float32,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    key: Optional[mx.array] = None
) -> mx.array:
    """
    Generate samples from a truncated normal distribution.
    
    This implementation matches JAX's behavior with proper variance correction
    to ensure the resulting distribution has the desired standard deviation.
    
    Args:
        shape: Output shape
        dtype: Data type of the output
        mean: Mean of the underlying normal distribution
        std: Standard deviation of the desired output distribution
        lower: Lower truncation bound (in units of std)
        upper: Upper truncation bound (in units of std)
        key: Random key for deterministic initialization
        
    Returns:
        Array of shape `shape` with samples from truncated normal distribution
    """
    if key is None:
        key = mx.random.key(0)
    
    # Generate uniform samples
    u = mx.random.uniform(shape=shape, key=key)
    
    # Convert bounds to normalized scale
    a_norm = (lower * std - mean) / std
    b_norm = (upper * std - mean) / std
    
    # Compute CDF values at bounds
    from scipy import special
    phi_a = 0.5 * (1 + special.erf(a_norm / math.sqrt(2)))
    phi_b = 0.5 * (1 + special.erf(b_norm / math.sqrt(2)))
    
    # Apply inverse CDF transform (truncated normal)
    # This ensures samples are within [a, b]
    z = phi_a + u * (phi_b - phi_a)
    
    # Convert back to normal scale using inverse error function
    x_normalized = math.sqrt(2) * special.erfinv(2 * z - 1)
    
    # Apply variance correction to maintain desired std
    # This is the key difference from PyTorch's implementation
    variance_factor = _compute_variance_factor(a_norm, b_norm)
    x_corrected = x_normalized / math.sqrt(variance_factor)
    
    # Scale and shift to desired mean and std
    result = mean + std * x_corrected
    
    return mx.array(result, dtype=dtype)

def _compute_variance_factor(a: float, b: float) -> float:
    """
    Compute the variance correction factor for truncated normal distribution.
    
    This factor ensures that after truncation, the distribution maintains
    the desired variance.
    """
    from scipy import special
    
    # PDF values at bounds
    phi_a = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a)
    phi_b = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * b * b)
    
    # CDF values at bounds
    Phi_a = 0.5 * (1 + special.erf(a / math.sqrt(2)))
    Phi_b = 0.5 * (1 + special.erf(b / math.sqrt(2)))
    
    # Variance factor calculation
    Z = Phi_b - Phi_a
    variance_factor = 1 + (a * phi_a - b * phi_b) / Z - ((phi_a - phi_b) / Z) ** 2
    
    return variance_factor
```

### MLX-Specific Optimizations

```python
def truncated_normal_mlx_optimized(
    shape: Tuple[int, ...],
    dtype: mx.Dtype = mx.float32,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    key: Optional[mx.array] = None
) -> mx.array:
    """
    MLX-optimized version using lazy evaluation and Metal acceleration.
    """
    if key is None:
        key = mx.random.key(0)
    
    # Use MLX's optimized random generation
    u = mx.random.uniform(shape=shape, key=key, dtype=dtype)
    
    # Precompute constants
    a_std = lower * std
    b_std = upper * std
    
    # Use MLX's native operations for better performance
    # Approximate inverse error function for Metal GPU
    z = _mlx_truncated_normal_transform(u, a_std, b_std, mean, std)
    
    return z

def _mlx_truncated_normal_transform(u, a, b, mean, std):
    """MLX-native implementation of truncated normal transform."""
    # This would use MLX's compiled operations
    # Implementation details depend on MLX's available ops
    pass
```

## 3. Module Integration

### Custom Linear Layer with Truncated Normal Initialization

```python
class LinearTruncNormal(nn.Module):
    """Linear layer with truncated normal initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: Optional[float] = None,
        key: Optional[mx.array] = None
    ):
        super().__init__()
        
        if std is None:
            # LeCun normal initialization
            std = math.sqrt(1.0 / in_features)
        
        if key is None:
            key = mx.random.key(0)
        
        # Split key for weight and bias
        key_w, key_b = mx.random.split(key)
        
        # Initialize weights with truncated normal
        self.weight = truncated_normal(
            shape=(out_features, in_features),
            std=std,
            key=key_w
        )
        
        if bias:
            # Initialize bias to zeros (standard practice)
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
    
    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
```

### Custom Embedding Layer

```python
class EmbeddingTruncNormal(nn.Module):
    """Embedding layer with truncated normal initialization."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        std: Optional[float] = None,
        key: Optional[mx.array] = None
    ):
        super().__init__()
        
        if std is None:
            std = 1.0 / math.sqrt(embedding_dim)
        
        if key is None:
            key = mx.random.key(0)
        
        self.weight = truncated_normal(
            shape=(num_embeddings, embedding_dim),
            std=std,
            key=key
        )
    
    def __call__(self, indices):
        return self.weight[indices]
```

## 4. Comparison with PyTorch and JAX

### PyTorch's Implementation Issues

PyTorch's `trunc_normal_` has a mathematical flaw where it doesn't properly correct for the variance change due to truncation. This results in:

```python
# PyTorch's approach (incorrect variance)
x = torch.normal(mean, std, size)
x = torch.clamp(x, min=a, max=b)  # Simple clipping changes variance!
```

### JAX's Correct Implementation

JAX properly accounts for variance correction:

```python
# JAX's approach (correct variance)
def truncated_normal_jax(key, shape, lower=-2, upper=2):
    # Proper inverse CDF method with variance correction
    return jax.random.truncated_normal(key, lower, upper, shape)
```

### Our MLX Implementation

Our implementation follows JAX's mathematically correct approach:

1. Uses inverse CDF method for proper truncation
2. Applies variance correction factor
3. Ensures deterministic initialization with explicit keys
4. Leverages MLX's Metal acceleration

## 5. Testing Strategy

### Statistical Property Tests

```python
def test_truncated_normal_statistics():
    """Test that truncated normal has correct statistical properties."""
    key = mx.random.key(42)
    shape = (10000,)
    std = 0.02
    
    # Generate samples
    samples = truncated_normal(shape, std=std, key=key)
    
    # Test mean (should be close to 0)
    assert abs(mx.mean(samples).item()) < 0.001
    
    # Test standard deviation (should be close to specified std)
    assert abs(mx.std(samples).item() - std) < 0.001
    
    # Test bounds (should be within [-2*std, 2*std])
    assert mx.min(samples).item() >= -2 * std
    assert mx.max(samples).item() <= 2 * std
```

### Determinism Tests

```python
def test_deterministic_initialization():
    """Test that same key produces same initialization."""
    key = mx.random.key(42)
    shape = (100, 100)
    
    # Generate twice with same key
    init1 = truncated_normal(shape, key=key)
    init2 = truncated_normal(shape, key=key)
    
    # Should be identical
    assert mx.array_equal(init1, init2)
```

### Numerical Comparison with JAX

```python
def test_compare_with_jax():
    """Compare numerical outputs with JAX implementation."""
    import jax
    import jax.numpy as jnp
    
    seed = 42
    shape = (1000,)
    
    # JAX implementation
    jax_key = jax.random.PRNGKey(seed)
    jax_samples = jax.random.truncated_normal(jax_key, -2, 2, shape)
    
    # MLX implementation
    mlx_key = mx.random.key(seed)
    mlx_samples = truncated_normal(shape, key=mlx_key)
    
    # Compare statistics (not exact values due to different RNGs)
    jax_mean = jnp.mean(jax_samples)
    mlx_mean = mx.mean(mlx_samples)
    assert abs(jax_mean - mlx_mean.item()) < 0.01
```

### Integration Tests

```python
def test_model_initialization():
    """Test full model initialization with truncated normal."""
    key = mx.random.key(42)
    
    # Create model with custom initialization
    model = nn.Sequential([
        LinearTruncNormal(784, 256, key=key),
        nn.ReLU(),
        LinearTruncNormal(256, 10, key=key)
    ])
    
    # Test forward pass
    x = mx.random.normal((32, 784))
    y = model(x)
    assert y.shape == (32, 10)
```

## 6. Performance Considerations

### Optimization Opportunities

1. **Batch Initialization**: Initialize multiple layers in a single call
2. **Compiled Functions**: Use MLX's JIT compilation for initialization
3. **Memory Efficiency**: Use in-place operations where possible
4. **GPU Acceleration**: Leverage Metal for large tensors

### Benchmarking Code

```python
def benchmark_initialization():
    """Benchmark initialization performance."""
    import time
    
    sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]
    
    for size in sizes:
        # Time truncated normal
        start = time.time()
        _ = truncated_normal(size)
        mlx_time = time.time() - start
        
        # Time standard normal (for comparison)
        start = time.time()
        _ = mx.random.normal(size)
        normal_time = time.time() - start
        
        print(f"Size {size}: TruncNormal {mlx_time:.3f}s, Normal {normal_time:.3f}s")
```

## 7. Integration Guide

### Model Definition

```python
class HRMModel(nn.Module):
    """Example HRM model with truncated normal initialization."""
    
    def __init__(self, config, key=None):
        super().__init__()
        
        if key is None:
            key = mx.random.key(0)
        
        # Split keys for different components
        keys = mx.random.split(key, num=4)
        
        # Token embeddings with truncated normal
        self.token_embeddings = EmbeddingTruncNormal(
            config.vocab_size,
            config.hidden_dim,
            std=config.initializer_range,
            key=keys[0]
        )
        
        # Position embeddings
        self.position_embeddings = EmbeddingTruncNormal(
            config.max_position_embeddings,
            config.hidden_dim,
            std=config.initializer_range,
            key=keys[1]
        )
        
        # Transformer layers
        self.layers = []
        for i in range(config.num_layers):
            layer_key = mx.random.split(keys[2], num=config.num_layers)[i]
            self.layers.append(
                TransformerLayer(config, key=layer_key)
            )
        
        # Output projection
        self.output_projection = LinearTruncNormal(
            config.hidden_dim,
            config.vocab_size,
            std=config.initializer_range,
            key=keys[3]
        )
```

### Training Script Integration

```python
def create_model_with_seed(config, seed=42):
    """Create model with deterministic initialization."""
    key = mx.random.key(seed)
    model = HRMModel(config, key=key)
    return model

# Training loop
def train():
    # Ensure reproducibility
    seed = 42
    model = create_model_with_seed(config, seed)
    
    # Rest of training code...
```

## 8. Migration Checklist

- [ ] Implement core `truncated_normal` function
- [ ] Add variance correction calculation
- [ ] Create MLX-optimized version
- [ ] Implement custom Linear layer
- [ ] Implement custom Embedding layer
- [ ] Write comprehensive tests
- [ ] Benchmark performance vs standard initialization
- [ ] Document API and usage examples
- [ ] Integrate with existing HRM codebase
- [ ] Validate numerical equivalence with JAX

## 9. Example Usage

### Basic Usage

```python
import mlx.core as mx
from mlx_hrm.initializers import truncated_normal

# Simple initialization
weights = truncated_normal((256, 128), std=0.02)

# With custom bounds
weights = truncated_normal((256, 128), std=0.02, lower=-3.0, upper=3.0)

# Deterministic initialization
key = mx.random.key(42)
weights = truncated_normal((256, 128), std=0.02, key=key)
```

### Model Usage

```python
from mlx_hrm.layers import LinearTruncNormal, EmbeddingTruncNormal

# Create layers with truncated normal init
linear = LinearTruncNormal(784, 256, std=0.02)
embedding = EmbeddingTruncNormal(10000, 512, std=0.01)

# Use in model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = EmbeddingTruncNormal(10000, 512)
        self.fc1 = LinearTruncNormal(512, 256)
        self.fc2 = LinearTruncNormal(256, 10)
    
    def __call__(self, x):
        x = self.embed(x)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)
```

## 10. References

1. [JAX Truncated Normal Implementation](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.truncated_normal.html)
2. [The Truncated Normal Distribution](https://en.wikipedia.org/wiki/Truncated_normal_distribution)
3. [On the Variance of Truncated Normal Distribution](https://stats.stackexchange.com/questions/380040/)
4. [MLX Documentation](https://ml-explore.github.io/mlx/)

## Appendix: Mathematical Derivations

### Variance Correction Factor Derivation

For a truncated normal distribution with bounds [a, b], the variance is:

```
Var[X] = σ² * [1 + (a*φ(a) - b*φ(b))/(Φ(b) - Φ(a)) - ((φ(a) - φ(b))/(Φ(b) - Φ(a)))²]
```

To maintain variance σ² after truncation, we need to scale the underlying normal distribution by 1/√(variance_factor).

### Numerical Stability Considerations

When implementing the inverse error function, care must be taken near the bounds to avoid numerical instability. The implementation should handle edge cases where:
- Values are very close to ±1
- The truncation bounds are very tight
- The standard deviation is very small

---

This implementation plan provides a complete roadmap for porting the HRM custom initialization to MLX while maintaining mathematical correctness and performance.