# MLX RMSNorm Implementation Plan for HRM

## Executive Summary

This document provides the implementation plan for porting RMSNorm (Root Mean Square Normalization) from PyTorch to MLX. RMSNorm is a critical normalization layer in the HRM architecture that provides computational efficiency compared to LayerNorm while maintaining training stability.

## Component Overview

### What is RMSNorm?

RMSNorm normalizes activations using only the root mean square statistic, without centering (no mean subtraction). This makes it computationally more efficient than LayerNorm while providing similar or better training stability.

**Key Benefits:**
- Faster computation (no mean calculation)
- Fewer parameters (no bias term)
- Better numerical stability in mixed precision training
- Lower memory footprint

### Current Implementation Details
- **Location**: Used throughout HRM blocks in place of LayerNorm
- **PyTorch Implementation**: Custom implementation with learnable scale parameter
- **Key Features**:
  - Operates on the last dimension by default
  - Uses float32 for normalization computation (numerical stability)
  - Learnable weight parameter for scaling

## Mathematical Formulation

Given input tensor `x` with shape `[..., hidden_dim]`:

```
RMS(x) = sqrt(mean(x²))
RMSNorm(x) = (x / (RMS(x) + ε)) * γ
```

Where:
- `ε` is a small constant for numerical stability (typically 1e-5 or 1e-6)
- `γ` is a learnable scale parameter with shape `[hidden_dim]`

## Implementation Strategy

### Option 1: Use MLX's Built-in RMSNorm (Recommended)

MLX provides an optimized RMSNorm implementation:

```python
import mlx.nn as nn

# Direct usage
rmsnorm = nn.RMSNorm(dims=hidden_size, eps=1e-5)
```

### Option 2: Custom Implementation for Compatibility

If we need exact PyTorch interface compatibility:

```python
import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm layer maintaining PyTorch interface compatibility."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))
        
    def _norm(self, x):
        """Compute RMS normalization."""
        # Cast to float32 for stability
        x_float = x.astype(mx.float32)
        
        # Compute RMS
        variance = mx.mean(mx.square(x_float), axis=-1, keepdims=True)
        x_normed = x_float * mx.rsqrt(variance + self.eps)
        
        # Cast back to original dtype
        return x_normed.astype(x.dtype)
    
    def __call__(self, x):
        """Forward pass with weight scaling."""
        output = self._norm(x)
        return self.weight * output
```

## Key Differences Between PyTorch and MLX

### 1. API Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Module creation | `nn.Module` | `nn.Module` (same) |
| Parameter registration | `nn.Parameter()` | Direct attribute assignment |
| Forward method | `forward()` | `__call__()` |

### 2. Computation Differences

```python
# PyTorch
variance = x.pow(2).mean(-1, keepdim=True)
x_normed = x / torch.sqrt(variance + self.eps)

# MLX
variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
x_normed = x * mx.rsqrt(variance + self.eps)  # rsqrt is more efficient
```

### 3. Type Casting

```python
# PyTorch
x_fp32 = x.float()
normed = compute_norm(x_fp32)
output = normed.type_as(x)

# MLX
x_fp32 = x.astype(mx.float32)
normed = compute_norm(x_fp32)
output = normed.astype(x.dtype)
```

## Testing Strategy

### 1. Unit Tests

```python
def test_rmsnorm_shapes():
    """Test that RMSNorm preserves input shapes."""
    test_shapes = [
        (32, 768),           # 2D input
        (4, 128, 768),       # 3D input (batch, seq, hidden)
        (2, 8, 64, 768),     # 4D input
    ]
    
    for shape in test_shapes:
        x = mx.random.normal(shape)
        norm = nn.RMSNorm(shape[-1])
        output = norm(x)
        assert output.shape == shape

def test_rmsnorm_normalization():
    """Test that output has unit RMS."""
    batch_size, seq_len, hidden_dim = 2, 10, 64
    x = mx.random.normal((batch_size, seq_len, hidden_dim))
    
    norm = nn.RMSNorm(hidden_dim)
    output = norm(x)
    
    # Compute RMS of output
    rms = mx.sqrt(mx.mean(output**2, axis=-1))
    
    # Should be close to 1 (accounting for learnable weight)
    expected = mx.ones_like(rms)
    assert mx.allclose(rms, expected, atol=1e-3)
```

### 2. Numerical Accuracy Tests

```python
def test_numerical_stability():
    """Test RMSNorm with extreme values."""
    hidden_dim = 64
    norm = nn.RMSNorm(hidden_dim)
    
    # Test with very small values
    x_small = mx.full((2, 10, hidden_dim), 1e-8)
    output_small = norm(x_small)
    assert not mx.any(mx.isnan(output_small))
    assert not mx.any(mx.isinf(output_small))
    
    # Test with very large values
    x_large = mx.full((2, 10, hidden_dim), 1e8)
    output_large = norm(x_large)
    assert not mx.any(mx.isnan(output_large))
    assert not mx.any(mx.isinf(output_large))

def test_gradient_flow():
    """Test gradient flow through RMSNorm."""
    hidden_dim = 64
    norm = nn.RMSNorm(hidden_dim)
    
    def loss_fn(x):
        return mx.sum(norm(x))
    
    x = mx.random.normal((2, 10, hidden_dim))
    grad_fn = mx.grad(loss_fn)
    grad = grad_fn(x)
    
    assert grad.shape == x.shape
    assert not mx.any(mx.isnan(grad))
```

### 3. Performance Benchmarks

```python
import time

def benchmark_rmsnorm():
    """Benchmark RMSNorm performance."""
    configs = [
        (32, 512, 768),    # Typical transformer
        (64, 128, 1024),   # Larger hidden dim
        (16, 1024, 512),   # Longer sequence
    ]
    
    for batch, seq_len, hidden_dim in configs:
        # Create layer and input
        norm = nn.RMSNorm(hidden_dim)
        x = mx.random.normal((batch, seq_len, hidden_dim))
        
        # Warmup
        for _ in range(10):
            _ = norm(x)
        mx.eval(x)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            output = norm(x)
            mx.eval(output)
        elapsed = time.time() - start
        
        print(f"Config {(batch, seq_len, hidden_dim)}: "
              f"{elapsed/100*1000:.2f} ms/iter")
```

## Performance Considerations

### 1. Memory Efficiency
- RMSNorm uses less memory than LayerNorm (no running statistics)
- MLX's implementation is optimized for Apple Silicon
- In-place operations where possible

### 2. Computation Optimization
- Use `mx.rsqrt()` instead of `1/mx.sqrt()` for efficiency
- MLX automatically fuses operations when possible
- Consider using `mx.compile()` for frequently called normalization

### 3. Mixed Precision
- Always compute normalization in float32 for stability
- MLX handles dtype casting efficiently
- No need for manual autocast regions like PyTorch

## Integration Examples

### 1. In Transformer Block

```python
class TransformerBlock(nn.Module):
    """Transformer block with RMSNorm."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def __call__(self, x):
        # Pre-norm architecture
        h = x + self.attention(self.norm1(x))
        out = h + self.feed_forward(self.norm2(h))
        return out
```

### 2. In HRM-Specific Architecture

```python
class HRMBlock(nn.Module):
    """HRM block with RMSNorm."""
    
    def __init__(self, config):
        super().__init__()
        self.high_level = HighLevelModule(config)
        self.low_level = LowLevelModule(config)
        
        # RMSNorm for each module
        self.norm_high = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.norm_low = nn.RMSNorm(config.hidden_size, eps=1e-5)
    
    def __call__(self, x):
        # Hierarchical processing with normalization
        high_out = self.norm_high(self.high_level(x))
        low_out = self.norm_low(self.low_level(high_out))
        return low_out
```

### 3. Standalone Usage

```python
# Simple normalization
hidden_size = 768
norm = nn.RMSNorm(hidden_size)

# Process a batch
x = mx.random.normal((32, 128, hidden_size))
normalized = norm(x)

# With custom epsilon
norm_stable = nn.RMSNorm(hidden_size, eps=1e-6)
normalized_stable = norm_stable(x)
```

## Migration Checklist

- [ ] Decide between MLX built-in vs custom implementation
- [ ] Set appropriate epsilon value (1e-5 or 1e-6)
- [ ] Update all LayerNorm instances to RMSNorm
- [ ] Verify weight initialization (should be ones)
- [ ] Test numerical stability with edge cases
- [ ] Benchmark performance vs PyTorch
- [ ] Validate gradient flow in training
- [ ] Update model config to use `rms_norm_eps` parameter
- [ ] Document any API differences for users

## Common Pitfalls and Solutions

### 1. Epsilon Value
- **Issue**: Different epsilon values can affect training stability
- **Solution**: Use 1e-5 for most cases, 1e-6 for extra stability

### 2. Weight Initialization
- **Issue**: Incorrect initialization can cause gradient issues
- **Solution**: Always initialize weights to ones

### 3. Mixed Precision
- **Issue**: Half precision can cause numerical instability
- **Solution**: Always compute norm in float32, cast result back

### 4. Batch Processing
- **Issue**: Variable batch sizes might cause recompilation
- **Solution**: Use dynamic shapes or pad to fixed sizes

## Conclusion

RMSNorm is a straightforward component to port to MLX, with excellent built-in support. The key advantages:

1. **Simplicity**: MLX provides `nn.RMSNorm` out of the box
2. **Performance**: Optimized for Apple Silicon
3. **Compatibility**: Easy to maintain PyTorch-like interface
4. **Stability**: Proper float32 computation for numerical stability

The implementation requires minimal code changes and should provide equivalent or better performance compared to the PyTorch version.