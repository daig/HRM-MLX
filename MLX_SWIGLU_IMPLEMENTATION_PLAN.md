# MLX SwiGLU Implementation Plan for HRM

## Executive Summary

This document provides the implementation plan for porting SwiGLU (Swish-Gated Linear Unit) from PyTorch to MLX. SwiGLU is a critical activation function used in the feed-forward networks of the HRM architecture, providing superior performance compared to standard ReLU-based FFNs.

## Component Overview

### What is SwiGLU?

SwiGLU is a gated activation function that combines the Swish (SiLU) activation with a gating mechanism. It splits the input into two paths (gate and up), applies Swish activation to the gate, and multiplies it element-wise with the up projection before a final down projection.

**Architecture:**
- Input → Linear projection to 2×intermediate_dim
- Split into gate and up components
- Apply SiLU (Swish) to gate: `x * sigmoid(x)`
- Element-wise multiply: `SiLU(gate) * up`
- Down projection to original dimension

**Key Benefits:**
- Better gradient flow than ReLU
- Smoother activation landscape
- Improved model capacity
- State-of-the-art performance in transformers

### Current Implementation Details
- **Location**: Feed-forward networks in HRM transformer blocks
- **PyTorch Implementation**: Custom module with fused gate/up projection
- **Key Features**:
  - Intermediate dimension calculation with multiple-of constraint
  - Efficient single projection for gate and up
  - No bias terms (following modern transformer practices)

## Mathematical Formulation

Given input `x` with hidden dimension `d`:

```
# Intermediate dimension calculation
intermediate_dim = int(d * expansion_factor * 2/3)
intermediate_dim = round_to_multiple(intermediate_dim, multiple_of)

# Projections
gate_up = xW_{gate_up}  # Shape: [batch, seq, 2 * intermediate_dim]
gate, up = split(gate_up, 2, dim=-1)

# SiLU activation (Swish)
SiLU(x) = x * sigmoid(x)

# SwiGLU operation
h = SiLU(gate) ⊙ up  # Element-wise multiplication

# Down projection
output = hW_{down}  # Shape: [batch, seq, d]
```

The 2/3 factor accounts for the parameter increase from having both gate and up projections while maintaining similar total parameters to a standard FFN.

## Implementation

```python
import mlx.core as mx
import mlx.nn as nn

def _find_multiple(n: int, multiple: int) -> int:
    """Find the smallest multiple of 'multiple' that is >= n."""
    return (n + multiple - 1) // multiple * multiple

class SwiGLU(nn.Module):
    """SwiGLU activation function for MLX.
    
    Implements the SwiGLU activation as used in modern transformers.
    Combines gating with Swish (SiLU) activation for improved performance.
    """
    
    def __init__(
        self, 
        dim: int, 
        expansion_factor: float = 4.0, 
        multiple_of: int = 256,
        bias: bool = False
    ):
        """
        Initialize SwiGLU module.
        
        Args:
            dim: Input/output dimension
            expansion_factor: Expansion factor for intermediate dimension
            multiple_of: Round intermediate dimension to multiple of this
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        # Calculate intermediate dimension
        # 2/3 factor accounts for the gating overhead
        intermediate_dim = int(dim * expansion_factor * 2 / 3)
        intermediate_dim = _find_multiple(intermediate_dim, multiple_of)
        
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        
        # Combined gate and up projection for efficiency
        self.gate_up_proj = nn.Linear(dim, intermediate_dim * 2, bias=bias)
        
        # Down projection
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Standard deviation for initialization
        std = (2.0 / (5 * self.dim)) ** 0.5
        
        # Initialize gate_up projection
        nn.init.normal(self.gate_up_proj.weight, std=std)
        
        # Initialize down projection with smaller std
        nn.init.normal(self.down_proj.weight, std=std / (2 ** 0.5))
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through SwiGLU.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Output tensor of shape [..., dim]
        """
        # Project to gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split into gate and up components
        gate, up = mx.split(gate_up, 2, axis=-1)
        
        # Apply SiLU (Swish) activation to gate
        # SiLU(x) = x * sigmoid(x)
        gate_activated = gate * mx.sigmoid(gate)
        
        # Element-wise multiplication with up
        hidden = gate_activated * up
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output
```

## Key Implementation Details

### 1. Intermediate Dimension Calculation

```python
def calculate_intermediate_dim(dim: int, expansion_factor: float, multiple_of: int) -> int:
    """Calculate the intermediate dimension for SwiGLU."""
    # Account for gate overhead with 2/3 factor
    intermediate = int(dim * expansion_factor * 2 / 3)
    
    # Round to multiple for hardware efficiency
    return _find_multiple(intermediate, multiple_of)
```

### 2. Efficient Projection Strategy

Instead of separate gate and up projections:
```python
# Inefficient: Two separate projections
self.gate_proj = nn.Linear(dim, intermediate_dim)
self.up_proj = nn.Linear(dim, intermediate_dim)
gate = self.gate_proj(x)
up = self.up_proj(x)
```

We use a single fused projection:
```python
# Efficient: Single projection then split
self.gate_up_proj = nn.Linear(dim, intermediate_dim * 2)
gate_up = self.gate_up_proj(x)
gate, up = mx.split(gate_up, 2, axis=-1)
```

### 3. SiLU (Swish) Activation

```python
def silu(x: mx.array) -> mx.array:
    """SiLU (Swish) activation function."""
    return x * mx.sigmoid(x)
```

## Testing Strategy

### 1. Unit Tests

```python
def test_swiglu_shapes():
    """Test that SwiGLU preserves the input dimension."""
    test_configs = [
        (512, 4.0, 256),    # Standard config
        (768, 4.0, 256),    # Larger dimension
        (256, 8.0, 128),    # Higher expansion
    ]
    
    for dim, expansion, multiple in test_configs:
        swiglu = SwiGLU(dim, expansion, multiple)
        
        # Test different input shapes
        for shape in [(32, dim), (4, 128, dim), (2, 8, 64, dim)]:
            x = mx.random.normal(shape)
            output = swiglu(x)
            assert output.shape == shape
            
def test_intermediate_dimension():
    """Test intermediate dimension calculation."""
    swiglu = SwiGLU(dim=768, expansion_factor=4.0, multiple_of=256)
    
    # Check intermediate dimension
    expected = _find_multiple(int(768 * 4.0 * 2 / 3), 256)
    assert swiglu.intermediate_dim == expected
    
    # Verify projection shapes
    assert swiglu.gate_up_proj.weight.shape == (expected * 2, 768)
    assert swiglu.down_proj.weight.shape == (768, expected)
```

### 2. Numerical Behavior Tests

```python
def test_swiglu_activation():
    """Test SwiGLU activation behavior."""
    dim = 64
    swiglu = SwiGLU(dim, expansion_factor=4.0)
    
    # Test with zeros (should produce near-zero output)
    x_zeros = mx.zeros((2, 10, dim))
    output_zeros = swiglu(x_zeros)
    assert mx.max(mx.abs(output_zeros)) < 0.1
    
    # Test gradient flow
    def loss_fn(x):
        return mx.sum(swiglu(x) ** 2)
    
    x = mx.random.normal((2, 10, dim))
    grad_fn = mx.grad(loss_fn)
    grad = grad_fn(x)
    
    assert grad.shape == x.shape
    assert not mx.any(mx.isnan(grad))
    assert mx.max(mx.abs(grad)) > 0  # Non-zero gradients

def test_gating_mechanism():
    """Test that gating mechanism works correctly."""
    dim = 64
    swiglu = SwiGLU(dim)
    
    # Create input where we can track the gating effect
    x = mx.ones((1, 1, dim))
    
    # Get gate and up values manually
    gate_up = swiglu.gate_up_proj(x)
    gate, up = mx.split(gate_up, 2, axis=-1)
    gate_activated = gate * mx.sigmoid(gate)
    
    # The gating should modulate the signal
    assert not mx.allclose(gate_activated, up)
```

### 3. Performance Benchmarks

```python
import time

def benchmark_swiglu():
    """Benchmark SwiGLU performance."""
    configs = [
        (768, 4.0, 256, 32, 512),     # Standard transformer
        (1024, 4.0, 256, 16, 1024),   # Larger model
        (512, 8.0, 128, 64, 256),     # Higher expansion
    ]
    
    for dim, expansion, multiple, batch, seq_len in configs:
        swiglu = SwiGLU(dim, expansion, multiple)
        x = mx.random.normal((batch, seq_len, dim))
        
        # Warmup
        for _ in range(10):
            _ = swiglu(x)
        mx.eval(x)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            output = swiglu(x)
            mx.eval(output)
        elapsed = time.time() - start
        
        # Calculate FLOPS
        intermediate_dim = swiglu.intermediate_dim
        flops_per_token = (
            2 * dim * intermediate_dim * 2 +  # gate_up projection
            2 * intermediate_dim * dim +      # down projection
            intermediate_dim * 3              # activations
        )
        total_flops = flops_per_token * batch * seq_len * 100
        gflops = total_flops / elapsed / 1e9
        
        print(f"Config (dim={dim}, expansion={expansion}): "
              f"{elapsed/100*1000:.2f} ms/iter, {gflops:.1f} GFLOPS")
```

### 4. Comparison Tests

```python
def test_swiglu_vs_mlp():
    """Compare SwiGLU with standard MLP."""
    dim = 512
    expansion = 4.0
    
    # SwiGLU
    swiglu = SwiGLU(dim, expansion)
    
    # Standard MLP with same parameter count
    intermediate_dim = int(dim * expansion)
    mlp = nn.Sequential(
        nn.Linear(dim, intermediate_dim),
        nn.GELU(),
        nn.Linear(intermediate_dim, dim)
    )
    
    # Compare parameter counts
    swiglu_params = sum(p.size for p in [swiglu.gate_up_proj.weight, 
                                          swiglu.down_proj.weight])
    mlp_params = sum(p.size for layer in mlp.layers 
                     for p in [layer.weight] if hasattr(layer, 'weight'))
    
    print(f"SwiGLU parameters: {swiglu_params:,}")
    print(f"MLP parameters: {mlp_params:,}")
    
    # Test forward pass
    x = mx.random.normal((32, 128, dim))
    swiglu_out = swiglu(x)
    mlp_out = mlp(x)
    
    assert swiglu_out.shape == mlp_out.shape
```

## Performance Considerations

### 1. Memory Efficiency
- Fused gate/up projection reduces memory bandwidth
- No intermediate storage for separate projections
- MLX's unified memory architecture benefits from fewer allocations

### 2. Computation Optimization
- Single matrix multiplication for gate/up projection
- MLX automatically fuses element-wise operations
- Consider using `mx.compile()` for static shapes

### 3. Hardware Utilization
- Round intermediate dimensions to multiples of 256 for Apple Silicon
- Larger multiples (256, 512) often give better performance
- Balance between dimension size and memory usage

## Integration Examples

### 1. In Transformer Block

```python
class TransformerBlock(nn.Module):
    """Transformer block with SwiGLU FFN."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(
            dim=config.hidden_size,
            expansion_factor=config.expansion_factor,
            multiple_of=config.multiple_of
        )
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
    
    def __call__(self, x):
        # Attention block
        h = x + self.attention(self.norm1(x))
        # FFN block with SwiGLU
        out = h + self.feed_forward(self.norm2(h))
        return out
```

### 2. With Dropout and Residual

```python
class SwiGLUWithDropout(nn.Module):
    """SwiGLU with dropout for regularization."""
    
    def __init__(self, dim: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.swiglu = SwiGLU(dim, **kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x):
        return self.dropout(self.swiglu(x))
```

### 3. Conditional Computation

```python
class ConditionalSwiGLU(nn.Module):
    """SwiGLU with conditional computation based on routing."""
    
    def __init__(self, dim: int, num_experts: int = 4, **kwargs):
        super().__init__()
        self.experts = [SwiGLU(dim, **kwargs) for _ in range(num_experts)]
        self.router = nn.Linear(dim, num_experts)
    
    def __call__(self, x):
        # Compute routing weights
        routes = mx.softmax(self.router(x), axis=-1)
        
        # Apply experts
        output = mx.zeros_like(x)
        for i, expert in enumerate(self.experts):
            weight = routes[..., i:i+1]
            output = output + weight * expert(x)
        
        return output
```

## Migration Checklist

- [ ] Calculate appropriate intermediate dimensions
- [ ] Choose suitable `multiple_of` value (256 recommended)
- [ ] Decide on expansion factor (typically 4.0)
- [ ] Implement weight initialization strategy
- [ ] Test shape preservation across various inputs
- [ ] Verify gradient flow through the module
- [ ] Benchmark against PyTorch implementation
- [ ] Test memory usage patterns
- [ ] Validate numerical stability
- [ ] Document any parameter changes

## Common Pitfalls and Solutions

### 1. Intermediate Dimension
- **Issue**: Wrong intermediate dimension calculation
- **Solution**: Always use the 2/3 factor for SwiGLU

### 2. Activation Function
- **Issue**: Using wrong activation (ReLU instead of SiLU)
- **Solution**: Ensure using `x * mx.sigmoid(x)` for SiLU

### 3. Parameter Count
- **Issue**: Parameter explosion with high expansion factors
- **Solution**: Monitor total parameters, adjust expansion factor

### 4. Numerical Stability
- **Issue**: Gradient vanishing/explosion
- **Solution**: Proper weight initialization and normalization

## Optimization Tips

### 1. Compile for Static Shapes
```python
@mx.compile
def optimized_swiglu_forward(swiglu, x):
    return swiglu(x)
```

### 2. Batch Processing
```python
# Process multiple sequences efficiently
batch_x = mx.concatenate([x1, x2, x3], axis=0)
batch_out = swiglu(batch_x)
out1, out2, out3 = mx.split(batch_out, 3, axis=0)
```

### 3. Memory Preallocation
```python
# For iterative processing
output_buffer = mx.zeros((batch, seq_len, dim))
for i in range(iterations):
    output_buffer = swiglu(input_sequence[i])
```

## Conclusion

SwiGLU is a powerful activation function that significantly improves transformer performance. The MLX implementation:

1. **Efficiency**: Fused projections reduce memory bandwidth
2. **Simplicity**: Clean implementation with minimal dependencies
3. **Performance**: Optimized for Apple Silicon architecture
4. **Flexibility**: Easy to integrate into existing models

The key to successful implementation is maintaining the correct intermediate dimension calculation and ensuring proper weight initialization. With these considerations, SwiGLU in MLX should match or exceed PyTorch performance.