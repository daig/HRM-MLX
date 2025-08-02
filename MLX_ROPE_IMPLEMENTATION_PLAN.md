# MLX RoPE Implementation Plan for HRM

## Executive Summary

This document provides the implementation plan for porting Rotary Position Embeddings (RoPE) from PyTorch to MLX. RoPE is a critical position encoding mechanism in the HRM architecture that enables the model to understand both absolute and relative positions without requiring learned position embeddings.

## Component Overview

### What is RoPE?

RoPE (Rotary Position Embeddings) encodes absolute positions using rotation matrices and naturally captures relative position information. It works by rotating query and key vectors in the complex plane based on their positions, allowing the model to understand sequential relationships.

**Key Features:**
- No learnable parameters (deterministic computation)
- Extrapolates to longer sequences than trained on
- Preserves inner product relationships
- Computationally efficient
- Natural encoding of relative positions

### How RoPE Works

1. Split feature dimensions into pairs
2. Treat each pair as a complex number
3. Rotate by angle proportional to position
4. Rotation frequency varies by dimension (higher dims rotate slower)

### Current Implementation Details
- **Location**: Used in attention layers throughout HRM
- **PyTorch Implementation**: Precomputed cos/sin cache with apply function
- **Integration**: Applied to Q and K vectors before attention computation

## Mathematical Formulation

For position `m` and dimension pair `i`:

### Frequency Calculation
```
θ_i = base^(-2i/d)
```
Where `base` is typically 10000 and `d` is the head dimension.

### Rotation Angle
```
φ_{m,i} = m * θ_i
```

### Rotation Application
For query/key vector `x = [x_0, x_1, ..., x_{d-1}]`, split into pairs and rotate:
```
x'_{2i} = x_{2i} * cos(φ_{m,i}) - x_{2i+1} * sin(φ_{m,i})
x'_{2i+1} = x_{2i} * sin(φ_{m,i}) + x_{2i+1} * cos(φ_{m,i})
```

This is equivalent to complex multiplication:
```
(x_{2i} + ix_{2i+1}) * e^{iφ_{m,i}}
```

## Implementation

```python
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for MLX.
    
    Implements the rotary position embedding mechanism that encodes
    absolute positions while naturally capturing relative position information.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_type: str = "default"
    ):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimension of the embeddings (head_dim)
            max_position_embeddings: Maximum sequence length to cache
            base: Base for the geometric progression
            scaling_factor: Scale positions for length extrapolation
            rope_type: Type of RoPE ('default' or 'linear_scaling')
        """
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        
        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.inv_freq = inv_freq
        
        # Precompute and cache cos/sin values
        self._set_cos_sin_cache()
    
    def _compute_inv_freq(self) -> mx.array:
        """Compute the inverse frequencies for rotary embeddings."""
        # Standard RoPE frequencies
        inv_freq = 1.0 / (self.base ** (
            mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        ))
        return inv_freq
    
    def _set_cos_sin_cache(self):
        """Precompute and cache cosine and sine values."""
        # Create position indices
        t = mx.arange(self.max_position_embeddings, dtype=mx.float32)
        
        # Apply scaling for length extrapolation
        if self.rope_type == "linear_scaling":
            t = t / self.scaling_factor
        
        # Compute frequencies for each position
        # Shape: [max_position_embeddings, dim//2]
        freqs = mx.outer(t, self.inv_freq)
        
        # Different reshaping strategy than original paper but equivalent
        # This makes it easier to apply rotations
        # Shape: [max_position_embeddings, dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        
        # Cache cosine and sine values
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)
    
    def __call__(
        self, 
        seq_len: Optional[int] = None,
        dtype: Optional[mx.Dtype] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Get cosine and sine values for the given sequence length.
        
        Args:
            seq_len: Sequence length (uses cached length if None)
            dtype: Output dtype (uses cached dtype if None)
            
        Returns:
            Tuple of (cos, sin) arrays with shape [seq_len, dim]
        """
        if seq_len is None:
            cos, sin = self.cos_cached, self.sin_cached
        else:
            # Return only up to seq_len
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        # Cast to requested dtype if specified
        if dtype is not None:
            cos = cos.astype(dtype)
            sin = sin.astype(dtype)
            
        return cos, sin
    
    def extend_cache(self, max_position_embeddings: int):
        """Extend the cached positions if needed."""
        if max_position_embeddings <= self.max_position_embeddings:
            return
            
        # Update max position embeddings
        old_max = self.max_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        
        # Compute new positions
        t_new = mx.arange(old_max, max_position_embeddings, dtype=mx.float32)
        if self.rope_type == "linear_scaling":
            t_new = t_new / self.scaling_factor
            
        # Compute new frequencies
        freqs_new = mx.outer(t_new, self.inv_freq)
        emb_new = mx.concatenate([freqs_new, freqs_new], axis=-1)
        
        # Extend cache
        self.cos_cached = mx.concatenate([
            self.cos_cached, 
            mx.cos(emb_new)
        ], axis=0)
        self.sin_cached = mx.concatenate([
            self.sin_cached, 
            mx.sin(emb_new)
        ], axis=0)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor with x2 negated and swapped with x1
    """
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, 
    k: mx.array, 
    cos: mx.array, 
    sin: mx.array,
    unsqueeze_dim: int = -2
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
    
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Store original dtype for restoration
    orig_dtype = q.dtype
    
    # Cast to float32 for numerical stability
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    
    # Expand cos/sin for broadcasting with q/k
    # cos/sin: [seq_len, head_dim] -> [seq_len, 1, head_dim]
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    
    # Apply rotation using the rotation matrix formula
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # Cast back to original dtype
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


# Alternative implementation using complex numbers (more elegant but potentially slower)
def apply_rotary_pos_emb_complex(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE using complex number representation."""
    # Reshape to complex pairs
    q_complex = q.reshape(*q.shape[:-1], -1, 2)
    k_complex = k.reshape(*k.shape[:-1], -1, 2)
    
    # Create complex rotation
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    cos_complex = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_complex = sin.reshape(*sin.shape[:-1], -1, 2)
    
    # Apply complex multiplication
    # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    q_real = q_complex[..., 0] * cos_complex[..., 0] - q_complex[..., 1] * sin_complex[..., 0]
    q_imag = q_complex[..., 0] * sin_complex[..., 0] + q_complex[..., 1] * cos_complex[..., 0]
    
    k_real = k_complex[..., 0] * cos_complex[..., 0] - k_complex[..., 1] * sin_complex[..., 0]
    k_imag = k_complex[..., 0] * sin_complex[..., 0] + k_complex[..., 1] * cos_complex[..., 0]
    
    # Stack and reshape back
    q_rot = mx.stack([q_real, q_imag], axis=-1).reshape(*q.shape)
    k_rot = mx.stack([k_real, k_imag], axis=-1).reshape(*k.shape)
    
    return q_rot, k_rot
```

## Advanced RoPE Variants

### 1. Dynamic NTK-Aware Scaling

```python
class DynamicNTKRoPE(RotaryEmbedding):
    """RoPE with dynamic NTK-aware scaling for length extrapolation."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, 
                 base: float = 10000.0, scaling_alpha: float = 1.0):
        self.scaling_alpha = scaling_alpha
        super().__init__(dim, max_position_embeddings, base)
    
    def _compute_inv_freq(self) -> mx.array:
        """Compute inverse frequencies with NTK-aware scaling."""
        if self.scaling_alpha != 1.0:
            base = self.base * self.scaling_alpha ** (self.dim / (self.dim - 2))
        else:
            base = self.base
            
        return 1.0 / (base ** (
            mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        ))
```

### 2. 2D RoPE for Vision Tasks

```python
class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for image transformers."""
    
    def __init__(self, dim: int, max_size: int = 224):
        super().__init__()
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        self.dim = dim
        self.max_size = max_size
        
        # Half dimensions for each spatial axis
        half_dim = dim // 2
        self.rope_h = RotaryEmbedding(half_dim, max_size)
        self.rope_w = RotaryEmbedding(half_dim, max_size)
    
    def __call__(self, h: int, w: int, dtype: Optional[mx.Dtype] = None):
        """Get 2D positional embeddings."""
        cos_h, sin_h = self.rope_h(h, dtype)
        cos_w, sin_w = self.rope_w(w, dtype)
        
        # Broadcast and concatenate
        cos_h = mx.expand_dims(cos_h, axis=1)  # [h, 1, dim//2]
        sin_h = mx.expand_dims(sin_h, axis=1)
        cos_w = mx.expand_dims(cos_w, axis=0)  # [1, w, dim//2]
        sin_w = mx.expand_dims(sin_w, axis=0)
        
        cos = mx.concatenate([
            mx.broadcast_to(cos_h, (h, w, self.dim//2)),
            mx.broadcast_to(cos_w, (h, w, self.dim//2))
        ], axis=-1)
        
        sin = mx.concatenate([
            mx.broadcast_to(sin_h, (h, w, self.dim//2)),
            mx.broadcast_to(sin_w, (h, w, self.dim//2))
        ], axis=-1)
        
        return cos.reshape(h * w, self.dim), sin.reshape(h * w, self.dim)
```

## Testing Strategy

### 1. Unit Tests

```python
def test_rope_basic():
    """Test basic RoPE functionality."""
    dim = 64
    seq_len = 128
    
    rope = RotaryEmbedding(dim, max_position_embeddings=256)
    cos, sin = rope(seq_len)
    
    # Check shapes
    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)
    
    # Check values are in valid range
    assert mx.all(mx.abs(cos) <= 1.0)
    assert mx.all(mx.abs(sin) <= 1.0)
    
    # Check orthogonality (cos² + sin² = 1)
    magnitude = cos**2 + sin**2
    expected = mx.ones_like(magnitude)
    assert mx.allclose(magnitude, expected, atol=1e-6)

def test_rotation_properties():
    """Test that rotation preserves magnitude and relative positions."""
    batch, seq_len, num_heads, head_dim = 2, 64, 8, 64
    
    # Create test tensors
    q = mx.random.normal((batch, seq_len, num_heads, head_dim))
    k = mx.random.normal((batch, seq_len, num_heads, head_dim))
    
    # Apply RoPE
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    # Check magnitude preservation
    q_magnitude = mx.sqrt(mx.sum(q**2, axis=-1))
    q_rot_magnitude = mx.sqrt(mx.sum(q_rot**2, axis=-1))
    assert mx.allclose(q_magnitude, q_rot_magnitude, rtol=1e-5)
    
    # Check shape preservation
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
```

### 2. Position Encoding Tests

```python
def test_relative_position_encoding():
    """Test that RoPE encodes relative positions correctly."""
    head_dim = 64
    rope = RotaryEmbedding(head_dim)
    
    # Get embeddings for different positions
    cos1, sin1 = rope(10)
    cos2, sin2 = rope(20)
    
    # Create dummy queries at different positions
    q1 = mx.ones((1, 1, 1, head_dim))
    q2 = mx.ones((1, 1, 1, head_dim))
    
    # Apply rotations for positions 5 and 15
    q1_rot, _ = apply_rotary_pos_emb(q1, q1, cos1[5:6], sin1[5:6])
    q2_rot, _ = apply_rotary_pos_emb(q2, q2, cos2[15:16], sin2[15:16])
    
    # The relative position (15-5=10) should be encoded
    # This is a property of RoPE that we can verify
    relative_encoding = mx.sum(q1_rot * q2_rot, axis=-1)
    
    # Compare with direct relative position 10
    q_rel, _ = apply_rotary_pos_emb(q1, q1, cos1[10:11], sin1[10:11])
    expected = mx.sum(q1 * q_rel, axis=-1)
    
    assert mx.allclose(relative_encoding, expected, rtol=1e-4)

def test_extrapolation():
    """Test RoPE extrapolation to longer sequences."""
    head_dim = 64
    max_train_len = 512
    
    # Create RoPE with limited training length
    rope = RotaryEmbedding(head_dim, max_position_embeddings=max_train_len)
    
    # Extend to longer sequence
    test_len = 1024
    rope.extend_cache(test_len)
    
    # Get embeddings
    cos, sin = rope(test_len)
    
    # Verify extended positions
    assert cos.shape[0] == test_len
    assert sin.shape[0] == test_len
    
    # Check that original positions match
    cos_orig, sin_orig = rope(max_train_len)
    assert mx.allclose(cos[:max_train_len], cos_orig)
    assert mx.allclose(sin[:max_train_len], sin_orig)
```

### 3. Performance Benchmarks

```python
import time

def benchmark_rope():
    """Benchmark RoPE performance."""
    configs = [
        (32, 512, 12, 64),    # Small model
        (32, 1024, 16, 128),  # Medium model
        (16, 2048, 20, 128),  # Large sequence
    ]
    
    for batch, seq_len, num_heads, head_dim in configs:
        # Initialize RoPE
        rope = RotaryEmbedding(head_dim, max_position_embeddings=seq_len)
        
        # Create tensors
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        
        # Warmup
        for _ in range(10):
            cos, sin = rope(seq_len)
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        mx.eval(q_rot)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            cos, sin = rope(seq_len)
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            mx.eval(q_rot)
        elapsed = time.time() - start
        
        print(f"Config (B={batch}, L={seq_len}, H={num_heads}, D={head_dim}): "
              f"{elapsed/100*1000:.2f} ms/iter")
```

### 4. Integration Tests

```python
def test_rope_in_attention():
    """Test RoPE integration with attention mechanism."""
    from mlx_attention import MLXAttention  # Assumes attention is implemented
    
    batch, seq_len, hidden_size = 4, 128, 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # Create attention with RoPE
    attention = MLXAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim
    )
    
    # Create RoPE
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(seq_len)
    
    # Forward pass
    x = mx.random.normal((batch, seq_len, hidden_size))
    output = attention(x, rope_cos_sin=(cos, sin))
    
    assert output.shape == (batch, seq_len, hidden_size)
    assert not mx.any(mx.isnan(output))
```

## Performance Considerations

### 1. Caching Strategy
- Precompute cos/sin for maximum expected sequence length
- Cache in appropriate dtype to avoid repeated casting
- Extend cache dynamically only when needed

### 2. Memory Optimization
```python
# Share RoPE instance across layers
rope = RotaryEmbedding(head_dim, max_position_embeddings=2048)

# Use in multiple attention layers
for layer in attention_layers:
    layer.rope = rope  # Share the same instance
```

### 3. Computation Optimization
- Use `rotate_half` implementation (faster than complex numbers)
- Minimize dtype conversions
- Leverage MLX's automatic operation fusion

### 4. Batching Considerations
- RoPE computation is position-dependent but batch-independent
- Can broadcast efficiently across batch dimension
- No need to recompute for different batch sizes

## Integration Guide

### 1. With Attention Module

```python
class AttentionWithRoPE(nn.Module):
    """Attention module with integrated RoPE."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Initialize projections
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False
        )
        
        # Initialize RoPE
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
    
    def __call__(self, x, attention_mask=None):
        batch, seq_len, _ = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(seq_len, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Continue with attention computation...
        return self._compute_attention(q, k, v, attention_mask)
```

### 2. Configuration Management

```python
@dataclass
class RoPEConfig:
    """Configuration for RoPE."""
    rope_type: str = "default"
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 2048
    
    def create_rope(self, dim: int) -> RotaryEmbedding:
        """Create RoPE instance from config."""
        kwargs = {
            "dim": dim,
            "max_position_embeddings": self.max_position_embeddings,
            "base": self.rope_theta,
        }
        
        if self.rope_scaling is not None:
            kwargs["scaling_factor"] = self.rope_scaling.get("factor", 1.0)
            kwargs["rope_type"] = self.rope_scaling.get("type", "linear_scaling")
            
        return RotaryEmbedding(**kwargs)
```

## Common Issues and Solutions

### 1. Dimension Mismatch
- **Issue**: RoPE dimension doesn't match head dimension
- **Solution**: Ensure `rope_dim = head_dim` in initialization

### 2. Position Overflow
- **Issue**: Sequence longer than cached positions
- **Solution**: Use `extend_cache()` method or initialize with larger max

### 3. Numerical Precision
- **Issue**: Gradients become unstable with long sequences
- **Solution**: Compute in float32, use proper scaling factors

### 4. Memory Usage
- **Issue**: Large cache for very long sequences
- **Solution**: Share RoPE instances, use dynamic caching

## Conclusion

RoPE is an elegant and efficient position encoding mechanism that's crucial for modern transformers. The MLX implementation provides:

1. **Efficiency**: Precomputed caches and optimized rotations
2. **Flexibility**: Support for various RoPE variants and scaling methods
3. **Simplicity**: Clean API that integrates easily with attention
4. **Performance**: Leverages MLX's strengths for Apple Silicon

The key to successful implementation is proper caching strategy and careful handling of dimensions and dtypes. With these considerations, RoPE in MLX should match or exceed PyTorch performance while maintaining numerical accuracy.