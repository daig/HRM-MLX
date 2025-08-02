# MLX Attention Implementation Plan for HRM

## Executive Summary

This document outlines the implementation plan for porting the HRM (Hierarchical Reasoning Model) Attention mechanism from PyTorch with FlashAttention to MLX framework. The key objective is to replace FlashAttention with MLX's `scaled_dot_product_attention` while maintaining the model's functionality and performance characteristics.

## Dependencies

This implementation depends on:
- **MLX_ROPE_IMPLEMENTATION_PLAN.md** - For Rotary Position Embeddings implementation
- **MLX_RMSNORM_IMPLEMENTATION_PLAN.md** - For RMSNorm layers used in transformer blocks

## Component Overview

The HRM Attention mechanism is a critical component of the Hierarchical Reasoning Model that implements:
- Non-causal multi-head self-attention for bidirectional reasoning
- Support for both standard MHA (Multi-Head Attention) and GQA (Grouped Query Attention)
- Integration with RoPE (Rotary Position Embeddings) for position encoding
- Efficient memory usage through fused attention computation

### Current Implementation Details
- **Location**: `/models/layers.py` (lines 168-243)
- **Class**: `Attention`
- **Dependencies**: FlashAttention v2/v3, PyTorch, custom layers (CastedLinear)
- **Key Features**:
  - Non-causal attention (bidirectional context)
  - Automatic dtype casting for mixed precision training
  - Single QKV projection for efficiency
  - Support for different numbers of key/value heads (GQA)

## Step-by-Step Implementation Guide

### Step 1: Environment Setup and Dependencies

```python
# Add MLX imports to replace PyTorch components
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
```

### Step 2: Replace CastedLinear with MLX Linear Layer

```python
class MLXLinear(nn.Module):
    """MLX Linear layer with automatic dtype casting support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        # MLX uses different initialization by default
        scale = 1.0 / (in_features ** 0.5)
        self.weight = mx.random.truncated_normal(
            shape=(out_features, in_features),
            dtype=mx.float32,
            lower=-2 * scale,
            upper=2 * scale
        )
        self.bias = mx.zeros((out_features,)) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        # MLX handles dtype casting automatically in most cases
        # but we can explicitly cast if needed
        weight = self.weight.astype(x.dtype)
        bias = self.bias.astype(x.dtype) if self.bias is not None else None
        return mx.matmul(x, weight.T) + (bias if bias is not None else 0)
```

### Step 3: Implement MLX Attention Class

```python
class MLXAttention(nn.Module):
    """Multi-head attention using MLX's scaled_dot_product_attention."""
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Single projection for Q, K, V
        self.qkv_proj = MLXLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        
        # Output projection
        self.o_proj = MLXLinear(self.output_size, self.hidden_size, bias=False)
    
    def __call__(self, cos_sin: tuple, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through attention layer.
        
        Args:
            cos_sin: Precomputed RoPE values (cos, sin) or None
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        
        # Reshape to separate heads
        qkv = qkv.reshape(
            batch_size, seq_len, 
            self.num_heads + 2 * self.num_key_value_heads, 
            self.head_dim
        )
        
        # Split into Q, K, V
        query = qkv[:, :, :self.num_heads, :]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads, :]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:, :]
        
        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            # Use apply_rotary_pos_emb from MLX_ROPE_IMPLEMENTATION_PLAN.md
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Reshape for MLX's scaled_dot_product_attention
        # MLX expects: [batch, num_heads, seq_len, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Apply scaled dot product attention
        # For non-causal attention, mask=None
        mask = "causal" if self.causal else None
        
        attn_output = mx.fast.scaled_dot_product_attention(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            mask=mask
        )
        
        # Reshape back to [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        # Flatten heads
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        
        # Output projection
        return self.o_proj(attn_output)
```

### Step 4: Integrate RoPE from Dedicated Implementation

The Rotary Position Embeddings implementation is provided in `MLX_ROPE_IMPLEMENTATION_PLAN.md`. Import and use:

```python
# Import from the RoPE implementation
from mlx_rope import RotaryEmbedding, apply_rotary_pos_emb

# In the attention initialization
self.rope = RotaryEmbedding(
    dim=self.head_dim,
    max_position_embeddings=max_position_embeddings,
    base=rope_theta
)

# In the forward pass
cos, sin = self.rope(seq_len, dtype=hidden_states.dtype)
query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

### Step 5: Handle GQA (Grouped Query Attention)

MLX's `scaled_dot_product_attention` natively supports GQA without pre-tiling. The implementation above already handles this correctly by:
- Keeping separate `num_heads` and `num_key_value_heads` parameters
- Not expanding K and V to match Q's head count
- Letting MLX handle the broadcasting internally

## Key Differences Between PyTorch and MLX Implementations

### 1. **Tensor Operations**
- **PyTorch**: Uses `torch.Tensor`, `torch.cat`, `torch.chunk`, etc.
- **MLX**: Uses `mx.array`, `mx.concatenate`, array slicing

### 2. **Linear Layers**
- **PyTorch**: Custom `CastedLinear` with manual dtype casting
- **MLX**: Built-in `nn.Linear` or custom `MLXLinear` with simpler casting

### 3. **Attention Computation**
- **PyTorch**: FlashAttention with `flash_attn_func`
- **MLX**: `mx.fast.scaled_dot_product_attention` with similar interface

### 4. **Shape Conventions**
- **PyTorch FlashAttention**: Expects [batch, seq_len, num_heads, head_dim]
- **MLX**: Expects [batch, num_heads, seq_len, head_dim]

### 5. **Masking**
- **PyTorch**: Causal flag in `flash_attn_func`
- **MLX**: String "causal" or array mask in `scaled_dot_product_attention`

### 6. **Memory Management**
- **PyTorch**: Explicit `.detach()`, `.requires_grad`
- **MLX**: Automatic gradient tracking with `mx.grad`

## Testing Strategy

### 1. **Unit Tests for Components**

```python
def test_attention_shapes():
    """Test that attention produces correct output shapes."""
    batch_size, seq_len, hidden_size = 2, 128, 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )
    
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(None, x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)

def test_gqa_attention():
    """Test grouped query attention with fewer KV heads."""
    batch_size, seq_len, hidden_size = 2, 128, 512
    num_heads = 8
    num_kv_heads = 2  # GQA with 4:1 ratio
    head_dim = hidden_size // num_heads
    
    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        causal=False
    )
    
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(None, x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
```

### 2. **Numerical Validation**

```python
def test_attention_numerical():
    """Compare MLX attention with reference implementation."""
    # Create small test case
    batch_size, seq_len, hidden_size = 1, 4, 64
    num_heads = 4
    head_dim = hidden_size // num_heads
    
    # Initialize with same weights
    attn_mlx = MLXAttention(...)
    attn_ref = ReferenceAttention(...)  # Simple numpy implementation
    
    # Compare outputs
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output_mlx = attn_mlx(None, x)
    output_ref = attn_ref(None, x)
    
    assert mx.allclose(output_mlx, output_ref, atol=1e-5)
```

### 3. **Integration Tests**

- Test full HRM block with MLX attention
- Verify gradient flow through attention layers
- Test with RoPE position embeddings
- Validate ACT (Adaptive Computation Time) compatibility

### 4. **Performance Benchmarks**

```python
def benchmark_attention():
    """Benchmark MLX attention performance."""
    configs = [
        (32, 512, 768, 12, 12),   # Standard MHA
        (32, 512, 768, 12, 3),    # GQA 4:1
        (16, 1024, 1024, 16, 16), # Larger model
    ]
    
    for batch, seq_len, hidden, n_heads, n_kv_heads in configs:
        attn = MLXAttention(...)
        x = mx.random.normal((batch, seq_len, hidden))
        
        # Warm up
        for _ in range(10):
            _ = attn(None, x)
        mx.eval(x)
        
        # Time execution
        start = time.time()
        for _ in range(100):
            output = attn(None, x)
            mx.eval(output)
        elapsed = time.time() - start
        
        print(f"Config {config}: {elapsed:.3f}s for 100 iterations")
```

## Performance Considerations

### 1. **Memory Efficiency**
- MLX's `scaled_dot_product_attention` uses similar memory optimizations as FlashAttention
- No need to materialize full attention matrix for long sequences
- Automatic fusion of operations for better memory bandwidth utilization

### 2. **Dtype Handling**
- MLX automatically handles mixed precision in many cases
- Softmax in attention is computed in float32 for stability
- Consider matching mask dtype to input dtype for better performance

### 3. **Compilation and Optimization**
- MLX uses lazy evaluation and JIT compilation
- Use `mx.eval()` to force evaluation when benchmarking
- Consider using `mx.compile()` decorator for frequently called functions

### 4. **Batch Processing**
- MLX is optimized for larger batch sizes
- Consider batching strategies for inference
- Use `mx.vmap` for efficient batch operations when applicable

## Example Usage

### Basic Attention Usage

```python
# Initialize model components
hidden_size = 768
num_heads = 12
head_dim = hidden_size // num_heads

attention = MLXAttention(
    hidden_size=hidden_size,
    head_dim=head_dim,
    num_heads=num_heads,
    num_key_value_heads=num_heads,  # Standard MHA
    causal=False  # Non-causal for bidirectional attention
)

# Example forward pass
batch_size, seq_len = 4, 256
x = mx.random.normal((batch_size, seq_len, hidden_size))

# Without position embeddings
output = attention(None, x)

# With RoPE (using implementation from MLX_ROPE_IMPLEMENTATION_PLAN.md)
from mlx_rope import RotaryEmbedding
rope = RotaryEmbedding(head_dim, max_position_embeddings=512, base=10000)
cos, sin = rope(seq_len)
output = attention((cos, sin), x)
```

### GQA (Grouped Query Attention) Usage

```python
# GQA with 4:1 ratio (4 query heads per KV head)
attention_gqa = MLXAttention(
    hidden_size=hidden_size,
    head_dim=head_dim,
    num_heads=16,
    num_key_value_heads=4,  # GQA
    causal=False
)

output = attention_gqa(None, x)
```

### Integration with HRM Block

```python
class MLXHRMBlock(nn.Module):
    """HRM transformer block with MLX attention."""
    
    def __init__(self, config):
        super().__init__()
        self.self_attn = MLXAttention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = MLXSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )
        self.norm_eps = config.rms_norm_eps
    
    def __call__(self, cos_sin, hidden_states):
        # Self-attention with residual
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin, hidden_states),
            self.norm_eps
        )
        
        # MLP with residual
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            self.norm_eps
        )
        
        return hidden_states
```

## Migration Checklist

- [ ] Set up MLX development environment
- [ ] Implement MLXLinear layer with dtype casting
- [ ] Implement MLXAttention class
- [ ] Import RoPE implementation from MLX_ROPE_IMPLEMENTATION_PLAN.md
- [ ] Import RMSNorm implementation from MLX_RMSNORM_IMPLEMENTATION_PLAN.md
- [ ] Import SwiGLU implementation from MLX_SWIGLU_IMPLEMENTATION_PLAN.md
- [ ] Create unit tests for attention shapes
- [ ] Create numerical validation tests
- [ ] Implement GQA support tests
- [ ] Integrate attention into HRM block
- [ ] Run performance benchmarks
- [ ] Validate gradient flow
- [ ] Test with full HRM model
- [ ] Document API differences
- [ ] Create migration guide for existing code

## Conclusion

This implementation plan provides a comprehensive approach to porting the HRM Attention mechanism from PyTorch with FlashAttention to MLX. The key advantages of this migration include:

1. **Simplified API**: MLX's `scaled_dot_product_attention` provides a cleaner interface
2. **Native GQA Support**: No manual tiling needed for grouped query attention
3. **Automatic Optimizations**: MLX handles many optimizations automatically
4. **Unified Framework**: Easier deployment on Apple Silicon with Metal acceleration

The implementation maintains all critical features of the original HRM attention while leveraging MLX's optimized primitives for efficient execution.