"""Example usage of RMSNorm implementations."""

import mlx.core as mx
import mlx.nn as nn
from mlx_hrm.layers import rms_norm, RMSNorm, create_rms_norm


def main():
    """Demonstrate different ways to use RMSNorm."""
    
    # Configuration
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    
    # Create sample input
    x = mx.random.normal((batch_size, seq_len, hidden_dim))
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    
    print("\n" + "="*50)
    print("1. Functional RMSNorm (matches original HRM)")
    print("="*50)
    
    # Use functional implementation (no learnable parameters)
    output_functional = rms_norm(x, variance_epsilon=1e-5)
    print(f"Output shape: {output_functional.shape}")
    
    # Verify normalization property
    rms = mx.sqrt(mx.mean(mx.square(output_functional), axis=-1))
    print(f"Output RMS (should be ~1): {mx.mean(rms):.4f}")
    
    print("\n" + "="*50)
    print("2. Module without learnable parameters")
    print("="*50)
    
    # Create module that behaves like functional version
    norm_no_scale = RMSNorm(hidden_dim, use_scale=False)
    output_no_scale = norm_no_scale(x)
    
    # Should match functional output
    print(f"Matches functional: {mx.allclose(output_functional, output_no_scale)}")
    
    print("\n" + "="*50)
    print("3. Module with learnable scale (standard)")
    print("="*50)
    
    # Create module with learnable parameters
    norm_with_scale = RMSNorm(hidden_dim, use_scale=True)
    output_with_scale = norm_with_scale(x)
    print(f"Output shape: {output_with_scale.shape}")
    
    # This uses MLX's optimized implementation
    print("Using MLX's optimized RMSNorm")
    
    print("\n" + "="*50)
    print("4. Using factory function")
    print("="*50)
    
    # Create using factory function
    norm_factory = create_rms_norm(dims=hidden_dim, use_scale=True)
    output_factory = norm_factory(x)
    print(f"Created module type: {type(norm_factory)}")
    
    print("\n" + "="*50)
    print("5. Integration in a model")
    print("="*50)
    
    class SimpleTransformerBlock(nn.Module):
        """Example transformer block using RMSNorm."""
        
        def __init__(self, hidden_dim: int):
            super().__init__()
            # For HRM compatibility, use functional or no-scale version
            self.norm1 = RMSNorm(hidden_dim, use_scale=False)
            self.norm2 = RMSNorm(hidden_dim, use_scale=False)
            
            # Placeholder for attention and FFN
            self.attention = nn.Linear(hidden_dim, hidden_dim)
            self.ffn = nn.Linear(hidden_dim, hidden_dim)
        
        def __call__(self, x):
            # Pre-norm architecture
            h = x + self.attention(self.norm1(x))
            out = h + self.ffn(self.norm2(h))
            return out
    
    # Create and use the block
    block = SimpleTransformerBlock(hidden_dim)
    block_output = block(x)
    print(f"Block output shape: {block_output.shape}")
    
    print("\n" + "="*50)
    print("6. Mixed precision example")
    print("="*50)
    
    # Create float16 input
    x_fp16 = x.astype(mx.float16)
    
    # RMSNorm handles mixed precision internally
    output_fp16 = rms_norm(x_fp16)
    print(f"Input dtype: {x_fp16.dtype}")
    print(f"Output dtype: {output_fp16.dtype}")
    print("Internal computation done in float32 for stability")


if __name__ == "__main__":
    main()