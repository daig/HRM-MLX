"""Example usage of SwiGLU activation in MLX HRM."""

import mlx.core as mx
import mlx.nn as nn
from mlx_hrm.layers import SwiGLU, SwiGLUFactory, RMSNorm


def basic_usage():
    """Basic SwiGLU usage example."""
    print("Basic SwiGLU Usage")
    print("=" * 50)
    
    # Create SwiGLU layer
    hidden_dim = 768
    swiglu = SwiGLU(hidden_dim, expansion=4.0)
    
    # Create input
    batch_size = 2
    seq_len = 128
    x = mx.random.normal((batch_size, seq_len, hidden_dim))
    
    # Forward pass
    output = swiglu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Intermediate dimension: {swiglu.intermediate_dim}")
    print(f"Parameter count: {swiglu.gate_up_proj.weight.size + swiglu.down_proj.weight.size:,}")


def transformer_block_example():
    """Example of SwiGLU in a transformer block."""
    print("\n" + "=" * 50)
    print("Transformer Block with SwiGLU")
    print("=" * 50)
    
    class TransformerBlock(nn.Module):
        """Transformer block using SwiGLU FFN."""
        
        def __init__(self, hidden_dim: int, num_heads: int = 8, expansion: float = 4.0):
            super().__init__()
            
            # Multi-head attention (placeholder - would use real attention)
            self.attention = nn.Linear(hidden_dim, hidden_dim)
            
            # Normalization layers (using RMSNorm like HRM)
            self.norm1 = RMSNorm(hidden_dim, use_scale=False)
            self.norm2 = RMSNorm(hidden_dim, use_scale=False)
            
            # SwiGLU FFN
            self.ffn = SwiGLU(hidden_dim, expansion=expansion)
        
        def __call__(self, x):
            # Attention block with residual
            attn_out = self.attention(self.norm1(x))
            x = x + attn_out
            
            # FFN block with residual
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out
            
            return x
    
    # Create and use the block
    hidden_dim = 512
    block = TransformerBlock(hidden_dim)
    
    # Process some input
    x = mx.random.normal((4, 64, hidden_dim))
    output = block(x)
    
    print(f"Created transformer block with hidden_dim={hidden_dim}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"FFN intermediate dim: {block.ffn.intermediate_dim}")


def factory_usage():
    """Example using SwiGLU factory methods."""
    print("\n" + "=" * 50)
    print("SwiGLU Factory Usage")
    print("=" * 50)
    
    hidden_dim = 1024
    x = mx.random.normal((2, 128, hidden_dim))
    
    # Different factory configurations
    configs = [
        ("Default", SwiGLUFactory.create_default(hidden_dim)),
        ("Large", SwiGLUFactory.create_large(hidden_dim)),
        ("Efficient", SwiGLUFactory.create_efficient(hidden_dim)),
    ]
    
    print(f"\nComparing different SwiGLU configurations (hidden_dim={hidden_dim}):")
    print("-" * 70)
    print(f"{'Config':>12} | {'Intermediate':>12} | {'Parameters':>12} | {'Output Mean':>12}")
    print("-" * 70)
    
    for name, swiglu in configs:
        output = swiglu(x)
        params = swiglu.gate_up_proj.weight.size + swiglu.down_proj.weight.size
        out_mean = float(mx.mean(mx.abs(output)))
        
        print(f"{name:>12} | {swiglu.intermediate_dim:>12} | {params:>12,} | {out_mean:>12.6f}")


def custom_configuration():
    """Example with custom SwiGLU configuration."""
    print("\n" + "=" * 50)
    print("Custom SwiGLU Configuration")
    print("=" * 50)
    
    # Custom configuration for specific use case
    hidden_dim = 256
    
    # Small model with high expansion
    swiglu_high_exp = SwiGLU(
        hidden_dim=hidden_dim,
        expansion=8.0,        # Higher expansion for more capacity
        multiple_of=128,      # Smaller multiple for efficiency
        bias=False,           # No bias (modern practice)
        init_std=0.02        # Custom initialization
    )
    
    # Process input
    x = mx.random.normal((8, 64, hidden_dim))
    output = swiglu_high_exp(x)
    
    print(f"Custom SwiGLU configuration:")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Expansion factor: 8.0")
    print(f"  Intermediate dimension: {swiglu_high_exp.intermediate_dim}")
    print(f"  Multiple of: 128")
    print(f"  Total parameters: {swiglu_high_exp.gate_up_proj.weight.size + swiglu_high_exp.down_proj.weight.size:,}")


def mixed_precision_example():
    """Example of SwiGLU with mixed precision."""
    print("\n" + "=" * 50)
    print("Mixed Precision Usage")
    print("=" * 50)
    
    hidden_dim = 512
    swiglu = SwiGLU(hidden_dim)
    
    # Note: Current implementation promotes to float32 internally
    # This matches PyTorch HRM behavior for stability
    
    dtypes = [mx.float32, mx.float16]
    if hasattr(mx, 'bfloat16'):
        dtypes.append(mx.bfloat16)
    
    print(f"\nProcessing with different dtypes:")
    print("-" * 50)
    
    for dtype in dtypes:
        x = mx.random.normal((4, 32, hidden_dim), dtype=dtype)
        output = swiglu(x)
        
        print(f"Input dtype: {dtype}, Output dtype: {output.dtype}")
        print(f"  Output mean: {float(mx.mean(mx.abs(output))):.6f}")


def performance_tips():
    """Performance optimization tips for SwiGLU."""
    print("\n" + "=" * 50)
    print("Performance Optimization Tips")
    print("=" * 50)
    
    print("\n1. Choose appropriate expansion factor:")
    print("   - 4.0 is standard (matches most transformers)")
    print("   - 2.667 gives clean intermediate dims after 2/3 factor")
    print("   - Higher values increase capacity but also compute")
    
    print("\n2. Round to efficient multiples:")
    print("   - 256 is optimal for most Apple Silicon")
    print("   - 128 can be faster for smaller models")
    print("   - Powers of 2 generally perform best")
    
    print("\n3. Batch processing:")
    hidden_dim = 256
    swiglu = SwiGLU(hidden_dim)
    
    # Process multiple sequences at once
    sequences = [mx.random.normal((1, 64, hidden_dim)) for _ in range(4)]
    batched = mx.concatenate(sequences, axis=0)
    output = swiglu(batched)
    print(f"   - Batched shape: {batched.shape} → {output.shape}")
    
    print("\n4. Consider compilation for static shapes:")
    print("   - Use @mx.compile for repeated operations")
    print("   - Significant speedup for inference")


def main():
    """Run all examples."""
    basic_usage()
    transformer_block_example()
    factory_usage()
    custom_configuration()
    mixed_precision_example()
    performance_tips()
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("\n✅ SwiGLU provides better gradient flow than ReLU-based FFNs")
    print("✅ Parameter overhead (~10-15%) is offset by improved performance")
    print("✅ Matches PyTorch HRM implementation exactly")
    print("✅ Optimized for Apple Silicon with proper dimension rounding")


if __name__ == "__main__":
    main()