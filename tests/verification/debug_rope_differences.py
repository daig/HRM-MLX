"""
Debug script to understand numerical differences between PyTorch and MLX RoPE.
"""

import numpy as np
import torch
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from mlx_hrm.modules.rope import RotaryEmbedding as MLXRoPE


def analyze_differences():
    """Analyze where differences come from."""
    print("=== Analyzing RoPE Implementation Differences ===\n")
    
    dim = 8  # Small for easy debugging
    max_pos = 10
    base = 10000.0
    
    print(f"Config: dim={dim}, max_pos={max_pos}, base={base}\n")
    
    # Step 1: Check inv_freq calculation
    print("1. Inverse Frequency Calculation:")
    
    # PyTorch
    torch_arange = torch.arange(0, dim, 2, dtype=torch.float32)
    torch_inv_freq = 1.0 / (base ** (torch_arange / dim))
    print(f"PyTorch arange: {torch_arange.numpy()}")
    print(f"PyTorch inv_freq: {torch_inv_freq.numpy()}")
    
    # MLX
    mlx_arange = mx.arange(0, dim, 2, dtype=mx.float32)
    mlx_inv_freq = 1.0 / (base ** (mlx_arange / dim))
    print(f"MLX arange: {np.array(mlx_arange)}")
    print(f"MLX inv_freq: {np.array(mlx_inv_freq)}")
    
    print(f"Inv freq diff: {np.max(np.abs(torch_inv_freq.numpy() - np.array(mlx_inv_freq))):.2e}\n")
    
    # Step 2: Check position indices
    print("2. Position Indices:")
    torch_t = torch.arange(max_pos, dtype=torch.float32)
    mlx_t = mx.arange(max_pos, dtype=mx.float32)
    print(f"Position diff: {np.max(np.abs(torch_t.numpy() - np.array(mlx_t))):.2e}\n")
    
    # Step 3: Check outer product
    print("3. Outer Product (freqs):")
    torch_freqs = torch.outer(torch_t, torch_inv_freq)
    mlx_freqs = mx.outer(mlx_t, mlx_inv_freq)
    
    print(f"PyTorch freqs shape: {torch_freqs.shape}")
    print(f"MLX freqs shape: {mlx_freqs.shape}")
    print(f"Freqs diff: {np.max(np.abs(torch_freqs.numpy() - np.array(mlx_freqs))):.2e}\n")
    
    # Step 4: Check concatenation
    print("4. Concatenation:")
    torch_emb = torch.cat((torch_freqs, torch_freqs), dim=-1)
    mlx_emb = mx.concatenate([mlx_freqs, mlx_freqs], axis=-1)
    print(f"Emb diff: {np.max(np.abs(torch_emb.numpy() - np.array(mlx_emb))):.2e}\n")
    
    # Step 5: Check cos/sin
    print("5. Cos/Sin Computation:")
    torch_cos = torch_emb.cos()
    torch_sin = torch_emb.sin()
    mlx_cos = mx.cos(mlx_emb)
    mlx_sin = mx.sin(mlx_emb)
    
    cos_diff = np.max(np.abs(torch_cos.numpy() - np.array(mlx_cos)))
    sin_diff = np.max(np.abs(torch_sin.numpy() - np.array(mlx_sin)))
    
    print(f"Max cos diff: {cos_diff:.2e}")
    print(f"Max sin diff: {sin_diff:.2e}")
    
    # Find where max difference occurs
    cos_diff_array = np.abs(torch_cos.numpy() - np.array(mlx_cos))
    max_idx = np.unravel_index(np.argmax(cos_diff_array), cos_diff_array.shape)
    print(f"\nMax difference at position {max_idx}:")
    print(f"PyTorch cos: {torch_cos[max_idx].item():.10f}")
    print(f"MLX cos: {np.array(mlx_cos)[max_idx]:.10f}")
    print(f"Input angle: {torch_emb[max_idx].item():.10f}")
    
    # Check if it's a precision issue
    print("\n6. Precision Analysis:")
    # Compute cos using numpy as reference
    np_cos = np.cos(torch_emb.numpy())
    print(f"NumPy vs PyTorch cos diff: {np.max(np.abs(torch_cos.numpy() - np_cos)):.2e}")
    print(f"NumPy vs MLX cos diff: {np.max(np.abs(np.array(mlx_cos) - np_cos)):.2e}")
    
    # Test with larger values to see if differences scale
    print("\n7. Testing with larger dimensions:")
    for test_dim in [64, 128, 256]:
        rope_torch = torch.nn.Module()
        inv_freq = 1.0 / (base ** (torch.arange(0, test_dim, 2, dtype=torch.float32) / test_dim))
        t = torch.arange(100, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_torch = emb.cos()
        
        rope_mlx = MLXRoPE(test_dim, 100, base)
        cos_mlx, _ = rope_mlx()
        
        diff = np.max(np.abs(cos_torch.numpy() - np.array(cos_mlx)))
        print(f"Dim {test_dim}: max cos diff = {diff:.2e}")


if __name__ == "__main__":
    analyze_differences()