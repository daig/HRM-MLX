"""
Standalone verification of RoPE implementation against PyTorch reference.

This creates a minimal PyTorch RoPE implementation matching HRM's exact behavior
to verify our MLX implementation without requiring the full HRM codebase.
"""

import os
import sys
import unittest
import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from mlx_hrm.modules.rope import RotaryEmbedding as MLXRoPE
from mlx_hrm.modules.rope import apply_rotary_pos_emb as mlx_apply_rope
from mlx_hrm.modules.rope import rotate_half as mlx_rotate_half


# Minimal PyTorch RoPE implementation matching HRM exactly
def pytorch_rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def pytorch_apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                                cos: torch.Tensor, sin: torch.Tensor):
    """Apply Rotary Position Embeddings to query and key tensors."""
    orig_dtype = q.dtype
    # Cast to cos/sin dtype for computation
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    
    # Apply rotation using complex number multiplication formula
    q_embed = (q * cos.unsqueeze(-2)) + (pytorch_rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (pytorch_rotate_half(k) * sin.unsqueeze(-2))
    
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class PyTorchRotaryEmbedding(torch.nn.Module):
    """PyTorch RoPE matching HRM implementation exactly."""
    
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        
        # Calculate frequency bands for rotations
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        
        # Position indices
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        
        # Outer product gives frequencies for each position
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate frequencies for both sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register as buffers (not parameters)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self):
        return self.cos_cached, self.sin_cached


class TestRoPEStandaloneVerification(unittest.TestCase):
    """Verify MLX RoPE against minimal PyTorch reference."""
    
    def test_initialization_exact_match(self):
        """Test that cos/sin values match exactly at initialization."""
        print("\n=== Initialization Exact Match Test ===")
        
        configs = [
            (64, 512, 10000.0),
            (128, 2048, 10000.0),
            (256, 4096, 10000.0),
        ]
        
        for dim, max_pos, base in configs:
            with self.subTest(dim=dim, max_pos=max_pos, base=base):
                # Create both versions
                torch_rope = PyTorchRotaryEmbedding(dim, max_pos, base)
                mlx_rope = MLXRoPE(dim, max_pos, base)
                
                # Get values
                torch_cos, torch_sin = torch_rope()
                mlx_cos, mlx_sin = mlx_rope()
                
                # Convert to numpy
                torch_cos_np = torch_cos.detach().cpu().numpy()
                torch_sin_np = torch_sin.detach().cpu().numpy()
                mlx_cos_np = np.array(mlx_cos)
                mlx_sin_np = np.array(mlx_sin)
                
                # Check shapes
                self.assertEqual(torch_cos_np.shape, mlx_cos_np.shape)
                self.assertEqual(torch_sin_np.shape, mlx_sin_np.shape)
                
                # Check exact values
                max_cos_diff = np.max(np.abs(torch_cos_np - mlx_cos_np))
                max_sin_diff = np.max(np.abs(torch_sin_np - mlx_sin_np))
                
                print(f"Config (dim={dim}, max_pos={max_pos}):")
                print(f"  Max cos diff: {max_cos_diff:.2e}")
                print(f"  Max sin diff: {max_sin_diff:.2e}")
                
                self.assertLess(max_cos_diff, 1e-6)
                self.assertLess(max_sin_diff, 1e-6)
    
    def test_rotate_half_exact_match(self):
        """Test rotate_half function exact match."""
        print("\n=== rotate_half Exact Match Test ===")
        
        # Test multiple shapes
        test_cases = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.random.randn(8, 16),
            np.random.randn(2, 4, 8, 32),
            np.random.randn(1, 512, 8, 128),
        ]
        
        for i, data in enumerate(test_cases):
            with self.subTest(case=i, shape=data.shape):
                # PyTorch
                torch_input = torch.from_numpy(data.astype(np.float32))
                torch_output = pytorch_rotate_half(torch_input)
                
                # MLX
                mlx_input = mx.array(data.astype(np.float32))
                mlx_output = mlx_rotate_half(mlx_input)
                
                # Compare
                np.testing.assert_allclose(
                    torch_output.numpy(),
                    np.array(mlx_output),
                    rtol=1e-7,
                    atol=1e-7,
                    err_msg=f"Mismatch for shape {data.shape}"
                )
    
    def test_apply_rope_exact_match(self):
        """Test apply_rotary_pos_emb exact match."""
        print("\n=== apply_rotary_pos_emb Exact Match Test ===")
        
        # Test configuration
        batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
        
        # Create identical random inputs
        np.random.seed(42)
        q_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        k_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        
        # Create RoPE
        torch_rope = PyTorchRotaryEmbedding(head_dim, seq_len, 10000.0)
        mlx_rope = MLXRoPE(head_dim, seq_len, 10000.0)
        
        # Get cos/sin
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        # Apply - PyTorch
        torch_q = torch.from_numpy(q_data)
        torch_k = torch.from_numpy(k_data)
        torch_q_rot, torch_k_rot = pytorch_apply_rotary_pos_emb(
            torch_q, torch_k, torch_cos, torch_sin
        )
        
        # Apply - MLX
        mlx_q = mx.array(q_data)
        mlx_k = mx.array(k_data)
        mlx_q_rot, mlx_k_rot = mlx_apply_rope(
            mlx_q, mlx_k, mlx_cos, mlx_sin
        )
        
        # Compare
        max_q_diff = np.max(np.abs(torch_q_rot.numpy() - np.array(mlx_q_rot)))
        max_k_diff = np.max(np.abs(torch_k_rot.numpy() - np.array(mlx_k_rot)))
        
        print(f"Max Q difference: {max_q_diff:.2e}")
        print(f"Max K difference: {max_k_diff:.2e}")
        
        self.assertLess(max_q_diff, 1e-5)
        self.assertLess(max_k_diff, 1e-5)
    
    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        print("\n=== Dtype Preservation Test ===")
        
        # Test with float16
        q_f16 = np.random.randn(1, 32, 4, 16).astype(np.float16)
        k_f16 = np.random.randn(1, 32, 4, 16).astype(np.float16)
        
        # Create RoPE
        torch_rope = PyTorchRotaryEmbedding(16, 32, 10000.0)
        mlx_rope = MLXRoPE(16, 32, 10000.0)
        
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        # Apply with float16 input
        torch_q = torch.from_numpy(q_f16)
        torch_k = torch.from_numpy(k_f16)
        torch_q_rot, torch_k_rot = pytorch_apply_rotary_pos_emb(
            torch_q, torch_k, torch_cos, torch_sin
        )
        
        mlx_q = mx.array(q_f16)
        mlx_k = mx.array(k_f16)
        mlx_q_rot, mlx_k_rot = mlx_apply_rope(
            mlx_q, mlx_k, mlx_cos, mlx_sin
        )
        
        # Check dtypes preserved
        self.assertEqual(torch_q_rot.dtype, torch.float16)
        self.assertEqual(mlx_q_rot.dtype, mx.float16)
        
        print("✓ Dtype preservation verified")
    
    def test_frequency_formula(self):
        """Test that frequency formula matches exactly."""
        print("\n=== Frequency Formula Test ===")
        
        dim = 128
        base = 10000.0
        
        # Expected frequencies: 1.0 / (base ** (arange(0, dim, 2) / dim))
        expected_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        
        # Create RoPE and extract frequencies from position 1
        torch_rope = PyTorchRotaryEmbedding(dim, 10, base)
        mlx_rope = MLXRoPE(dim, 10, base)
        
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        # At position 1, angle = frequency
        torch_angles = np.arctan2(torch_sin[1].numpy(), torch_cos[1].numpy())
        mlx_angles = np.arctan2(np.array(mlx_sin[1]), np.array(mlx_cos[1]))
        
        # Extract unique frequencies (first half before duplication)
        torch_freqs = torch_angles[:dim//2]
        mlx_freqs = mlx_angles[:dim//2]
        
        # Check against expected
        np.testing.assert_allclose(torch_freqs, expected_freqs, rtol=1e-5)
        np.testing.assert_allclose(mlx_freqs, expected_freqs, rtol=1e-5)
        
        print("✓ Frequency formula verified")
    
    def test_critical_properties(self):
        """Test critical mathematical properties."""
        print("\n=== Critical Properties Test ===")
        
        dim = 64
        rope_torch = PyTorchRotaryEmbedding(dim, 100, 10000.0)
        rope_mlx = MLXRoPE(dim, 100, 10000.0)
        
        cos_torch, sin_torch = rope_torch()
        cos_mlx, sin_mlx = rope_mlx()
        
        # Property 1: cos² + sin² = 1
        magnitude_torch = cos_torch**2 + sin_torch**2
        magnitude_mlx = mx.square(cos_mlx) + mx.square(sin_mlx)
        
        np.testing.assert_allclose(
            magnitude_torch.numpy(),
            np.ones_like(magnitude_torch.numpy()),
            atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(magnitude_mlx),
            np.ones_like(np.array(magnitude_mlx)),
            atol=1e-6
        )
        
        # Property 2: Position 0 is identity
        np.testing.assert_allclose(cos_torch[0].numpy(), np.ones(dim), atol=1e-7)
        np.testing.assert_allclose(sin_torch[0].numpy(), np.zeros(dim), atol=1e-7)
        np.testing.assert_allclose(np.array(cos_mlx[0]), np.ones(dim), atol=1e-7)
        np.testing.assert_allclose(np.array(sin_mlx[0]), np.zeros(dim), atol=1e-7)
        
        print("✓ All critical properties verified")


def main():
    """Run standalone verification."""
    print("\n" + "="*60)
    print("STANDALONE ROPE VERIFICATION")
    print("="*60)
    
    # Run tests
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()