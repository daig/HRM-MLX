"""
Integration tests comparing MLX RoPE implementation with PyTorch HRM reference.

This test requires both PyTorch and MLX to be installed and verifies numerical
accuracy between the two implementations.
"""

import os
import sys
import unittest
import numpy as np

# Add paths for both MLX and PyTorch HRM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../HRM')))

try:
    import torch
    import mlx.core as mx
    from models.layers import RotaryEmbedding as PyTorchRoPE
    from models.layers import apply_rotary_pos_emb as pytorch_apply_rope
    from models.layers import rotate_half as pytorch_rotate_half
    
    from mlx_hrm.modules.rope import RotaryEmbedding as MLXRoPE
    from mlx_hrm.modules.rope import apply_rotary_pos_emb as mlx_apply_rope
    from mlx_hrm.modules.rope import rotate_half as mlx_rotate_half
    
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"Skipping PyTorch parity tests: {e}")
    PYTORCH_AVAILABLE = False


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
class TestPyTorchParity(unittest.TestCase):
    """Test exact numerical parity with PyTorch HRM implementation."""
    
    def setUp(self):
        """Set random seeds for reproducibility."""
        np.random.seed(42)
        if PYTORCH_AVAILABLE:
            torch.manual_seed(42)
    
    def test_rope_initialization_parity(self):
        """Test that RoPE initialization matches PyTorch exactly."""
        dim = 64
        max_pos = 512
        base = 10000.0
        
        # Create PyTorch version
        torch_rope = PyTorchRoPE(dim, max_pos, base)
        torch_cos, torch_sin = torch_rope()
        
        # Create MLX version
        mlx_rope = MLXRoPE(dim, max_pos, base)
        mlx_cos, mlx_sin = mlx_rope()
        
        # Convert to numpy for comparison
        torch_cos_np = torch_cos.detach().numpy()
        torch_sin_np = torch_sin.detach().numpy()
        mlx_cos_np = np.array(mlx_cos)
        mlx_sin_np = np.array(mlx_sin)
        
        # Check shapes
        self.assertEqual(torch_cos_np.shape, mlx_cos_np.shape)
        self.assertEqual(torch_sin_np.shape, mlx_sin_np.shape)
        
        # Check values (should be exactly equal for deterministic computation)
        np.testing.assert_allclose(torch_cos_np, mlx_cos_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(torch_sin_np, mlx_sin_np, rtol=1e-6, atol=1e-6)
    
    def test_rotate_half_parity(self):
        """Test that rotate_half matches PyTorch implementation."""
        # Test various shapes
        test_shapes = [
            (8,),
            (4, 16),
            (2, 3, 4, 32),
            (1, 10, 8, 64)
        ]
        
        for shape in test_shapes:
            # Create random tensor
            data = np.random.randn(*shape).astype(np.float32)
            
            # PyTorch version
            torch_input = torch.from_numpy(data)
            torch_output = pytorch_rotate_half(torch_input)
            
            # MLX version
            mlx_input = mx.array(data)
            mlx_output = mlx_rotate_half(mlx_input)
            
            # Compare
            np.testing.assert_allclose(
                torch_output.numpy(), 
                np.array(mlx_output), 
                rtol=1e-6, 
                atol=1e-6,
                err_msg=f"Mismatch for shape {shape}"
            )
    
    def test_apply_rope_parity(self):
        """Test that apply_rotary_pos_emb matches PyTorch exactly."""
        batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
        
        # Create test data
        q_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        k_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        
        # Create RoPE
        torch_rope = PyTorchRoPE(head_dim, seq_len, 10000.0)
        mlx_rope = MLXRoPE(head_dim, seq_len, 10000.0)
        
        # Get cos/sin
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        # Apply RoPE - PyTorch
        torch_q = torch.from_numpy(q_data)
        torch_k = torch.from_numpy(k_data)
        torch_q_rot, torch_k_rot = pytorch_apply_rope(
            torch_q, torch_k, 
            torch_cos[:seq_len], torch_sin[:seq_len]
        )
        
        # Apply RoPE - MLX
        mlx_q = mx.array(q_data)
        mlx_k = mx.array(k_data)
        mlx_q_rot, mlx_k_rot = mlx_apply_rope(
            mlx_q, mlx_k,
            mlx_cos[:seq_len], mlx_sin[:seq_len]
        )
        
        # Compare results
        np.testing.assert_allclose(
            torch_q_rot.numpy(),
            np.array(mlx_q_rot),
            rtol=1e-5,
            atol=1e-5
        )
        np.testing.assert_allclose(
            torch_k_rot.numpy(),
            np.array(mlx_k_rot),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_dtype_casting_parity(self):
        """Test that dtype casting behavior matches PyTorch."""
        # Test with float16 input
        q_data = np.random.randn(1, 8, 2, 16).astype(np.float16)
        k_data = np.random.randn(1, 8, 2, 16).astype(np.float16)
        
        # Create RoPE with float32 cos/sin
        torch_rope = PyTorchRoPE(16, 8, 10000.0)
        mlx_rope = MLXRoPE(16, 8, 10000.0)
        
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        # Apply with dtype preservation
        torch_q = torch.from_numpy(q_data)
        torch_k = torch.from_numpy(k_data)
        torch_q_rot, torch_k_rot = pytorch_apply_rope(
            torch_q, torch_k, torch_cos, torch_sin
        )
        
        mlx_q = mx.array(q_data)
        mlx_k = mx.array(k_data)
        mlx_q_rot, mlx_k_rot = mlx_apply_rope(
            mlx_q, mlx_k, mlx_cos, mlx_sin
        )
        
        # Check dtype preservation
        self.assertEqual(torch_q_rot.dtype, torch.float16)
        self.assertEqual(mlx_q_rot.dtype, mx.float16)
        
        # Check numerical accuracy (with higher tolerance for float16)
        np.testing.assert_allclose(
            torch_q_rot.numpy(),
            np.array(mlx_q_rot),
            rtol=1e-3,
            atol=1e-3
        )
    
    def test_frequency_computation_parity(self):
        """Test that frequency computation exactly matches PyTorch."""
        for dim in [32, 64, 128]:
            for base in [10000.0, 50000.0, 100000.0]:
                # Create both versions
                torch_rope = PyTorchRoPE(dim, 10, base)
                mlx_rope = MLXRoPE(dim, 10, base)
                
                # Get first position (cos/sin at position 1 reveals frequencies)
                torch_cos, torch_sin = torch_rope()
                mlx_cos, mlx_sin = mlx_rope()
                
                # Compare at position 1 (where angle = frequency)
                torch_cos1 = torch_cos[1].detach().numpy()
                torch_sin1 = torch_sin[1].detach().numpy()
                mlx_cos1 = np.array(mlx_cos[1])
                mlx_sin1 = np.array(mlx_sin[1])
                
                np.testing.assert_allclose(
                    torch_cos1, mlx_cos1, 
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"cos mismatch for dim={dim}, base={base}"
                )
                np.testing.assert_allclose(
                    torch_sin1, mlx_sin1,
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"sin mismatch for dim={dim}, base={base}"
                )
    
    def test_edge_cases_parity(self):
        """Test edge cases match between implementations."""
        # Test with dim=1 (minimal case)
        torch_rope = PyTorchRoPE(1, 5, 10000.0)
        mlx_rope = MLXRoPE(1, 5, 10000.0)
        
        torch_cos, torch_sin = torch_rope()
        mlx_cos, mlx_sin = mlx_rope()
        
        np.testing.assert_allclose(
            torch_cos.detach().numpy(),
            np.array(mlx_cos),
            rtol=1e-6
        )
        
        # Test with very small base
        torch_rope_small = PyTorchRoPE(8, 10, 1.1)
        mlx_rope_small = MLXRoPE(8, 10, 1.1)
        
        torch_cos_small, _ = torch_rope_small()
        mlx_cos_small, _ = mlx_rope_small()
        
        np.testing.assert_allclose(
            torch_cos_small.detach().numpy(),
            np.array(mlx_cos_small),
            rtol=1e-5  # Slightly higher tolerance for extreme values
        )


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not available")
class TestRelativePositionProperty(unittest.TestCase):
    """Test that both implementations encode relative positions identically."""
    
    def test_relative_position_encoding(self):
        """Verify relative position encoding property is preserved."""
        dim = 128
        rope_torch = PyTorchRoPE(dim, 100, 10000.0)
        rope_mlx = MLXRoPE(dim, 100, 10000.0)
        
        # Test positions 10 and 25 (relative distance = 15)
        pos1, pos2 = 10, 25
        
        # Create identical vectors
        vec = np.ones((1, 1, 1, dim), dtype=np.float32)
        vec_torch = torch.from_numpy(vec)
        vec_mlx = mx.array(vec)
        
        # Get cos/sin for both positions
        cos_torch, sin_torch = rope_torch()
        cos_mlx, sin_mlx = rope_mlx()
        
        # Apply RoPE at position 1
        vec1_torch, _ = pytorch_apply_rope(
            vec_torch, vec_torch,
            cos_torch[pos1:pos1+1], sin_torch[pos1:pos1+1]
        )
        vec1_mlx, _ = mlx_apply_rope(
            vec_mlx, vec_mlx,
            cos_mlx[pos1:pos1+1], sin_mlx[pos1:pos1+1]
        )
        
        # Apply RoPE at position 2
        vec2_torch, _ = pytorch_apply_rope(
            vec_torch, vec_torch,
            cos_torch[pos2:pos2+1], sin_torch[pos2:pos2+1]
        )
        vec2_mlx, _ = mlx_apply_rope(
            vec_mlx, vec_mlx,
            cos_mlx[pos2:pos2+1], sin_mlx[pos2:pos2+1]
        )
        
        # Compute dot products (should encode relative position)
        dot_torch = torch.sum(vec1_torch * vec2_torch, dim=-1).item()
        dot_mlx = mx.sum(vec1_mlx * vec2_mlx, axis=-1).item()
        
        # Should be very close
        self.assertAlmostEqual(dot_torch, dot_mlx, places=5)
        
        # Also verify against direct relative encoding
        rel_pos = pos2 - pos1
        vec_rel_torch, _ = pytorch_apply_rope(
            vec_torch, vec_torch,
            cos_torch[rel_pos:rel_pos+1], sin_torch[rel_pos:rel_pos+1]
        )
        vec_rel_mlx, _ = mlx_apply_rope(
            vec_mlx, vec_mlx,
            cos_mlx[rel_pos:rel_pos+1], sin_mlx[rel_pos:rel_pos+1]
        )
        
        dot_rel_torch = torch.sum(vec_torch * vec_rel_torch, dim=-1).item()
        dot_rel_mlx = mx.sum(vec_mlx * vec_rel_mlx, axis=-1).item()
        
        # All should match
        self.assertAlmostEqual(dot_torch, dot_rel_torch, places=4)
        self.assertAlmostEqual(dot_mlx, dot_rel_mlx, places=4)


if __name__ == "__main__":
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Install PyTorch to run parity tests.")
        print("These tests verify exact numerical match with PyTorch HRM.")
    else:
        unittest.main()