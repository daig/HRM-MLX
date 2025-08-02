"""
Critical verification test to ensure MLX RoPE implementation
behaves EXACTLY like the original PyTorch HRM implementation.

This test creates identical inputs and verifies outputs match
to within floating point precision limits.
"""

import os
import sys
import unittest
import numpy as np

# Add paths for both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../HRM')))

try:
    import torch
    import mlx.core as mx
    
    # Import PyTorch HRM implementation
    from models.layers import RotaryEmbedding as PyTorchRoPE
    from models.layers import apply_rotary_pos_emb as pytorch_apply_rope
    from models.layers import rotate_half as pytorch_rotate_half
    
    # Import MLX implementation
    from mlx_hrm.modules.rope import RotaryEmbedding as MLXRoPE
    from mlx_hrm.modules.rope import apply_rotary_pos_emb as mlx_apply_rope
    from mlx_hrm.modules.rope import rotate_half as mlx_rotate_half
    
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"Cannot run verification: {e}")
    PYTORCH_AVAILABLE = False


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch HRM not available")
class TestRoPEExactMatch(unittest.TestCase):
    """Verify EXACT match with PyTorch HRM implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use multiple random seeds to ensure robustness
        self.random_seeds = [42, 123, 456, 789, 1234]
        
    def test_rope_initialization_exact_match(self):
        """Test that initialization produces EXACTLY the same cos/sin values."""
        print("\n=== Testing RoPE Initialization Exact Match ===")
        
        # Test various configurations from HRM
        configs = [
            (64, 512, 10000.0),    # Small
            (128, 2048, 10000.0),  # Medium (HRM default head_dim)
            (256, 8192, 10000.0),  # Large
            (64, 512, 50000.0),    # Different base
        ]
        
        for dim, max_pos, base in configs:
            with self.subTest(dim=dim, max_pos=max_pos, base=base):
                # Create both implementations
                torch_rope = PyTorchRoPE(dim, max_pos, base)
                mlx_rope = MLXRoPE(dim, max_pos, base)
                
                # Get cos/sin values
                torch_cos, torch_sin = torch_rope()
                mlx_cos, mlx_sin = mlx_rope()
                
                # Convert to numpy
                torch_cos_np = torch_cos.detach().cpu().numpy()
                torch_sin_np = torch_sin.detach().cpu().numpy()
                mlx_cos_np = np.array(mlx_cos)
                mlx_sin_np = np.array(mlx_sin)
                
                # Check exact match
                max_cos_diff = np.max(np.abs(torch_cos_np - mlx_cos_np))
                max_sin_diff = np.max(np.abs(torch_sin_np - mlx_sin_np))
                
                print(f"Config (dim={dim}, max_pos={max_pos}, base={base}):")
                print(f"  Max cos difference: {max_cos_diff:.2e}")
                print(f"  Max sin difference: {max_sin_diff:.2e}")
                
                # Should match to within float32 precision
                self.assertLess(max_cos_diff, 1e-6, 
                    f"cos values don't match for dim={dim}, max_pos={max_pos}, base={base}")
                self.assertLess(max_sin_diff, 1e-6,
                    f"sin values don't match for dim={dim}, max_pos={max_pos}, base={base}")
    
    def test_rotate_half_exact_match(self):
        """Test that rotate_half produces EXACTLY the same output."""
        print("\n=== Testing rotate_half Exact Match ===")
        
        test_shapes = [
            (8,),                    # 1D
            (16, 32),               # 2D
            (4, 8, 16, 64),         # 4D (typical attention shape)
            (2, 512, 8, 128),       # Larger 4D
            (1, 2048, 16, 256),     # Even larger
        ]
        
        for seed in self.random_seeds[:2]:  # Test with multiple seeds
            np.random.seed(seed)
            
            for shape in test_shapes:
                with self.subTest(seed=seed, shape=shape):
                    # Create identical random input
                    data = np.random.randn(*shape).astype(np.float32)
                    
                    # PyTorch version
                    torch_input = torch.from_numpy(data)
                    torch_output = pytorch_rotate_half(torch_input)
                    
                    # MLX version
                    mlx_input = mx.array(data)
                    mlx_output = mlx_rotate_half(mlx_input)
                    
                    # Compare
                    torch_np = torch_output.numpy()
                    mlx_np = np.array(mlx_output)
                    
                    max_diff = np.max(np.abs(torch_np - mlx_np))
                    
                    self.assertLess(max_diff, 1e-7,
                        f"rotate_half mismatch for shape {shape}, seed {seed}. Max diff: {max_diff}")
    
    def test_apply_rope_exact_match(self):
        """Test that apply_rotary_pos_emb produces EXACTLY the same output."""
        print("\n=== Testing apply_rotary_pos_emb Exact Match ===")
        
        # Test configurations matching HRM usage
        configs = [
            (1, 128, 8, 64),      # Small batch
            (8, 512, 8, 64),      # Medium
            (32, 2048, 16, 128),  # Large
            (16, 1024, 12, 64),   # Typical HRM config
        ]
        
        for seed in self.random_seeds[:3]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            for batch, seq_len, num_heads, head_dim in configs:
                with self.subTest(seed=seed, batch=batch, seq_len=seq_len, 
                                 num_heads=num_heads, head_dim=head_dim):
                    
                    # Create identical inputs
                    q_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
                    k_data = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
                    
                    # Create RoPE instances
                    torch_rope = PyTorchRoPE(head_dim, seq_len, 10000.0)
                    mlx_rope = MLXRoPE(head_dim, seq_len, 10000.0)
                    
                    # Get cos/sin
                    torch_cos, torch_sin = torch_rope()
                    mlx_cos, mlx_sin = mlx_rope()
                    
                    # Apply RoPE - PyTorch
                    torch_q = torch.from_numpy(q_data)
                    torch_k = torch.from_numpy(k_data)
                    torch_q_rot, torch_k_rot = pytorch_apply_rope(
                        torch_q, torch_k, torch_cos, torch_sin
                    )
                    
                    # Apply RoPE - MLX
                    mlx_q = mx.array(q_data)
                    mlx_k = mx.array(k_data)
                    mlx_q_rot, mlx_k_rot = mlx_apply_rope(
                        mlx_q, mlx_k, mlx_cos, mlx_sin
                    )
                    
                    # Compare outputs
                    torch_q_np = torch_q_rot.numpy()
                    torch_k_np = torch_k_rot.numpy()
                    mlx_q_np = np.array(mlx_q_rot)
                    mlx_k_np = np.array(mlx_k_rot)
                    
                    max_q_diff = np.max(np.abs(torch_q_np - mlx_q_np))
                    max_k_diff = np.max(np.abs(torch_k_np - mlx_k_np))
                    
                    print(f"Config (B={batch}, L={seq_len}, H={num_heads}, D={head_dim}, seed={seed}):")
                    print(f"  Max Q difference: {max_q_diff:.2e}")
                    print(f"  Max K difference: {max_k_diff:.2e}")
                    
                    self.assertLess(max_q_diff, 1e-5,
                        f"Q rotation mismatch. Max diff: {max_q_diff}")
                    self.assertLess(max_k_diff, 1e-5,
                        f"K rotation mismatch. Max diff: {max_k_diff}")
    
    def test_dtype_behavior_exact_match(self):
        """Test that dtype casting behavior matches EXACTLY."""
        print("\n=== Testing Dtype Behavior Exact Match ===")
        
        # Test configurations
        seq_len, num_heads, head_dim = 64, 4, 32
        
        for input_dtype in [np.float16, np.float32]:
            with self.subTest(input_dtype=input_dtype):
                # Create test data
                q_data = np.random.randn(1, seq_len, num_heads, head_dim).astype(input_dtype)
                k_data = np.random.randn(1, seq_len, num_heads, head_dim).astype(input_dtype)
                
                # Create RoPE (cos/sin always float32 initially)
                torch_rope = PyTorchRoPE(head_dim, seq_len, 10000.0)
                mlx_rope = MLXRoPE(head_dim, seq_len, 10000.0)
                
                torch_cos, torch_sin = torch_rope()
                mlx_cos, mlx_sin = mlx_rope()
                
                # PyTorch version
                torch_q = torch.from_numpy(q_data)
                torch_k = torch.from_numpy(k_data)
                torch_q_rot, torch_k_rot = pytorch_apply_rope(
                    torch_q, torch_k, torch_cos, torch_sin
                )
                
                # MLX version
                mlx_q = mx.array(q_data)
                mlx_k = mx.array(k_data)
                mlx_q_rot, mlx_k_rot = mlx_apply_rope(
                    mlx_q, mlx_k, mlx_cos, mlx_sin
                )
                
                # Check dtype preservation
                self.assertEqual(str(torch_q_rot.dtype), f'torch.{input_dtype.__name__}')
                self.assertEqual(str(mlx_q_rot.dtype), f'mlx.core.{input_dtype.__name__}')
                
                # Check numerical match (higher tolerance for float16)
                rtol = 1e-3 if input_dtype == np.float16 else 1e-5
                np.testing.assert_allclose(
                    torch_q_rot.numpy(),
                    np.array(mlx_q_rot),
                    rtol=rtol,
                    err_msg=f"Mismatch with input dtype {input_dtype}"
                )
    
    def test_frequency_computation_exact_match(self):
        """Test that frequency computation matches EXACTLY."""
        print("\n=== Testing Frequency Computation Exact Match ===")
        
        # Test edge cases and typical values
        test_cases = [
            (2, 10000.0),      # Minimal dimension
            (64, 10000.0),     # Typical small
            (128, 10000.0),    # HRM default
            (256, 10000.0),    # Large
            (128, 1.0),        # Base = 1 (edge case)
            (128, 2.0),        # Small base
            (128, 1e6),        # Large base
        ]
        
        for dim, base in test_cases:
            with self.subTest(dim=dim, base=base):
                # Create both versions
                torch_rope = PyTorchRoPE(dim, 10, base)
                mlx_rope = MLXRoPE(dim, 10, base)
                
                # Get position 1 (reveals frequencies)
                torch_cos, torch_sin = torch_rope()
                mlx_cos, mlx_sin = mlx_rope()
                
                # Extract frequencies from position 1
                torch_cos1 = torch_cos[1].detach().numpy()
                torch_sin1 = torch_sin[1].detach().numpy()
                mlx_cos1 = np.array(mlx_cos[1])
                mlx_sin1 = np.array(mlx_sin[1])
                
                # Compute angles (frequencies)
                torch_angles = np.arctan2(torch_sin1, torch_cos1)
                mlx_angles = np.arctan2(mlx_sin1, mlx_cos1)
                
                # Check exact match
                max_angle_diff = np.max(np.abs(torch_angles - mlx_angles))
                
                print(f"Frequency test (dim={dim}, base={base}):")
                print(f"  Max angle difference: {max_angle_diff:.2e}")
                
                self.assertLess(max_angle_diff, 1e-6,
                    f"Frequency mismatch for dim={dim}, base={base}")
    
    def test_edge_cases_exact_match(self):
        """Test edge cases to ensure robust exact matching."""
        print("\n=== Testing Edge Cases Exact Match ===")
        
        # Test 1: Very small dimensions
        for dim in [1, 2, 3, 4]:
            torch_rope = PyTorchRoPE(dim, 10, 10000.0)
            mlx_rope = MLXRoPE(dim, 10, 10000.0)
            
            torch_cos, torch_sin = torch_rope()
            mlx_cos, mlx_sin = mlx_rope()
            
            np.testing.assert_allclose(
                torch_cos.detach().numpy(),
                np.array(mlx_cos),
                rtol=1e-6,
                err_msg=f"Mismatch for dim={dim}"
            )
        
        # Test 2: Position 0 (identity)
        rope_torch = PyTorchRoPE(64, 10, 10000.0)
        rope_mlx = MLXRoPE(64, 10, 10000.0)
        
        cos_torch, sin_torch = rope_torch()
        cos_mlx, sin_mlx = rope_mlx()
        
        # At position 0: cos=1, sin=0
        np.testing.assert_allclose(cos_torch[0].numpy(), np.ones(64), atol=1e-7)
        np.testing.assert_allclose(sin_torch[0].numpy(), np.zeros(64), atol=1e-7)
        np.testing.assert_allclose(np.array(cos_mlx[0]), np.ones(64), atol=1e-7)
        np.testing.assert_allclose(np.array(sin_mlx[0]), np.zeros(64), atol=1e-7)
        
        print("✓ All edge cases pass exact match test")
    
    def test_cache_extension_exact_match(self):
        """Test that cache extension produces exact same results."""
        print("\n=== Testing Cache Extension Exact Match ===")
        
        dim = 128
        initial_max = 512
        extended_max = 1024
        
        # Create and extend PyTorch version
        torch_rope = PyTorchRoPE(dim, initial_max, 10000.0)
        # PyTorch version doesn't have extend_cache, so recreate
        torch_rope_extended = PyTorchRoPE(dim, extended_max, 10000.0)
        
        # Create and extend MLX version
        mlx_rope = MLXRoPE(dim, initial_max, 10000.0)
        mlx_rope.extend_cache(extended_max)
        
        # Compare extended portions
        torch_cos_ext, torch_sin_ext = torch_rope_extended()
        mlx_cos_ext, mlx_sin_ext = mlx_rope(extended_max)
        
        # Check that extension matches
        np.testing.assert_allclose(
            torch_cos_ext.detach().numpy(),
            np.array(mlx_cos_ext),
            rtol=1e-6,
            err_msg="Extended cos cache mismatch"
        )
        np.testing.assert_allclose(
            torch_sin_ext.detach().numpy(),
            np.array(mlx_sin_ext),
            rtol=1e-6,
            err_msg="Extended sin cache mismatch"
        )
        
        print("✓ Cache extension produces exact match")


def run_critical_verification():
    """Run the critical verification tests with detailed output."""
    print("\n" + "="*60)
    print("CRITICAL VERIFICATION: MLX RoPE vs PyTorch HRM")
    print("="*60)
    
    if not PYTORCH_AVAILABLE:
        print("\n⚠️  WARNING: Cannot verify exact match - PyTorch HRM not available")
        print("Please ensure PyTorch and the HRM model are accessible")
        return False
    
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoPEExactMatch)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ VERIFICATION PASSED: MLX RoPE matches PyTorch HRM exactly!")
    else:
        print("❌ VERIFICATION FAILED: Differences detected!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_critical_verification()
    sys.exit(0 if success else 1)