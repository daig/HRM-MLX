"""
Unit tests for Rotary Position Embeddings (RoPE) implementation.

Tests verify:
1. Basic functionality and shapes
2. Mathematical properties (orthogonality, magnitude preservation)
3. Relative position encoding
4. Extrapolation capabilities
5. Exact match with PyTorch HRM reference
"""

import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from mlx_hrm.modules.rope import RotaryEmbedding, rotate_half, apply_rotary_pos_emb


class TestRotaryEmbedding(unittest.TestCase):
    """Test suite for RotaryEmbedding class."""
    
    def test_initialization(self):
        """Test RoPE initialization with various configurations."""
        # Test basic initialization
        dim = 64
        max_pos = 512
        base = 10000.0
        
        rope = RotaryEmbedding(dim, max_pos, base)
        
        # Check attributes
        self.assertEqual(rope.dim, dim)
        self.assertEqual(rope.max_position_embeddings, max_pos)
        self.assertEqual(rope.base, base)
        
        # Check cached values shape
        self.assertEqual(rope.cos_cached.shape, (max_pos, dim))
        self.assertEqual(rope.sin_cached.shape, (max_pos, dim))
        
        # Check dtype
        self.assertEqual(rope.cos_cached.dtype, mx.float32)
        self.assertEqual(rope.sin_cached.dtype, mx.float32)
    
    def test_frequency_calculation(self):
        """Test that frequency calculation matches expected values."""
        dim = 8
        base = 10000.0
        rope = RotaryEmbedding(dim, 10, base)
        
        # Manually compute expected frequencies
        # inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        expected_inv_freq = np.array([
            1.0,  # base^(0/8) = base^0 = 1
            1.0 / (base ** (2.0/8)),  # base^(0.25)
            1.0 / (base ** (4.0/8)),  # base^(0.5)
            1.0 / (base ** (6.0/8)),  # base^(0.75)
        ])
        
        # Extract frequencies from cached cos/sin
        # At position 1, frequencies are just inv_freq
        cos_pos1 = np.array(rope.cos_cached[1])
        sin_pos1 = np.array(rope.sin_cached[1])
        
        # Verify using arctan2 to extract angles
        angles = np.arctan2(sin_pos1[:4], cos_pos1[:4])
        computed_inv_freq = angles  # At position 1, angle = inv_freq
        
        np.testing.assert_allclose(computed_inv_freq, expected_inv_freq, rtol=1e-5)
    
    def test_cos_sin_properties(self):
        """Test mathematical properties of cached cos/sin values."""
        rope = RotaryEmbedding(64, 128)
        cos, sin = rope()
        
        # Check values are in valid range
        self.assertTrue(mx.all(mx.abs(cos) <= 1.0).item())
        self.assertTrue(mx.all(mx.abs(sin) <= 1.0).item())
        
        # Check orthogonality (cos² + sin² = 1)
        magnitude = cos**2 + sin**2
        expected = mx.ones_like(magnitude)
        
        # Use higher tolerance as trig operations can accumulate small errors
        self.assertTrue(mx.allclose(magnitude, expected, atol=1e-6).item())
    
    def test_forward_call(self):
        """Test forward call with different sequence lengths."""
        rope = RotaryEmbedding(64, 256)
        
        # Test full cache return
        cos_full, sin_full = rope()
        self.assertEqual(cos_full.shape, (256, 64))
        self.assertEqual(sin_full.shape, (256, 64))
        
        # Test partial sequence
        seq_len = 128
        cos_partial, sin_partial = rope(seq_len)
        self.assertEqual(cos_partial.shape, (seq_len, 64))
        self.assertEqual(sin_partial.shape, (seq_len, 64))
        
        # Verify partial is subset of full
        self.assertTrue(mx.allclose(cos_partial, cos_full[:seq_len]).item())
        self.assertTrue(mx.allclose(sin_partial, sin_full[:seq_len]).item())
    
    def test_dtype_casting(self):
        """Test dtype casting in forward call."""
        rope = RotaryEmbedding(32, 64)
        
        # Test different dtypes
        for dtype in [mx.float16, mx.float32]:
            cos, sin = rope(dtype=dtype)
            self.assertEqual(cos.dtype, dtype)
            self.assertEqual(sin.dtype, dtype)
    
    def test_extend_cache(self):
        """Test cache extension for longer sequences."""
        dim = 64
        initial_max = 128
        rope = RotaryEmbedding(dim, initial_max)
        
        # Verify initial cache size
        self.assertEqual(rope.cos_cached.shape[0], initial_max)
        
        # Extend cache
        new_max = 256
        rope.extend_cache(new_max)
        
        # Check extended size
        self.assertEqual(rope.max_position_embeddings, new_max)
        self.assertEqual(rope.cos_cached.shape, (new_max, dim))
        self.assertEqual(rope.sin_cached.shape, (new_max, dim))
        
        # Verify original positions unchanged
        rope_fresh = RotaryEmbedding(dim, initial_max)
        cos_orig, sin_orig = rope_fresh()
        cos_extended, sin_extended = rope(initial_max)
        
        self.assertTrue(mx.allclose(cos_orig, cos_extended, atol=1e-6).item())
        self.assertTrue(mx.allclose(sin_orig, sin_extended, atol=1e-6).item())
    
    def test_no_extend_when_sufficient(self):
        """Test that extend_cache does nothing when cache is sufficient."""
        rope = RotaryEmbedding(32, 256)
        original_cos = rope.cos_cached
        original_sin = rope.sin_cached
        
        # Try to extend to smaller size
        rope.extend_cache(128)
        
        # Verify nothing changed
        self.assertTrue(mx.array_equal(rope.cos_cached, original_cos))
        self.assertTrue(mx.array_equal(rope.sin_cached, original_sin))


class TestRotateHalf(unittest.TestCase):
    """Test suite for rotate_half function."""
    
    def test_basic_rotation(self):
        """Test basic rotate_half functionality."""
        # Test 1D case
        x = mx.array([1.0, 2.0, 3.0, 4.0])
        rotated = rotate_half(x)
        expected = mx.array([-3.0, -4.0, 1.0, 2.0])
        self.assertTrue(mx.allclose(rotated, expected).item())
        
        # Test 2D case
        x = mx.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]])
        rotated = rotate_half(x)
        expected = mx.array([[-3.0, -4.0, 1.0, 2.0],
                             [-7.0, -8.0, 5.0, 6.0]])
        self.assertTrue(mx.allclose(rotated, expected).item())
    
    def test_odd_dimension(self):
        """Test rotate_half with odd dimensions (should handle floor division)."""
        x = mx.array([1.0, 2.0, 3.0])
        rotated = rotate_half(x)
        # With dim=3, first half is [1] (index 0), second half is [2, 3] (indices 1, 2)
        # But when we split in half: x1 = x[:1] = [1], x2 = x[1:] = [2, 3]
        # Result should be concatenate([-x2, x1]) = concatenate([[-2, -3], [1]]) = [-2, -3, 1]
        expected = mx.array([-2.0, -3.0, 1.0])
        self.assertEqual(rotated.shape[0], 3)  
        self.assertTrue(mx.allclose(rotated, expected).item())
    
    def test_higher_dimensions(self):
        """Test rotate_half with higher dimensional tensors."""
        batch, seq_len, num_heads, head_dim = 2, 4, 3, 8
        x = mx.random.normal((batch, seq_len, num_heads, head_dim))
        
        rotated = rotate_half(x)
        
        # Check shape preserved
        self.assertEqual(rotated.shape, x.shape)
        
        # Manually verify rotation for first element
        x_np = np.array(x)
        expected_first_half = -x_np[..., head_dim//2:]
        expected_second_half = x_np[..., :head_dim//2]
        
        rotated_np = np.array(rotated)
        np.testing.assert_allclose(rotated_np[..., :head_dim//2], expected_first_half)
        np.testing.assert_allclose(rotated_np[..., head_dim//2:], expected_second_half)


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test suite for apply_rotary_pos_emb function."""
    
    def test_basic_application(self):
        """Test basic RoPE application."""
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 32
        
        # Create test tensors
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        
        # Create RoPE and get cos/sin
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(seq_len)
        
        # Apply RoPE
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Check shapes preserved
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
    
    def test_dtype_preservation(self):
        """Test that apply_rotary_pos_emb preserves input dtype."""
        q = mx.random.normal((1, 4, 2, 16), dtype=mx.float16)
        k = mx.random.normal((1, 4, 2, 16), dtype=mx.float16)
        
        rope = RotaryEmbedding(16)
        cos, sin = rope(4, dtype=mx.float32)  # Different dtype for cos/sin
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Should preserve original dtype
        self.assertEqual(q_rot.dtype, mx.float16)
        self.assertEqual(k_rot.dtype, mx.float16)
    
    def test_magnitude_preservation(self):
        """Test that rotation preserves vector magnitudes."""
        batch, seq_len, num_heads, head_dim = 1, 16, 1, 64
        
        # Create normalized vectors for easier testing
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        
        # Normalize
        q = q / mx.sqrt(mx.sum(q**2, axis=-1, keepdims=True))
        k = k / mx.sqrt(mx.sum(k**2, axis=-1, keepdims=True))
        
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(seq_len)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Check magnitude preservation
        q_magnitude = mx.sqrt(mx.sum(q**2, axis=-1))
        q_rot_magnitude = mx.sqrt(mx.sum(q_rot**2, axis=-1))
        
        k_magnitude = mx.sqrt(mx.sum(k**2, axis=-1))
        k_rot_magnitude = mx.sqrt(mx.sum(k_rot**2, axis=-1))
        
        self.assertTrue(mx.allclose(q_magnitude, q_rot_magnitude, rtol=1e-5).item())
        self.assertTrue(mx.allclose(k_magnitude, k_rot_magnitude, rtol=1e-5).item())
    
    def test_rotation_formula(self):
        """Test that rotation follows the expected formula."""
        # Simple case with known values
        head_dim = 4
        q = mx.ones((1, 1, 1, head_dim))
        k = mx.ones((1, 1, 1, head_dim))
        
        # Create simple cos/sin values
        cos = mx.array([[0.0, 1.0, 0.0, 1.0]])  # cos(90°), cos(0°), cos(90°), cos(0°)
        sin = mx.array([[1.0, 0.0, 1.0, 0.0]])  # sin(90°), sin(0°), sin(90°), sin(0°)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expected: q_rot = q * cos + rotate_half(q) * sin
        # rotate_half([1,1,1,1]) = [-1,-1,1,1]
        # q * cos = [1,1,1,1] * [0,1,0,1] = [0,1,0,1]
        # rotate_half(q) * sin = [-1,-1,1,1] * [1,0,1,0] = [-1,0,1,0]
        # q_rot = [0,1,0,1] + [-1,0,1,0] = [-1,1,1,1]
        expected = mx.array([[[-1.0, 1.0, 1.0, 1.0]]])
        
        self.assertTrue(mx.allclose(q_rot, expected, atol=1e-6).item())
    
    def test_relative_position_property(self):
        """Test that RoPE encodes relative positions correctly."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim)
        
        # Create identical vectors at different positions
        q = mx.ones((1, 2, 1, head_dim))  # Two positions
        k = mx.ones((1, 2, 1, head_dim))
        
        # Get embeddings for positions [5, 15]
        cos_all, sin_all = rope()
        cos = mx.stack([cos_all[5], cos_all[15]], axis=0)
        sin = mx.stack([sin_all[5], sin_all[15]], axis=0)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # The dot product between positions should encode relative distance
        # This is a key property of RoPE
        dot_product = mx.sum(q_rot[0, 0] * q_rot[0, 1], axis=-1)
        
        # Compare with direct relative position encoding (position 10 = 15-5)
        q_rel = mx.ones((1, 1, 1, head_dim))
        cos_rel = cos_all[10:11]
        sin_rel = sin_all[10:11]
        q_rel_rot, _ = apply_rotary_pos_emb(q_rel, q_rel, cos_rel, sin_rel)
        
        expected_dot = mx.sum(mx.ones((1, 1, head_dim)) * q_rel_rot[0, 0], axis=-1)
        
        # Should be approximately equal (relative position property)
        self.assertTrue(mx.allclose(dot_product, expected_dot[0, 0], rtol=1e-4).item())


class TestPyTorchCompatibility(unittest.TestCase):
    """Test exact compatibility with PyTorch HRM implementation."""
    
    def test_frequency_match(self):
        """Test that frequency calculation exactly matches PyTorch."""
        dim = 128
        base = 10000.0
        max_pos = 2048
        
        rope = RotaryEmbedding(dim, max_pos, base)
        
        # PyTorch formula: inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        # Verify a few specific values
        inv_freq_expected = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        
        # Extract actual frequencies from first position cos/sin
        # At position 0, cos=1 and sin=0 for all frequencies
        # At position 1, the angle is exactly inv_freq
        cos1 = np.array(rope.cos_cached[1])
        sin1 = np.array(rope.sin_cached[1])
        
        # Extract angles from first half (before duplication)
        angles = np.arctan2(sin1[:dim//2], cos1[:dim//2])
        
        np.testing.assert_allclose(angles, inv_freq_expected, rtol=1e-5)
    
    def test_duplication_pattern(self):
        """Test that frequency duplication matches PyTorch pattern."""
        dim = 8
        rope = RotaryEmbedding(dim, 10)
        
        cos, sin = rope()
        
        # Check that second half is duplicate of first half
        cos_np = np.array(cos)
        sin_np = np.array(sin)
        
        # Each position should have pattern [f0, f1, f2, f3, f0, f1, f2, f3]
        for pos in range(10):
            np.testing.assert_allclose(
                cos_np[pos, :dim//2], 
                cos_np[pos, dim//2:],
                rtol=1e-6
            )
            np.testing.assert_allclose(
                sin_np[pos, :dim//2], 
                sin_np[pos, dim//2:],
                rtol=1e-6
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_zero_dimension(self):
        """Test behavior with zero dimension (should work but be trivial)."""
        rope = RotaryEmbedding(0, 10)
        cos, sin = rope()
        
        self.assertEqual(cos.shape, (10, 0))
        self.assertEqual(sin.shape, (10, 0))
    
    def test_single_position(self):
        """Test with single position."""
        rope = RotaryEmbedding(16, 1)
        cos, sin = rope()
        
        self.assertEqual(cos.shape, (1, 16))
        self.assertEqual(sin.shape, (1, 16))
        
        # First position should have cos=1, sin=0 pattern
        self.assertTrue(mx.allclose(cos[0, 0], mx.array(1.0)).item())
        self.assertTrue(mx.allclose(sin[0, 0], mx.array(0.0)).item())
    
    def test_very_large_base(self):
        """Test with very large base (approaches no rotation)."""
        rope = RotaryEmbedding(32, 100, base=1e10)
        cos, sin = rope()
        
        # With huge base, frequencies approach 0, so cos≈1, sin≈0
        # But the lowest frequencies (first dimensions) still have some rotation
        # Check that at position 0, cos=1 and sin=0 (initial position)
        self.assertTrue(mx.allclose(cos[0], mx.ones(32)).item())
        self.assertTrue(mx.allclose(sin[0], mx.zeros(32)).item())
        
        # Check that higher frequency dimensions have less rotation
        # Last quarter of dimensions should have minimal rotation at early positions
        high_freq_cos = cos[:5, 24:]  # First 5 positions, last 8 dimensions
        high_freq_sin = sin[:5, 24:]
        self.assertTrue(mx.all(high_freq_cos > 0.99).item())
        self.assertTrue(mx.all(mx.abs(high_freq_sin) < 0.1).item())
    
    def test_small_base(self):
        """Test with small base (rapid rotation)."""
        rope = RotaryEmbedding(32, 100, base=2.0)
        cos, sin = rope()
        
        # With small base, should see more variation in cos/sin
        cos_var = mx.var(cos).item()
        sin_var = mx.var(sin).item()
        
        # Should have significant variation
        self.assertGreater(cos_var, 0.1)
        self.assertGreater(sin_var, 0.1)


if __name__ == "__main__":
    unittest.main()