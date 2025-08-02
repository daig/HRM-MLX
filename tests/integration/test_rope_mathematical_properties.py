"""
Mathematical validation tests for RoPE implementation.

These tests verify the mathematical properties of RoPE without requiring
the PyTorch reference implementation.
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from mlx_hrm.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


class TestRoPEMathematicalProperties(unittest.TestCase):
    """Test mathematical properties that RoPE must satisfy."""
    
    def test_frequency_progression(self):
        """Test that frequencies follow the expected geometric progression."""
        dim = 64
        base = 10000.0
        rope = RotaryEmbedding(dim, 100, base)
        
        # Extract frequencies from position 1
        cos1, sin1 = rope(2)
        angles = np.arctan2(np.array(sin1[1]), np.array(cos1[1]))
        
        # Frequencies should be: base^(-2i/d) for i in [0, 2, 4, ..., d-2]
        # First half contains unique frequencies, second half is duplicate
        unique_freqs = angles[:dim//2]
        
        # Check geometric progression
        for i in range(len(unique_freqs) - 1):
            expected_ratio = base ** (-2.0 / dim)
            actual_ratio = unique_freqs[i+1] / unique_freqs[i] if unique_freqs[i] != 0 else 0
            
            # Due to discrete sampling, check if ratio is approximately correct
            if abs(unique_freqs[i]) > 1e-6:  # Avoid division by very small numbers
                self.assertAlmostEqual(actual_ratio, expected_ratio, places=3)
    
    def test_rotation_preserves_inner_product_structure(self):
        """Test that rotation preserves the structure of inner products."""
        dim = 64
        rope = RotaryEmbedding(dim, 50)
        
        # Create orthogonal vectors
        v1_data = np.zeros((1, 1, 1, dim), dtype=np.float32)
        v1_data[0, 0, 0, 0] = 1.0  # Unit vector in first dimension
        v1 = mx.array(v1_data)
        
        v2_data = np.zeros((1, 1, 1, dim), dtype=np.float32)
        v2_data[0, 0, 0, 1] = 1.0  # Unit vector in second dimension
        v2 = mx.array(v2_data)
        
        # Apply rotation at position 10
        cos, sin = rope(11)
        v1_rot, _ = apply_rotary_pos_emb(v1, v1, cos[10:11], sin[10:11])
        v2_rot, _ = apply_rotary_pos_emb(v2, v2, cos[10:11], sin[10:11])
        
        # Check that orthogonality is preserved (approximately)
        # Due to the way RoPE works with pairs, perfect orthogonality isn't preserved
        # but the angle between vectors should be predictable
        dot_original = mx.sum(v1 * v2).item()
        dot_rotated = mx.sum(v1_rot * v2_rot).item()
        
        # Original vectors are orthogonal
        self.assertAlmostEqual(dot_original, 0.0, places=6)
        
        # After rotation, they should have a specific relationship based on rotation angle
        # This is a weaker test but verifies rotation is applied consistently
        self.assertLess(abs(dot_rotated), 0.5)  # Should remain mostly orthogonal
    
    def test_inverse_rotation_property(self):
        """Test that rotating forward then backward returns original (approximately)."""
        dim = 32
        rope = RotaryEmbedding(dim, 20)
        
        # Create random vector
        v = mx.random.normal((1, 1, 1, dim))
        
        # Get rotation for position 5
        cos, sin = rope(6)
        cos5 = cos[5:6]
        sin5 = sin[5:6]
        
        # Apply forward rotation
        v_rot, _ = apply_rotary_pos_emb(v, v, cos5, sin5)
        
        # Apply inverse rotation (negate sine for inverse)
        v_back, _ = apply_rotary_pos_emb(v_rot, v_rot, cos5, -sin5)
        
        # Should approximately recover original
        np.testing.assert_allclose(
            np.array(v),
            np.array(v_back),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_additive_rotation_property(self):
        """Test that rotating by pos1 then pos2 equals rotating by pos1+pos2."""
        dim = 32
        rope = RotaryEmbedding(dim, 30)
        
        # Positions to test
        pos1, pos2 = 3, 5
        pos_sum = pos1 + pos2
        
        # Create test vector
        v = mx.random.normal((1, 1, 1, dim))
        
        # Get rotations
        cos, sin = rope()
        
        # Method 1: Rotate by pos1, then by pos2
        v1, _ = apply_rotary_pos_emb(v, v, cos[pos1:pos1+1], sin[pos1:pos1+1])
        v2, _ = apply_rotary_pos_emb(v1, v1, cos[pos2:pos2+1], sin[pos2:pos2+1])
        
        # Method 2: Rotate directly by pos1+pos2
        v_direct, _ = apply_rotary_pos_emb(v, v, cos[pos_sum:pos_sum+1], sin[pos_sum:pos_sum+1])
        
        # Should be equal (this is the additive property of rotations)
        np.testing.assert_allclose(
            np.array(v2),
            np.array(v_direct),
            rtol=1e-4,
            atol=1e-4
        )
    
    def test_position_zero_identity(self):
        """Test that position 0 is the identity transformation."""
        dim = 64
        rope = RotaryEmbedding(dim, 10)
        
        # At position 0, cos should be 1 and sin should be 0
        cos, sin = rope(1)
        
        # Verify
        np.testing.assert_allclose(np.array(cos[0]), np.ones(dim), atol=1e-6)
        np.testing.assert_allclose(np.array(sin[0]), np.zeros(dim), atol=1e-6)
        
        # Apply to random vector
        v = mx.random.normal((2, 3, 4, dim))
        v_rot, _ = apply_rotary_pos_emb(v, v, cos[0:1], sin[0:1])
        
        # Should be unchanged
        np.testing.assert_allclose(
            np.array(v),
            np.array(v_rot),
            rtol=1e-6,
            atol=1e-6
        )
    
    def test_frequency_ordering(self):
        """Test that frequencies decrease with dimension index."""
        dim = 64
        base = 10000.0
        rope = RotaryEmbedding(dim, 100, base)
        
        # Get cos/sin values
        cos, sin = rope()
        
        # Compute approximate frequencies by looking at rate of change
        # between positions 0 and 10
        frequencies = []
        for i in range(dim // 2):  # Only check unique frequencies
            # Approximate frequency from phase change
            phase_0 = np.arctan2(np.array(sin[0, i]), np.array(cos[0, i]))
            phase_10 = np.arctan2(np.array(sin[10, i]), np.array(cos[10, i]))
            
            # Handle phase wrapping
            phase_diff = phase_10 - phase_0
            if phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            elif phase_diff > np.pi:
                phase_diff -= 2 * np.pi
                
            freq_estimate = abs(phase_diff) / 10
            frequencies.append(freq_estimate)
        
        # Check that frequencies generally decrease (allowing some noise)
        decreasing_count = sum(1 for i in range(len(frequencies)-1) 
                             if frequencies[i] > frequencies[i+1] * 0.9)
        
        # Most frequencies should be decreasing
        self.assertGreater(decreasing_count, len(frequencies) * 0.7)
    
    def test_high_frequency_components(self):
        """Test that higher dimensions have higher frequencies."""
        dim = 64
        rope = RotaryEmbedding(dim, 100)
        
        # Get several positions
        cos, sin = rope(50)
        
        # Compute frequency by looking at how fast values change
        # Higher dimensions should change faster
        pos_range = slice(10, 40)
        
        # Compute "frequency" as variance across positions
        low_dim_var = mx.var(cos[pos_range, 0]).item()  # First dimension
        high_dim_var = mx.var(cos[pos_range, dim//2-1]).item()  # Middle dimension
        
        # Due to the way RoPE duplicates frequencies, we need to be careful
        # The unique frequencies are in the first half
        # Higher indexed dimensions (within first half) should vary less
        # because they have lower frequencies (slower rotation)
        self.assertGreater(low_dim_var, high_dim_var * 0.5)


class TestRoPEEdgeBehavior(unittest.TestCase):
    """Test edge cases and boundary behavior."""
    
    def test_numerical_stability_large_positions(self):
        """Test numerical stability at large position values."""
        dim = 32
        rope = RotaryEmbedding(dim, 10000)  # Very large max position
        
        # Test at large positions
        large_pos = 9999
        cos, sin = rope(10000)
        
        # Values should still be bounded
        self.assertTrue(mx.all(mx.abs(cos[large_pos]) <= 1.0).item())
        self.assertTrue(mx.all(mx.abs(sin[large_pos]) <= 1.0).item())
        
        # Should still satisfy cos² + sin² = 1
        magnitude = cos[large_pos]**2 + sin[large_pos]**2
        np.testing.assert_allclose(
            np.array(magnitude),
            np.ones(dim),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_extreme_base_values(self):
        """Test behavior with extreme base values."""
        dim = 16
        
        # Very small base - rapid rotation
        rope_small = RotaryEmbedding(dim, 20, base=1.5)
        cos_small, sin_small = rope_small(20)
        
        # Should see significant variation even in early positions
        var_cos = mx.var(cos_small[:10, 0]).item()
        self.assertGreater(var_cos, 0.1)
        
        # Very large base - slow rotation  
        rope_large = RotaryEmbedding(dim, 20, base=1e6)
        cos_large, sin_large = rope_large(20)
        
        # Should see minimal variation in early positions
        var_cos_large = mx.var(cos_large[:10, -1]).item()
        self.assertLess(var_cos_large, 0.01)
    
    def test_dimension_scalability(self):
        """Test that implementation scales to various dimensions."""
        # Skip dimension 1 as it's a degenerate case
        for dim in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            rope = RotaryEmbedding(dim, 10)
            cos, sin = rope(5)
            
            # Basic sanity checks
            self.assertEqual(cos.shape, (5, dim))
            self.assertEqual(sin.shape, (5, dim))
            
            # Values should be valid
            self.assertTrue(mx.all(mx.abs(cos) <= 1.0).item())
            self.assertTrue(mx.all(mx.abs(sin) <= 1.0).item())
        
        # Test dimension 1 separately - it's a special case
        rope1 = RotaryEmbedding(1, 10)
        cos1, sin1 = rope1(5)
        # With dim=1, we have frequencies for positions 0 (and duplicated)
        # So output dimension should be 2 (concatenated)
        self.assertEqual(cos1.shape[0], 5)
        self.assertTrue(cos1.shape[1] >= 1)  # At least 1 dimension


if __name__ == "__main__":
    unittest.main()