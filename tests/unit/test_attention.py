"""
Unit tests for the Attention module.

Tests cover:
- Shape preservation for various configurations
- Grouped Query Attention (GQA) functionality
- RoPE integration
- Causal vs non-causal masking
- Gradient flow
- Edge cases and error handling
"""

import unittest
import math

import mlx.core as mx
import mlx.nn as nn

from mlx_hrm.modules.attention import Attention, repeat_kv
from mlx_hrm.modules.rope import RotaryEmbedding


class TestAttention(unittest.TestCase):
    """Test cases for the Attention module."""
    
    def setUp(self):
        """Set random seed for reproducibility."""
        mx.random.seed(42)
    
    def test_repeat_kv_no_repetition(self):
        """Test repeat_kv when n_rep=1 (no repetition)."""
        batch, seq_len, n_heads, head_dim = 2, 8, 4, 16
        x = mx.random.normal((batch, seq_len, n_heads, head_dim))
        
        result = repeat_kv(x, n_rep=1)
        self.assertTrue(mx.array_equal(result, x))
    
    def test_repeat_kv_with_repetition(self):
        """Test repeat_kv with various repetition factors."""
        batch, seq_len, n_kv_heads, head_dim = 2, 8, 2, 16
        x = mx.random.normal((batch, seq_len, n_kv_heads, head_dim))
        
        # Test 2x repetition
        result = repeat_kv(x, n_rep=2)
        self.assertEqual(result.shape, (batch, seq_len, n_kv_heads * 2, head_dim))
        
        # Verify repetition pattern
        for i in range(n_kv_heads):
            self.assertTrue(mx.allclose(
                result[:, :, i*2], 
                result[:, :, i*2+1],
                atol=1e-6
            ))
        
        # Test 4x repetition
        result = repeat_kv(x, n_rep=4)
        self.assertEqual(result.shape, (batch, seq_len, n_kv_heads * 4, head_dim))
    
    def test_attention_initialization(self):
        """Test Attention module initialization."""
        hidden_size = 768
        num_heads = 12
        head_dim = hidden_size // num_heads
        
        # Standard MHA
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        self.assertEqual(attn.hidden_size, hidden_size)
        self.assertEqual(attn.head_dim, head_dim)
        self.assertEqual(attn.num_heads, num_heads)
        self.assertEqual(attn.num_key_value_heads, num_heads)
        self.assertEqual(attn.output_size, head_dim * num_heads)
        self.assertEqual(attn.n_rep, 1)
        self.assertFalse(attn.causal)
        
        # GQA with 4:1 ratio
        attn_gqa = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=16,
            num_key_value_heads=4,
            causal=True
        )
        
        self.assertEqual(attn_gqa.num_heads, 16)
        self.assertEqual(attn_gqa.num_key_value_heads, 4)
        self.assertEqual(attn_gqa.n_rep, 4)
        self.assertTrue(attn_gqa.causal)
    
    def test_attention_shapes(self):
        """Test that attention produces correct output shapes."""
        configs = [
            # (batch_size, seq_len, hidden_size, num_heads, num_kv_heads)
            (2, 128, 512, 8, 8),    # Standard MHA
            (4, 256, 768, 12, 12),  # Larger MHA
            (2, 128, 512, 8, 2),    # GQA 4:1
            (1, 512, 1024, 16, 4),  # GQA 4:1, larger
        ]
        
        for batch_size, seq_len, hidden_size, num_heads, num_kv_heads in configs:
            head_dim = hidden_size // num_heads
            
            attn = Attention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                causal=False
            )
            
            x = mx.random.normal((batch_size, seq_len, hidden_size))
            output = attn(None, x)
            
            self.assertEqual(
                output.shape, 
                (batch_size, seq_len, hidden_size),
                f"Failed for config: batch={batch_size}, seq={seq_len}, "
                f"hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}"
            )
    
    def test_attention_with_rope(self):
        """Test attention with RoPE integration."""
        batch_size, seq_len, hidden_size = 2, 64, 256
        num_heads = 8
        head_dim = hidden_size // num_heads
        
        # Create attention and RoPE
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        rope = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=128,
            base=10000.0
        )
        
        # Forward pass
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        cos, sin = rope(seq_len, dtype=x.dtype)
        output = attn((cos, sin), x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        self.assertTrue(mx.isfinite(output).all())
    
    def test_causal_vs_non_causal(self):
        """Test causal vs non-causal attention patterns."""
        batch_size, seq_len, hidden_size = 1, 8, 64
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        # Create both causal and non-causal attention
        attn_causal = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=True
        )
        
        attn_non_causal = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        # Same input
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        
        output_causal = attn_causal(None, x)
        output_non_causal = attn_non_causal(None, x)
        
        # Outputs should be different due to masking
        self.assertFalse(mx.allclose(output_causal, output_non_causal, atol=1e-3))
        
        # Both should have correct shapes
        self.assertEqual(output_causal.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(output_non_causal.shape, (batch_size, seq_len, hidden_size))
    
    def test_gqa_functionality(self):
        """Test grouped query attention with different configurations."""
        batch_size, seq_len, hidden_size = 2, 128, 512
        
        configs = [
            (8, 8, 1),   # Standard MHA
            (8, 4, 2),   # GQA 2:1
            (8, 2, 4),   # GQA 4:1
            (16, 4, 4),  # GQA 4:1, more heads
        ]
        
        for num_heads, num_kv_heads, expected_n_rep in configs:
            head_dim = hidden_size // num_heads
            
            attn = Attention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                causal=False
            )
            
            self.assertEqual(attn.n_rep, expected_n_rep)
            
            x = mx.random.normal((batch_size, seq_len, hidden_size))
            output = attn(None, x)
            
            self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
            self.assertTrue(mx.isfinite(output).all())
    
    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        batch_size, seq_len, hidden_size = 1, 32, 128
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        def loss_fn(attn, x):
            output = attn(None, x)
            return output.mean()
        
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        
        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(attn, x)
        
        # Check that gradients exist for all parameters
        self.assertIn("qkv_proj", grads)
        self.assertIn("o_proj", grads)
        
        # Check gradient shapes
        qkv_weight_grad = grads["qkv_proj"]["weight"]
        o_weight_grad = grads["o_proj"]["weight"]
        
        self.assertEqual(
            qkv_weight_grad.shape,
            attn.qkv_proj.weight.shape
        )
        self.assertEqual(
            o_weight_grad.shape,
            attn.o_proj.weight.shape
        )
        
        # Check gradients are finite
        self.assertTrue(mx.isfinite(qkv_weight_grad).all())
        self.assertTrue(mx.isfinite(o_weight_grad).all())
    
    def test_attention_numerical_stability(self):
        """Test attention with extreme values."""
        batch_size, seq_len, hidden_size = 1, 16, 64
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        # Test with very small values
        x_small = mx.ones((batch_size, seq_len, hidden_size)) * 1e-6
        output_small = attn(None, x_small)
        self.assertTrue(mx.isfinite(output_small).all())
        
        # Test with large values (but not too large to avoid overflow)
        x_large = mx.ones((batch_size, seq_len, hidden_size)) * 10
        output_large = attn(None, x_large)
        self.assertTrue(mx.isfinite(output_large).all())
    
    def test_attention_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test invalid GQA configuration
        with self.assertRaises(AssertionError):
            Attention(
                hidden_size=512,
                head_dim=64,
                num_heads=8,
                num_key_value_heads=3,  # 8 not divisible by 3
                causal=False
            )
    
    def test_attention_dtype_handling(self):
        """Test attention with different data types."""
        batch_size, seq_len, hidden_size = 1, 32, 128
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        # Test with float32
        x_f32 = mx.random.normal((batch_size, seq_len, hidden_size), dtype=mx.float32)
        output_f32 = attn(None, x_f32)
        self.assertEqual(output_f32.dtype, mx.float32)
        
        # Test with float16 - Note: MLX attention internally uses float32 for stability
        x_f16 = x_f32.astype(mx.float16)
        output_f16 = attn(None, x_f16)
        # Output dtype may be float32 due to internal computations
        self.assertIn(output_f16.dtype, [mx.float16, mx.float32])
        
        # Test with bfloat16 if available
        if hasattr(mx, 'bfloat16'):
            x_bf16 = x_f32.astype(mx.bfloat16)
            output_bf16 = attn(None, x_bf16)
            self.assertIn(output_bf16.dtype, [mx.bfloat16, mx.float32])
    
    def test_shape_info(self):
        """Test shape_info method."""
        # Standard MHA
        attn = Attention(
            hidden_size=768,
            head_dim=64,
            num_heads=12,
            num_key_value_heads=12,
            causal=False
        )
        info = attn.shape_info()
        self.assertIn("hidden_size=768", info)
        self.assertIn("num_heads=12", info)
        self.assertNotIn("GQA", info)
        self.assertNotIn("causal", info)
        
        # GQA with causal
        attn_gqa = Attention(
            hidden_size=512,
            head_dim=64,
            num_heads=8,
            num_key_value_heads=2,
            causal=True
        )
        info = attn_gqa.shape_info()
        self.assertIn("GQA 4:1", info)
        self.assertIn("causal", info)


if __name__ == "__main__":
    unittest.main()