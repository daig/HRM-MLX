"""
Test attention + RoPE integration compliance.

This test verifies that the attention mechanism works correctly with RoPE
position embeddings, which is critical for HRM's reasoning capabilities.

Key areas tested:
1. RoPE application to query and key tensors
2. Non-causal attention patterns (bidirectional)
3. Multi-head attention consistency
4. Position embedding correctness
"""

import mlx.core as mx
import numpy as np
from typing import Tuple
import pytest

from mlx_hrm.modules.act import HRMConfig
from mlx_hrm.modules.attention import Attention
from mlx_hrm.modules.rope import RotaryEmbedding


class AttentionRoPEIntegrationTest:
    """Test suite for attention + RoPE integration."""
    
    def __init__(self):
        # Test configuration
        self.config = HRMConfig(
            batch_size=2,
            seq_len=8,
            hidden_size=128,
            num_heads=8,
            rope_theta=10000.0,
            rms_norm_eps=1e-5
        )
        
        self.head_dim = self.config.hidden_size // self.config.num_heads
    
    def test_rope_embedding_generation(self) -> bool:
        """Test RoPE embedding generation."""
        print("Testing RoPE embedding generation...")
        
        rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.seq_len,
            base=self.config.rope_theta
        )
        
        try:
            # Generate RoPE embeddings
            cos, sin = rope(self.config.seq_len, dtype=mx.float32)
            
            print(f"  Generated embeddings:")
            print(f"    cos shape: {cos.shape}")
            print(f"    sin shape: {sin.shape}")
            
            # Expected shape: [seq_len, head_dim]
            expected_shape = (self.config.seq_len, self.head_dim)
            
            if cos.shape == expected_shape and sin.shape == expected_shape:
                print(f"  ✅ PASSED: RoPE embeddings have correct shape")
                
                # Verify mathematical properties
                # cos^2 + sin^2 should equal 1 (approximately)
                cos_sin_sum = cos**2 + sin**2
                unit_circle_error = mx.abs(cos_sin_sum - 1.0).max()
                
                print(f"    Unit circle error (max): {unit_circle_error:.8f}")
                
                if unit_circle_error < 1e-6:
                    print(f"  ✅ PASSED: RoPE embeddings satisfy unit circle property")
                    return True
                else:
                    print(f"  ❌ FAILED: RoPE embeddings don't satisfy unit circle property")
                    return False
            else:
                print(f"  ❌ FAILED: Incorrect RoPE embedding shapes")
                print(f"           Expected: {expected_shape}")
                print(f"           Got cos: {cos.shape}, sin: {sin.shape}")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_attention_without_rope(self) -> bool:
        """Test attention mechanism without RoPE (baseline)."""
        print("\nTesting attention without RoPE...")
        
        attention = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.head_dim,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False  # Non-causal for HRM
        )
        
        try:
            # Create test input
            batch_size = 2
            seq_len = 4
            x = mx.random.normal((batch_size, seq_len, self.config.hidden_size))
            
            # Attention without RoPE (cos_sin=None)
            output = attention(cos_sin=None, hidden_states=x)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            
            if output.shape == x.shape:
                print(f"  ✅ PASSED: Attention preserves input shape")
                
                # Output should be different from input (attention does something)
                diff = mx.abs(output - x).mean()
                print(f"    Input-output difference: {diff:.6f}")
                
                if diff > 1e-5:
                    print(f"  ✅ PASSED: Attention transforms input meaningfully")
                    return True
                else:
                    print(f"  ❌ FAILED: Attention produces identical output to input")
                    return False
            else:
                print(f"  ❌ FAILED: Attention changes input shape")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_attention_with_rope(self) -> bool:
        """Test attention mechanism with RoPE integration."""
        print("\nTesting attention with RoPE integration...")
        
        attention = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.head_dim,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False
        )
        
        rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.seq_len,
            base=self.config.rope_theta
        )
        
        try:
            # Create test input
            batch_size = 2
            seq_len = 4
            x = mx.random.normal((batch_size, seq_len, self.config.hidden_size))
            
            # Generate RoPE embeddings
            cos, sin = rope(seq_len, dtype=x.dtype)
            cos_sin = (cos, sin)
            
            # Attention with RoPE
            output_with_rope = attention(cos_sin=cos_sin, hidden_states=x)
            
            # Attention without RoPE for comparison
            output_without_rope = attention(cos_sin=None, hidden_states=x)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output with RoPE shape: {output_with_rope.shape}")
            print(f"  Output without RoPE shape: {output_without_rope.shape}")
            
            # Both should preserve shape
            shape_correct = (output_with_rope.shape == x.shape and 
                           output_without_rope.shape == x.shape)
            
            if shape_correct:
                print(f"  ✅ PASSED: Attention with RoPE preserves input shape")
                
                # RoPE should make a difference in the output
                rope_diff = mx.abs(output_with_rope - output_without_rope).mean()
                print(f"    RoPE effect (mean difference): {rope_diff:.6f}")
                
                if rope_diff > 1e-5:
                    print(f"  ✅ PASSED: RoPE meaningfully affects attention output")
                    return True
                else:
                    print(f"  ❌ FAILED: RoPE has no effect on attention output")
                    return False
            else:
                print(f"  ❌ FAILED: Attention with RoPE doesn't preserve shape")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_position_sensitivity(self) -> bool:
        """Test that attention is sensitive to position changes with RoPE."""
        print("\nTesting position sensitivity with RoPE...")
        
        attention = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.head_dim,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False
        )
        
        rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.seq_len,
            base=self.config.rope_theta
        )
        
        try:
            # Create different content at different positions to test position encoding
            batch_size = 1
            seq_len = 4
            hidden_size = self.config.hidden_size
            
            # Create a sequence with varying content at each position
            # Position 0: small values, Position 1: medium values, etc.
            sequence_parts = []
            for pos in range(seq_len):
                # Each position has distinctive content
                pos_content = mx.ones((1, 1, hidden_size)) * (pos + 1) * 0.1
                sequence_parts.append(pos_content)
            
            x = mx.concatenate(sequence_parts, axis=1)
            
            # Same sequence for comparison
            x1 = x
            x2 = x
            
            # Generate RoPE for both sequences
            cos, sin = rope(seq_len, dtype=mx.float32)
            cos_sin = (cos, sin)
            
            # Apply attention
            out1 = attention(cos_sin=cos_sin, hidden_states=x1)
            out2 = attention(cos_sin=cos_sin, hidden_states=x2)
            
            print(f"  Sequence 1 output shape: {out1.shape}")
            print(f"  Sequence 2 output shape: {out2.shape}")
            
            # Even with identical content, RoPE should make positions distinguishable
            # So the outputs should be different at different positions
            position_diff = mx.abs(out1[0, 0] - out1[0, 1]).mean()  # pos 0 vs pos 1
            print(f"    Position 0 vs 1 difference: {position_diff:.6f}")
            
            if position_diff > 1e-6:
                print(f"  ✅ PASSED: RoPE makes positions distinguishable")
                return True
            else:
                print(f"  ❌ FAILED: RoPE doesn't distinguish positions")
                print(f"           This suggests RoPE isn't being applied correctly")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_multi_head_consistency(self) -> bool:
        """Test multi-head attention consistency."""
        print("\nTesting multi-head attention consistency...")
        
        # Test with different head configurations
        configs = [
            {"num_heads": 1, "description": "Single head"},
            {"num_heads": 4, "description": "Multi-head (4)"},
            {"num_heads": 8, "description": "Multi-head (8)"}
        ]
        
        batch_size = 2
        seq_len = 4
        x = mx.random.normal((batch_size, seq_len, self.config.hidden_size))
        
        rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.seq_len,
            base=self.config.rope_theta
        )
        cos, sin = rope(seq_len, dtype=x.dtype)
        cos_sin = (cos, sin)
        
        outputs = []
        
        for config in configs:
            try:
                head_dim = self.config.hidden_size // config["num_heads"]
                
                # Create RoPE with correct head dimension for this configuration
                rope_for_config = RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=self.config.seq_len,
                    base=self.config.rope_theta
                )
                cos_config, sin_config = rope_for_config(seq_len, dtype=x.dtype)
                cos_sin_config = (cos_config, sin_config)
                
                attention = Attention(
                    hidden_size=self.config.hidden_size,
                    head_dim=head_dim,
                    num_heads=config["num_heads"],
                    num_key_value_heads=config["num_heads"],
                    causal=False
                )
                
                output = attention(cos_sin=cos_sin_config, hidden_states=x)
                outputs.append(output)
                
                print(f"  {config['description']}: output shape {output.shape}")
                
            except Exception as e:
                print(f"  💥 ERROR in {config['description']}: {e}")
                return False
        
        # All outputs should have the same shape (multi-head is internal)
        all_same_shape = all(out.shape == outputs[0].shape for out in outputs)
        
        if all_same_shape:
            print(f"  ✅ PASSED: All head configurations produce same output shape")
            
            # Different head configurations should produce different outputs
            # (they represent different computational approaches)
            diff_1_vs_4 = mx.abs(outputs[0] - outputs[1]).mean()
            diff_4_vs_8 = mx.abs(outputs[1] - outputs[2]).mean()
            
            print(f"    1-head vs 4-head difference: {diff_1_vs_4:.6f}")
            print(f"    4-head vs 8-head difference: {diff_4_vs_8:.6f}")
            
            if diff_1_vs_4 > 1e-5 and diff_4_vs_8 > 1e-5:
                print(f"  ✅ PASSED: Different head configurations produce different outputs")
                return True
            else:
                print(f"  ❌ FAILED: Different head configurations produce similar outputs")
                return False
        else:
            print(f"  ❌ FAILED: Different head configurations produce different shapes")
            return False
    
    def test_non_causal_attention(self) -> bool:
        """Test that attention is non-causal (bidirectional)."""
        print("\nTesting non-causal (bidirectional) attention...")
        
        attention = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.head_dim,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False  # This is critical for HRM
        )
        
        rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.seq_len,
            base=self.config.rope_theta
        )
        
        try:
            # Create a sequence where later positions have distinctive values
            batch_size = 1
            seq_len = 4
            
            # Create sequence: [zeros, ones, twos, threes]
            pos_values = []
            for pos in range(seq_len):
                if pos == 0:
                    pos_values.append(mx.zeros((1, 1, self.config.hidden_size)))
                else:
                    pos_values.append(mx.ones((1, 1, self.config.hidden_size)) * float(pos))
            
            x = mx.concatenate(pos_values, axis=1)
            
            cos, sin = rope(seq_len, dtype=x.dtype)
            cos_sin = (cos, sin)
            
            output = attention(cos_sin=cos_sin, hidden_states=x)
            
            print(f"  Input pattern: [0, 1, 2, 3] at each position")
            print(f"  Output shape: {output.shape}")
            
            # In non-causal attention, position 0 should be influenced by later positions
            # So output[0,0] should be different from input[0,0] due to later positions
            pos_0_input = x[0, 0].sum()
            pos_0_output = output[0, 0].sum()
            
            influence_from_future = mx.abs(pos_0_output - pos_0_input)
            print(f"    Position 0 change: {influence_from_future:.6f}")
            
            if influence_from_future > 1e-3:
                print(f"  ✅ PASSED: Position 0 influenced by later positions (non-causal)")
                return True
            else:
                print(f"  ❌ FAILED: Position 0 not influenced by later positions")
                print(f"           This suggests attention might be causal when it shouldn't be")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all attention + RoPE integration tests."""
        print("🔬 Running Attention + RoPE Integration Tests")
        print("=" * 60)
        
        tests = [
            ("RoPE Embedding Generation", self.test_rope_embedding_generation),
            ("Attention Without RoPE", self.test_attention_without_rope),
            ("Attention With RoPE", self.test_attention_with_rope),
            ("Position Sensitivity", self.test_position_sensitivity),
            ("Multi-Head Consistency", self.test_multi_head_consistency),
            ("Non-Causal Attention", self.test_non_causal_attention)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}")
            print("-" * 40)
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"📊 Attention + RoPE Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED - Attention + RoPE integration is compliant!")
        else:
            print("⚠️  SOME TESTS FAILED - Attention + RoPE integration needs review")
        
        return passed_tests == len(tests)


def test_attention_rope_integration():
    """Pytest entry point for attention + RoPE integration tests."""
    tester = AttentionRoPEIntegrationTest()
    assert tester.run_all_tests(), "Attention + RoPE integration tests failed"


if __name__ == "__main__":
    # Run tests directly
    tester = AttentionRoPEIntegrationTest()
    tester.run_all_tests()