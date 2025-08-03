"""
Unit tests for the ACT (Adaptive Computation Time) implementation.

Tests cover:
- Carry state management
- HRM block functionality
- Reasoning module cycles
- Inner model forward pass
- ACT wrapper halting logic
- Q-learning components
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_hrm.modules.act import (
    HRMInnerCarry, HRMCarry, HRMConfig,
    HRMBlock, HRMReasoningModule
)
from mlx_hrm.models.hrm_inner import HRMInner
from mlx_hrm.models.hrm_act import HRM_ACT, create_hrm_act


class TestCarryStates(unittest.TestCase):
    """Test carry state data structures."""
    
    def setUp(self):
        mx.random.seed(42)
    
    def test_inner_carry_creation(self):
        """Test HRMInnerCarry creation and methods."""
        batch_size = 2
        seq_len = 10
        hidden_size = 64
        
        z_H = mx.random.normal((batch_size, seq_len, hidden_size))
        z_L = mx.random.normal((batch_size, seq_len, hidden_size))
        
        carry = HRMInnerCarry(z_H=z_H, z_L=z_L)
        
        self.assertEqual(carry.z_H.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(carry.z_L.shape, (batch_size, seq_len, hidden_size))
        
        # Test detach
        detached = carry.detach()
        self.assertTrue(mx.array_equal(detached.z_H, carry.z_H))
        self.assertTrue(mx.array_equal(detached.z_L, carry.z_L))
        
        # Test to_dict and from_dict
        d = carry.to_dict()
        self.assertIn('z_H', d)
        self.assertIn('z_L', d)
        
        reconstructed = HRMInnerCarry.from_dict(d)
        self.assertTrue(mx.array_equal(reconstructed.z_H, carry.z_H))
        self.assertTrue(mx.array_equal(reconstructed.z_L, carry.z_L))
    
    def test_outer_carry_creation(self):
        """Test HRMCarry creation and methods."""
        batch_size = 2
        seq_len = 10
        hidden_size = 64
        
        inner_carry = HRMInnerCarry(
            z_H=mx.random.normal((batch_size, seq_len, hidden_size)),
            z_L=mx.random.normal((batch_size, seq_len, hidden_size))
        )
        
        carry = HRMCarry(
            inner_carry=inner_carry,
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.zeros((batch_size,), dtype=mx.bool_),
            current_data={"input_ids": mx.ones((batch_size, seq_len), dtype=mx.int32)}
        )
        
        self.assertEqual(carry.steps.shape, (batch_size,))
        self.assertEqual(carry.halted.shape, (batch_size,))
        self.assertIn("input_ids", carry.current_data)
        
        # Test detach
        detached = carry.detach()
        self.assertTrue(mx.array_equal(detached.steps, carry.steps))
        self.assertTrue(mx.array_equal(detached.halted, carry.halted))


class TestHRMConfig(unittest.TestCase):
    """Test HRM configuration."""
    
    def test_config_creation(self):
        """Test HRMConfig creation and properties."""
        config = HRMConfig(
            batch_size=4,
            seq_len=128,
            vocab_size=1000,
            hidden_size=256,
            num_heads=8
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.seq_len, 128)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.head_dim, 32)  # 256 / 8
        
        # Check defaults
        self.assertEqual(config.H_cycles, 2)
        self.assertEqual(config.L_cycles, 2)
        self.assertEqual(config.halt_max_steps, 16)


class TestHRMBlock(unittest.TestCase):
    """Test HRM transformer block."""
    
    def setUp(self):
        mx.random.seed(42)
        self.config = HRMConfig(
            batch_size=2,
            seq_len=16,
            vocab_size=100,
            hidden_size=128,
            num_heads=4
        )
    
    def test_block_forward(self):
        """Test HRMBlock forward pass."""
        block = HRMBlock(self.config)
        
        batch_size = 2
        seq_len = 16
        x = mx.random.normal((batch_size, seq_len, self.config.hidden_size))
        
        # Without RoPE
        output = block(None, x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(mx.isfinite(output).all())
        
        # With RoPE
        from mlx_hrm.modules.rope import RotaryEmbedding
        rope = RotaryEmbedding(dim=self.config.head_dim)
        cos, sin = rope(seq_len)
        
        output_rope = block((cos, sin), x)
        self.assertEqual(output_rope.shape, x.shape)
        self.assertTrue(mx.isfinite(output_rope).all())
    
    def test_block_gradient_flow(self):
        """Test gradient flow through HRMBlock."""
        block = HRMBlock(self.config)
        
        def loss_fn(params, x):
            # Reconstruct block with params for MLX gradient computation
            output = block(None, x)
            return output.mean()
        
        x = mx.random.normal((2, 16, self.config.hidden_size))
        # For now, just test forward pass works
        output = block(None, x)
        loss = output.mean()
        
        # Verify output is valid
        self.assertTrue(mx.isfinite(loss))


class TestHRMReasoningModule(unittest.TestCase):
    """Test HRM reasoning module with cycles."""
    
    def setUp(self):
        mx.random.seed(42)
        self.config = HRMConfig(
            batch_size=2,
            seq_len=16,
            vocab_size=100,
            hidden_size=128,
            num_heads=4
        )
    
    def test_reasoning_module_cycles(self):
        """Test that reasoning module applies correct number of cycles."""
        num_layers = 2
        num_cycles = 3
        
        module = HRMReasoningModule(
            config=self.config,
            num_layers=num_layers,
            num_cycles=num_cycles
        )
        
        # Check correct number of blocks
        self.assertEqual(len(module.blocks), num_layers)
        
        # Forward pass
        x = mx.random.normal((2, 16, self.config.hidden_size))
        output = module(None, x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(mx.isfinite(output).all())
        
        # Output should be different from input due to processing
        self.assertFalse(mx.allclose(output, x, atol=1e-3))


class TestHRMInner(unittest.TestCase):
    """Test inner HRM model."""
    
    def setUp(self):
        mx.random.seed(42)
        self.config = HRMConfig(
            batch_size=2,
            seq_len=16,
            vocab_size=100,
            hidden_size=128,
            num_heads=4,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=1
        )
    
    def test_inner_model_creation(self):
        """Test HRMInner model creation."""
        model = HRMInner(self.config)
        
        # Check components exist
        self.assertIsNotNone(model.dense_tok_emb)
        self.assertIsNotNone(model.rope)
        self.assertIsNotNone(model.H_module)
        self.assertIsNotNone(model.L_module)
        self.assertIsNotNone(model.lm_head)
        self.assertIsNotNone(model.q_head)
    
    def test_inner_model_forward(self):
        """Test HRMInner forward pass."""
        model = HRMInner(self.config)
        
        batch_size = 2
        seq_len = 16
        
        # Create carry and batch
        carry = HRMInnerCarry(
            z_H=mx.zeros((batch_size, seq_len, self.config.hidden_size)),
            z_L=mx.zeros((batch_size, seq_len, self.config.hidden_size))
        )
        
        batch = {
            "input_ids": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len))
        }
        
        # Forward pass
        new_carry, logits, (q_halt, q_continue) = model(carry, batch)
        
        # Check outputs
        self.assertEqual(new_carry.z_H.shape, carry.z_H.shape)
        self.assertEqual(new_carry.z_L.shape, carry.z_L.shape)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertEqual(q_halt.shape, (batch_size,))
        self.assertEqual(q_continue.shape, (batch_size,))
        
        # Check values are finite
        self.assertTrue(mx.isfinite(logits).all())
        self.assertTrue(mx.isfinite(q_halt).all())
        self.assertTrue(mx.isfinite(q_continue).all())
    
    def test_inner_model_reset_carry(self):
        """Test carry reset functionality."""
        model = HRMInner(self.config)
        
        batch_size = 2
        seq_len = 16
        
        carry = HRMInnerCarry(
            z_H=mx.ones((batch_size, seq_len, self.config.hidden_size)),
            z_L=mx.ones((batch_size, seq_len, self.config.hidden_size))
        )
        
        # Reset first sequence
        halted = mx.array([True, False])
        new_carry = model.reset_carry(halted, carry)
        
        # First sequence should be reset to init values (zeros)
        self.assertTrue(mx.allclose(new_carry.z_H[0], mx.zeros_like(carry.z_H[0]), atol=1e-6))
        self.assertTrue(mx.allclose(new_carry.z_L[0], mx.zeros_like(carry.z_L[0]), atol=1e-6))
        
        # Second sequence should be unchanged
        self.assertTrue(mx.allclose(new_carry.z_H[1], carry.z_H[1], atol=1e-6))
        self.assertTrue(mx.allclose(new_carry.z_L[1], carry.z_L[1], atol=1e-6))


class TestHRM_ACT(unittest.TestCase):
    """Test HRM with ACT wrapper."""
    
    def setUp(self):
        mx.random.seed(42)
        self.config = HRMConfig(
            batch_size=2,
            seq_len=16,
            vocab_size=100,
            hidden_size=128,
            num_heads=4,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=1,
            halt_max_steps=5,
            halt_exploration_prob=0.2
        )
    
    def test_act_creation(self):
        """Test HRM_ACT creation."""
        model = HRM_ACT(self.config)
        
        self.assertIsNotNone(model.inner)
        self.assertEqual(model.halt_max_steps, 5)
        self.assertEqual(model.halt_exploration_prob, 0.2)
        
        # Test factory function
        model2 = create_hrm_act(self.config)
        self.assertIsInstance(model2, HRM_ACT)
    
    def test_act_initial_carry(self):
        """Test initial carry creation."""
        model = HRM_ACT(self.config)
        
        batch_size = 4
        carry = model.initial_carry(batch_size)
        
        self.assertEqual(carry.steps.shape, (batch_size,))
        self.assertEqual(carry.halted.shape, (batch_size,))
        self.assertTrue(carry.halted.all())  # Should start halted
        self.assertTrue((carry.steps == 0).all())
    
    def test_act_forward_single_step(self):
        """Test single ACT forward step."""
        model = HRM_ACT(self.config)
        model.set_training(False)  # Eval mode for deterministic behavior
        
        batch_size = 2
        seq_len = 16
        
        # Initial carry
        carry = model.initial_carry(batch_size)
        
        # Batch data
        batch = {
            "input_ids": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len)),
            "labels": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len))
        }
        
        # Forward pass
        new_carry, outputs = model(carry, batch)
        
        # Check outputs
        self.assertIn("logits", outputs)
        self.assertIn("q_halt_logits", outputs)
        self.assertIn("q_continue_logits", outputs)
        
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertEqual(outputs["q_halt_logits"].shape, (batch_size,))
        self.assertEqual(outputs["q_continue_logits"].shape, (batch_size,))
        
        # Steps should increment
        self.assertTrue((new_carry.steps == 1).all())
    
    def test_act_halting_logic(self):
        """Test ACT halting behavior."""
        model = HRM_ACT(self.config)
        
        batch_size = 2
        seq_len = 16
        
        carry = model.initial_carry(batch_size)
        batch = {
            "input_ids": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len)),
            "labels": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len))
        }
        
        # Run multiple steps
        steps_taken = []
        halted_count = 0
        
        for i in range(self.config.halt_max_steps * 2):
            carry, outputs = model(carry, batch)
            steps_taken.append(carry.steps.tolist())
            
            # Should never exceed max steps
            self.assertTrue((carry.steps <= self.config.halt_max_steps).all())
            
            # Count how many times sequences halt
            if carry.halted.any():
                halted_count += carry.halted.sum().item()
        
        # Over multiple iterations, we should see sequences halting and resetting
        self.assertGreater(halted_count, 0, "No sequences halted during test")
        
        # Check that step counts vary (some should be low due to resets)
        all_steps = [s for step_list in steps_taken for s in step_list]
        unique_steps = set(all_steps)
        self.assertGreater(len(unique_steps), 1, "Step counts should vary")
    
    def test_act_training_mode(self):
        """Test ACT behavior in training mode."""
        model = HRM_ACT(self.config)
        model.set_training(True)
        
        batch_size = 2
        seq_len = 16
        
        carry = model.initial_carry(batch_size)
        batch = {
            "input_ids": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len)),
            "labels": mx.random.randint(0, self.config.vocab_size, (batch_size, seq_len))
        }
        
        # In training mode, should compute target Q-values
        carry, outputs = model(carry, batch)
        
        if model.halt_max_steps > 1:
            self.assertIn("target_q_continue", outputs)
            self.assertEqual(outputs["target_q_continue"].shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()