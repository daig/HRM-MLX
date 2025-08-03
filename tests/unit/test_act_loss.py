"""Unit tests for ACT loss head."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_hrm.training import ACTLossHead, IGNORE_LABEL_ID
from mlx_hrm.models import create_hrm


class MockHRMModel(nn.Module):
    """Mock HRM model for testing ACT loss head."""
    
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        
    def initial_carry(self, batch_size):
        """Create mock initial carry state."""
        return type('MockCarry', (), {
            'halted': mx.zeros((batch_size,), dtype=mx.bool_),
            'steps': mx.ones((batch_size,))
        })()
    
    def __call__(self, carry, batch):
        """Mock forward pass."""
        batch_size, seq_len = batch['input_ids'].shape
        
        # Create mock outputs
        outputs = {
            'logits': mx.random.normal((batch_size, seq_len, self.vocab_size)),
            'q_halt_logits': mx.random.normal((batch_size,)),
            'q_continue_logits': mx.random.normal((batch_size,)),
            'target_q_continue': mx.random.uniform(0, 1, (batch_size,))
        }
        
        # Update carry
        new_carry = type('MockCarry', (), {
            'halted': mx.ones((batch_size,), dtype=mx.bool_),  # All halted
            'steps': mx.full((batch_size,), 5.0)  # 5 steps each
        })()
        
        return new_carry, outputs


class TestACTLossHead:
    """Test ACT loss head functionality."""
    
    def test_initialization(self):
        """Test ACT loss head initialization."""
        model = MockHRMModel()
        
        # Test with stablemax
        loss_head = ACTLossHead(model, loss_type='stablemax')
        assert loss_head.model is model
        
        # Test with softmax
        loss_head = ACTLossHead(model, loss_type='softmax')
        assert loss_head.model is model
        
        # Test invalid loss type
        with pytest.raises(ValueError):
            ACTLossHead(model, loss_type='invalid')
    
    def test_forward_pass(self):
        """Test complete forward pass through loss head."""
        model = MockHRMModel()
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        
        batch = {
            'input_ids': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
            'labels': mx.random.randint(0, vocab_size, (batch_size, seq_len))
        }
        
        carry = loss_head.initial_carry(batch_size)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        # Check outputs
        assert isinstance(total_loss, mx.array)
        assert total_loss.shape == ()  # Scalar
        assert mx.isfinite(total_loss)
        
        # Check metrics
        assert 'lm_loss' in metrics
        assert 'q_halt_loss' in metrics
        assert 'q_continue_loss' in metrics
        assert 'count' in metrics
        assert 'accuracy' in metrics
        assert 'exact_accuracy' in metrics
        assert 'q_halt_accuracy' in metrics
        assert 'steps' in metrics
        
        # Check outputs are detached
        assert all(mx.stop_gradient(v).item() == v.item() for v in outputs.values())
    
    def test_loss_computation(self):
        """Test that losses are computed correctly."""
        model = MockHRMModel()
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        batch = {
            'input_ids': mx.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'labels': mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        }
        
        carry = loss_head.initial_carry(2)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        # Total loss should be sum of components with weights
        expected_total = metrics['lm_loss'] + 0.5 * (metrics['q_halt_loss'] + metrics['q_continue_loss'])
        assert mx.allclose(total_loss, expected_total, rtol=1e-5)
    
    def test_masking(self):
        """Test that masking works correctly with ignore index."""
        model = MockHRMModel()
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        # Create batch with some ignored positions
        batch = {
            'input_ids': mx.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'labels': mx.array([[1, 2, IGNORE_LABEL_ID, IGNORE_LABEL_ID], 
                               [5, 6, 7, IGNORE_LABEL_ID]])
        }
        
        carry = loss_head.initial_carry(2)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        # Loss should still be computed correctly
        assert mx.isfinite(total_loss)
        assert total_loss > 0
    
    def test_metrics_computation(self):
        """Test that metrics are computed correctly."""
        model = MockHRMModel()
        loss_head = ACTLossHead(model, loss_type='softmax')
        
        batch_size = 8
        seq_len = 10
        
        batch = {
            'input_ids': mx.random.randint(0, 100, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 100, (batch_size, seq_len))
        }
        
        carry = loss_head.initial_carry(batch_size)
        
        # Test with metrics
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch, return_metrics=True)
        assert len(metrics) > 5  # Should have many metrics
        
        # Test without metrics
        new_carry2, total_loss2, metrics2, outputs2 = loss_head(carry, batch, return_metrics=False)
        assert len(metrics2) >= 2  # Should still have loss metrics
        assert 'count' not in metrics2
        assert 'accuracy' not in metrics2
    
    def test_different_loss_types(self):
        """Test that different loss types produce different results."""
        model = MockHRMModel()
        
        batch = {
            'input_ids': mx.array([[1, 2, 3, 4]]),
            'labels': mx.array([[1, 2, 3, 4]])
        }
        
        # Stablemax loss
        loss_head_stable = ACTLossHead(model, loss_type='stablemax')
        carry = loss_head_stable.initial_carry(1)
        _, loss_stable, _, _ = loss_head_stable(carry, batch)
        
        # Softmax loss
        loss_head_soft = ACTLossHead(model, loss_type='softmax')
        carry = loss_head_soft.initial_carry(1)
        _, loss_soft, _, _ = loss_head_soft(carry, batch)
        
        # Losses should be different but both valid
        assert not mx.allclose(loss_stable, loss_soft)
        assert mx.isfinite(loss_stable)
        assert mx.isfinite(loss_soft)


class TestACTLossHeadIntegration:
    """Integration tests with real model."""
    
    @pytest.mark.slow
    def test_with_real_model(self):
        """Test ACT loss head with actual HRM model."""
        # Create tiny model for testing
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        batch_size = 2
        seq_len = 8
        
        batch = {
            'input_ids': mx.random.randint(0, 100, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 100, (batch_size, seq_len)),
            'puzzle_id': mx.zeros((batch_size,), dtype=mx.int32)
        }
        
        carry = loss_head.initial_carry(batch_size)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        # Verify all components work together
        assert mx.isfinite(total_loss)
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, model.config.vocab_size)
        
        # Check gradient flow
        def loss_fn(model, batch):
            carry = model.initial_carry(batch['input_ids'].shape[0])
            _, loss, _, _ = model(carry, batch)
            return loss
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        loss_val, grads = value_and_grad_fn(loss_head, batch)
        
        # Gradients should exist
        assert grads is not None
        assert mx.isfinite(loss_val)


if __name__ == "__main__":
    pytest.main([__file__])