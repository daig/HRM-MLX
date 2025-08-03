"""Integration tests for loss functions with HRM model."""

import pytest
import mlx.core as mx
from mlx_hrm.models import create_hrm
from mlx_hrm.training import ACTLossHead, MetricsTracker


class TestLossIntegration:
    """Test loss functions integrated with full HRM model."""
    
    def test_full_training_step(self):
        """Test a complete training step with loss computation."""
        # Create small model
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        # Create batch
        batch_size = 2
        seq_len = 16
        vocab_size = model.config.vocab_size
        
        batch = {
            'input_ids': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
            'labels': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
            'puzzle_id': mx.zeros((batch_size,), dtype=mx.int32)
        }
        
        # Forward pass
        carry = loss_head.initial_carry(batch_size)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        # Verify outputs
        assert mx.isfinite(total_loss)
        assert total_loss > 0
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
        
        # Verify metrics
        assert all(mx.isfinite(v) for v in metrics.values())
        assert metrics['lm_loss'] > 0
        assert metrics['q_halt_loss'] > 0
        
        # Test gradient computation
        def loss_fn(model, batch):
            carry = model.initial_carry(batch['input_ids'].shape[0])
            _, loss, _, _ = model(carry, batch)
            return loss
        
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(loss_head, batch)
        
        # Check some gradients exist and are finite
        has_gradients = False
        for module_name, module_grads in grads.items():
            if module_grads:
                has_gradients = True
                for param_name, grad in module_grads.items():
                    assert mx.all(mx.isfinite(grad))
        
        assert has_gradients, "No gradients computed"
    
    def test_multiple_act_steps(self):
        """Test loss computation over multiple ACT steps."""
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        batch_size = 4
        seq_len = 8
        
        batch = {
            'input_ids': mx.random.randint(0, 100, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 100, (batch_size, seq_len)),
            'puzzle_id': mx.zeros((batch_size,), dtype=mx.int32)
        }
        
        carry = loss_head.initial_carry(batch_size)
        
        # Run multiple steps (simulating ACT)
        total_steps = 0
        for step in range(5):
            new_carry, loss, metrics, outputs = loss_head(carry, batch)
            
            # Verify step tracking
            if 'steps' in metrics:
                step_count = metrics['steps'].item()
                if metrics['count'].item() > 0:
                    avg_steps = step_count / metrics['count'].item()
                    assert avg_steps > 0
            
            carry = new_carry
            total_steps += 1
            
            # Check if all sequences have halted
            if mx.all(new_carry.halted):
                break
        
        assert total_steps >= 1, "Should run at least one step"
    
    def test_loss_with_different_configs(self):
        """Test loss computation with different model configurations."""
        configs = ['tiny', 'small']
        loss_types = ['stablemax', 'softmax']
        
        for config in configs:
            for loss_type in loss_types:
                model = create_hrm(config)
                loss_head = ACTLossHead(model, loss_type=loss_type)
                
                batch = {
                    'input_ids': mx.array([[1, 2, 3, 4]]),
                    'labels': mx.array([[1, 2, 3, 4]]),
                    'puzzle_id': mx.array([0])
                }
                
                carry = loss_head.initial_carry(1)
                _, loss, metrics, _ = loss_head(carry, batch)
                
                assert mx.isfinite(loss)
                assert loss > 0
                print(f"Config: {config}, Loss type: {loss_type}, Loss: {loss.item():.4f}")
    
    def test_metrics_tracking_integration(self):
        """Test metrics tracking over multiple batches."""
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        tracker = MetricsTracker()
        
        # Run several batches
        num_batches = 5
        for i in range(num_batches):
            batch = {
                'input_ids': mx.random.randint(0, 100, (2, 8)),
                'labels': mx.random.randint(0, 100, (2, 8)),
                'puzzle_id': mx.array([i % 3, i % 3])  # Cycle through 3 puzzle types
            }
            
            carry = loss_head.initial_carry(2)
            _, loss, metrics, _ = loss_head(carry, batch)
            
            tracker.update(metrics)
        
        # Check aggregated metrics
        current = tracker.get_current()
        assert current['count'] > 0
        assert 'lm_loss' in current
        assert 'accuracy' in current
        
        # Log epoch
        epoch_metrics = tracker.log_epoch()
        assert len(epoch_metrics) > 5
        
        # Verify history
        assert len(tracker.history) == 1
        assert tracker.history[0] == epoch_metrics
    
    def test_gradient_accumulation_pattern(self):
        """Test pattern for gradient accumulation."""
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        # Accumulate gradients over multiple batches
        accumulated_loss = 0.0
        num_accumulation_steps = 4
        
        for i in range(num_accumulation_steps):
            batch = {
                'input_ids': mx.random.randint(0, 100, (1, 8)),
                'labels': mx.random.randint(0, 100, (1, 8)),
                'puzzle_id': mx.array([0])
            }
            
            carry = loss_head.initial_carry(1)
            _, loss, _, _ = loss_head(carry, batch)
            
            # Accumulate loss (in real training, this would be used for gradients)
            accumulated_loss = accumulated_loss + loss / num_accumulation_steps
        
        assert mx.isfinite(accumulated_loss)
        assert accumulated_loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])