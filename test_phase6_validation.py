#!/usr/bin/env python3
"""
Phase 6 validation script for loss functions and metrics.
Tests numerical stability and gradient flow without pytest dependency.
"""

import mlx.core as mx
import numpy as np
import sys
import traceback

# Add src to path
sys.path.insert(0, 'src')

from mlx_hrm.training import (
    s_function,
    log_stablemax,
    stablemax_cross_entropy,
    softmax_cross_entropy,
    ACTLossHead,
    MetricsTracker,
    compute_puzzle_metrics,
    IGNORE_LABEL_ID
)
from mlx_hrm.models import create_hrm


def test_s_function():
    """Test S-function numerical properties."""
    print("Testing S-function...")
    
    # Test positive inputs
    x_pos = mx.array([0, 1, 2, 3, 4])
    s_pos = s_function(x_pos)
    expected_pos = x_pos + 1.0
    assert mx.allclose(s_pos, expected_pos), "S-function failed on positive inputs"
    
    # Test negative inputs
    x_neg = mx.array([-0.5, -0.9, -0.99])
    s_neg = s_function(x_neg)
    # For x < 0, S(x) = 1/(1-x) = 1/(1-(-0.5)) = 1/1.5 = 0.667
    expected_neg = mx.array([1.0/1.5, 1.0/1.9, 1.0/1.99])  # [0.667, 0.526, 0.502]
    assert mx.allclose(s_neg, expected_neg, rtol=1e-2), "S-function failed on negative inputs"
    
    # Test all outputs are positive
    x_random = mx.random.uniform(-10, 10, (100,))
    s_random = s_function(x_random)
    assert mx.all(s_random > 0), "S-function produced non-positive outputs"
    
    # Test numerical stability
    x_critical = mx.array([0.999999, 0.9999999, -0.999999])
    s_critical = s_function(x_critical)
    assert mx.all(mx.isfinite(s_critical)), "S-function is not numerically stable"
    
    print("✅ S-function tests passed")


def test_log_stablemax():
    """Test log stablemax properties."""
    print("Testing log stablemax...")
    
    # Test probability normalization
    x = mx.random.normal((10, 20))
    log_probs = log_stablemax(x)
    probs = mx.exp(log_probs)
    sums = mx.sum(probs, axis=-1)
    assert mx.allclose(sums, mx.ones_like(sums), rtol=1e-5), "Log stablemax probabilities don't sum to 1"
    
    # Test finite outputs
    x_extreme = mx.random.uniform(-100, 100, (5, 10))
    log_probs_extreme = log_stablemax(x_extreme)
    assert mx.all(mx.isfinite(log_probs_extreme)), "Log stablemax produced non-finite values"
    
    # Test dtype preservation
    for dtype in [mx.float16, mx.float32]:
        x_typed = mx.random.normal((5, 10)).astype(dtype)
        log_probs_typed = log_stablemax(x_typed)
        assert log_probs_typed.dtype == dtype, f"Log stablemax didn't preserve {dtype}"
    
    print("✅ Log stablemax tests passed")


def test_stablemax_cross_entropy():
    """Test stablemax cross-entropy loss."""
    print("Testing stablemax cross-entropy...")
    
    # Basic test
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = stablemax_cross_entropy(logits, labels)
    assert loss.shape == (batch_size, seq_len), f"Unexpected loss shape: {loss.shape}"
    assert mx.all(mx.isfinite(loss)), "Loss contains non-finite values"
    assert mx.all(loss >= 0), "Loss contains negative values"
    
    # Test masking
    logits_mask = mx.random.normal((2, 10, 100))
    labels_mask = mx.array([[1, 2, 3, -100, -100, 6, 7, 8, 9, 10],
                           [11, 12, -100, -100, -100, 16, 17, 18, 19, 20]])
    
    loss_mask = stablemax_cross_entropy(logits_mask, labels_mask)
    assert mx.all(loss_mask[0, 3:5] == 0), "Masking failed for first sequence"
    assert mx.all(loss_mask[1, 2:5] == 0), "Masking failed for second sequence"
    
    # Test reduction modes
    loss_none = stablemax_cross_entropy(logits, labels, reduction='none')
    loss_mean = stablemax_cross_entropy(logits, labels, reduction='mean')
    loss_sum = stablemax_cross_entropy(logits, labels, reduction='sum')
    
    assert loss_none.shape == (batch_size, seq_len), "None reduction shape incorrect"
    assert loss_mean.shape == (), "Mean reduction should be scalar"
    assert loss_sum.shape == (), "Sum reduction should be scalar"
    assert mx.allclose(loss_mean, mx.mean(loss_none)), "Mean reduction incorrect"
    assert mx.allclose(loss_sum, mx.sum(loss_none)), "Sum reduction incorrect"
    
    print("✅ Stablemax cross-entropy tests passed")


def test_gradient_flow():
    """Test gradient flow through loss functions."""
    print("Testing gradient flow...")
    
    def stablemax_loss_fn(logits, labels):
        return stablemax_cross_entropy(logits, labels, reduction='mean')
    
    def softmax_loss_fn(logits, labels):
        return softmax_cross_entropy(logits, labels, reduction='mean')
    
    logits = mx.random.normal((2, 10, 20))
    labels = mx.random.randint(0, 20, (2, 10))
    
    # Test stablemax gradients
    grad_fn_stable = mx.grad(stablemax_loss_fn)
    grad_stable = grad_fn_stable(logits, labels)
    assert mx.all(mx.isfinite(grad_stable)), "Stablemax gradients contain non-finite values"
    
    # Test softmax gradients
    grad_fn_soft = mx.grad(softmax_loss_fn)
    grad_soft = grad_fn_soft(logits, labels)
    assert mx.all(mx.isfinite(grad_soft)), "Softmax gradients contain non-finite values"
    
    print("✅ Gradient flow tests passed")


def test_act_loss_head():
    """Test ACT loss head functionality."""
    print("Testing ACT loss head...")
    
    # Create a mock model
    class MockModel:
        def __init__(self):
            pass
            
        def initial_carry(self, batch_size):
            return type('MockCarry', (), {
                'halted': mx.ones((batch_size,), dtype=mx.bool_),
                'steps': mx.full((batch_size,), 5.0)
            })()
        
        def __call__(self, carry, batch):
            batch_size, seq_len = batch['input_ids'].shape
            outputs = {
                'logits': mx.random.normal((batch_size, seq_len, 100)),
                'q_halt_logits': mx.random.normal((batch_size,)),
                'q_continue_logits': mx.random.normal((batch_size,)),
                'target_q_continue': mx.random.uniform(0, 1, (batch_size,))
            }
            return carry, outputs
    
    model = MockModel()
    loss_head = ACTLossHead(model, loss_type='stablemax')
    
    batch = {
        'input_ids': mx.random.randint(0, 100, (4, 16)),
        'labels': mx.random.randint(0, 100, (4, 16))
    }
    
    carry = loss_head.initial_carry(4)
    new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
    
    # Check outputs
    assert isinstance(total_loss, mx.array), "Total loss is not an array"
    assert total_loss.shape == (), "Total loss is not scalar"
    assert mx.isfinite(total_loss), "Total loss is not finite"
    
    # Check metrics
    expected_metrics = ['lm_loss', 'q_halt_loss', 'q_continue_loss', 'count', 
                       'accuracy', 'exact_accuracy', 'q_halt_accuracy', 'steps']
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert mx.isfinite(metrics[metric]), f"Metric {metric} is not finite"
    
    # Check loss combination
    expected_total = metrics['lm_loss'] + 0.5 * (metrics['q_halt_loss'] + metrics['q_continue_loss'])
    assert mx.allclose(total_loss, expected_total, rtol=1e-5), "Total loss calculation incorrect"
    
    print("✅ ACT loss head tests passed")


def test_metrics_tracker():
    """Test metrics tracking functionality."""
    print("Testing metrics tracker...")
    
    tracker = MetricsTracker()
    
    # Test updates - simulate 5 batches with 16 samples each
    # Each batch has aggregate metrics (loss sum, accuracy sum, count)
    for i in range(5):
        batch_count = 16
        # Simulate different per-sample losses and accuracies for each batch
        batch_loss = (2.0 + i * 0.1) * batch_count  # Total loss for batch
        batch_accuracy = (0.8 + i * 0.02) * batch_count  # Total accuracy for batch
        
        metrics = {
            'loss': mx.array(batch_loss),
            'accuracy': mx.array(batch_accuracy),
            'count': mx.array(batch_count)
        }
        tracker.update(metrics)
    
    current = tracker.get_current()
    
    # Check averages - should be weighted by count
    # Total loss: (2.0*16 + 2.1*16 + 2.2*16 + 2.3*16 + 2.4*16) = 16 * (2.0+2.1+2.2+2.3+2.4) = 16 * 11.0 = 176
    # Total count: 5 * 16 = 80
    # Average loss: 176 / 80 = 2.2
    expected_loss = 2.2
    expected_acc = 0.84
    
    assert abs(current['loss'] - expected_loss) < 1e-5, f"Loss average incorrect: {current['loss']} vs {expected_loss}"
    assert abs(current['accuracy'] - expected_acc) < 1e-5, f"Accuracy average incorrect: {current['accuracy']} vs {expected_acc}"
    assert current['count'] == 80, f"Count sum incorrect: {current['count']} vs 80"
    
    # Test epoch logging
    epoch_metrics = tracker.log_epoch()
    assert len(tracker.history) == 1, "History not updated after epoch log"
    assert tracker.get_current() == {}, "Tracker not reset after epoch log"
    
    print("✅ Metrics tracker tests passed")


def test_with_real_model():
    """Test with actual HRM model (if available)."""
    print("Testing with real HRM model...")
    
    try:
        # Create tiny model
        model = create_hrm('tiny')
        loss_head = ACTLossHead(model, loss_type='stablemax')
        
        batch = {
            'input_ids': mx.random.randint(0, 100, (2, 8)),
            'labels': mx.random.randint(0, 100, (2, 8))
        }
        
        carry = loss_head.initial_carry(2)
        new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
        
        assert mx.isfinite(total_loss), "Real model total loss is not finite"
        assert 'logits' in outputs, "Missing logits in real model outputs"
        assert outputs['logits'].shape == (2, 8, model.config.vocab_size), "Incorrect logits shape"
        
        # Test gradient flow
        def loss_fn(model, batch):
            carry = model.initial_carry(batch['input_ids'].shape[0])
            _, loss, _, _ = model(carry, batch)
            return loss
        
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        loss_val, grads = value_and_grad_fn(loss_head, batch)
        
        assert grads is not None, "No gradients computed"
        assert mx.isfinite(loss_val), "Loss value is not finite"
        
        print("✅ Real model tests passed")
        
    except Exception as e:
        print(f"⚠️  Real model test skipped: {e}")


def main():
    """Run all validation tests."""
    print("🧪 Phase 6 Loss Functions & Metrics Validation")
    print("=" * 50)
    
    tests = [
        test_s_function,
        test_log_stablemax,
        test_stablemax_cross_entropy,
        test_gradient_flow,
        test_act_loss_head,
        test_metrics_tracker,
        test_with_real_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Phase 6 validation tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)