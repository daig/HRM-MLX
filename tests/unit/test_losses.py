"""Unit tests for loss functions."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_hrm.training import (
    IGNORE_LABEL_ID,
    s_function,
    log_stablemax,
    stablemax_cross_entropy,
    softmax_cross_entropy,
)


class TestSFunction:
    """Test the S-function component of stablemax."""
    
    def test_s_function_positive(self):
        """Test S-function on positive inputs."""
        x = mx.array([0, 1, 2, 3, 4])
        s_x = s_function(x)
        expected = x + 1.0  # For x >= 0, S(x) = x + 1
        assert mx.allclose(s_x, expected)
    
    def test_s_function_negative(self):
        """Test S-function on negative inputs."""
        x = mx.array([-0.5, -0.9, -0.99])
        s_x = s_function(x)
        # For x < 0, S(x) = 1/(1-x)
        expected = mx.array([2.0, 10.0, 100.0])
        assert mx.allclose(s_x, expected, rtol=1e-2)
    
    def test_s_function_all_positive_output(self):
        """Verify S-function always produces positive outputs."""
        x = mx.random.uniform(-10, 10, (100,))
        s_x = s_function(x)
        assert mx.all(s_x > 0)
    
    def test_s_function_stability(self):
        """Test numerical stability near critical points."""
        # Test values very close to 1
        x = mx.array([0.999999, 0.9999999, -0.999999])
        s_x = s_function(x)
        assert mx.all(mx.isfinite(s_x))


class TestLogStablemax:
    """Test log stablemax computation."""
    
    def test_log_stablemax_sums_to_one(self):
        """Verify that exp(log_stablemax) sums to 1."""
        x = mx.random.normal((10, 20))
        log_probs = log_stablemax(x)
        probs = mx.exp(log_probs)
        sums = mx.sum(probs, axis=-1)
        assert mx.allclose(sums, mx.ones_like(sums), rtol=1e-5)
    
    def test_log_stablemax_finite(self):
        """Test that log stablemax produces finite values."""
        # Test with extreme values
        x = mx.random.uniform(-100, 100, (5, 10))
        log_probs = log_stablemax(x)
        assert mx.all(mx.isfinite(log_probs))
    
    def test_log_stablemax_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        for dtype in [mx.float16, mx.float32]:
            x = mx.random.normal((5, 10)).astype(dtype)
            log_probs = log_stablemax(x)
            assert log_probs.dtype == dtype


class TestStablemaxCrossEntropy:
    """Test stablemax cross-entropy loss."""
    
    def test_stablemax_ce_basic(self):
        """Test basic stablemax cross-entropy computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = stablemax_cross_entropy(logits, labels)
        assert loss.shape == (batch_size, seq_len)
        assert mx.all(mx.isfinite(loss))
        assert mx.all(loss >= 0)
    
    def test_stablemax_ce_masking(self):
        """Test that ignore_index works correctly."""
        logits = mx.random.normal((2, 10, 100))
        labels = mx.array([[1, 2, 3, -100, -100, 6, 7, 8, 9, 10],
                          [11, 12, -100, -100, -100, 16, 17, 18, 19, 20]])
        
        loss = stablemax_cross_entropy(logits, labels)
        # Positions with -100 should have 0 loss
        assert mx.all(loss[0, 3:5] == 0)
        assert mx.all(loss[1, 2:5] == 0)
        # Other positions should have non-zero loss
        assert mx.all(loss[0, :3] > 0)
        assert mx.all(loss[0, 5:] > 0)
    
    def test_stablemax_ce_reduction(self):
        """Test different reduction modes."""
        logits = mx.random.normal((2, 10, 50))
        labels = mx.random.randint(0, 50, (2, 10))
        
        # Test 'none' reduction
        loss_none = stablemax_cross_entropy(logits, labels, reduction='none')
        assert loss_none.shape == (2, 10)
        
        # Test 'mean' reduction
        loss_mean = stablemax_cross_entropy(logits, labels, reduction='mean')
        assert loss_mean.shape == ()
        assert mx.allclose(loss_mean, mx.mean(loss_none))
        
        # Test 'sum' reduction
        loss_sum = stablemax_cross_entropy(logits, labels, reduction='sum')
        assert loss_sum.shape == ()
        assert mx.allclose(loss_sum, mx.sum(loss_none))
    
    def test_stablemax_ce_perfect_prediction(self):
        """Test loss when predictions are perfect."""
        vocab_size = 10
        logits = mx.zeros((1, 5, vocab_size))
        labels = mx.array([[2, 3, 4, 5, 6]])
        
        # Set correct labels to high values
        for i in range(5):
            logits[0, i, labels[0, i].item()] = 100.0
        
        loss = stablemax_cross_entropy(logits, labels, reduction='mean')
        # Loss should be very low for perfect predictions
        assert loss < 0.01


class TestSoftmaxCrossEntropy:
    """Test standard softmax cross-entropy (baseline)."""
    
    def test_softmax_ce_basic(self):
        """Test basic softmax cross-entropy computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = softmax_cross_entropy(logits, labels)
        assert loss.shape == (batch_size, seq_len)
        assert mx.all(mx.isfinite(loss))
        assert mx.all(loss >= 0)
    
    def test_softmax_ce_masking(self):
        """Test that ignore_index works correctly."""
        logits = mx.random.normal((2, 10, 100))
        labels = mx.array([[1, 2, 3, -100, -100, 6, 7, 8, 9, 10],
                          [11, 12, -100, -100, -100, 16, 17, 18, 19, 20]])
        
        loss = softmax_cross_entropy(logits, labels)
        # Positions with -100 should have 0 loss
        assert mx.all(loss[0, 3:5] == 0)
        assert mx.all(loss[1, 2:5] == 0)
    
    def test_softmax_ce_reduction(self):
        """Test different reduction modes."""
        logits = mx.random.normal((2, 10, 50))
        labels = mx.random.randint(0, 50, (2, 10))
        
        # Test all reduction modes
        loss_none = softmax_cross_entropy(logits, labels, reduction='none')
        loss_mean = softmax_cross_entropy(logits, labels, reduction='mean')
        loss_sum = softmax_cross_entropy(logits, labels, reduction='sum')
        
        assert loss_none.shape == (2, 10)
        assert loss_mean.shape == ()
        assert loss_sum.shape == ()


class TestComparisonStablemaxSoftmax:
    """Compare stablemax and softmax losses."""
    
    def test_similar_magnitude(self):
        """Test that stablemax and softmax losses are in similar ranges."""
        logits = mx.random.normal((4, 20, 50))
        labels = mx.random.randint(0, 50, (4, 20))
        
        stablemax_loss = stablemax_cross_entropy(logits, labels, reduction='mean')
        softmax_loss = softmax_cross_entropy(logits, labels, reduction='mean')
        
        # They should be in the same order of magnitude
        ratio = stablemax_loss / softmax_loss
        assert 0.1 < ratio < 10.0
    
    def test_gradient_flow(self):
        """Test that gradients flow through both loss functions."""
        def loss_fn(logits, labels, use_stablemax=True):
            if use_stablemax:
                return stablemax_cross_entropy(logits, labels, reduction='mean')
            else:
                return softmax_cross_entropy(logits, labels, reduction='mean')
        
        logits = mx.random.normal((2, 10, 20))
        labels = mx.random.randint(0, 20, (2, 10))
        
        # Test stablemax gradients
        loss_grad_fn = mx.grad(loss_fn)
        grad_stablemax = loss_grad_fn(logits, labels, True)
        assert mx.all(mx.isfinite(grad_stablemax))
        
        # Test softmax gradients
        grad_softmax = loss_grad_fn(logits, labels, False)
        assert mx.all(mx.isfinite(grad_softmax))


if __name__ == "__main__":
    pytest.main([__file__])