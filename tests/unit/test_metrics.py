"""Unit tests for metrics tracking."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_hrm.training import MetricsTracker, compute_puzzle_metrics, IGNORE_LABEL_ID


class TestMetricsTracker:
    """Test MetricsTracker functionality."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert tracker.metrics_sum == {}
        assert tracker.metrics_count == {}
        assert tracker.history == []
    
    def test_single_update(self):
        """Test updating with a single batch of metrics."""
        tracker = MetricsTracker()
        
        metrics = {
            'loss': mx.array(2.5),
            'accuracy': mx.array(0.85),
            'count': mx.array(32)
        }
        
        tracker.update(metrics)
        
        current = tracker.get_current()
        assert current['loss'] == 2.5
        assert current['accuracy'] == 0.85
        assert current['count'] == 32
    
    def test_multiple_updates(self):
        """Test updating with multiple batches."""
        tracker = MetricsTracker()
        
        # Update with multiple batches
        for i in range(5):
            metrics = {
                'loss': mx.array(2.0 + i * 0.1),
                'accuracy': mx.array(0.8 + i * 0.02),
                'count': mx.array(16)
            }
            tracker.update(metrics)
        
        current = tracker.get_current()
        
        # Loss should be averaged: (2.0 + 2.1 + 2.2 + 2.3 + 2.4) / 5 = 2.2
        assert abs(current['loss'] - 2.2) < 1e-5
        
        # Accuracy should be averaged: (0.8 + 0.82 + 0.84 + 0.86 + 0.88) / 5 = 0.84
        assert abs(current['accuracy'] - 0.84) < 1e-5
        
        # Count should be summed: 16 * 5 = 80
        assert current['count'] == 80
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        
        # Add some metrics
        tracker.update({'loss': mx.array(1.5), 'count': mx.array(10)})
        assert tracker.get_current()['loss'] == 1.5
        
        # Reset
        tracker.reset()
        assert tracker.metrics_sum == {}
        assert tracker.metrics_count == {}
        assert tracker.get_current() == {}
    
    def test_log_epoch(self):
        """Test logging epoch metrics."""
        tracker = MetricsTracker()
        
        # Update metrics for epoch 1
        tracker.update({'loss': mx.array(2.0), 'accuracy': mx.array(0.7)})
        tracker.update({'loss': mx.array(1.8), 'accuracy': mx.array(0.75)})
        
        # Log epoch 1
        epoch1_metrics = tracker.log_epoch()
        assert abs(epoch1_metrics['loss'] - 1.9) < 1e-5
        assert abs(epoch1_metrics['accuracy'] - 0.725) < 1e-5
        
        # Tracker should be reset after logging
        assert tracker.get_current() == {}
        
        # Update metrics for epoch 2
        tracker.update({'loss': mx.array(1.5), 'accuracy': mx.array(0.8)})
        epoch2_metrics = tracker.log_epoch()
        
        # Check history
        assert len(tracker.history) == 2
        assert tracker.history[0] == epoch1_metrics
        assert tracker.history[1] == epoch2_metrics
    
    def test_get_history(self):
        """Test getting metric history."""
        tracker = MetricsTracker()
        
        # Create some epochs
        for epoch in range(3):
            for batch in range(5):
                tracker.update({
                    'loss': mx.array(2.0 - epoch * 0.5 + batch * 0.01),
                    'accuracy': mx.array(0.6 + epoch * 0.1)
                })
            tracker.log_epoch()
        
        # Get loss history
        loss_history = tracker.get_history('loss')
        assert len(loss_history) == 3
        assert loss_history[0] > loss_history[1] > loss_history[2]  # Decreasing
        
        # Get accuracy history
        acc_history = tracker.get_history('accuracy')
        assert len(acc_history) == 3
        assert acc_history[0] < acc_history[1] < acc_history[2]  # Increasing
        
        # Get non-existent metric
        missing_history = tracker.get_history('missing_metric')
        assert missing_history == [0.0, 0.0, 0.0]
    
    def test_weighted_averaging(self):
        """Test that metrics are properly weighted by count."""
        tracker = MetricsTracker()
        
        # Batch 1: 10 samples with loss 1.0
        tracker.update({
            'loss': mx.array(10.0),  # Total loss for batch
            'count': mx.array(10)
        })
        
        # Batch 2: 30 samples with loss 2.0
        tracker.update({
            'loss': mx.array(60.0),  # Total loss for batch
            'count': mx.array(30)
        })
        
        current = tracker.get_current()
        # Weighted average: (10.0 + 60.0) / (10 + 30) = 70/40 = 1.75
        assert abs(current['loss'] - 1.75) < 1e-5
        assert current['count'] == 40


class TestComputePuzzleMetrics:
    """Test puzzle-specific metrics computation."""
    
    def test_basic_puzzle_metrics(self):
        """Test basic puzzle accuracy computation."""
        batch_size = 4
        seq_len = 10
        vocab_size = 50
        
        # Create predictions and labels
        predictions = mx.random.normal((batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Make some predictions correct
        for i in range(batch_size):
            for j in range(seq_len):
                # Set correct label to have highest logit
                predictions[i, j, labels[i, j]] += 10.0
        
        metrics = compute_puzzle_metrics(predictions, labels)
        
        assert 'puzzle_accuracy' in metrics
        assert mx.allclose(metrics['puzzle_accuracy'], mx.array(1.0))
    
    def test_puzzle_metrics_with_masking(self):
        """Test puzzle metrics with ignored positions."""
        predictions = mx.zeros((2, 5, 10))
        labels = mx.array([
            [1, 2, 3, IGNORE_LABEL_ID, IGNORE_LABEL_ID],
            [4, 5, IGNORE_LABEL_ID, IGNORE_LABEL_ID, IGNORE_LABEL_ID]
        ])
        
        # Set correct predictions
        predictions[0, 0, 1] = 10.0
        predictions[0, 1, 2] = 10.0
        predictions[0, 2, 3] = 10.0
        predictions[1, 0, 4] = 10.0
        predictions[1, 1, 5] = 10.0
        
        metrics = compute_puzzle_metrics(predictions, labels)
        
        # Both sequences should be correct (ignoring masked positions)
        assert mx.allclose(metrics['puzzle_accuracy'], mx.array(1.0))
    
    def test_puzzle_metrics_partial_correct(self):
        """Test puzzle metrics with partially correct sequences."""
        predictions = mx.zeros((4, 3, 10))
        labels = mx.array([
            [1, 2, 3],  # Will be all correct
            [4, 5, 6],  # Will have 2/3 correct
            [7, 8, 9],  # Will have 1/3 correct
            [0, 1, 2],  # Will be all wrong
        ])
        
        # Set predictions
        # Sequence 0: all correct
        predictions[0, 0, 1] = 10.0
        predictions[0, 1, 2] = 10.0
        predictions[0, 2, 3] = 10.0
        
        # Sequence 1: 2/3 correct
        predictions[1, 0, 4] = 10.0
        predictions[1, 1, 5] = 10.0
        predictions[1, 2, 0] = 10.0  # Wrong
        
        # Sequence 2: 1/3 correct
        predictions[2, 0, 7] = 10.0
        predictions[2, 1, 0] = 10.0  # Wrong
        predictions[2, 2, 0] = 10.0  # Wrong
        
        # Sequence 3: all wrong
        predictions[3, 0, 9] = 10.0
        predictions[3, 1, 9] = 10.0
        predictions[3, 2, 9] = 10.0
        
        metrics = compute_puzzle_metrics(predictions, labels)
        
        # Only sequence 0 is fully correct
        assert mx.allclose(metrics['puzzle_accuracy'], mx.array(0.25))
    
    def test_per_puzzle_metrics(self):
        """Test per-puzzle ID metrics."""
        batch_size = 6
        seq_len = 5
        vocab_size = 20
        
        predictions = mx.zeros((batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        puzzle_ids = mx.array([0, 0, 1, 1, 2, 2])  # 3 different puzzles, 2 samples each
        
        # Make puzzles 0 and 2 correct, puzzle 1 incorrect
        for i in [0, 1, 4, 5]:  # Indices for puzzles 0 and 2
            for j in range(seq_len):
                predictions[i, j, labels[i, j]] = 10.0
        
        metrics = compute_puzzle_metrics(predictions, labels, puzzle_ids)
        
        # Check overall accuracy
        assert mx.allclose(metrics['puzzle_accuracy'], mx.array(4/6))
        
        # Check per-puzzle accuracy
        assert 'puzzle_0_accuracy' in metrics
        assert 'puzzle_1_accuracy' in metrics
        assert 'puzzle_2_accuracy' in metrics
        
        assert mx.allclose(metrics['puzzle_0_accuracy'], mx.array(1.0))
        assert mx.allclose(metrics['puzzle_1_accuracy'], mx.array(0.0))
        assert mx.allclose(metrics['puzzle_2_accuracy'], mx.array(1.0))


class TestMetricsIntegration:
    """Test metrics integration with loss head."""
    
    def test_metrics_from_loss_head(self):
        """Test that metrics from ACT loss head work with tracker."""
        from mlx_hrm.training import ACTLossHead
        from test_act_loss import MockHRMModel
        
        model = MockHRMModel()
        loss_head = ACTLossHead(model)
        tracker = MetricsTracker()
        
        # Run multiple batches
        for i in range(10):
            batch = {
                'input_ids': mx.random.randint(0, 100, (4, 16)),
                'labels': mx.random.randint(0, 100, (4, 16))
            }
            
            carry = loss_head.initial_carry(4)
            _, _, metrics, _ = loss_head(carry, batch)
            
            tracker.update(metrics)
        
        # Check that all expected metrics are tracked
        current = tracker.get_current()
        expected_metrics = ['lm_loss', 'q_halt_loss', 'q_continue_loss', 
                          'accuracy', 'exact_accuracy', 'q_halt_accuracy', 'steps']
        
        for metric in expected_metrics:
            assert metric in current
            assert current[metric] > 0


if __name__ == "__main__":
    pytest.main([__file__])