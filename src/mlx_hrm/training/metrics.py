"""Metrics tracking for HRM training."""

import mlx.core as mx
from typing import Dict, List, Optional
import numpy as np
from .losses import IGNORE_LABEL_ID


class MetricsTracker:
    """
    Track and aggregate training metrics over time.
    
    Handles:
    - Running averages
    - Per-epoch statistics
    - Metric history
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics_sum = {}
        self.metrics_count = {}
        self.history = []
    
    def update(self, metrics: Dict[str, mx.array]):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values from ACTLossHead
        """
        for key, value in metrics.items():
            # Convert to Python scalar for tracking
            value_scalar = float(value.item())
            
            if key == 'count':
                # Special handling for count
                if 'count' not in self.metrics_count:
                    self.metrics_count['count'] = 0
                self.metrics_count['count'] += value_scalar
            else:
                # Sum-based metrics
                if key not in self.metrics_sum:
                    self.metrics_sum[key] = 0.0
                    self.metrics_count[key] = 0
                
                self.metrics_sum[key] += value_scalar
                
                # Use count for proper averaging
                if 'count' in metrics:
                    self.metrics_count[key] += float(metrics['count'].item())
                else:
                    self.metrics_count[key] += 1
    
    def get_current(self) -> Dict[str, float]:
        """Get current averaged metrics."""
        result = {}
        
        for key in self.metrics_sum:
            if self.metrics_count[key] > 0:
                result[key] = self.metrics_sum[key] / self.metrics_count[key]
            else:
                result[key] = 0.0
        
        # Add count
        if 'count' in self.metrics_count:
            result['count'] = self.metrics_count['count']
        
        return result
    
    def log_epoch(self):
        """Log current metrics and reset for next epoch."""
        current = self.get_current()
        self.history.append(current)
        self.reset()
        return current
    
    def get_history(self, metric: str) -> List[float]:
        """Get history of a specific metric."""
        return [epoch.get(metric, 0.0) for epoch in self.history]


def compute_puzzle_metrics(
    predictions: mx.array,
    labels: mx.array,
    puzzle_ids: Optional[mx.array] = None
) -> Dict[str, mx.array]:
    """
    Compute puzzle-specific metrics.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth [batch_size, seq_len]
        puzzle_ids: Optional puzzle identifiers for per-puzzle metrics
        
    Returns:
        Dictionary of puzzle-specific metrics
    """
    # Get predicted tokens
    pred_tokens = mx.argmax(predictions, axis=-1)
    
    # Compute exact match for each sequence
    mask = labels != IGNORE_LABEL_ID
    matches = (pred_tokens == labels) | ~mask
    exact_match = mx.all(matches, axis=-1)
    
    metrics = {
        'puzzle_accuracy': mx.mean(exact_match)
    }
    
    # Per-puzzle metrics if puzzle IDs provided
    if puzzle_ids is not None:
        unique_puzzles = mx.unique(puzzle_ids)
        for puzzle_id in unique_puzzles:
            puzzle_mask = puzzle_ids == puzzle_id
            puzzle_acc = mx.mean(exact_match[puzzle_mask])
            metrics[f'puzzle_{puzzle_id}_accuracy'] = puzzle_acc
    
    return metrics