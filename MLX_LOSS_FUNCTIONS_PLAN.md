# MLX Loss Functions & Metrics Implementation Plan

## Overview
This document provides the detailed implementation plan for HRM-specific loss functions and training metrics in MLX. This corresponds to Phase 6 of the master implementation plan.

### Goal
Implement the novel Stablemax loss function, ACT-specific loss components, and comprehensive metrics tracking to enable proper training of the HRM model.

### Key Components
1. **Stablemax Loss**: HRM's novel alternative to softmax cross-entropy
2. **ACT Loss Components**: Combined language modeling and Q-learning losses
3. **Training Metrics**: Comprehensive tracking of model performance

## Technical Background

### Stablemax Function
The Stablemax is a novel activation function that provides better numerical stability and gradient properties compared to softmax:

```
S(x) = { 1/(1-x) if x < 0
       { x + 1   if x ≥ 0

stablemax(x) = S(x) / Σ S(x_i)
```

### ACT Loss Structure
The total loss combines three components:
1. **Language Modeling Loss**: Token prediction accuracy
2. **Q-halt Loss**: Learning when to stop computation
3. **Q-continue Loss**: TD-learning for value estimation

Total Loss = LM_loss + 0.5 * (Q_halt_loss + Q_continue_loss)

## Implementation Components

### 1. Stablemax Activation and Loss
**File**: `src/mlx_hrm/training/losses.py`

```python
"""Loss functions for HRM training."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

# Special token ID for positions to ignore in loss
IGNORE_LABEL_ID = -100


def s_function(x: mx.array, epsilon: float = 1e-30) -> mx.array:
    """
    S-function for stablemax: Maps inputs to positive values.
    
    This function ensures numerical stability by mapping:
    - Negative values: 1/(1-x) (approaches infinity as x -> 1 from below)
    - Non-negative values: x + 1
    
    Args:
        x: Input tensor
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Transformed tensor with all positive values
    """
    return mx.where(
        x < 0,
        1.0 / (1.0 - x + epsilon),
        x + 1.0
    )


def log_stablemax(x: mx.array, axis: int = -1) -> mx.array:
    """
    Compute log probabilities using stablemax activation.
    
    Stablemax is an alternative to softmax that can provide better
    numerical stability and gradient properties for certain tasks.
    
    Args:
        x: Input logits
        axis: Dimension along which to compute stablemax
        
    Returns:
        Log probabilities computed via stablemax
    """
    # Cast to float64 for numerical precision
    x_f64 = x.astype(mx.float64)
    
    # Apply S-function
    s_x = s_function(x_f64)
    
    # Compute log probabilities
    log_probs = mx.log(s_x) - mx.log(mx.sum(s_x, axis=axis, keepdims=True))
    
    # Cast back to original dtype
    return log_probs.astype(x.dtype)


def stablemax_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_LABEL_ID,
    reduction: str = 'none'
) -> mx.array:
    """
    Compute cross-entropy loss using stablemax activation.
    
    This is HRM's novel loss function that can improve training stability
    compared to standard softmax cross-entropy.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        ignore_index: Label value to ignore in loss computation
        reduction: 'none', 'mean', or 'sum'
        
    Returns:
        Loss values (shape depends on reduction)
    """
    # Get dimensions
    batch_size, seq_len, vocab_size = logits.shape
    
    # Compute log probabilities with stablemax
    log_probs = log_stablemax(logits, axis=-1)
    
    # Create mask for valid positions
    valid_mask = labels != ignore_index
    
    # Replace invalid labels with 0 for gathering
    safe_labels = mx.where(valid_mask, labels, 0)
    
    # Gather log probabilities for true labels
    # Need to reshape for gathering
    flat_log_probs = log_probs.reshape(-1, vocab_size)
    flat_labels = safe_labels.reshape(-1)
    indices = mx.stack([mx.arange(flat_labels.shape[0]), flat_labels], axis=1)
    
    # Gather and reshape back
    prediction_log_probs = flat_log_probs[indices[:, 0], indices[:, 1]]
    prediction_log_probs = prediction_log_probs.reshape(batch_size, seq_len)
    
    # Apply mask and compute negative log likelihood
    loss = -mx.where(valid_mask, prediction_log_probs, 0.0)
    
    # Apply reduction
    if reduction == 'mean':
        return mx.mean(loss)
    elif reduction == 'sum':
        return mx.sum(loss)
    else:  # 'none'
        return loss


def softmax_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_LABEL_ID,
    reduction: str = 'none'
) -> mx.array:
    """
    Standard softmax cross-entropy loss (alternative to stablemax).
    
    Provided as a baseline and for comparison with stablemax.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        ignore_index: Label value to ignore in loss computation
        reduction: 'none', 'mean', or 'sum'
        
    Returns:
        Loss values (shape depends on reduction)
    """
    # Create mask for valid positions
    valid_mask = labels != ignore_index
    
    # Use MLX's built-in cross entropy
    # Flatten for computation
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    
    # Compute loss
    losses = nn.losses.cross_entropy(flat_logits, flat_labels, reduction='none')
    losses = losses.reshape(labels.shape)
    
    # Apply mask
    losses = mx.where(valid_mask, losses, 0.0)
    
    # Apply reduction
    if reduction == 'mean':
        return mx.mean(losses)
    elif reduction == 'sum':
        return mx.sum(losses)
    else:  # 'none'
        return losses
```

### 2. ACT Loss Components
**File**: `src/mlx_hrm/training/act_loss.py`

```python
"""ACT-specific loss components for HRM training."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional, Sequence
from .losses import stablemax_cross_entropy, softmax_cross_entropy, IGNORE_LABEL_ID


class ACTLossHead(nn.Module):
    """
    Loss computation head for ACT (Adaptive Computation Time) models.
    
    This module wraps the HRM model and computes multiple losses:
    1. Language modeling loss (prediction accuracy)
    2. Q-halt loss (learning when to stop computation)
    3. Q-continue loss (bootstrapping Q-values)
    
    It also tracks various metrics for monitoring training progress.
    """
    
    def __init__(self, model: nn.Module, loss_type: str = 'stablemax'):
        """
        Initialize ACT loss head.
        
        Args:
            model: The HRM model to wrap
            loss_type: 'stablemax' or 'softmax'
        """
        super().__init__()
        self.model = model
        
        # Select loss function
        if loss_type == 'stablemax':
            self.loss_fn = stablemax_cross_entropy
        elif loss_type == 'softmax':
            self.loss_fn = softmax_cross_entropy
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def initial_carry(self, batch_size: int):
        """Forward initial carry creation to wrapped model."""
        return self.model.initial_carry(batch_size)
    
    def __call__(
        self,
        carry,
        batch: Dict[str, mx.array],
        return_metrics: bool = True
    ) -> Tuple[any, mx.array, Dict[str, mx.array], Dict[str, mx.array]]:
        """
        Forward pass computing losses and metrics.
        
        Args:
            carry: Model carry state
            batch: Input batch with 'input_ids' and 'labels'
            return_metrics: Whether to compute and return metrics
            
        Returns:
            - new_carry: Updated carry state
            - total_loss: Combined loss for backpropagation
            - metrics: Dictionary of metrics for logging
            - outputs: Model outputs (detached)
        """
        # Run model forward pass
        new_carry, outputs = self.model(carry, batch)
        
        # Extract labels from batch
        labels = batch['labels']
        logits = outputs['logits']
        
        # Initialize metrics dict
        metrics = {}
        
        # Compute accuracy metrics if requested
        if return_metrics:
            with mx.no_grad():
                # Mask for valid (non-ignored) positions
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mx.sum(mask, axis=-1)  # Valid tokens per sequence
                
                # Divisor for averaging, avoid division by zero
                loss_divisor = mx.maximum(loss_counts, 1).reshape(-1, 1)
                
                # Check which predictions are correct
                predictions = mx.argmax(logits, axis=-1)
                is_correct = mask & (predictions == labels)
                
                # A sequence is correct only if ALL tokens are correct
                seq_is_correct = mx.sum(is_correct, axis=-1) == loss_counts
                
                # Only compute metrics for halted sequences with valid tokens
                valid_metrics = new_carry.halted & (loss_counts > 0)
                
                metrics = {
                    # Number of valid sequences
                    'count': mx.sum(valid_metrics),
                    
                    # Token-level accuracy
                    'accuracy': mx.sum(
                        mx.where(
                            valid_metrics.reshape(-1, 1),
                            mx.sum(is_correct.astype(mx.float32), axis=-1, keepdims=True) / loss_divisor,
                            0.0
                        )
                    ),
                    
                    # Sequence-level accuracy
                    'exact_accuracy': mx.sum(valid_metrics & seq_is_correct),
                    
                    # How well Q-halt predicts sequence correctness
                    'q_halt_accuracy': mx.sum(
                        valid_metrics & ((outputs['q_halt_logits'] >= 0) == seq_is_correct)
                    ),
                    
                    # Average computation steps used
                    'steps': mx.sum(mx.where(valid_metrics, new_carry.steps, 0))
                }
        
        # Compute losses
        
        # 1. Language modeling loss
        # Normalize by number of valid tokens per sequence
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mx.sum(mask, axis=-1, keepdims=True)
        loss_divisor = mx.maximum(loss_counts, 1)
        
        lm_losses = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, reduction='none')
        lm_loss = mx.sum(lm_losses / loss_divisor)
        
        # 2. Q-halt loss: Train Q-values to predict if sequence will be correct
        if return_metrics:
            # Use computed seq_is_correct
            q_halt_loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(outputs['q_halt_logits']),
                seq_is_correct.astype(mx.float32),
                reduction='sum'
            )
        else:
            # Recompute if metrics not requested
            mask = labels != IGNORE_LABEL_ID
            predictions = mx.argmax(logits, axis=-1)
            is_correct = mask & (predictions == labels)
            seq_is_correct = mx.all(is_correct | ~mask, axis=-1)
            
            q_halt_loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(outputs['q_halt_logits']),
                seq_is_correct.astype(mx.float32),
                reduction='sum'
            )
        
        # Add losses to metrics
        metrics['lm_loss'] = mx.stop_gradient(lm_loss)
        metrics['q_halt_loss'] = mx.stop_gradient(q_halt_loss)
        
        # 3. Q-continue loss: Bootstrap Q-values
        q_continue_loss = mx.array(0.0)
        if 'target_q_continue' in outputs:
            q_continue_loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(outputs['q_continue_logits']),
                outputs['target_q_continue'],
                reduction='sum'
            )
            metrics['q_continue_loss'] = mx.stop_gradient(q_continue_loss)
        
        # Combine losses with weights
        # LM loss is primary, Q-losses are auxiliary (0.5 weight)
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        
        # Detach outputs to prevent gradient flow
        detached_outputs = {k: mx.stop_gradient(v) for k, v in outputs.items()}
        
        return new_carry, total_loss, metrics, detached_outputs
```

### 3. Training Metrics
**File**: `src/mlx_hrm/training/metrics.py`

```python
"""Metrics tracking for HRM training."""

import mlx.core as mx
from typing import Dict, List
import numpy as np


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
```

## Testing Strategy

### 1. Loss Function Tests
**File**: `tests/unit/test_losses.py`

```python
def test_stablemax_function():
    """Test S-function and stablemax computation."""
    # Test S-function properties
    x = mx.array([-2, -1, 0, 1, 2])
    s_x = s_function(x)
    assert mx.all(s_x > 0)  # All positive
    
def test_stablemax_cross_entropy():
    """Test stablemax loss computation."""
    logits = mx.random.normal((2, 10, 100))
    labels = mx.random.randint(0, 100, (2, 10))
    
    loss = stablemax_cross_entropy(logits, labels)
    assert loss.shape == (2, 10)
    assert mx.all(mx.isfinite(loss))
    
def test_loss_masking():
    """Test that ignore_index works correctly."""
    logits = mx.random.normal((2, 10, 100))
    labels = mx.array([[1, 2, 3, -100, -100, 6, 7, 8, 9, 10],
                      [11, 12, -100, -100, -100, 16, 17, 18, 19, 20]])
    
    loss = stablemax_cross_entropy(logits, labels)
    # Positions with -100 should have 0 loss
    assert mx.all(loss[:, 3:5] == 0)
```

### 2. ACT Loss Tests
**File**: `tests/unit/test_act_loss.py`

```python
def test_act_loss_head():
    """Test complete ACT loss computation."""
    from mlx_hrm.models import create_hrm
    
    model = create_hrm('tiny')
    loss_head = ACTLossHead(model, loss_type='stablemax')
    
    batch = {
        'input_ids': mx.random.randint(0, 100, (2, 32)),
        'labels': mx.random.randint(0, 100, (2, 32))
    }
    
    carry = loss_head.initial_carry(2)
    new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
    
    assert mx.isfinite(total_loss)
    assert 'lm_loss' in metrics
    assert 'q_halt_loss' in metrics
```

### 3. Metrics Tests
**File**: `tests/unit/test_metrics.py`

```python
def test_metrics_tracker():
    """Test metrics tracking and averaging."""
    tracker = MetricsTracker()
    
    # Update with multiple batches
    for i in range(10):
        metrics = {
            'loss': mx.array(1.0 + i * 0.1),
            'accuracy': mx.array(0.8 + i * 0.01),
            'count': mx.array(32)
        }
        tracker.update(metrics)
    
    current = tracker.get_current()
    assert 'loss' in current
    assert 'accuracy' in current
    assert current['count'] == 320  # 10 * 32
```

## Performance Considerations

### 1. Numerical Stability
- Use float64 for stablemax computation
- Careful handling of edge cases (x ≈ 1 for negative branch)
- Gradient clipping may be needed

### 2. Efficiency
- Vectorized operations throughout
- Avoid unnecessary type conversions
- Cache computed values when possible

### 3. Memory Usage
- Metrics tracking uses Python scalars to avoid GPU memory
- Detached outputs to prevent gradient accumulation
- Efficient masking operations

## Implementation Timeline

### Day 1: Core Losses (3-4 hours)
- [ ] Implement stablemax function and loss
- [ ] Add softmax baseline
- [ ] Create comprehensive tests
- [ ] Verify numerical stability

### Day 2: ACT Components (3-4 hours)
- [ ] Implement ACT loss head
- [ ] Add Q-learning losses
- [ ] Create metrics tracking
- [ ] Integration testing

## Success Criteria

1. **Correctness**
   - ✅ Losses match PyTorch implementation
   - ✅ Proper gradient flow
   - ✅ Numerically stable
   - ✅ Handles edge cases

2. **Performance**
   - ✅ Efficient computation
   - ✅ Minimal memory overhead
   - ✅ Fast enough for training
   - ✅ Scales with model size

3. **Usability**
   - ✅ Clear API
   - ✅ Good error messages
   - ✅ Easy integration
   - ✅ Comprehensive metrics

## Next Steps

After completing loss functions:
1. Move to Phase 7: Training Infrastructure
2. Integrate losses with optimizer
3. Begin full training runs