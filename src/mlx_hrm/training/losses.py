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
    # Use float32 for computation (MLX doesn't support float64 on GPU)
    x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
    
    # Apply S-function
    s_x = s_function(x_f32)
    
    # Compute log probabilities
    # Promote to FP32 for sum (matches PyTorch mixed precision behavior)
    s_x_sum = mx.sum(s_x.astype(mx.float32), axis=axis, keepdims=True).astype(s_x.dtype)
    log_probs = mx.log(s_x) - mx.log(s_x_sum)
    
    # Cast back to original dtype if necessary
    if x.dtype != mx.float32:
        return log_probs.astype(x.dtype)
    return log_probs


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
        # Promote to FP32 for mean (matches PyTorch mixed precision behavior)
        return mx.mean(loss.astype(mx.float32)).astype(loss.dtype)
    elif reduction == 'sum':
        # Promote to FP32 for sum (matches PyTorch mixed precision behavior)
        return mx.sum(loss.astype(mx.float32)).astype(loss.dtype)
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
        # Promote to FP32 for mean (matches PyTorch mixed precision behavior)
        return mx.mean(losses.astype(mx.float32)).astype(losses.dtype)
    elif reduction == 'sum':
        # Promote to FP32 for sum (matches PyTorch mixed precision behavior)
        return mx.sum(losses.astype(mx.float32)).astype(losses.dtype)
    else:  # 'none'
        return losses