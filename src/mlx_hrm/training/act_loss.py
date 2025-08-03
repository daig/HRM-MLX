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