"""
HRM with Adaptive Computation Time (ACT) wrapper for MLX.

This module implements the outer ACT wrapper that manages dynamic computation
steps based on Q-learning. It wraps the inner HRM model and handles:
- Dynamic halting based on Q-values
- Exploration during training
- Carry state management across steps
"""

from typing import Dict, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.act import HRMConfig, HRMCarry, HRMInnerCarry
from .hrm_inner import HRMInner


class HRM_ACT(nn.Module):
    """
    HRM model with Adaptive Computation Time (ACT).
    
    This wrapper manages:
    1. Dynamic halting based on Q-learning
    2. Exploration during training for better Q-value estimation
    3. Efficient batching of sequences with different computation needs
    4. Carry state management across ACT steps
    
    During training, sequences can halt at different steps based on Q-values.
    During evaluation, all sequences run for max_steps for consistency.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        
        self.config = config
        self.inner = HRMInner(config)
        
        # ACT parameters
        self.halt_max_steps = config.halt_max_steps
        self.halt_exploration_prob = config.halt_exploration_prob
        
        # Training mode flag (will be set by framework)
        self._training = True
    
    @property
    def training(self) -> bool:
        """Check if model is in training mode."""
        return self._training
    
    def set_training(self, mode: bool):
        """Set training mode."""
        self._training = mode
    
    def initial_carry(self, batch_size: int) -> HRMCarry:
        """
        Create initial carry state for a batch.
        
        All sequences start as 'halted' so they will be reset with proper
        initialization on the first forward pass.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            Initial carry state
        """
        # Create empty inner carry (will be initialized on first reset)
        inner_carry = HRMInnerCarry(
            z_H=mx.zeros((batch_size, 1, self.config.hidden_size)),
            z_L=mx.zeros((batch_size, 1, self.config.hidden_size))
        )
        
        return HRMCarry(
            inner_carry=inner_carry,
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),  # Start halted
            current_data={}
        )
    
    def __call__(
        self,
        carry: HRMCarry,
        batch: Dict[str, mx.array]
    ) -> Tuple[HRMCarry, Dict[str, mx.array]]:
        """
        ACT forward pass managing adaptive computation.
        
        Key operations:
        1. Reset sequences that halted in previous step
        2. Run one step of the inner model
        3. Decide which sequences should halt based on Q-values
        4. Apply exploration for better Q-learning
        
        Args:
            carry: Current carry state
            batch: New batch data
            
        Returns:
            - new_carry: Updated carry state
            - outputs: Model outputs including logits and Q-values
        """
        # Reset halted sequences to process new examples
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        # Reset step counter for halted sequences
        new_steps = mx.where(carry.halted, 0, carry.steps)
        
        # Update batch data for halted sequences
        new_current_data = {}
        for k, v in batch.items():
            # Use new data for halted sequences, keep old data for continuing ones
            mask = carry.halted
            # Reshape mask to broadcast properly
            while mask.ndim < v.ndim:
                mask = mask.reshape(mask.shape + (1,))
            new_current_data[k] = mx.where(mask, v, carry.current_data.get(k, v))
        
        # Run one step of the inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # ACT halting logic (no gradients needed)
        # Increment step counter
        new_steps = mx.stop_gradient(new_steps + 1)
        is_last_step = new_steps >= self.halt_max_steps
        
        # Always halt at max steps
        halted = is_last_step
        
        # Training-specific halting logic
        if self.training and (self.halt_max_steps > 1):
            # Halt when Q(halt) > Q(continue)
            q_halt_stop = mx.stop_gradient(q_halt_logits)
            q_continue_stop = mx.stop_gradient(q_continue_logits)
            halted = halted | (q_halt_stop > q_continue_stop)
            
            # Exploration: Sometimes force minimum steps for better Q-learning
            if self.halt_exploration_prob > 0:
                # Random exploration mask
                explore = mx.random.uniform(shape=q_halt_logits.shape) < self.halt_exploration_prob
                
                # Random minimum steps for exploration
                min_steps = mx.random.randint(
                    low=2,
                    high=self.halt_max_steps + 1,
                    shape=new_steps.shape
                )
                
                # Apply exploration: only halt if we've reached minimum steps
                halted = halted & ((new_steps >= min_steps) | ~explore)
            
            # Compute target Q-values for training
            # Run another forward pass to get next step's Q-values
            # This is used for bootstrapping in Q-learning
            _, _, (next_q_halt, next_q_continue) = self.inner(
                new_inner_carry.detach(), new_current_data
            )
            
            # Target: max(Q_halt, Q_continue) for bootstrapping
            # On last step, only Q_halt is valid
            target_q = mx.where(
                is_last_step,
                next_q_halt,
                mx.maximum(next_q_halt, next_q_continue)
            )
            
            # Use sigmoid to convert to probability for stable training
            outputs["target_q_continue"] = mx.sigmoid(mx.stop_gradient(target_q))
        
        # Create new carry state
        new_carry = HRMCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )
        
        return new_carry, outputs


def create_hrm_act(config: HRMConfig) -> HRM_ACT:
    """
    Factory function to create an HRM model with ACT.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized HRM_ACT model
    """
    return HRM_ACT(config)