"""
Adaptive Computation Time (ACT) implementation for MLX HRM.

This module implements the ACT mechanism that allows the model to dynamically
adjust computation steps based on problem complexity. It includes:
- Carry state management for maintaining context between steps
- Halting mechanism based on Q-learning
- Hierarchical processing with H-level (planning) and L-level (computation)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, NamedTuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HRMInnerCarry:
    """
    Carry state for the inner HRM model between recurrent steps.
    
    Contains hidden states from both hierarchical levels:
    - z_H: High-level (planning) hidden states
    - z_L: Low-level (computation) hidden states
    
    These states are passed between ACT steps to maintain context.
    """
    z_H: mx.array  # Shape: [batch_size, seq_len + puzzle_emb_len, hidden_size]
    z_L: mx.array  # Shape: [batch_size, seq_len + puzzle_emb_len, hidden_size]
    
    def detach(self) -> 'HRMInnerCarry':
        """Create a gradient-free copy of the carry state."""
        return HRMInnerCarry(
            z_H=mx.stop_gradient(self.z_H),
            z_L=mx.stop_gradient(self.z_L)
        )
    
    def to_dict(self) -> Dict[str, mx.array]:
        """Convert to dictionary for MLX operations."""
        return {
            'z_H': self.z_H,
            'z_L': self.z_L
        }
    
    @staticmethod
    def from_dict(d: Dict[str, mx.array]) -> 'HRMInnerCarry':
        """Create from dictionary."""
        return HRMInnerCarry(z_H=d['z_H'], z_L=d['z_L'])


@dataclass
class HRMCarry:
    """
    Complete carry state for the ACT wrapper model.
    
    Includes:
    - inner_carry: The H/L level states from the inner model
    - steps: Number of computation steps taken per sequence
    - halted: Boolean mask indicating which sequences have halted
    - current_data: The current batch data being processed
    """
    inner_carry: HRMInnerCarry
    steps: mx.array         # Shape: [batch_size], counts computation steps
    halted: mx.array        # Shape: [batch_size], True if sequence has halted
    current_data: Dict[str, mx.array]  # Current batch data (inputs, labels, etc.)
    
    def detach(self) -> 'HRMCarry':
        """Create a gradient-free copy of the carry state."""
        return HRMCarry(
            inner_carry=self.inner_carry.detach(),
            steps=mx.stop_gradient(self.steps),
            halted=mx.stop_gradient(self.halted),
            current_data={k: mx.stop_gradient(v) for k, v in self.current_data.items()}
        )


class HRMConfig(NamedTuple):
    """
    Configuration for the HRM model with ACT.
    
    The model has two hierarchical levels:
    - H-level: High-level planning and reasoning
    - L-level: Low-level computation and execution
    """
    # Data configuration
    batch_size: int              # Training batch size
    seq_len: int                 # Maximum sequence length
    puzzle_emb_ndim: int = 0     # Dimension of puzzle-specific embeddings (0 = disabled)
    num_puzzle_identifiers: int = 0  # Number of unique puzzle types
    vocab_size: int = 0          # Size of token vocabulary
    
    # Hierarchical architecture
    H_cycles: int = 2            # Number of H-level processing cycles per ACT step
    L_cycles: int = 2            # Number of L-level processing cycles per ACT step
    H_layers: int = 4            # Number of transformer layers in H-level
    L_layers: int = 4            # Number of transformer layers in L-level
    
    # Transformer configuration
    hidden_size: int = 512       # Hidden dimension for all layers
    expansion: float = 4.0       # MLP expansion factor
    num_heads: int = 8           # Number of attention heads
    pos_encodings: str = "rope"  # Position encoding type: "rope" or "learned"
    
    rms_norm_eps: float = 1e-5   # Epsilon for RMS normalization
    rope_theta: float = 10000.0  # Base frequency for RoPE
    
    # ACT configuration
    halt_max_steps: int = 16     # Maximum number of ACT steps
    halt_exploration_prob: float = 0.1  # Exploration probability during training
    
    @property
    def head_dim(self) -> int:
        """Compute head dimension from hidden size and number of heads."""
        return self.hidden_size // self.num_heads


class HRMBlock(nn.Module):
    """
    Single transformer block used in both H-level and L-level modules.
    
    Implements a standard transformer block with:
    - Multi-head self-attention (non-causal for bidirectional reasoning)
    - SwiGLU MLP (more efficient than standard FFN)
    - Post-normalization with RMSNorm
    - Residual connections
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        
        from ..modules.attention import Attention
        from ..layers.activations import SwiGLU
        
        self.config = config
        self.norm_eps = config.rms_norm_eps
        
        # Non-causal attention for bidirectional reasoning
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        
        # SwiGLU MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )
    
    def __call__(self, cos_sin: Optional[Tuple[mx.array, mx.array]], x: mx.array) -> mx.array:
        """
        Forward pass through transformer block.
        
        Args:
            cos_sin: RoPE embeddings (cos, sin) or None
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor with same shape as input
        """
        from ..layers.normalization import rms_norm
        
        # Self-attention with residual and normalization
        x = rms_norm(x + self.self_attn(cos_sin, x), self.norm_eps)
        
        # MLP with residual and normalization
        x = rms_norm(x + self.mlp(x), self.norm_eps)
        
        return x


class HRMReasoningModule(nn.Module):
    """
    Reasoning module implementing either H-level or L-level processing.
    
    Each module consists of:
    - Multiple transformer blocks (layers)
    - Multiple processing cycles through the layers
    - Shared weights across cycles (recurrent processing)
    """
    
    def __init__(self, config: HRMConfig, num_layers: int, num_cycles: int):
        super().__init__()
        
        self.config = config
        self.num_cycles = num_cycles
        
        # Stack of transformer blocks
        self.blocks = [HRMBlock(config) for _ in range(num_layers)]
    
    def __call__(
        self, 
        cos_sin: Optional[Tuple[mx.array, mx.array]], 
        x: mx.array
    ) -> mx.array:
        """
        Process input through multiple cycles of transformer blocks.
        
        Args:
            cos_sin: RoPE embeddings or None
            x: Input tensor
            
        Returns:
            Processed tensor after all cycles
        """
        # Run multiple cycles through the same blocks (recurrent processing)
        for _ in range(self.num_cycles):
            for block in self.blocks:
                x = block(cos_sin, x)
        
        return x