"""Activation functions for MLX HRM.

This module provides custom activation functions including SwiGLU
for the HRM model implementation.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .initialization import LinearTruncNormal


def _find_multiple(n: int, multiple: int) -> int:
    """Find the smallest multiple of 'multiple' that is >= n.
    
    Used to ensure dimensions are multiples of certain values for
    hardware efficiency on Apple Silicon.
    
    Args:
        n: The number to round up
        multiple: The multiple to round to
        
    Returns:
        The smallest multiple of 'multiple' that is >= n
    """
    return ((n + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    """SwiGLU activation function matching PyTorch HRM implementation.
    
    SwiGLU is a gated linear unit that uses SiLU (Swish) as the activation.
    It's more parameter-efficient and performs better than standard FFN.
    
    Architecture:
    - Input → [Gate projection, Up projection] → SiLU(gate) * up → Down projection → Output
    
    The 2/3 factor maintains roughly the same parameter count as a standard FFN
    with 4x expansion when accounting for the gating overhead.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        expansion: float = 4.0,
        multiple_of: int = 256,
        bias: bool = False,
        init_std: Optional[float] = None
    ):
        """Initialize SwiGLU module.
        
        Args:
            hidden_size: Input/output dimension
            expansion: Expansion factor for intermediate dimension
            multiple_of: Round intermediate dimension to multiple of this
            bias: Whether to use bias in linear layers (default: False)
            init_std: Standard deviation for weight initialization
                     If None, uses (2 / (5 * hidden_size))^0.5
        """
        super().__init__()
        
        # Calculate intermediate dimension
        # 2/3 factor: SwiGLU has 3 weight matrices vs standard FFN's 2
        intermediate_dim = int(round(expansion * hidden_size * 2 / 3))
        intermediate_dim = _find_multiple(intermediate_dim, multiple_of)
        
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        
        # Compute initialization std if not provided
        if init_std is None:
            # Following the original HRM initialization strategy
            init_std = (2.0 / (5 * hidden_size)) ** 0.5
        
        # Combined gate and up projection for efficiency
        # Uses custom initialization matching PyTorch HRM
        self.gate_up_proj = LinearTruncNormal(
            hidden_size, 
            intermediate_dim * 2, 
            bias=bias,
            std=init_std
        )
        
        # Down projection back to model dimension
        # Initialize with smaller std as in original
        self.down_proj = LinearTruncNormal(
            intermediate_dim, 
            hidden_size, 
            bias=bias,
            std=init_std / (2 ** 0.5)
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through SwiGLU.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            Output tensor of shape [..., hidden_size]
        """
        # Project to gate and up (single matrix multiply)
        gate_up = self.gate_up_proj(x)
        
        # Split into gate and up components
        # Using split instead of chunk for MLX
        gate, up = mx.split(gate_up, 2, axis=-1)
        
        # Apply SiLU (Swish) activation to gate and multiply with up
        # SiLU(x) = x * sigmoid(x)
        gate_activated = gate * mx.sigmoid(gate)
        output = self.down_proj(gate_activated * up)
        
        return output


def silu(x: mx.array) -> mx.array:
    """SiLU (Swish) activation function.
    
    SiLU(x) = x * sigmoid(x)
    
    This is provided for reference, but mx.silu should be used directly
    for better performance.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * mx.sigmoid(x)


class SwiGLUFactory:
    """Factory for creating SwiGLU modules with common configurations."""
    
    @staticmethod
    def create_default(hidden_size: int, **kwargs) -> SwiGLU:
        """Create SwiGLU with default HRM configuration.
        
        Args:
            hidden_size: Model hidden dimension
            **kwargs: Additional arguments passed to SwiGLU
            
        Returns:
            SwiGLU module with default settings
        """
        defaults = {
            'expansion': 4.0,
            'multiple_of': 256,
            'bias': False
        }
        defaults.update(kwargs)
        return SwiGLU(hidden_size, **defaults)
    
    @staticmethod
    def create_large(hidden_size: int, **kwargs) -> SwiGLU:
        """Create SwiGLU with larger expansion for increased capacity.
        
        Args:
            hidden_size: Model hidden dimension
            **kwargs: Additional arguments passed to SwiGLU
            
        Returns:
            SwiGLU module with larger expansion
        """
        defaults = {
            'expansion': 8.0,
            'multiple_of': 256,
            'bias': False
        }
        defaults.update(kwargs)
        return SwiGLU(hidden_size, **defaults)
    
    @staticmethod
    def create_efficient(hidden_size: int, **kwargs) -> SwiGLU:
        """Create SwiGLU optimized for efficiency.
        
        Args:
            hidden_size: Model hidden dimension
            **kwargs: Additional arguments passed to SwiGLU
            
        Returns:
            SwiGLU module optimized for speed/memory
        """
        defaults = {
            'expansion': 2.667,  # Results in nice round numbers after 2/3
            'multiple_of': 128,  # Smaller multiple for efficiency
            'bias': False
        }
        defaults.update(kwargs)
        return SwiGLU(hidden_size, **defaults)