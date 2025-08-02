"""RMSNorm implementation for MLX HRM.

This module provides both functional and module-based RMSNorm implementations
to match the original PyTorch HRM behavior while leveraging MLX optimizations.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def rms_norm(hidden_states: mx.array, variance_epsilon: float = 1e-5) -> mx.array:
    """Functional RMSNorm matching PyTorch HRM implementation exactly.
    
    This is a direct port of the PyTorch functional RMSNorm used in HRM.
    It normalizes without learnable parameters, matching the original behavior.
    
    Args:
        hidden_states: Input tensor to normalize
        variance_epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same shape and dtype as input
    """
    input_dtype = hidden_states.dtype
    # Cast to float32 for numerical stability
    hidden_states = hidden_states.astype(mx.float32)
    
    # Compute RMS: sqrt(mean(x²))
    variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
    
    # Normalize by RMS using rsqrt for efficiency
    hidden_states = hidden_states * mx.rsqrt(variance + variance_epsilon)
    
    # Cast back to original dtype
    return hidden_states.astype(input_dtype)


class RMSNorm(nn.Module):
    """Module-based RMSNorm with optional learnable scale parameter.
    
    This provides a more flexible implementation that can either:
    1. Match the original HRM behavior (no learnable parameters)
    2. Use learnable scale parameters (standard transformer approach)
    """
    
    def __init__(
        self, 
        dims: int, 
        eps: float = 1e-5,
        use_scale: bool = True
    ):
        """Initialize RMSNorm layer.
        
        Args:
            dims: Dimension to normalize (typically hidden_size)
            eps: Epsilon for numerical stability
            use_scale: Whether to use learnable scale parameter
        """
        super().__init__()
        self.eps = eps
        self.use_scale = use_scale
        
        if use_scale:
            # Use MLX's built-in RMSNorm with learnable parameters
            self._norm = nn.RMSNorm(dims, eps=eps)
        else:
            # No learnable parameters, just store config
            self._norm = None
    
    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        if self.use_scale:
            # Use MLX's built-in implementation
            return self._norm(x)
        else:
            # Use functional implementation without learnable parameters
            return rms_norm(x, self.eps)


class RMSNormCompatible(nn.Module):
    """RMSNorm that exactly matches PyTorch HRM's functional behavior.
    
    This is a module wrapper around the functional implementation,
    providing the same interface as nn.Module while maintaining
    the exact behavior of the original PyTorch implementation.
    """
    
    def __init__(self, eps: float = 1e-5):
        """Initialize RMSNorm compatible layer.
        
        Args:
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization without learnable parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return rms_norm(x, self.eps)


def create_rms_norm(
    dims: Optional[int] = None,
    eps: float = 1e-5,
    use_scale: bool = False,
    functional_only: bool = False
) -> Optional[RMSNorm]:
    """Factory function to create appropriate RMSNorm variant.
    
    Args:
        dims: Hidden dimension (required if use_scale=True)
        eps: Epsilon for numerical stability
        use_scale: Whether to use learnable scale parameters
        functional_only: If True, returns None (use rms_norm function directly)
        
    Returns:
        RMSNorm module or None if functional_only=True
    """
    if functional_only:
        return None
    
    if use_scale and dims is None:
        raise ValueError("dims must be specified when use_scale=True")
    
    if use_scale:
        return RMSNorm(dims, eps=eps, use_scale=True)
    else:
        return RMSNormCompatible(eps=eps)