"""
Precision-aware layers for MLX HRM implementation.

This module implements layers that match the original PyTorch HRM's sophisticated
manual mixed precision system, where:
- Master weights are stored in FP32 for numerical stability
- Computation happens in BF16 for efficiency  
- Critical operations (normalization, loss) use higher precision as needed

Key components:
- MLXCastedLinear: FP32 master weights, cast to computation dtype
- precision_aware_rms_norm: Always FP32 computation regardless of input dtype
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional
from ..layers.initialization import truncated_normal


class MLXCastedLinear(nn.Module):
    """
    MLX equivalent of PyTorch HRM's CastedLinear layer.
    
    This layer stores weights in FP32 (master weights) for numerical stability
    but casts them to the input dtype during computation for efficiency.
    This matches the original HRM's precision architecture exactly.
    
    Key features:
    - Weights stored in FP32 for gradient stability
    - Dynamic casting to input dtype (typically BF16) during forward pass
    - Truncated LeCun normal initialization matching original HRM
    - Optional bias support (most HRM layers use bias=False)
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term (default: False, matching most HRM usage)
        master_dtype: Dtype for storing master weights (default: float32)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        master_dtype: mx.Dtype = mx.float32
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.master_dtype = master_dtype
        
        # Initialize weights with truncated LeCun normal in FP32 (master weights)
        # This matches the original HRM initialization exactly
        std = math.sqrt(1.0 / in_features)  # LeCun normal scaling
        
        self.weight = truncated_normal(
            shape=(out_features, in_features),
            dtype=master_dtype,  # Store in FP32 for stability
            std=std,
            lower=-2.0,
            upper=2.0
        )
        
        if bias:
            # Zero initialization for bias (standard practice)
            self.bias = mx.zeros((out_features,), dtype=master_dtype)
        else:
            self.bias = None
            
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with dynamic precision casting.
        
        This exactly matches the original PyTorch HRM's CastedLinear behavior:
        F.linear(input, self.weight.to(input.dtype), bias.to(input.dtype))
        
        Args:
            x: Input tensor (typically in BF16 for forward computation)
            
        Returns:
            Output tensor in same dtype as input
        """
        
        # Cast master weights to input dtype for computation
        # This is the key insight from the original HRM implementation
        weight_cast = self.weight.astype(x.dtype)
        
        # Linear transformation: y = x @ W^T
        y = x @ weight_cast.T
        
        # Add bias if present (also cast to input dtype)
        if self.bias is not None:
            bias_cast = self.bias.astype(x.dtype)
            y = y + bias_cast
            
        return y
    
    def __repr__(self):
        bias_str = ", bias=True" if self.bias is not None else ""
        return (f"MLXCastedLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}{bias_str}, "
                f"master_dtype={self.master_dtype})")


def precision_aware_rms_norm(
    x: mx.array, 
    weight: Optional[mx.array] = None, 
    eps: float = 1e-6
) -> mx.array:
    """
    RMSNorm that always computes in FP32 regardless of input dtype.
    
    This matches the original PyTorch HRM behavior where normalization
    operations are cast to FP32 for numerical stability:
    
    # Cast to float32 for stability in normalization computation
    hidden_states = hidden_states.to(torch.float32)
    
    Args:
        x: Input tensor (any dtype, typically BF16)
        weight: Optional scale parameter (stored in FP32)
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized tensor cast back to original input dtype
    """
    original_dtype = x.dtype
    
    # Cast to FP32 for stable computation (like original HRM)
    x_fp32 = x.astype(mx.float32)
    
    # Compute RMS normalization in FP32
    mean_square = mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    rms_inv = mx.rsqrt(mean_square + eps)
    normalized = x_fp32 * rms_inv
    
    # Apply scale if provided (weight should be in FP32)
    if weight is not None:
        weight_fp32 = weight.astype(mx.float32)
        normalized = normalized * weight_fp32
    
    # Cast back to original dtype for output
    return normalized.astype(original_dtype)


class PrecisionAwareRMSNorm(nn.Module):
    """
    RMSNorm module that always computes in FP32 for stability.
    
    This matches the original HRM's approach where normalization layers
    are precision-critical and always use FP32 computation regardless
    of the input dtype.
    
    Args:
        hidden_size: Size of the input features
        eps: Small epsilon for numerical stability
        learnable_scale: Whether to include learnable scale parameter
        master_dtype: Dtype for storing parameters (default: float32)
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        learnable_scale: bool = True,
        master_dtype: mx.Dtype = mx.float32
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.eps = eps
        self.master_dtype = master_dtype
        
        if learnable_scale:
            # Initialize scale parameter to ones in FP32 (master parameter)
            self.weight = mx.ones((hidden_size,), dtype=master_dtype)
        else:
            self.weight = None
            
    def __call__(self, x: mx.array) -> mx.array:
        """Apply precision-aware RMS normalization."""
        return precision_aware_rms_norm(x, self.weight, self.eps)
        
    def __repr__(self):
        learnable = "learnable_scale=True" if self.weight is not None else "learnable_scale=False"
        return (f"PrecisionAwareRMSNorm(hidden_size={self.hidden_size}, "
                f"eps={self.eps}, {learnable})")