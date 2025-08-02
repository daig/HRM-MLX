"""
Custom weight initialization for MLX HRM.

This module implements truncated normal initialization matching the PyTorch HRM
implementation, which is based on JAX's truncated normal initialization.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Union
import math


def truncated_normal(
    shape: Union[int, Tuple[int, ...]],
    dtype: mx.Dtype = mx.float32,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    key: Optional[mx.array] = None
) -> mx.array:
    """
    Generate samples from a truncated normal distribution.
    
    This implementation matches the PyTorch HRM code exactly, which is based on
    JAX's truncated normal initialization with proper variance correction.
    
    Args:
        shape: Output shape (int or tuple of ints)
        dtype: Data type of the output
        mean: Mean of the distribution (should be 0 for standard init)
        std: Standard deviation of the output distribution (after truncation)
        lower: Lower truncation bound (in units of std)
        upper: Upper truncation bound (in units of std)
        key: Random key for deterministic initialization
        
    Returns:
        Array of shape `shape` with samples from truncated normal distribution
    """
    if isinstance(shape, int):
        shape = (shape,)
    
    if key is None:
        key = mx.random.key(0)
    
    # Handle zero std case
    if std == 0:
        return mx.zeros(shape, dtype=dtype)
    
    # Following the exact algorithm from HRM/models/common.py
    sqrt2 = math.sqrt(2)
    
    # Compute erf values at bounds
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2
    
    # Compute PDF values at bounds
    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower ** 2)  # IMPORTANT: pdf_u is PDF at LOWER bound (confusing naming from PyTorch)
    pdf_l = c * math.exp(-0.5 * upper ** 2)  # IMPORTANT: pdf_l is PDF at UPPER bound (confusing naming from PyTorch)
    
    # Compute compensated std (this is the key variance correction)
    # Note the formula matches PyTorch exactly, with their confusing variable names
    comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
    
    # Generate uniform samples in [a, b]
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    u = mx.random.uniform(shape=(total_elements,), key=key)
    samples = a + u * (b - a)
    
    # Apply inverse error function
    # erfinv(x) = inverf(x) * sqrt(2)
    # We need to compute erfinv for each sample
    samples_list = samples.tolist()
    erfinv_list = []
    for x in samples_list:
        # Compute erfinv using a rational approximation
        erfinv_list.append(_erfinv(x))
    
    samples = mx.array(erfinv_list, dtype=dtype)
    
    # Scale by sqrt(2) * comp_std
    samples = samples * sqrt2 * comp_std
    
    # Clip to bounds
    samples = mx.clip(samples, lower * comp_std, upper * comp_std)
    
    # Add mean (if non-zero)
    if mean != 0:
        samples = samples + mean
    
    # Reshape back to original shape
    samples = mx.reshape(samples, shape)
    
    return samples


def _erfinv(x: float) -> float:
    """
    Compute the inverse error function.
    
    Uses a rational approximation that's accurate for the full range [-1, 1].
    """
    if x == 0:
        return 0
    if x == 1:
        return float('inf')
    if x == -1:
        return float('-inf')
    if abs(x) > 1:
        raise ValueError("erfinv input must be in [-1, 1]")
    
    # Use different approximations for different ranges
    if abs(x) <= 0.7:
        # For |x| <= 0.7, use a polynomial approximation
        x2 = x * x
        a0 = 0.886226899
        a1 = -1.645349621
        a2 = 0.914624893
        a3 = -0.140543331
        b0 = 1
        b1 = -2.118377725
        b2 = 1.442710462
        b3 = -0.329097515
        b4 = 0.012229801
        
        num = ((a3 * x2 + a2) * x2 + a1) * x2 + a0
        den = (((b4 * x2 + b3) * x2 + b2) * x2 + b1) * x2 + b0
        return x * num / den
    else:
        # For |x| > 0.7, use a different approximation
        if x > 0:
            z = math.sqrt(-math.log(0.5 * (1 - x)))
        else:
            z = math.sqrt(-math.log(0.5 * (1 + x)))
        
        c0 = 1.432788582
        c1 = -0.189269933
        c2 = 0.001308028
        d0 = 1
        d1 = -0.196854185
        d2 = 0.049519673
        
        num = (c2 * z + c1) * z + c0
        den = (d2 * z + d1) * z + d0
        result = z * num / den
        
        return result if x > 0 else -result


def init_truncated_normal(
    std: float = 0.02,
    mean: float = 0.0,
    lower: float = -2.0,
    upper: float = 2.0,
    key: Optional[mx.array] = None
):
    """
    Create an initializer function for truncated normal distribution.
    
    This returns a function that can be used with MLX layers.
    
    Args:
        std: Standard deviation of the distribution
        mean: Mean of the distribution
        lower: Lower bound in standard deviations
        upper: Upper bound in standard deviations
        key: Random key for deterministic initialization
        
    Returns:
        Initializer function compatible with MLX layers
    """
    def _init(shape: Tuple[int, ...], dtype: mx.Dtype = mx.float32) -> mx.array:
        return truncated_normal(
            shape=shape,
            dtype=dtype,
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            key=key
        )
    return _init


class LinearTruncNormal(nn.Module):
    """Linear layer with truncated normal initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: Optional[float] = None,
        key: Optional[mx.array] = None
    ):
        """
        Initialize a linear layer with truncated normal weights.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features  
            bias: Whether to include bias term
            std: Standard deviation for initialization (default: 1/sqrt(in_features))
            key: Random key for deterministic initialization
        """
        super().__init__()
        
        if std is None:
            # LeCun normal initialization
            std = math.sqrt(1.0 / in_features)
        
        if key is None:
            key = mx.random.key(0)
        
        # Split key for weight and bias
        key_w, key_b = mx.random.split(key)
        
        # Initialize weights with truncated normal
        self.weight = truncated_normal(
            shape=(out_features, in_features),
            std=std,
            key=key_w
        )
        
        if bias:
            # Initialize bias to zeros (standard practice)
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
    
    def __call__(self, x):
        """Forward pass through the linear layer."""
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class EmbeddingTruncNormal(nn.Module):
    """Embedding layer with truncated normal initialization."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        std: Optional[float] = None,
        key: Optional[mx.array] = None
    ):
        """
        Initialize an embedding layer with truncated normal weights.
        
        Args:
            num_embeddings: Number of embeddings (vocabulary size)
            embedding_dim: Dimension of each embedding vector
            std: Standard deviation for initialization (default: 1/sqrt(embedding_dim))
            key: Random key for deterministic initialization
        """
        super().__init__()
        
        if std is None:
            std = 1.0 / math.sqrt(embedding_dim)
        
        if key is None:
            key = mx.random.key(0)
        
        self.weight = truncated_normal(
            shape=(num_embeddings, embedding_dim),
            std=std,
            key=key
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def __call__(self, indices):
        """Look up embeddings for the given indices."""
        return self.weight[indices]