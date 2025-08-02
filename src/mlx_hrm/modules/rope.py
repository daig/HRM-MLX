"""
Rotary Position Embeddings (RoPE) implementation for MLX.

Exact behavioral match of the PyTorch HRM RoPE implementation, adapted for MLX's
array operations and optimizations.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input.
    
    This is a helper function for RoPE. It splits the tensor in half
    and rotates by swapping and negating. This creates the rotation
    effect needed for positional encoding.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor with x2 negated and swapped with x1
    """
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, 
    k: mx.array, 
    cos: mx.array, 
    sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply Rotary Position Embeddings to query and key tensors.
    
    RoPE encodes position information by rotating vectors in the complex plane.
    This allows the model to understand relative positions without explicit position embeddings.
    
    This implementation exactly matches PyTorch HRM's behavior, including:
    - Dtype casting to cos/sin dtype for computation
    - Unsqueezing at dimension -2 for broadcasting
    - Casting back to original dtype
    
    Args:
        q, k: Query and key tensors [batch_size, seq_len, num_heads, head_dim]
        cos, sin: Precomputed cosine and sine values [seq_len, head_dim]
    
    Returns:
        Rotated query and key tensors with position information encoded
    """
    # Store original dtype for restoration
    orig_dtype = q.dtype
    
    # Cast to cos/sin dtype for computation (matches PyTorch behavior)
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)
    
    # Expand cos/sin for broadcasting with q/k
    # cos/sin: [seq_len, head_dim] -> [seq_len, 1, head_dim]
    # This matches PyTorch's unsqueeze(-2)
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    
    # Apply rotation using complex number multiplication formula:
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # Cast back to original dtype
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for encoding position information.
    
    This is an exact port of PyTorch HRM's RotaryEmbedding implementation.
    RoPE encodes absolute positions using rotation matrices and naturally
    captures relative position information through the properties of rotations.
    
    Key advantages:
    - Can extrapolate to longer sequences than trained on
    - Captures relative positions naturally
    - No additional parameters to learn
    
    The implementation follows PyTorch HRM's approach:
    - Precomputes cos/sin values for efficiency
    - Uses the same frequency calculation
    - Duplicates frequencies for both sin and cos (differs from paper but same result)
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0
    ):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimension of the embeddings (head_dim)
            max_position_embeddings: Maximum sequence length to cache
            base: Base for the geometric progression (rope_theta in config)
        """
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate frequency bands for rotations
        # Higher dimensions rotate slower (lower frequency)
        # Exact match of PyTorch: 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        
        # Position indices
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        
        # Outer product gives frequencies for each position
        # Shape: [max_position_embeddings, dim//2]
        freqs = mx.outer(t, inv_freq)
        
        # Duplicate frequencies for both sin and cos
        # This differs from the paper but achieves the same result
        # Shape: [max_position_embeddings, dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        
        # Precompute and cache sin/cos values for efficiency
        # In MLX, we store these as attributes (no buffer concept)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)
    
    def __call__(
        self, 
        seq_len: Optional[int] = None,
        dtype: Optional[mx.Dtype] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Get cosine and sine values for the given sequence length.
        
        This matches PyTorch HRM's forward() method behavior.
        
        Args:
            seq_len: Sequence length (uses full cache if None)
            dtype: Output dtype (uses cached dtype if None)
            
        Returns:
            Tuple of (cos, sin) arrays with shape [seq_len, dim]
        """
        if seq_len is None:
            cos, sin = self.cos_cached, self.sin_cached
        else:
            # Return only up to seq_len
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        # Cast to requested dtype if specified
        if dtype is not None:
            cos = cos.astype(dtype)
            sin = sin.astype(dtype)
            
        return cos, sin
    
    def extend_cache(self, max_position_embeddings: int):
        """Extend the cached positions if needed.
        
        This allows the model to handle sequences longer than initially configured.
        """
        if max_position_embeddings <= self.max_position_embeddings:
            return
            
        # Update max position embeddings
        old_max = self.max_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        
        # Compute new positions
        t_new = mx.arange(old_max, max_position_embeddings, dtype=mx.float32)
        
        # Compute new frequencies
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        freqs_new = mx.outer(t_new, inv_freq)
        emb_new = mx.concatenate([freqs_new, freqs_new], axis=-1)
        
        # Extend cache
        self.cos_cached = mx.concatenate([
            self.cos_cached, 
            mx.cos(emb_new)
        ], axis=0)
        self.sin_cached = mx.concatenate([
            self.sin_cached, 
            mx.sin(emb_new)
        ], axis=0)


# Type alias for RoPE precomputed values (cosine and sine) - matches PyTorch HRM
CosSin = Tuple[mx.array, mx.array]