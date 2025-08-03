"""
Multi-Head Attention implementation for MLX HRM.

This module implements the attention mechanism used in the Hierarchical Reasoning Model,
replacing PyTorch's FlashAttention with MLX's scaled_dot_product_attention.

Supports:
- Multi-head attention (MHA) when num_heads == num_key_value_heads
- Grouped-query attention (GQA) when num_key_value_heads < num_heads
- Both causal and non-causal attention
- Integration with Rotary Position Embeddings (RoPE)
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..layers.initialization import LinearTruncNormal


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """
    Repeat key/value heads for grouped query attention.
    
    Args:
        x: Key or value tensor of shape [batch, seq_len, n_kv_heads, head_dim]
        n_rep: Number of times to repeat each KV head
        
    Returns:
        Repeated tensor of shape [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    
    batch, seq_len, n_kv_heads, head_dim = x.shape
    # Reshape to add repetition dimension
    x = x.reshape(batch, seq_len, n_kv_heads, 1, head_dim)
    # Broadcast to repeat
    x = mx.broadcast_to(x, (batch, seq_len, n_kv_heads, n_rep, head_dim))
    # Reshape back to merge repetitions with heads
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """
    Multi-head attention with MLX backend.
    
    This implements the standard multi-head attention mechanism but uses
    MLX's scaled_dot_product_attention for efficiency.
    
    Supports:
    - Multi-head attention (MHA) when num_heads == num_key_value_heads
    - Grouped-query attention (GQA) when num_key_value_heads < num_heads
    - Both causal and non-causal attention
    - Integration with RoPE for position encoding
    
    Args:
        hidden_size: Model hidden dimension
        head_dim: Dimension of each attention head
        num_heads: Number of query heads
        num_key_value_heads: Number of key/value heads (for GQA)
        causal: Whether to use causal masking
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.output_size = head_dim * num_heads
        
        # Ensure head dimensions are valid
        assert self.output_size == self.head_dim * self.num_heads, \
            f"output_size ({self.output_size}) != head_dim ({self.head_dim}) * num_heads ({self.num_heads})"
        
        # Check GQA configuration
        assert self.num_heads % self.num_key_value_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        
        self.n_rep = self.num_heads // self.num_key_value_heads
        
        # Fused QKV projection for efficiency
        # Output size = Q_size + K_size + V_size
        self.qkv_proj = LinearTruncNormal(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        
        # Output projection to map back to hidden size
        self.o_proj = LinearTruncNormal(
            self.output_size,
            self.hidden_size,
            bias=False
        )
    
    def __call__(
        self,
        cos_sin: Optional[Tuple[mx.array, mx.array]],
        hidden_states: mx.array
    ) -> mx.array:
        """
        Forward pass through attention layer.
        
        Args:
            cos_sin: Precomputed RoPE values (cos, sin) or None
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V in one operation for efficiency
        qkv = self.qkv_proj(hidden_states)
        
        # Reshape to separate heads
        # Shape: [batch_size, seq_len, num_heads + 2*num_kv_heads, head_dim]
        qkv = qkv.reshape(
            batch_size, seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim
        )
        
        # Split into Q, K, V
        # For GQA: K and V have fewer heads than Q
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # Apply RoPE if position encodings are provided
        if cos_sin is not None:
            from .rope import apply_rotary_pos_emb
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Repeat KV heads if using GQA
        if self.n_rep > 1:
            key = repeat_kv(key, self.n_rep)
            value = repeat_kv(value, self.n_rep)
        
        # Transpose for attention: [B, N_heads, Seq_len, Head_dim]
        # MLX's scaled_dot_product_attention expects this format
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute scaled dot product attention
        # Scale factor is 1/sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # MLX's fast attention with proper masking
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            mask="causal" if self.causal else None
        )
        
        # Transpose back and reshape
        # From [B, N_heads, Seq_len, Head_dim] to [B, Seq_len, N_heads, Head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        # Flatten heads: [B, Seq_len, N_heads * Head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        
        # Final output projection
        return self.o_proj(attn_output)
    
    def shape_info(self) -> str:
        """Return string describing the attention configuration."""
        gqa_info = f" (GQA {self.n_rep}:1)" if self.n_rep > 1 else ""
        causal_info = " (causal)" if self.causal else ""
        return (f"Attention: hidden_size={self.hidden_size}, "
                f"num_heads={self.num_heads}, num_kv_heads={self.num_key_value_heads}"
                f"{gqa_info}{causal_info}")