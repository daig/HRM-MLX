"""
Precision-aware Inner HRM model implementation for MLX.

This module implements the precision-aware core HRM model that processes inputs through
hierarchical H-level and L-level modules with exact HRM precision compliance.

Key features:
- FP32 master weights for all parameters
- Forward computation in configured dtype (typically BF16)
- Precision-aware components throughout
- Exact match to original PyTorch HRM precision architecture
"""

from typing import Dict, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.act import HRMConfig, HRMInnerCarry, PrecisionAwareHRMReasoningModule
from ..modules.rope import RotaryEmbedding
from ..layers.precision import MLXCastedLinear
from ..layers.embeddings import PrecisionAwareSparseEmbedding
from ..layers.normalization import rms_norm
from ..models.precision_config import MLXPrecisionConfig


class PrecisionAwareHRMInner(nn.Module):
    """
    Precision-aware Inner HRM model implementing the hierarchical reasoning architecture.
    
    This version uses precision-aware components from Phase 1 to match the original
    PyTorch HRM's manual mixed precision architecture exactly:
    - FP32 master weights for all parameters
    - Forward computation in configured dtype (typically BF16)
    - Precision-aware embeddings, attention, and MLPs
    - Exact numerical compliance with original implementation
    
    Architecture:
    1. Token embeddings (precision-aware sparse for puzzle-specific, dense for vocabulary)
    2. RoPE position embeddings
    3. H-level processing (precision-aware high-level planning)
    4. L-level processing (precision-aware low-level computation)
    5. Output projections for token prediction and Q-values (precision-aware)
    
    Args:
        config: HRM configuration
        precision_config: Precision configuration (defaults to standard HRM config)
    """
    
    def __init__(self, config: HRMConfig, precision_config: Optional[MLXPrecisionConfig] = None):
        super().__init__()
        
        self.config = config
        self.norm_eps = config.rms_norm_eps
        
        # Use standard HRM precision config if none provided
        self.precision_config = precision_config or MLXPrecisionConfig.create_herm_standard_config()
        
        # Token embeddings with precision awareness
        if config.puzzle_emb_ndim > 0:
            # Precision-aware sparse embeddings for puzzle-specific tokens
            self.sparse_tok_emb = PrecisionAwareSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.hidden_size,  # Must match hidden_size for concatenation
                batch_size=config.batch_size,
                init_std=0.02,
                precision_config=self.precision_config
            )
        else:
            self.sparse_tok_emb = None
        
        # Precision-aware dense embeddings for vocabulary
        # Using MLXCastedLinear as an embedding layer (no bias, FP32 master weights)
        if config.vocab_size > 0:
            self.dense_tok_emb = MLXCastedLinear(
                input_dims=config.vocab_size,
                output_dims=config.hidden_size,
                bias=False
            )
        else:
            self.dense_tok_emb = None
        
        # Position embeddings (RoPE doesn't have learnable parameters)
        if config.pos_encodings == "rope":
            self.rope = RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.seq_len + 128,  # Extra for puzzle embeddings
                base=config.rope_theta
            )
        else:
            # Learned position embeddings (not implemented yet)
            raise NotImplementedError("Only RoPE position encodings are currently supported")
        
        # Precision-aware hierarchical reasoning modules
        self.H_module = PrecisionAwareHRMReasoningModule(
            config=config,
            num_layers=config.H_layers,
            num_cycles=config.H_cycles,
            precision_config=self.precision_config
        )
        
        self.L_module = PrecisionAwareHRMReasoningModule(
            config=config,
            num_layers=config.L_layers,
            num_cycles=config.L_cycles,
            precision_config=self.precision_config
        )
        
        # Precision-aware output heads with FP32 master weights
        # Language modeling head for token prediction
        self.lm_head = MLXCastedLinear(
            input_dims=config.hidden_size,
            output_dims=config.vocab_size,
            bias=False
        )
        
        # Q-head for ACT halting decisions with FP32 master weights
        # Output[0]: Q-value for halting, Output[1]: Q-value for continuing
        self.q_head = MLXCastedLinear(
            input_dims=config.hidden_size,
            output_dims=2,
            bias=False
        )
        
        # Learned initial states for resetting (stored in FP32)
        # These are used when sequences halt and need to be reset
        self.z_init_H = mx.zeros((1, 1, config.hidden_size), dtype=mx.float32)
        self.z_init_L = mx.zeros((1, 1, config.hidden_size), dtype=mx.float32)
    
    def embed_tokens(self, batch: Dict[str, mx.array]) -> mx.array:
        """
        Embed input tokens combining sparse and dense embeddings with precision handling.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Embedded tokens [batch_size, seq_len + puzzle_emb_len, hidden_size] in forward dtype
        """
        embeddings = []
        forward_dtype = self.precision_config.get_forward_dtype()
        
        # Precision-aware sparse puzzle embeddings
        if self.sparse_tok_emb is not None:
            puzzle_ids = batch.get("puzzle_ids")
            if puzzle_ids is not None:
                # PrecisionAwareSparseEmbedding automatically casts to forward dtype
                puzzle_emb = self.sparse_tok_emb(puzzle_ids)  # [batch, hidden_size] in forward_dtype
                # Add sequence dimension
                puzzle_emb = puzzle_emb.reshape(puzzle_emb.shape[0], 1, puzzle_emb.shape[1])
                embeddings.append(puzzle_emb)
        
        # Precision-aware dense token embeddings
        if self.dense_tok_emb is not None:
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                # Create one-hot encoding for embedding lookup
                batch_size, seq_len = input_ids.shape
                one_hot = mx.zeros((batch_size, seq_len, self.config.vocab_size), dtype=forward_dtype)
                one_hot = mx.scatter(one_hot, input_ids.reshape(batch_size, seq_len, 1), 1.0, axis=2)
                
                # Use MLXCastedLinear for embedding (FP32 master weights -> forward dtype)
                token_emb = []
                for i in range(seq_len):
                    emb_i = self.dense_tok_emb(one_hot[:, i])  # [batch, hidden_size]
                    token_emb.append(emb_i)
                token_emb = mx.stack(token_emb, axis=1)  # [batch, seq_len, hidden_size]
                embeddings.append(token_emb)
        
        # Concatenate all embeddings along sequence dimension
        if len(embeddings) > 1:
            x = mx.concatenate(embeddings, axis=1)
        else:
            x = embeddings[0]
        
        # Ensure output is in forward dtype
        return x.astype(forward_dtype)
    
    def reset_carry(self, halted: mx.array, carry: HRMInnerCarry) -> HRMInnerCarry:
        """
        Reset carry states for sequences that have halted with precision handling.
        
        When a sequence completes (halts), we reset its hidden states to learned
        initial values. This prepares it to process a new sequence in the next step.
        
        Args:
            halted: Boolean mask [batch_size] indicating which sequences halted
            carry: Current carry state
            
        Returns:
            Updated carry state with reset sequences in forward dtype
        """
        batch_size = halted.shape[0]
        seq_len = carry.z_H.shape[1]
        forward_dtype = self.precision_config.get_forward_dtype()
        
        # Cast initial states to forward dtype and broadcast to match dimensions
        z_init_H = self.z_init_H.astype(forward_dtype)
        z_init_L = self.z_init_L.astype(forward_dtype)
        
        z_init_H = mx.broadcast_to(z_init_H, (batch_size, seq_len, self.config.hidden_size))
        z_init_L = mx.broadcast_to(z_init_L, (batch_size, seq_len, self.config.hidden_size))
        
        # Reset halted sequences
        halted_mask = halted.reshape(-1, 1, 1)
        new_z_H = mx.where(halted_mask, z_init_H, carry.z_H.astype(forward_dtype))
        new_z_L = mx.where(halted_mask, z_init_L, carry.z_L.astype(forward_dtype))
        
        return HRMInnerCarry(z_H=new_z_H, z_L=new_z_L)
    
    def __call__(
        self,
        carry: HRMInnerCarry,
        batch: Dict[str, mx.array]
    ) -> Tuple[HRMInnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass through the precision-aware inner HRM model.
        
        Args:
            carry: Current carry state with H/L hidden states
            batch: Input batch data
            
        Returns:
            - new_carry: Updated carry state in forward dtype
            - logits: Token predictions [batch, seq_len, vocab_size] in forward dtype
            - (q_halt, q_continue): Q-values for ACT halting decision in forward dtype
        """
        forward_dtype = self.precision_config.get_forward_dtype()
        
        # Embed input tokens (returns forward dtype)
        x = self.embed_tokens(batch)
        
        # Get sequence length (including puzzle embedding if present)
        seq_len = x.shape[1]
        
        # Get RoPE embeddings (no precision conversion needed - just cos/sin values)
        cos, sin = self.rope(seq_len, dtype=forward_dtype)
        cos_sin = (cos, sin)
        
        # Ensure carry states are in forward dtype
        carry_z_H = carry.z_H.astype(forward_dtype)
        carry_z_L = carry.z_L.astype(forward_dtype)
        
        # H-level processing: Planning and high-level reasoning
        # All computation in forward dtype with FP32 master weights
        z_H = self.H_module(cos_sin, x + carry_z_H)
        
        # L-level processing: Low-level computation
        # All computation in forward dtype with FP32 master weights
        z_L = self.L_module(cos_sin, z_H + carry_z_L)
        
        # Output projections using precision-aware linear layers
        # Use only the token positions (skip puzzle embedding if present)
        if self.config.puzzle_emb_ndim > 0 and batch.get("puzzle_ids") is not None:
            # Skip first position (puzzle embedding)
            output_positions = z_L[:, 1:]
        else:
            output_positions = z_L
        
        # Token predictions using MLXCastedLinear (FP32 master weights -> forward dtype)
        logits = self.lm_head(output_positions)
        
        # Q-values for halting decision (use mean pooling over sequence)
        # This gives us a single halt/continue decision per sequence
        pooled = output_positions.mean(axis=1)  # [batch, hidden_size]
        q_logits = self.q_head(pooled)  # [batch, 2] - MLXCastedLinear handles precision
        
        # Split Q-values
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # Create new carry state in forward dtype
        new_carry = HRMInnerCarry(z_H=z_H, z_L=z_L)
        
        return new_carry, logits, (q_halt, q_continue)
    
    def precision_info(self) -> str:
        """Return string describing the precision configuration."""
        num_params = sum(p.size for p in self.parameters().values() if isinstance(p, mx.array))
        return (f"PrecisionAwareHRMInner: {num_params:,} parameters, "
                f"forward: {self.precision_config.forward_dtype}, "
                f"master: {self.precision_config.master_weights_dtype}")