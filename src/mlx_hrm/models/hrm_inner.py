"""
Inner HRM model implementation for MLX.

This module implements the core HRM model that processes inputs through
hierarchical H-level and L-level modules and produces outputs including
Q-values for ACT halting decisions.
"""

from typing import Dict, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.act import HRMConfig, HRMInnerCarry, HRMReasoningModule
from ..modules.rope import RotaryEmbedding
from ..layers.initialization import LinearTruncNormal, EmbeddingTruncNormal
from ..layers.embeddings import CastedSparseEmbedding
from ..layers.normalization import rms_norm


class HRMInner(nn.Module):
    """
    Inner HRM model implementing the hierarchical reasoning architecture.
    
    Architecture:
    1. Token embeddings (sparse for puzzle-specific, dense for vocabulary)
    2. RoPE position embeddings
    3. H-level processing (high-level planning)
    4. L-level processing (low-level computation)
    5. Output projections for token prediction and Q-values
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        
        self.config = config
        self.norm_eps = config.rms_norm_eps
        
        # Token embeddings
        if config.puzzle_emb_ndim > 0:
            # Sparse embeddings for puzzle-specific tokens
            self.sparse_tok_emb = CastedSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                num_embeddings_per_sample=128,  # Fixed for HRM
                init_scale=0.02
            )
            total_emb_dim = config.puzzle_emb_ndim
        else:
            self.sparse_tok_emb = None
            total_emb_dim = 0
        
        # Dense embeddings for vocabulary
        if config.vocab_size > 0:
            self.dense_tok_emb = EmbeddingTruncNormal(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                std=0.02
            )
            total_emb_dim += config.hidden_size
        else:
            self.dense_tok_emb = None
        
        # Project embeddings to hidden size if needed
        if total_emb_dim != config.hidden_size:
            self.emb_proj = LinearTruncNormal(
                total_emb_dim,
                config.hidden_size,
                bias=False
            )
        else:
            self.emb_proj = None
        
        # Position embeddings
        if config.pos_encodings == "rope":
            self.rope = RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.seq_len + 128,  # Extra for puzzle embeddings
                base=config.rope_theta
            )
        else:
            # Learned position embeddings (not implemented yet)
            raise NotImplementedError("Only RoPE position encodings are currently supported")
        
        # Hierarchical reasoning modules
        self.H_module = HRMReasoningModule(
            config=config,
            num_layers=config.H_layers,
            num_cycles=config.H_cycles
        )
        
        self.L_module = HRMReasoningModule(
            config=config,
            num_layers=config.L_layers,
            num_cycles=config.L_cycles
        )
        
        # Output heads
        # Language modeling head for token prediction
        self.lm_head = LinearTruncNormal(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
        
        # Q-head for ACT halting decisions
        # Output[0]: Q-value for halting, Output[1]: Q-value for continuing
        self.q_head = LinearTruncNormal(
            config.hidden_size,
            2,
            bias=False
        )
        
        # Learned initial states for resetting
        # These are used when sequences halt and need to be reset
        self.z_init_H = mx.zeros((1, 1, config.hidden_size))
        self.z_init_L = mx.zeros((1, 1, config.hidden_size))
    
    def embed_tokens(self, batch: Dict[str, mx.array]) -> mx.array:
        """
        Embed input tokens combining sparse and dense embeddings.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Embedded tokens [batch_size, seq_len + puzzle_emb_len, hidden_size]
        """
        embeddings = []
        
        # Sparse puzzle embeddings
        if self.sparse_tok_emb is not None:
            puzzle_ids = batch.get("puzzle_ids")
            if puzzle_ids is not None:
                puzzle_emb = self.sparse_tok_emb(puzzle_ids)  # [batch, puzzle_emb_ndim]
                # Add sequence dimension
                puzzle_emb = puzzle_emb.reshape(puzzle_emb.shape[0], 1, puzzle_emb.shape[1])
                embeddings.append(puzzle_emb)
        
        # Dense token embeddings
        if self.dense_tok_emb is not None:
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                token_emb = self.dense_tok_emb(input_ids)  # [batch, seq_len, hidden_size]
                embeddings.append(token_emb)
        
        # Concatenate all embeddings along sequence dimension
        if len(embeddings) > 1:
            x = mx.concatenate(embeddings, axis=1)
        else:
            x = embeddings[0]
        
        # Project to hidden size if needed
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        
        return x
    
    def reset_carry(self, halted: mx.array, carry: HRMInnerCarry) -> HRMInnerCarry:
        """
        Reset carry states for sequences that have halted.
        
        When a sequence completes (halts), we reset its hidden states to learned
        initial values. This prepares it to process a new sequence in the next step.
        
        Args:
            halted: Boolean mask [batch_size] indicating which sequences halted
            carry: Current carry state
            
        Returns:
            Updated carry state with reset sequences
        """
        batch_size = halted.shape[0]
        seq_len = carry.z_H.shape[1]
        
        # Broadcast initial states to match dimensions
        z_init_H = mx.broadcast_to(self.z_init_H, (batch_size, seq_len, self.config.hidden_size))
        z_init_L = mx.broadcast_to(self.z_init_L, (batch_size, seq_len, self.config.hidden_size))
        
        # Reset halted sequences
        halted_mask = halted.reshape(-1, 1, 1)
        new_z_H = mx.where(halted_mask, z_init_H, carry.z_H)
        new_z_L = mx.where(halted_mask, z_init_L, carry.z_L)
        
        return HRMInnerCarry(z_H=new_z_H, z_L=new_z_L)
    
    def __call__(
        self,
        carry: HRMInnerCarry,
        batch: Dict[str, mx.array]
    ) -> Tuple[HRMInnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass through the inner HRM model.
        
        Args:
            carry: Current carry state with H/L hidden states
            batch: Input batch data
            
        Returns:
            - new_carry: Updated carry state
            - logits: Token predictions [batch, seq_len, vocab_size]
            - (q_halt, q_continue): Q-values for ACT halting decision
        """
        # Embed input tokens
        x = self.embed_tokens(batch)
        
        # Get sequence length (including puzzle embedding if present)
        seq_len = x.shape[1]
        
        # Get RoPE embeddings
        cos, sin = self.rope(seq_len, dtype=x.dtype)
        cos_sin = (cos, sin)
        
        # H-level processing: Planning and high-level reasoning
        # Input: embedded tokens + previous H-level state
        z_H = self.H_module(cos_sin, x + carry.z_H)
        
        # L-level processing: Low-level computation
        # Input: H-level output + previous L-level state
        z_L = self.L_module(cos_sin, z_H + carry.z_L)
        
        # Output projections
        # Use only the token positions (skip puzzle embedding if present)
        if self.config.puzzle_emb_ndim > 0:
            # Skip first position (puzzle embedding)
            output_positions = z_L[:, 1:]
        else:
            output_positions = z_L
        
        # Token predictions
        logits = self.lm_head(output_positions)
        
        # Q-values for halting decision (use mean pooling over sequence)
        # This gives us a single halt/continue decision per sequence
        pooled = output_positions.mean(axis=1)  # [batch, hidden_size]
        q_logits = self.q_head(pooled)  # [batch, 2]
        
        # Split Q-values
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # Create new carry state
        new_carry = HRMInnerCarry(z_H=z_H, z_L=z_L)
        
        return new_carry, logits, (q_halt, q_continue)