"""
Complete Hierarchical Reasoning Model for MLX.

This is the main user-facing API that wraps the ACT model
and provides convenient methods for training and inference.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from ..modules.act import HRMConfig, HRMCarry
from .hrm_act import HRM_ACT


class HRM(nn.Module):
    """
    Complete Hierarchical Reasoning Model for MLX.
    
    This is the main user-facing API that wraps the ACT model
    and provides convenient methods for training and inference.
    
    Args:
        config: HRMConfig object, string preset name, or dict
        
    Example:
        >>> # Using a preset
        >>> model = HRM('small')
        
        >>> # Using a custom config
        >>> config = HRMConfig(hidden_size=512, num_heads=8, ...)
        >>> model = HRM(config)
        
        >>> # Forward pass
        >>> batch = {'input_ids': mx.array([[1, 2, 3, 4]])}
        >>> carry, outputs = model(batch)
        >>> logits = outputs['logits']
    """
    
    def __init__(self, config: Union[HRMConfig, str, Dict]):
        """
        Initialize HRM model.
        
        Args:
            config: HRMConfig object, string preset name, or dict
        """
        super().__init__()
        
        # Handle different config types
        if isinstance(config, str):
            from ..configs.model_presets import get_preset_config
            config = get_preset_config(config)
        elif isinstance(config, dict):
            config = HRMConfig(**config)
        
        self.config = config
        self.model = HRM_ACT(config)
        
    def __call__(
        self, 
        carry: Optional[HRMCarry],
        batch: Dict[str, mx.array]
    ) -> Tuple[HRMCarry, Dict[str, mx.array]]:
        """
        Standard forward pass.
        
        Args:
            batch: Dictionary containing 'input_ids' and optionally 'labels'
            carry: Optional carry state from previous forward pass
            
        Returns:
            new_carry: Updated carry state
            outputs: Dictionary containing:
                - logits: Language modeling predictions [batch_size, seq_len, vocab_size]
                - q_halt_logits: Q-values for halting decision [batch_size]
                - q_continue_logits: Q-values for continuation [batch_size]
                - halted: Whether each sequence halted [batch_size]
        """
        if carry is None:
            carry = self.initial_carry(batch['input_ids'].shape[0])
        return self.model(carry, batch)
    
    def initial_carry(self, batch_size: int) -> HRMCarry:
        """
        Create initial carry state for a batch.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            Initial carry state
        """
        return self.model.initial_carry(batch_size)
    
    def named_parameters(self):
        """
        Return named parameters like PyTorch for test compatibility.
        
        Returns:
            Iterator of (name, parameter) tuples
        """
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                name = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from flatten_dict(v, name)
                elif isinstance(v, mx.array):
                    yield (name, v)
        
        return flatten_dict(self.parameters())
    
    def forward_single(
        self, 
        input_ids: mx.array,
        max_length: Optional[int] = None
    ) -> mx.array:
        """
        Convenience method for single sequence inference.
        
        Args:
            input_ids: Token IDs [seq_len]
            max_length: Maximum sequence length (for padding)
            
        Returns:
            Logits for next token prediction [seq_len, vocab_size]
        """
        # Add batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        # Pad if needed
        if max_length and input_ids.shape[1] < max_length:
            padding = mx.zeros((1, max_length - input_ids.shape[1]), dtype=input_ids.dtype)
            input_ids = mx.concatenate([input_ids, padding], axis=1)
        
        # Forward pass
        batch = {'input_ids': input_ids}
        _, outputs = self(None, batch)
        
        # Remove batch dimension
        return outputs['logits'][0]
    
    def generate(
        self,
        prompt: mx.array,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token_id: Optional[int] = None
    ) -> mx.array:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Initial token IDs [seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering (None for no filtering)
            stop_token_id: Token ID to stop generation
            
        Returns:
            Generated token IDs [generated_len]
        """
        # Ensure prompt is 2D
        if prompt.ndim == 1:
            prompt = prompt.reshape(1, -1)
        
        generated = prompt
        carry = self.initial_carry(1)
        
        for _ in range(max_length - prompt.shape[1]):
            # Forward pass - create new carry for each step due to changing sequence length
            batch = {'input_ids': generated}
            carry = self.initial_carry(1)  # Fresh carry each time
            carry, outputs = self.model(carry, batch)
            
            # Get next token logits
            next_logits = outputs['logits'][0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_indices = mx.argsort(next_logits)[-top_k:]
                mask = mx.ones_like(next_logits) * float('-inf')
                mask[top_k_indices] = 0
                next_logits = next_logits + mask
            
            # Sample next token
            probs = mx.softmax(next_logits)
            next_token = mx.random.categorical(mx.log(probs))
            
            # Check for stop token
            if stop_token_id is not None and next_token.item() == stop_token_id:
                break
            
            # Append to generated sequence
            generated = mx.concatenate([generated, next_token.reshape(1, 1)], axis=1)
            
            # Check if model halted
            if carry.halted[0]:
                break
        
        return generated[0]
    
    def save_weights(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Direct equivalent of PyTorch's torch.save(model.state_dict(), path)
        # Just save the parameters dict directly
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.parameters(), f)
    
    def load_weights(self, path: str):
        """
        Load model weights.
        
        Args:
            path: Path to load weights from
        """
        path = Path(path)
        
        # Direct equivalent of PyTorch's model.load_state_dict(torch.load(path))
        import pickle
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        
        # Update model parameters
        self.update(weights)
    
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        
        def count_params(params):
            """Recursively count parameters in nested dict."""
            nonlocal total
            for k, v in params.items():
                if isinstance(v, dict):
                    count_params(v)
                elif isinstance(v, mx.array):
                    total += v.size
                    
        count_params(self.parameters())
        return total
    
    def __repr__(self) -> str:
        """String representation of model."""
        return (
            f"HRM(\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  H_layers={self.config.H_layers},\n"
            f"  L_layers={self.config.L_layers},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  num_parameters={self.num_parameters:,}\n"
            f")"
        )