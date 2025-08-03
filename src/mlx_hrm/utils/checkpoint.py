"""
Checkpoint saving and loading utilities for HRM models.

This module provides functions for saving and loading model checkpoints
with support for configuration, weights, and metadata.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

from ..modules.act import HRMConfig


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[HRMConfig] = None
):
    """
    Save model checkpoint with weights and configuration.
    
    The checkpoint is saved as a directory with the following structure:
    checkpoint_dir/
    ├── config.json      # Model configuration
    ├── weights.npz      # Model weights
    └── metadata.json    # Training metadata and version info
    
    Args:
        model: Model to save
        path: Path to checkpoint directory
        metadata: Optional metadata to save (e.g., training step, loss)
        config: Optional model configuration (will try to get from model.config)
        
    Example:
        >>> save_checkpoint(
        ...     model, 
        ...     'checkpoints/step_1000',
        ...     metadata={'step': 1000, 'loss': 0.234}
        ... )
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    if config is None and hasattr(model, 'config'):
        config = model.config
    
    if config is not None:
        # Save configuration
        config_dict = config._asdict() if hasattr(config, '_asdict') else vars(config)
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Save weights
    weights = dict(model.parameters())
    
    # Flatten nested parameter dictionaries
    flat_weights = {}
    
    def flatten_dict(d: Dict, prefix: str = ''):
        """Recursively flatten nested dictionary."""
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            elif isinstance(v, list):
                # Handle lists (e.g., blocks)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        flatten_dict(item, f"{key}.{i}")
                    else:
                        flat_weights[f"{key}.{i}"] = item
            elif isinstance(v, mx.array):
                flat_weights[key] = v
            # Skip non-array values
    
    flatten_dict(weights)
    # Save using MLX's safetensors format which supports dictionaries
    mx.save_safetensors(str(path / 'weights.safetensors'), flat_weights)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    # Add default metadata
    metadata.setdefault('timestamp', datetime.now().isoformat())
    metadata.setdefault('mlx_version', mx.__version__)
    metadata.setdefault('model_class', model.__class__.__name__)
    
    if hasattr(model, 'num_parameters'):
        metadata['num_parameters'] = model.num_parameters
    
    with open(path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Tuple[Dict[str, mx.array], HRMConfig, Dict[str, Any]]:
    """
    Load checkpoint components.
    
    Args:
        path: Path to checkpoint directory
        
    Returns:
        Tuple of (weights, config, metadata)
        
    Example:
        >>> weights, config, metadata = load_checkpoint('checkpoints/step_1000')
        >>> model = create_hrm(config)
        >>> model.load_weights(weights)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    # Handle both directory and single file
    if path.is_file():
        # Assume it's just weights
        weights = mx.load(str(path))
        return weights, None, {}
    
    # Load config
    config = None
    config_path = path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = HRMConfig(**config_dict)
    
    # Load weights
    weights_path = path / 'weights.safetensors'
    if not weights_path.exists():
        # Try old numpy format for backward compatibility
        weights_path = path / 'weights.npz'
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found at {path}")
    
    # Load with MLX
    if weights_path.suffix == '.safetensors':
        flat_weights = mx.load_safetensors(str(weights_path))
    else:
        # Fallback to npz format
        flat_weights = mx.load(str(weights_path))
    
    # Unflatten weights to nested structure
    weights = {}
    for key, value in flat_weights.items():
        parts = key.split('.')
        current = weights
        for i, part in enumerate(parts[:-1]):
            # Check if this part is a number (array index)
            if part.isdigit():
                # Convert parent dict to list if needed
                parent_key = parts[i-1] if i > 0 else None
                if parent_key and not isinstance(current, list):
                    # We need to convert the parent to a list
                    parent = weights
                    for p in parts[:i-1]:
                        parent = parent[p]
                    parent[parts[i-1]] = []
                    current = parent[parts[i-1]]
                
                # Extend list if needed
                idx = int(part)
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the value
        final_key = parts[-1]
        if final_key.isdigit() and isinstance(current, list):
            idx = int(final_key)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[final_key] = value
    
    # Load metadata
    metadata = {}
    metadata_path = path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return weights, config, metadata


def save_training_checkpoint(
    model: nn.Module,
    optimizer: mlx.optimizers.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str,
    keep_last_n: int = 5
):
    """
    Save a training checkpoint with automatic naming and cleanup.
    
    Args:
        model: Model to save
        optimizer: Optimizer with training state
        step: Current training step
        loss: Current loss value
        checkpoint_dir: Base directory for checkpoints
        keep_last_n: Number of recent checkpoints to keep
        
    Example:
        >>> save_training_checkpoint(
        ...     model, optimizer, step=1000, loss=0.234,
        ...     checkpoint_dir='checkpoints/run1'
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint name
    checkpoint_name = f"step_{step:08d}"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Save checkpoint
    metadata = {
        'step': step,
        'loss': loss,
        'optimizer_state': 'included'  # In full implementation, would save optimizer state
    }
    
    save_checkpoint(model, str(checkpoint_path), metadata)
    
    # Clean up old checkpoints
    if keep_last_n > 0:
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('step_')],
            key=lambda x: x.name
        )
        
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                # Remove old checkpoint
                for file in old_checkpoint.iterdir():
                    file.unlink()
                old_checkpoint.rmdir()
                print(f"Removed old checkpoint: {old_checkpoint.name}")


def convert_state_dict(
    state_dict: Dict[str, Any],
    source_format: str = 'pytorch',
    target_format: str = 'mlx'
) -> Dict[str, mx.array]:
    """
    Convert state dict between different formats.
    
    Args:
        state_dict: Original state dictionary
        source_format: Source format ('pytorch', 'mlx')
        target_format: Target format ('pytorch', 'mlx')
        
    Returns:
        Converted state dictionary
        
    Note:
        This is a placeholder for the full implementation that would
        handle PyTorch <-> MLX conversion with proper parameter mapping.
    """
    if source_format == target_format:
        return state_dict
    
    if source_format == 'pytorch' and target_format == 'mlx':
        # Convert PyTorch tensors to MLX arrays
        # In full implementation, would also handle parameter name mapping
        converted = {}
        for key, value in state_dict.items():
            if hasattr(value, 'detach'):  # PyTorch tensor
                # Convert via numpy as intermediate format
                import numpy as np
                np_array = value.detach().cpu().numpy()
                converted[key] = mx.array(np_array)
            else:
                converted[key] = value
        return converted
    
    raise NotImplementedError(f"Conversion from {source_format} to {target_format} not implemented")


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading weights.
    
    Args:
        path: Path to checkpoint
        
    Returns:
        Dictionary with checkpoint information
        
    Example:
        >>> info = get_checkpoint_info('checkpoints/step_1000')
        >>> print(f"Step: {info['step']}, Loss: {info['loss']}")
    """
    path = Path(path)
    
    info = {'path': str(path)}
    
    # Load config
    config_path = path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        info['config'] = config_dict
        info['model_size'] = config_dict.get('hidden_size', 'unknown')
    
    # Load metadata
    metadata_path = path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        info.update(metadata)
    
    # Check weights file
    weights_path = path / 'weights.safetensors'
    if not weights_path.exists():
        weights_path = path / 'weights.npz'
    if weights_path.exists():
        info['weights_size_mb'] = weights_path.stat().st_size / 1024 / 1024
    
    return info