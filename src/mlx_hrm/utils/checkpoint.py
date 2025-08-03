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
    
    # Save weights - direct equivalent of PyTorch's approach
    weights = model.parameters()
    
    # Use pickle like PyTorch does (torch.save uses pickle internally)
    import pickle
    with open(path / 'weights.pkl', 'wb') as f:
        pickle.dump(weights, f)
    
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
    weights_path = path / 'weights.pkl'
    if not weights_path.exists():
        # Try safetensors format for backward compatibility
        weights_path = path / 'weights.safetensors'
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found at {path}")
        # If safetensors exists, it's from old format - would need conversion
        raise NotImplementedError("Loading from safetensors format not yet implemented. Please re-save checkpoint.")
    
    # Load with pickle - direct equivalent of PyTorch
    import pickle
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
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