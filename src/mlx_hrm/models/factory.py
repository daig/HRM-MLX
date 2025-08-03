"""
Factory functions for creating HRM models.

This module provides convenient functions for creating HRM models
with different configurations and from checkpoints.
"""

from typing import Union, Dict, List, Optional
from pathlib import Path
import mlx.core as mx
import mlx.optimizers

from ..modules.act import HRMConfig
from .hrm_complete import HRM
from ..configs.model_presets import PRESET_CONFIGS, get_preset_info


def create_hrm(config: Union[str, Dict, HRMConfig]) -> HRM:
    """
    Create an HRM model with the given configuration.
    
    Args:
        config: One of:
            - Preset name ('tiny', 'small', 'base', 'large')
            - Dictionary of configuration parameters
            - HRMConfig object
        
    Returns:
        Initialized HRM model
        
    Examples:
        >>> # Using a preset
        >>> model = create_hrm('small')  # 27M parameter model
        
        >>> # Using custom config dict
        >>> model = create_hrm({'hidden_size': 512, 'num_heads': 8, ...})
        
        >>> # Using HRMConfig object
        >>> config = HRMConfig(hidden_size=512, num_heads=8, ...)
        >>> model = create_hrm(config)
    """
    return HRM(config)


def create_hrm_from_checkpoint(
    path: str,
    config: Optional[Union[str, Dict, HRMConfig]] = None
) -> HRM:
    """
    Load a pretrained HRM model from checkpoint.
    
    Args:
        path: Path to checkpoint directory or weights file
        config: Optional configuration override. If None, will try to load
                config from checkpoint directory.
                
    Returns:
        HRM model loaded with pretrained weights
        
    Examples:
        >>> # Load from checkpoint directory (contains config.json)
        >>> model = create_hrm_from_checkpoint('checkpoints/hrm_small/')
        
        >>> # Load weights only with explicit config
        >>> model = create_hrm_from_checkpoint('weights.npz', config='small')
    """
    path = Path(path)
    
    # Determine config
    if config is None:
        # Try to load config from checkpoint directory
        config_path = path / 'config.json' if path.is_dir() else path.parent / 'config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = HRMConfig(**config_dict)
        else:
            raise ValueError(
                f"No config provided and no config.json found at {config_path}. "
                "Please provide a configuration."
            )
    
    # Create model
    model = create_hrm(config)
    
    # Load weights
    if path.is_dir():
        weights_path = path / 'weights.safetensors'
        if not weights_path.exists():
            weights_path = path / 'weights.npz'
    else:
        weights_path = path
    
    if weights_path.exists():
        model.load_weights(str(weights_path))
    else:
        raise FileNotFoundError(f"No weights found at {weights_path}")
    
    return model


def list_available_presets() -> List[str]:
    """
    List available model presets.
    
    Returns:
        List of preset names
        
    Example:
        >>> presets = list_available_presets()
        >>> print(presets)
        ['tiny', 'small', 'base', 'large']
    """
    return list(PRESET_CONFIGS.keys())


def get_model_info(name_or_config: Union[str, HRMConfig]) -> Dict:
    """
    Get information about a model configuration.
    
    Args:
        name_or_config: Preset name or HRMConfig object
        
    Returns:
        Dictionary with model information including estimated parameter count
        
    Example:
        >>> info = get_model_info('small')
        >>> print(f"{info['name']}: {info['estimated_params_millions']}M parameters")
        small: 27.1M parameters
    """
    if isinstance(name_or_config, str):
        return get_preset_info(name_or_config)
    else:
        # Calculate info for custom config
        config = name_or_config
        
        # Same parameter estimation as in model_presets.py
        dense_emb_params = config.vocab_size * config.hidden_size
        # Sparse embeddings now use hidden_size directly
        sparse_emb_params = config.num_puzzle_identifiers * config.hidden_size if config.puzzle_emb_ndim > 0 else 0
        
        # Calculate head_dim from hidden_size and num_heads
        head_dim = config.hidden_size // config.num_heads
        
        attention_params_per_layer = (
            config.hidden_size * head_dim * config.num_heads * 3 +
            config.hidden_size * config.hidden_size
        )
        
        ffn_params_per_layer = int(
            config.hidden_size * config.hidden_size * config.expansion * 2
        )
        
        total_layers = config.H_layers + config.L_layers
        
        total_params = (
            dense_emb_params +
            sparse_emb_params +
            (attention_params_per_layer + ffn_params_per_layer) * total_layers +
            config.hidden_size * config.vocab_size * 2
        )
        
        return {
            'name': 'custom',
            'hidden_size': config.hidden_size,
            'num_heads': config.num_heads,
            'H_layers': config.H_layers,
            'L_layers': config.L_layers,
            'vocab_size': config.vocab_size,
            'estimated_params': total_params,
            'estimated_params_millions': round(total_params / 1e6, 1)
        }


def create_model_for_training(
    config: Union[str, Dict, HRMConfig],
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1
) -> tuple[HRM, mlx.optimizers.Adam]:
    """
    Create a model and optimizer configured for training.
    
    Args:
        config: Model configuration
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay coefficient
        
    Returns:
        Tuple of (model, optimizer)
        
    Example:
        >>> model, optimizer = create_model_for_training('small')
        >>> # Ready for training loop
    """
    model = create_hrm(config)
    
    # Create optimizer
    # Note: In full implementation, we'd use sparse-aware optimizer
    # For now, using standard Adam
    optimizer = mlx.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    return model, optimizer