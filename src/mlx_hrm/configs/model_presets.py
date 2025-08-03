"""
Predefined model configurations for common use cases.

This module provides preset configurations for different model sizes,
matching the configurations used in the original HRM paper.
"""

from ..modules.act import HRMConfig


# 7M parameter model for testing and development
HRM_TINY = HRMConfig(
    # Data
    batch_size=32,
    seq_len=256,
    vocab_size=1000,
    
    # Architecture  
    hidden_size=256,
    num_heads=8,
    H_layers=2,
    L_layers=2,
    H_cycles=1,
    L_cycles=1,
    
    # ACT
    halt_max_steps=8,
    halt_exploration_prob=0.1,
    
    # Training
    expansion=4.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    
    # Sparse embeddings
    puzzle_emb_ndim=64,
    num_puzzle_identifiers=32
)

# 27M parameter model (matches paper)
HRM_SMALL = HRMConfig(
    # Data
    batch_size=768,
    seq_len=512,
    vocab_size=1000,
    
    # Architecture
    hidden_size=512,
    num_heads=8,
    H_layers=4,
    L_layers=4,
    H_cycles=2,
    L_cycles=2,
    
    # ACT
    halt_max_steps=64,
    halt_exploration_prob=0.1,
    
    # Training
    expansion=4.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    
    # Sparse embeddings
    puzzle_emb_ndim=128,
    num_puzzle_identifiers=64
)

# 100M parameter model for scaling experiments
HRM_BASE = HRMConfig(
    # Data
    batch_size=512,
    seq_len=1024,
    vocab_size=1000,
    
    # Architecture
    hidden_size=768,
    num_heads=12,
    H_layers=8,
    L_layers=8,
    H_cycles=2,
    L_cycles=2,
    
    # ACT
    halt_max_steps=128,
    halt_exploration_prob=0.1,
    
    # Training
    expansion=4.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    
    # Sparse embeddings
    puzzle_emb_ndim=128,
    num_puzzle_identifiers=64
)

# 200M parameter model
HRM_LARGE = HRMConfig(
    # Data
    batch_size=256,
    seq_len=2048,
    vocab_size=1000,
    
    # Architecture
    hidden_size=1024,
    num_heads=16,
    H_layers=12,
    L_layers=12,
    H_cycles=2,
    L_cycles=2,
    
    # ACT
    halt_max_steps=256,
    halt_exploration_prob=0.1,
    
    # Training
    expansion=4.0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    
    # Sparse embeddings
    puzzle_emb_ndim=256,
    num_puzzle_identifiers=128
)


PRESET_CONFIGS = {
    'tiny': HRM_TINY,
    'small': HRM_SMALL,
    'base': HRM_BASE,
    'large': HRM_LARGE
}


def get_preset_config(name: str) -> HRMConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name ('tiny', 'small', 'base', 'large')
        
    Returns:
        HRMConfig for the requested preset
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )
    return PRESET_CONFIGS[name]


def list_presets() -> list[str]:
    """
    List available preset names.
    
    Returns:
        List of preset configuration names
    """
    return list(PRESET_CONFIGS.keys())


def get_preset_info(name: str) -> dict:
    """
    Get information about a preset configuration.
    
    Args:
        name: Preset name
        
    Returns:
        Dictionary with preset information
    """
    config = get_preset_config(name)
    
    # Estimate parameter count
    # Dense embeddings: vocab_size * hidden_size
    # Sparse embeddings: num_puzzle_identifiers * puzzle_emb_ndim * hidden_size
    # Attention: (hidden_size * head_dim * num_heads * 3 + hidden_size * hidden_size) per layer
    # FFN: hidden_size * hidden_size * expansion * 2 per layer
    # Output: hidden_size * vocab_size
    
    dense_emb_params = config.vocab_size * config.hidden_size
    # Sparse embeddings now use hidden_size directly
    sparse_emb_params = config.num_puzzle_identifiers * config.hidden_size if config.puzzle_emb_ndim > 0 else 0
    
    # Calculate head_dim from hidden_size and num_heads
    head_dim = config.hidden_size // config.num_heads
    
    attention_params_per_layer = (
        config.hidden_size * head_dim * config.num_heads * 3 +  # QKV
        config.hidden_size * config.hidden_size  # O projection
    )
    
    ffn_params_per_layer = int(
        config.hidden_size * config.hidden_size * config.expansion * 2
    )
    
    total_layers = config.H_layers + config.L_layers
    
    total_params = (
        dense_emb_params +
        sparse_emb_params +
        (attention_params_per_layer + ffn_params_per_layer) * total_layers +
        config.hidden_size * config.vocab_size * 2  # LM head + Q head
    )
    
    return {
        'name': name,
        'hidden_size': config.hidden_size,
        'num_heads': config.num_heads,
        'H_layers': config.H_layers,
        'L_layers': config.L_layers,
        'vocab_size': config.vocab_size,
        'estimated_params': total_params,
        'estimated_params_millions': round(total_params / 1e6, 1)
    }