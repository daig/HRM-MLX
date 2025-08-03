# MLX Model Integration Implementation Plan

## Overview
This document provides the detailed implementation plan for integrating all MLX HRM components into a complete, production-ready model. This corresponds to Phase 5 of the master implementation plan.

### Goal
Create a unified HRM model with clean APIs that integrates all previously implemented components, provides easy-to-use interfaces, and supports both training and inference workflows.

### Dependencies
- All Phase 1-4 components completed:
  - Custom initialization
  - RMSNorm
  - SwiGLU activation
  - Sparse embeddings
  - RoPE
  - Multi-head attention
  - ACT mechanism

## Architecture Overview

### Component Integration Hierarchy
```
HRM (User-facing API)
├── HRM_ACT (ACT wrapper)
│   └── HRMInner (Core model)
│       ├── Embeddings (Sparse + Dense)
│       ├── RoPE (Position encoding)
│       ├── H-Module (Planning)
│       │   └── HRMBlock × H_layers × H_cycles
│       │       ├── Attention
│       │       └── SwiGLU
│       └── L-Module (Computation)
│           └── HRMBlock × L_layers × L_cycles
│               ├── Attention
│               └── SwiGLU
└── Configuration System
```

## Implementation Components

### 1. Complete Model Wrapper
**File**: `src/mlx_hrm/models/hrm_complete.py`

```python
class HRM(nn.Module):
    """
    Complete Hierarchical Reasoning Model for MLX.
    
    This is the main user-facing API that wraps the ACT model
    and provides convenient methods for training and inference.
    """
    
    def __init__(self, config: Union[HRMConfig, str, dict]):
        """
        Initialize HRM model.
        
        Args:
            config: HRMConfig object, string preset name, or dict
        """
        super().__init__()
        
        # Handle different config types
        if isinstance(config, str):
            config = get_preset_config(config)
        elif isinstance(config, dict):
            config = HRMConfig(**config)
        
        self.config = config
        self.model = HRM_ACT(config)
        
    def forward(self, batch: Dict[str, mx.array], 
                carry: Optional[HRMCarry] = None) -> Tuple[HRMCarry, Dict[str, mx.array]]:
        """Standard forward pass."""
        if carry is None:
            carry = self.initial_carry(batch['input_ids'].shape[0])
        return self.model(carry, batch)
    
    def generate(self, prompt: mx.array, max_length: int, 
                 temperature: float = 1.0) -> mx.array:
        """Generate text autoregressively."""
        # Implement generation logic
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Save weights, config, and metadata
        
    @classmethod
    def from_checkpoint(cls, path: str) -> 'HRM':
        """Load model from checkpoint."""
        # Load and instantiate model
```

### 2. Configuration System
**File**: `src/mlx_hrm/configs/model_presets.py`

```python
"""Predefined model configurations for common use cases."""

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
    rope_theta=10000.0
)

# 27M parameter model (matches paper)
HRM_SMALL = HRMConfig(
    # Data
    batch_size=768,
    seq_len=512,
    vocab_size=1000,
    puzzle_emb_ndim=128,
    num_puzzle_identifiers=64,
    
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
    rope_theta=10000.0
)

# 100M parameter model for scaling experiments
HRM_BASE = HRMConfig(
    # Data
    batch_size=512,
    seq_len=1024,
    vocab_size=1000,
    puzzle_emb_ndim=128,
    num_puzzle_identifiers=64,
    
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
    rope_theta=10000.0
)

PRESET_CONFIGS = {
    'tiny': HRM_TINY,
    'small': HRM_SMALL,
    'base': HRM_BASE
}

def get_preset_config(name: str) -> HRMConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]
```

### 3. Model Factory Functions
**File**: `src/mlx_hrm/models/factory.py`

```python
"""Factory functions for creating HRM models."""

def create_hrm(config: Union[str, dict, HRMConfig]) -> HRM:
    """
    Create an HRM model with the given configuration.
    
    Args:
        config: Preset name ('tiny', 'small', 'base'), dict, or HRMConfig
        
    Returns:
        Initialized HRM model
        
    Examples:
        >>> model = create_hrm('small')  # 27M parameter model
        >>> model = create_hrm({'hidden_size': 512, ...})  # Custom config
    """
    return HRM(config)

def create_hrm_from_checkpoint(path: str) -> HRM:
    """Load a pretrained HRM model from checkpoint."""
    return HRM.from_checkpoint(path)

def list_available_presets() -> List[str]:
    """List available model presets."""
    return list(PRESET_CONFIGS.keys())
```

### 4. Checkpoint Management
**File**: `src/mlx_hrm/utils/checkpoint.py`

```python
"""Checkpoint saving and loading utilities."""

import json
from pathlib import Path
import mlx.core as mx

def save_checkpoint(model: HRM, path: str, metadata: Optional[Dict] = None):
    """
    Save model checkpoint with weights and configuration.
    
    Checkpoint format:
    checkpoint_dir/
    ├── config.json
    ├── weights.npz
    └── metadata.json
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = model.config._asdict()
    with open(path / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save weights
    weights = model.state_dict()
    mx.save(str(path / 'weights.npz'), weights)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    metadata['model_version'] = '1.0'
    metadata['mlx_version'] = mx.__version__
    with open(path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def load_checkpoint(path: str) -> Tuple[Dict, HRMConfig, Dict]:
    """Load checkpoint components."""
    path = Path(path)
    
    # Load config
    with open(path / 'config.json', 'r') as f:
        config_dict = json.load(f)
    config = HRMConfig(**config_dict)
    
    # Load weights
    weights = mx.load(str(path / 'weights.npz'))
    
    # Load metadata
    with open(path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return weights, config, metadata
```

## Integration Testing Strategy

### 1. Component Integration Tests
**File**: `tests/integration/test_model_integration.py`

```python
def test_full_model_creation():
    """Test that full model can be created with all components."""
    # Test with each preset
    for preset in ['tiny', 'small', 'base']:
        model = create_hrm(preset)
        assert isinstance(model, HRM)
        
def test_forward_pass():
    """Test complete forward pass through integrated model."""
    model = create_hrm('tiny')
    batch = create_test_batch(batch_size=2, seq_len=32)
    
    carry = model.initial_carry(2)
    new_carry, outputs = model(batch, carry)
    
    assert 'logits' in outputs
    assert 'q_halt_logits' in outputs
    assert outputs['logits'].shape == (2, 32, model.config.vocab_size)

def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    # Create and save model
    model1 = create_hrm('tiny')
    save_checkpoint(model1, 'test_checkpoint')
    
    # Load model
    model2 = create_hrm_from_checkpoint('test_checkpoint')
    
    # Compare outputs
    batch = create_test_batch(2, 32)
    outputs1 = model1(batch)[1]
    outputs2 = model2(batch)[1]
    assert mx.allclose(outputs1['logits'], outputs2['logits'])
```

### 2. API Usability Tests
**File**: `tests/integration/test_api_usability.py`

```python
def test_simple_inference():
    """Test that inference API is simple to use."""
    model = create_hrm('small')
    
    # Single sequence
    tokens = mx.array([1, 2, 3, 4, 5])
    output = model.forward_single(tokens)
    assert output.shape == (5, model.config.vocab_size)

def test_generation_api():
    """Test text generation interface."""
    model = create_hrm('small')
    
    prompt = mx.array([1, 2, 3])
    generated = model.generate(prompt, max_length=10)
    assert generated.shape[0] <= 10
```

## Performance Considerations

### 1. Memory Optimization
- Lazy weight loading for large models
- Efficient state management in carry objects
- Option to offload sparse embeddings

### 2. Inference Optimization
- KV-cache for generation (future)
- Batch inference support
- Dynamic batching for ACT

### 3. Training Optimization
- Gradient checkpointing support
- Mixed precision by default
- Efficient data loading

## Implementation Timeline

### Day 1: Core Integration (4-6 hours)
- [ ] Create HRM wrapper class
- [ ] Implement forward pass
- [ ] Add configuration handling
- [ ] Basic checkpoint support

### Day 2: Configuration & Factory (3-4 hours)
- [ ] Define model presets
- [ ] Create factory functions
- [ ] Add configuration validation
- [ ] Implement preset loading

### Day 3: Testing & Polish (4-5 hours)
- [ ] Write integration tests
- [ ] Add generation methods
- [ ] Create usage examples
- [ ] Performance profiling

## Success Criteria

1. **Functional Completeness**
   - ✅ All components properly integrated
   - ✅ Clean, intuitive API
   - ✅ Checkpoint save/load working
   - ✅ Multiple model sizes supported

2. **Usability**
   - ✅ Simple to create models
   - ✅ Easy inference interface
   - ✅ Clear error messages
   - ✅ Good documentation

3. **Performance**
   - ✅ Efficient memory usage
   - ✅ Fast model creation
   - ✅ No performance regressions
   - ✅ Scales to large models

## Example Usage

### Basic Training
```python
# Create model
model = create_hrm('small')

# Training loop
optimizer = mlx.optimizers.Adam(learning_rate=1e-4)
for batch in dataloader:
    loss, grads = mx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
```

### Inference
```python
# Load pretrained model
model = create_hrm_from_checkpoint('path/to/checkpoint')

# Single prediction
tokens = tokenizer.encode("Solve this puzzle:")
output = model.forward_single(tokens)
prediction = mx.argmax(output[-1])

# Generation
generated = model.generate(tokens, max_length=100)
text = tokenizer.decode(generated)
```

### Custom Configuration
```python
# Create custom model
config = HRMConfig(
    hidden_size=384,
    num_heads=6,
    H_layers=3,
    L_layers=3,
    # ... other params
)
model = create_hrm(config)
```

## Next Steps

After completing model integration:
1. Move to Phase 6: Loss Functions & Metrics
2. Begin integration with training infrastructure
3. Prepare for validation against PyTorch

This plan ensures a smooth integration of all components with a focus on usability and maintainability.