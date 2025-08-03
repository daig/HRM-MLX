# MLX HRM Final Phases Implementation Plan

## Overview
This document provides a detailed implementation plan for the final phases of the MLX HRM port. With core components complete (Phases 1-4), we now focus on integration, training, and validation.

### Completed Components:
- ✅ Custom initialization (truncated normal)
- ✅ RMSNorm layer 
- ✅ SwiGLU activation
- ✅ Sparse embeddings with SignSGD
- ✅ Rotary Position Embeddings (RoPE)
- ✅ Multi-Head Attention with GQA
- ✅ ACT mechanism with Q-learning

### Remaining Work:
- Phase 5: Model Integration
- Phase 6: Loss Functions & Metrics
- Phase 7: Training Infrastructure  
- Phase 8: Validation & Testing
- Phase 9: Documentation & Examples

## Phase 5: Model Integration (2-3 days)

### Goal
Integrate all components into a complete, usable HRM model with clean APIs.

### 5.1 Complete Model Assembly
**File**: `src/mlx_hrm/models/hrm_complete.py`

```python
class HRM:
    """Complete HRM model with all components integrated."""
    def __init__(self, config: HRMConfig):
        # Integrate HRM_ACT as the main model
        # Add methods for inference, checkpoint loading
        # Provide clean API for users
```

**Tasks**:
- [ ] Create unified model interface
- [ ] Add model factory functions for different sizes
- [ ] Implement checkpoint loading/saving
- [ ] Add inference-specific methods
- [ ] Create model configuration presets

### 5.2 Configuration System
**File**: `src/mlx_hrm/configs/model_configs.py`

```python
# Predefined configurations
HRM_TINY = HRMConfig(...)    # 7M params for testing
HRM_SMALL = HRMConfig(...)   # 27M params (main model)
HRM_BASE = HRMConfig(...)    # 100M params
```

**Tasks**:
- [ ] Define standard model configurations
- [ ] Create configuration validation
- [ ] Add configuration serialization
- [ ] Support loading from YAML/JSON

### 5.3 Integration Tests
**File**: `tests/integration/test_full_model.py`

**Tasks**:
- [ ] Test full model forward pass
- [ ] Test model with different configurations
- [ ] Test checkpoint save/load
- [ ] Test inference mode vs training mode
- [ ] Memory usage profiling

## Phase 6: Loss Functions & Metrics (1-2 days)

### Goal
Implement HRM-specific loss functions and training metrics.

### 6.1 Stablemax Loss
**File**: `src/mlx_hrm/training/losses.py`

```python
def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    """HRM's novel stablemax loss function."""
    # Port from PyTorch implementation
    # Ensure numerical stability in MLX
```

**Tasks**:
- [ ] Implement stablemax activation
- [ ] Create stablemax cross-entropy loss
- [ ] Add standard cross-entropy alternative
- [ ] Test numerical stability
- [ ] Benchmark vs softmax

### 6.2 ACT Loss Components
**File**: `src/mlx_hrm/training/act_loss.py`

```python
class ACTLossHead:
    """Combined loss for language modeling + Q-learning."""
    def __init__(self, model, loss_type='stablemax'):
        # Wrap model with loss computation
        # Track metrics during training
```

**Tasks**:
- [ ] Implement language modeling loss
- [ ] Add Q-halt loss (binary cross-entropy)
- [ ] Add Q-continue loss (TD learning)
- [ ] Combine losses with proper weighting
- [ ] Track accuracy metrics

### 6.3 Training Metrics
**File**: `src/mlx_hrm/training/metrics.py`

**Tasks**:
- [ ] Token-level accuracy
- [ ] Sequence-level exact match
- [ ] Average ACT steps used
- [ ] Q-value accuracy
- [ ] Loss component tracking

## Phase 7: Training Infrastructure (3-4 days)

### Goal
Build complete training pipeline with MLX-specific optimizations.

### 7.1 Optimizer Implementation
**File**: `src/mlx_hrm/training/optimizers.py`

```python
class AdamAtan2:
    """Adam-atan2 optimizer for sparse embeddings."""
    # Port from PyTorch implementation
    # Integrate with MLX optimizer interface
```

**Tasks**:
- [ ] Port Adam-atan2 optimizer
- [ ] Integrate with MLX's optimizer framework
- [ ] Support parameter groups
- [ ] Add gradient clipping
- [ ] Test convergence properties

### 7.2 Training Loop
**File**: `src/mlx_hrm/training/trainer.py`

```python
class HRMTrainer:
    """Training loop for HRM with MLX."""
    def __init__(self, model, config):
        # Setup optimizers, loss functions
        # Handle data loading
        # Manage checkpointing
```

**Tasks**:
- [ ] Implement training step function
- [ ] Add validation loop
- [ ] Gradient accumulation support
- [ ] Mixed precision training
- [ ] Checkpoint management
- [ ] Early stopping logic

### 7.3 Data Pipeline
**File**: `src/mlx_hrm/data/dataset.py`

**Tasks**:
- [ ] Create dataset loader for puzzles
- [ ] Implement data preprocessing
- [ ] Add data augmentation support
- [ ] Efficient batching for MLX
- [ ] Support for different puzzle types

## Phase 8: Validation & Testing (2-3 days)

### Goal
Ensure MLX implementation matches PyTorch performance and behavior.

### 8.1 Checkpoint Conversion
**File**: `scripts/convert_checkpoint.py`

```python
def convert_pytorch_to_mlx(pytorch_path, output_path):
    """Convert PyTorch HRM checkpoint to MLX format."""
    # Load PyTorch state dict
    # Map parameter names
    # Convert tensor formats
    # Save in MLX format
```

**Tasks**:
- [ ] Implement checkpoint converter
- [ ] Handle parameter name mapping
- [ ] Convert optimizer states
- [ ] Validate converted weights
- [ ] Support both directions (PyTorch ↔ MLX)

### 8.2 Numerical Validation
**File**: `tests/validation/test_numerical_parity.py`

**Tasks**:
- [ ] Compare forward pass outputs
- [ ] Validate gradient computations
- [ ] Check loss values match
- [ ] Test on multiple batch sizes
- [ ] Verify ACT behavior matches

### 8.3 Performance Benchmarks
**File**: `benchmarks/benchmark_full_model.py`

**Tasks**:
- [ ] Benchmark training throughput
- [ ] Measure inference speed
- [ ] Profile memory usage
- [ ] Compare with PyTorch baseline
- [ ] Test on different Apple Silicon chips

### 8.4 Accuracy Validation
**File**: `tests/validation/test_puzzle_accuracy.py`

**Tasks**:
- [ ] Test on ARC-1 dataset
- [ ] Validate on Sudoku puzzles
- [ ] Check maze solving accuracy
- [ ] Compare with published results
- [ ] Statistical significance testing

## Phase 9: Documentation & Examples (1-2 days)

### Goal
Create comprehensive documentation and usage examples.

### 9.1 API Documentation
**Tasks**:
- [ ] Document all public APIs
- [ ] Add docstrings to all modules
- [ ] Create API reference guide
- [ ] Add type hints throughout

### 9.2 Usage Examples
**Files**: `examples/`

```python
# examples/train_arc.py
# examples/inference_demo.py
# examples/custom_dataset.py
```

**Tasks**:
- [ ] Basic training example
- [ ] Inference example
- [ ] Custom dataset integration
- [ ] Fine-tuning example
- [ ] Checkpoint manipulation

### 9.3 Tutorials
**Files**: `docs/tutorials/`

**Tasks**:
- [ ] Getting started guide
- [ ] Understanding ACT mechanism
- [ ] Custom puzzle integration
- [ ] Performance optimization tips
- [ ] Debugging guide

## Implementation Priority & Dependencies

### Critical Path:
1. **Model Integration** → Required for everything else
2. **Loss Functions** → Required for training
3. **Checkpoint Conversion** → Enables validation
4. **Training Loop** → Core functionality
5. **Documentation** → User adoption

### Parallel Work:
- Optimizers can be developed alongside model integration
- Benchmarks can start once model is integrated
- Examples can be created incrementally

## Success Criteria

### Phase 5: Model Integration
- ✅ Full model runs without errors
- ✅ All components properly connected
- ✅ Clean, intuitive API
- ✅ Memory efficient operation

### Phase 6: Loss Functions
- ✅ Losses numerically match PyTorch
- ✅ Stable training dynamics
- ✅ Metrics properly tracked
- ✅ Efficient computation

### Phase 7: Training Infrastructure
- ✅ Model trains successfully
- ✅ Checkpointing works reliably
- ✅ Good training throughput
- ✅ Convergence matches PyTorch

### Phase 8: Validation
- ✅ Numerical outputs within tolerance
- ✅ Accuracy matches published results
- ✅ Performance meets targets
- ✅ All tests passing

### Phase 9: Documentation
- ✅ Complete API documentation
- ✅ Working examples for all use cases
- ✅ Clear installation instructions
- ✅ Troubleshooting guide

## Risk Mitigation

### Technical Risks:
1. **Numerical Differences**: 
   - Mitigation: Extensive validation tests
   - Fallback: Adjustable tolerance levels

2. **Performance Issues**:
   - Mitigation: Early benchmarking
   - Fallback: Optimization phase if needed

3. **API Incompatibilities**:
   - Mitigation: Wrapper layers
   - Fallback: Custom implementations

### Schedule Risks:
1. **Integration Complexity**:
   - Buffer: +1 day for integration
   - Early testing of component interactions

2. **Training Instability**:
   - Buffer: +2 days for debugging
   - Multiple optimizer options

## Estimated Timeline

### Total: 10-15 days

**Week 1:**
- Days 1-3: Model Integration (Phase 5)
- Days 3-4: Loss Functions (Phase 6)
- Days 5-7: Start Training Infrastructure (Phase 7)

**Week 2:**
- Days 8-10: Complete Training Infrastructure
- Days 11-12: Validation & Testing (Phase 8)
- Days 13-14: Documentation & Examples (Phase 9)
- Day 15: Buffer for issues

## Next Steps

1. **Immediate Priority**: Start Phase 5 model integration
2. **Prepare**: Set up test datasets for validation
3. **Plan**: Schedule time for thorough testing
4. **Document**: Keep notes on any deviations from PyTorch

This plan provides a clear roadmap to complete the MLX HRM implementation with proper validation and documentation.