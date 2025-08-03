# MLX HRM Master Implementation Plan

## Progress Status: Phase 5 Complete (Model Integration) 
**Last Updated**: 2025-08-03

### Phase Completion:
- ✅ **Phase 1**: Foundation Components (Initialization + RMSNorm) - COMPLETE
- ✅ **Phase 2**: Core Components - COMPLETE
  - ✅ SwiGLU Activation
  - ✅ Sparse Embeddings
  - ✅ Rotary Position Embeddings (RoPE)
- ✅ **Phase 3**: Complex Modules - COMPLETE
  - ✅ Multi-Head Attention with GQA support
- ✅ **Phase 4**: Advanced Features - COMPLETE
  - ✅ Adaptive Computation Time (ACT) mechanism
- ✅ **Phase 5**: Model Integration - COMPLETE
  - ✅ Complete HRM wrapper
  - ✅ Configuration system with presets
  - ✅ Factory functions
  - ✅ Checkpoint management
- ⏹️ **Phases 6-9**: Pending

## Executive Summary

This master plan provides a comprehensive roadmap for implementing the Hierarchical Reasoning Model (HRM) in MLX from scratch. It follows a bottom-up approach, starting with foundational components and building up to the complete model. Each phase references specific implementation plans and ensures proper testing and validation at every step.

### Current Status
- ✅ **Phase 1**: Foundation Components - COMPLETE
  - ✅ Custom Weight Initialization
  - ✅ RMSNorm Layer
- ✅ **Phase 2**: Core Components - COMPLETE
  - ✅ SwiGLU Activation
  - ✅ Sparse Embeddings
  - ✅ Rotary Position Embeddings
- ✅ **Phase 3**: Complex Modules - COMPLETE
  - ✅ Multi-Head Attention with GQA
- ✅ **Phase 4**: Advanced Features - COMPLETE
  - ✅ Adaptive Computation Time (ACT)
- ✅ **Phase 5**: Model Integration - COMPLETE
  - ✅ Complete model assembly
  - ✅ Configuration system
  - ✅ Factory functions
  - ✅ Checkpoint management
- ⏳ **Phase 6**: Loss Functions & Metrics - READY TO START
  - ⏹️ Stablemax loss (next)
  - ⏹️ ACT loss components
- ⏹️ **Phases 7-9**: Detailed plans created

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Implementation Phases](#implementation-phases)
4. [Testing Strategy](#testing-strategy)
5. [Integration and Validation](#integration-and-validation)
6. [Project Timeline](#project-timeline)

## Project Overview

### Goals
- Port HRM from PyTorch to MLX with full feature parity
- Optimize for Apple Silicon performance
- Maintain numerical accuracy and training stability
- Create modular, well-tested components
- Enable easy experimentation and extension

### Key Principles
1. **Bottom-up Development**: Start with independent components, build up to complex systems
2. **Test-Driven**: Each component must have comprehensive tests before integration
3. **Modular Design**: Components should be independently usable
4. **Performance Focus**: Leverage MLX optimizations at every level
5. **Clear Documentation**: Each module should be self-documenting

### Reference Documents
- **Requirements**: `MLX_PORTING_REQUIREMENTS.md` - Overall architecture and requirements
- **Component Plans**: Individual implementation plans for each module
- **Original HRM**: Located in sibling directory `../HRM/` - PyTorch reference implementation

## Repository Structure

```
MLX/  # Located at ~/Documents/MLX/
├── README.md
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── src/
│   └── mlx_hrm/
│       ├── __init__.py
│       ├── layers/             # Phase 1-2 components
│       │   ├── __init__.py
│       │   ├── initialization.py
│       │   ├── normalization.py
│       │   ├── activations.py
│       │   └── embeddings.py
│       ├── modules/            # Phase 3-4 components
│       │   ├── __init__.py
│       │   ├── rope.py
│       │   ├── attention.py
│       │   └── act.py
│       ├── models/             # Phase 5 integration
│       │   ├── __init__.py
│       │   ├── hrm_block.py
│       │   └── hrm_model.py
│       ├── training/           # Phase 6 training
│       │   ├── __init__.py
│       │   ├── optimizer.py
│       │   ├── trainer.py
│       │   └── utils.py
│       └── configs/            # Configuration files
│           ├── __init__.py
│           └── model_configs.py
├── tests/                      # Comprehensive test suite
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/                   # Usage examples
│   ├── inference.py
│   └── training.py
└── scripts/                    # Utility scripts
    ├── convert_checkpoint.py
    └── benchmark.py
```

## Implementation Phases

### Phase 1: Foundation Components (Week 1) ✅ COMPLETED

**Goal**: Implement basic building blocks with no dependencies

#### 1.1 Custom Weight Initialization ✅
- **Plan**: `MLX_CUSTOM_INITIALIZATION_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/initialization.py`
- **Key Tasks**:
  - [x] Implement truncated normal initialization
  - [x] Match JAX/PyTorch initialization behavior
  - [x] Create comprehensive unit tests
  - [x] Benchmark against PyTorch reference

**Completed Details**:
- Exact port of PyTorch HRM's truncated normal with variance correction
- Implemented `LinearTruncNormal` and `EmbeddingTruncNormal` layers
- 21 comprehensive unit tests covering all edge cases
- Documented PyTorch's std inflation quirk (29% for default bounds)
- Verified exact behavioral match including clipping at `bounds * comp_std`

#### 1.2 RMSNorm Layer ✅
- **Plan**: `MLX_RMSNORM_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/normalization.py`
- **Key Tasks**:
  - [x] Implement RMSNorm with MLX backend
  - [x] Ensure numerical stability in mixed precision
  - [x] Test shape preservation and gradient flow
  - [x] Performance benchmarking

**Completed Details**:
- Implemented both functional and module-based RMSNorm variants
- Functional version exactly matches PyTorch HRM behavior (no learnable params)
- Module version supports both with/without learnable scale parameters
- 22 comprehensive unit tests including edge cases and gradient flow
- Benchmarked performance: MLX built-in is ~2x faster than functional
- Mixed precision support with internal float32 computation for stability

### Phase 2: Core Components (Week 1-2)

**Goal**: Build essential layers that depend only on Phase 1

#### 2.1 SwiGLU Activation ✅
- **Plan**: `MLX_SWIGLU_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/activations.py`
- **Dependencies**: Custom initialization
- **Key Tasks**:
  - [x] Implement SwiGLU with fused projections
  - [x] Optimize intermediate dimension calculation
  - [x] Test activation behavior and gradients
  - [x] Benchmark memory usage and speed

**Completed Details**:
- Exact port of PyTorch HRM's SwiGLU with fused gate/up projection
- Proper intermediate dimension calculation with 2/3 factor and rounding
- Uses custom LinearTruncNormal for exact initialization match
- 22 comprehensive unit tests covering all functionality
- Verified parameter efficiency (~10-15% overhead vs standard FFN)
- Stable gradient flow across different input scales

#### 2.2 Sparse Embeddings ✅
- **Plan**: `MLX_SPARSE_EMBEDDINGS_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/embeddings.py`
- **Dependencies**: Custom initialization
- **Key Tasks**:
  - [x] Implement sparse embedding layer
  - [x] Port SignSGD optimizer
  - [x] Test vocabulary scaling
  - [x] Memory optimization for large vocabularies

**Completed Details**:
- Exact port of PyTorch HRM's CastedSparseEmbedding with local workspace
- SignSGD optimizer with proper sign-based updates and weight decay
- Support for mixed precision training (cast_to parameter)
- 15 comprehensive unit tests covering all functionality
- Memory-efficient design - only loads embeddings used in current batch
- Compatible with MLX's immutable array semantics

#### 2.3 Rotary Position Embeddings ✅
- **Plan**: `MLX_ROPE_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/modules/rope.py`
- **Dependencies**: None (standalone)
- **Key Tasks**:
  - [x] Implement RoPE with caching
  - [x] Support length extrapolation
  - [x] Test position encoding properties
  - [x] Optimize for different sequence lengths

**Completed Details**:
- Exact port of PyTorch HRM's RotaryEmbedding with precomputed cos/sin cache
- Implemented rotate_half and apply_rotary_pos_emb functions
- Support for dynamic cache extension and dtype casting
- 21 unit tests + 10 mathematical property tests
- Performance: 2.4+ GFLOPS throughput, 2x speedup with float16
- Memory efficient caching with lazy evaluation

### Phase 3: Complex Modules (Week 2-3)

**Goal**: Implement modules that integrate Phase 1-2 components

#### 3.1 Multi-Head Attention ✅
- **Plan**: `MLX_ATTENTION_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/modules/attention.py`
- **Dependencies**: RoPE, RMSNorm, Custom initialization
- **Key Tasks**:
  - [x] Implement attention with MLX scaled_dot_product_attention
  - [x] Support both MHA and GQA
  - [x] Integrate RoPE for position encoding
  - [x] Test attention patterns and gradients
  - [x] Benchmark against FlashAttention

**Completed Details**:
- Exact port of PyTorch HRM's Attention with fused QKV projection
- Full support for Grouped Query Attention (GQA) with KV head repetition
- Integrated with existing RoPE implementation
- 12 comprehensive unit tests covering all functionality
- Performance benchmarks show excellent scaling:
  - GQA provides expected speedups (4:1 = ~40% faster)
  - Causal masking has minimal overhead
  - Float16/bfloat16 support with stable numerics

### Phase 4: Advanced Features (Week 3-4)

**Goal**: Implement the hierarchical reasoning mechanisms

#### 4.1 Adaptive Computation Time (ACT) ✅
- **Plan**: `MLX_ACT_MECHANISM_IMPLEMENTATION_PLAN.md`
- **Modules**: 
  - `src/mlx_hrm/modules/act.py` - Carry states and base components
  - `src/mlx_hrm/models/hrm_inner.py` - Inner HRM model
  - `src/mlx_hrm/models/hrm_act.py` - ACT wrapper
- **Dependencies**: All previous components
- **Key Tasks**:
  - [x] Implement ACT controller
  - [x] Build high-level and low-level modules
  - [x] Create differentiable halting mechanism
  - [x] Test adaptive computation behavior
  - [x] Optimize for dynamic computation graphs

**Completed Details**:
- Implemented complete ACT mechanism with Q-learning based halting
- Created hierarchical H-level (planning) and L-level (computation) modules
- Carry state management for maintaining context between ACT steps
- Dynamic halting with exploration during training
- 14 comprehensive unit tests covering all functionality
- Full integration with existing components (Attention, RoPE, SwiGLU)

### Phase 5: Model Integration (Week 4-5) ✅ COMPLETED

**Goal**: Assemble all components into the complete HRM model

#### 5.1 Complete Model Assembly ✅
- **Plan**: `MLX_MODEL_INTEGRATION_PLAN.md`
- **Module**: `src/mlx_hrm/models/hrm_complete.py`
- **Dependencies**: All Phase 1-4 components
- **Key Tasks**:
  - [x] Create unified HRM wrapper class
  - [x] Integrate HRM_ACT as main model
  - [x] Add model factory functions
  - [x] Implement checkpoint loading/saving
  - [x] Create inference-specific methods

**Completed Details**:
- Created HRM wrapper class with clean, user-friendly API
- Integrated all components (ACT, attention, sparse embeddings, etc.)
- Implemented generation with temperature and top-k sampling
- Added convenient methods like forward_single()
- Created factory functions for easy model creation
- Full checkpoint save/load functionality with pickle format

#### 5.2 Configuration System ✅
- **Plan**: `MLX_MODEL_INTEGRATION_PLAN.md`
- **Module**: `src/mlx_hrm/configs/model_presets.py`
- **Dependencies**: HRMConfig from ACT module
- **Key Tasks**:
  - [x] Define model presets (tiny, small, base, large)
  - [x] Create configuration validation
  - [x] Support dict/string/HRMConfig inputs
  - [x] Add preset management functions
  - [x] Test configuration system

**Completed Details**:
- Created 4 model presets: tiny (7M), small (27M), base (100M), large (200M)
- Factory functions support string presets, dicts, and HRMConfig objects
- Added parameter counting and model info utilities
- Integration tests verify all configuration methods
- Created comprehensive usage examples in examples/basic_usage.py

### Phase 6: Loss Functions & Metrics (Week 5)

**Goal**: Implement HRM-specific loss functions and training metrics

#### 6.1 Stablemax Loss
- **Plan**: `MLX_LOSS_FUNCTIONS_PLAN.md`
- **Module**: `src/mlx_hrm/training/losses.py`
- **Dependencies**: None (standalone)
- **Key Tasks**:
  - [ ] Implement S-function for stablemax
  - [ ] Create stablemax cross-entropy loss
  - [ ] Add softmax baseline for comparison
  - [ ] Test numerical stability
  - [ ] Benchmark performance

#### 6.2 ACT Loss Components
- **Plan**: `MLX_LOSS_FUNCTIONS_PLAN.md`
- **Module**: `src/mlx_hrm/training/act_loss.py`
- **Dependencies**: Loss functions, HRM model
- **Key Tasks**:
  - [ ] Implement ACTLossHead wrapper
  - [ ] Add language modeling loss
  - [ ] Create Q-halt loss (binary cross-entropy)
  - [ ] Implement Q-continue loss (TD learning)
  - [ ] Track training metrics

### Phase 7: Training Infrastructure (Week 5-6)

**Goal**: Build complete training pipeline with MLX optimizations

#### 7.1 Adam-atan2 Optimizer
- **Plan**: `MLX_TRAINING_INFRASTRUCTURE_PLAN.md`
- **Module**: `src/mlx_hrm/training/optimizers.py`
- **Dependencies**: MLX optimizer framework
- **Key Tasks**:
  - [ ] Port Adam-atan2 for sparse embeddings
  - [ ] Create sparse-aware optimizer wrapper
  - [ ] Support parameter groups
  - [ ] Add gradient clipping
  - [ ] Test convergence properties

#### 7.2 Training Loop
- **Plan**: `MLX_TRAINING_INFRASTRUCTURE_PLAN.md`
- **Module**: `src/mlx_hrm/training/trainer.py`
- **Dependencies**: Model, optimizers, losses
- **Key Tasks**:
  - [ ] Implement HRMTrainer class
  - [ ] Add gradient accumulation
  - [ ] Create validation loop
  - [ ] Implement checkpointing
  - [ ] Add early stopping

#### 7.3 Data Pipeline
- **Plan**: `MLX_TRAINING_INFRASTRUCTURE_PLAN.md`
- **Module**: `src/mlx_hrm/data/dataset.py`
- **Dependencies**: None (standalone)
- **Key Tasks**:
  - [ ] Create PuzzleDataset class
  - [ ] Support ARC/Sudoku/Maze formats
  - [ ] Implement efficient DataLoader
  - [ ] Add data augmentation
  - [ ] Test loading performance

### Phase 8: Validation & Testing (Week 6)

**Goal**: Ensure MLX implementation matches PyTorch performance

#### 8.1 Checkpoint Conversion
- **Plan**: `MLX_VALIDATION_PLAN.md`
- **Module**: `scripts/convert_checkpoint.py`
- **Dependencies**: Model structure
- **Key Tasks**:
  - [ ] Implement bidirectional converter
  - [ ] Handle parameter name mapping
  - [ ] Convert optimizer states
  - [ ] Validate converted weights
  - [ ] Test on multiple model sizes

#### 8.2 Numerical Validation
- **Plan**: `MLX_VALIDATION_PLAN.md`
- **Module**: `tests/validation/test_numerical_parity.py`
- **Dependencies**: Both MLX and PyTorch models
- **Key Tasks**:
  - [ ] Compare forward pass outputs
  - [ ] Validate gradient computations
  - [ ] Check loss values match
  - [ ] Test ACT behavior equivalence
  - [ ] Profile numerical differences

#### 8.3 Performance & Accuracy
- **Plan**: `MLX_VALIDATION_PLAN.md`
- **Modules**: 
  - `benchmarks/benchmark_full_model.py`
  - `tests/validation/test_puzzle_accuracy.py`
- **Key Tasks**:
  - [ ] Benchmark training throughput
  - [ ] Measure inference speed
  - [ ] Test on ARC-1 dataset (target: 42%)
  - [ ] Validate on Sudoku (target: 98%)
  - [ ] Compare with published results

### Phase 9: Documentation & Examples (Week 6)

**Goal**: Create comprehensive documentation and usage examples

#### 9.1 API Documentation
- **Modules**: All public APIs
- **Key Tasks**:
  - [ ] Document all public interfaces
  - [ ] Add comprehensive docstrings
  - [ ] Create API reference guide
  - [ ] Add type hints throughout
  - [ ] Generate API documentation

#### 9.2 Usage Examples
- **Directory**: `examples/`
- **Key Tasks**:
  - [ ] Basic training example
  - [ ] Inference demonstration
  - [ ] Custom dataset integration
  - [ ] Fine-tuning tutorial
  - [ ] Checkpoint manipulation guide

#### 9.3 Tutorials
- **Directory**: `docs/tutorials/`
- **Key Tasks**:
  - [ ] Getting started guide
  - [ ] Understanding ACT mechanism
  - [ ] Custom puzzle integration
  - [ ] Performance optimization tips
  - [ ] Migration from PyTorch guide

## Testing Strategy

### Unit Testing (Continuous)

Each component must have:
1. **Shape Tests**: Verify input/output dimensions
2. **Numerical Tests**: Check mathematical correctness
3. **Gradient Tests**: Ensure proper backpropagation
4. **Edge Case Tests**: Handle extreme values
5. **Performance Tests**: Benchmark speed and memory

### Integration Testing (After each phase)

1. **Component Integration**: Test component interactions
2. **Forward Pass**: Verify full model forward pass
3. **Backward Pass**: Check gradient flow through model
4. **Checkpoint Compatibility**: Ensure weight loading works

### System Testing (Phase 6)

1. **Training Convergence**: Verify model trains properly
2. **Accuracy Validation**: Match PyTorch performance
3. **Memory Profiling**: Ensure efficient memory usage
4. **Speed Benchmarking**: Compare with PyTorch baseline

## Integration and Validation

### Checkpoint Conversion

```python
# scripts/convert_checkpoint.py
def convert_pytorch_to_mlx(pytorch_path, mlx_path):
    """Convert PyTorch checkpoint to MLX format.
    
    PyTorch models typically stored in ../HRM/checkpoints/
    """
    # Load PyTorch weights from HRM directory
    # Map to MLX model structure
    # Save in MLX format
```

### Validation Pipeline

1. **Weight Validation**: Ensure converted weights match
2. **Output Validation**: Compare model outputs with PyTorch
3. **Gradient Validation**: Verify gradient computation
4. **Training Validation**: Compare training dynamics

### Performance Targets

- **Inference Speed**: ≥ PyTorch performance on M1/M2
- **Memory Usage**: ≤ PyTorch memory footprint
- **Numerical Accuracy**: < 1e-5 relative error
- **Training Stability**: Match PyTorch convergence

## Project Timeline

### Week 1: Foundation ✅ COMPLETE
- Days 1-2: Setup and initialization module ✅ COMPLETE
- Days 3-4: RMSNorm implementation ✅ COMPLETE
- Day 5: Testing and benchmarking ✅ COMPLETE

### Week 2: Core Components ✅ COMPLETE
- Days 1-2: SwiGLU activation ✅ COMPLETE
- Days 3-4: Sparse embeddings and RoPE ✅ COMPLETE
- Day 5: Integration testing ✅ COMPLETE

### Week 3: Complex Modules ✅ COMPLETE
- Days 1-3: Attention mechanism ✅ COMPLETE
- Days 4-5: Initial ACT implementation ✅ COMPLETE

### Week 4: Advanced Features ✅ COMPLETE
- Days 1-3: Complete ACT mechanism ✅ COMPLETE
- Days 4-5: Integration and testing ✅ COMPLETE

### Week 5: Model Integration & Loss Functions
- Days 1-2: Complete model assembly (Phase 5.1)
- Days 3: Configuration system (Phase 5.2)
- Days 4-5: Loss functions & metrics (Phase 6)

### Week 6: Training Infrastructure
- Days 1-2: Optimizers and training loop (Phase 7.1-7.2)
- Days 3: Data pipeline (Phase 7.3)
- Days 4-5: Initial validation and testing (Phase 8)

### Week 7: Validation & Documentation
- Days 1-2: Complete validation suite (Phase 8)
- Days 3-4: Documentation and examples (Phase 9)
- Day 5: Final testing and release preparation

## Success Criteria

1. **Feature Parity**: All HRM features implemented
2. **Performance**: Meet or exceed PyTorch baseline
3. **Accuracy**: < 0.1% difference in task performance
4. **Stability**: Reliable training across different configs
5. **Usability**: Clear API and documentation

## Risk Mitigation

### Technical Risks
1. **Numerical Differences**: Maintain reference implementations for validation
2. **Performance Issues**: Profile early and often
3. **API Mismatches**: Create compatibility layers where needed

### Schedule Risks
1. **Complexity**: ACT mechanism may require extra time
2. **Testing**: Allocate buffer time for debugging
3. **Integration**: Plan for unexpected integration issues

## Conclusion

This master plan provides a systematic approach to implementing HRM in MLX. By following a bottom-up methodology with comprehensive testing at each stage, we ensure a robust and performant implementation. The modular design allows for parallel development and easy debugging, while the phased approach manages complexity and risk.

Each phase builds on the previous ones, creating a solid foundation for the complete model. Regular validation against the PyTorch reference ensures correctness, while performance benchmarking guides optimization efforts.

## Quick Reference

### Implementation Order
1. **Initialization** ✅ → 2. **RMSNorm** ✅ → 3. **SwiGLU** ✅ → 4. **Sparse Embeddings** ✅ → 5. **RoPE** ✅ → 6. **Attention** ✅ → 7. **ACT** ✅ → 8. **HRM Model** ⏳ → 9. **Loss Functions** → 10. **Training** → 11. **Validation** → 12. **Documentation**

### Key Dependencies
- Attention needs: RoPE, RMSNorm, Custom initialization
- ACT needs: All layers and attention
- HRM Model needs: All components
- Loss Functions need: Complete model
- Training needs: Model + Losses + Optimizers
- Validation needs: Training infrastructure

### Critical Path
Initialization → RMSNorm → RoPE → Attention → ACT → Model Integration → Loss Functions → Training Infrastructure → Validation

### Detailed Implementation Plans
- **Phase 5**: `MLX_MODEL_INTEGRATION_PLAN.md`
- **Phase 6**: `MLX_LOSS_FUNCTIONS_PLAN.md`
- **Phase 7**: `MLX_TRAINING_INFRASTRUCTURE_PLAN.md`
- **Phase 8**: `MLX_VALIDATION_PLAN.md`