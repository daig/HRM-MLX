# MLX HRM Master Implementation Plan

## Executive Summary

This master plan provides a comprehensive roadmap for implementing the Hierarchical Reasoning Model (HRM) in MLX from scratch. It follows a bottom-up approach, starting with foundational components and building up to the complete model. Each phase references specific implementation plans and ensures proper testing and validation at every step.

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

### Phase 1: Foundation Components (Week 1)

**Goal**: Implement basic building blocks with no dependencies

#### 1.1 Custom Weight Initialization
- **Plan**: `MLX_CUSTOM_INITIALIZATION_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/initialization.py`
- **Key Tasks**:
  - [ ] Implement truncated normal initialization
  - [ ] Match JAX/PyTorch initialization behavior
  - [ ] Create comprehensive unit tests
  - [ ] Benchmark against PyTorch reference

#### 1.2 RMSNorm Layer
- **Plan**: `MLX_RMSNORM_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/normalization.py`
- **Key Tasks**:
  - [ ] Implement RMSNorm with MLX backend
  - [ ] Ensure numerical stability in mixed precision
  - [ ] Test shape preservation and gradient flow
  - [ ] Performance benchmarking

### Phase 2: Core Components (Week 1-2)

**Goal**: Build essential layers that depend only on Phase 1

#### 2.1 SwiGLU Activation
- **Plan**: `MLX_SWIGLU_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/activations.py`
- **Dependencies**: Custom initialization
- **Key Tasks**:
  - [ ] Implement SwiGLU with fused projections
  - [ ] Optimize intermediate dimension calculation
  - [ ] Test activation behavior and gradients
  - [ ] Benchmark memory usage and speed

#### 2.2 Sparse Embeddings
- **Plan**: `MLX_SPARSE_EMBEDDINGS_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/layers/embeddings.py`
- **Dependencies**: Custom initialization
- **Key Tasks**:
  - [ ] Implement sparse embedding layer
  - [ ] Port SignSGD optimizer
  - [ ] Test vocabulary scaling
  - [ ] Memory optimization for large vocabularies

#### 2.3 Rotary Position Embeddings
- **Plan**: `MLX_ROPE_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/modules/rope.py`
- **Dependencies**: None (standalone)
- **Key Tasks**:
  - [ ] Implement RoPE with caching
  - [ ] Support length extrapolation
  - [ ] Test position encoding properties
  - [ ] Optimize for different sequence lengths

### Phase 3: Complex Modules (Week 2-3)

**Goal**: Implement modules that integrate Phase 1-2 components

#### 3.1 Multi-Head Attention
- **Plan**: `MLX_ATTENTION_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/modules/attention.py`
- **Dependencies**: RoPE, RMSNorm, Custom initialization
- **Key Tasks**:
  - [ ] Implement attention with MLX scaled_dot_product_attention
  - [ ] Support both MHA and GQA
  - [ ] Integrate RoPE for position encoding
  - [ ] Test attention patterns and gradients
  - [ ] Benchmark against FlashAttention

### Phase 4: Advanced Features (Week 3-4)

**Goal**: Implement the hierarchical reasoning mechanisms

#### 4.1 Adaptive Computation Time (ACT)
- **Plan**: `MLX_ACT_MECHANISM_IMPLEMENTATION_PLAN.md`
- **Module**: `src/mlx_hrm/modules/act.py`
- **Dependencies**: All previous components
- **Key Tasks**:
  - [ ] Implement ACT controller
  - [ ] Build high-level and low-level modules
  - [ ] Create differentiable halting mechanism
  - [ ] Test adaptive computation behavior
  - [ ] Optimize for dynamic computation graphs

### Phase 5: Model Integration (Week 4-5)

**Goal**: Assemble all components into the complete HRM model

#### 5.1 HRM Block
- **Module**: `src/mlx_hrm/models/hrm_block.py`
- **Dependencies**: All components
- **Key Tasks**:
  - [ ] Integrate attention and FFN modules
  - [ ] Implement residual connections
  - [ ] Add ACT mechanism integration
  - [ ] Test block functionality

#### 5.2 Complete HRM Model
- **Module**: `src/mlx_hrm/models/hrm_model.py`
- **Dependencies**: HRM blocks, embeddings
- **Key Tasks**:
  - [ ] Stack HRM blocks into full model
  - [ ] Add input/output projections
  - [ ] Implement model configuration system
  - [ ] Create model factory functions
  - [ ] Test full forward pass

### Phase 6: Training Infrastructure (Week 5-6)

**Goal**: Build training and evaluation capabilities

#### 6.1 Training Utilities
- **Module**: `src/mlx_hrm/training/`
- **Key Tasks**:
  - [ ] Port Adam-atan2 optimizer
  - [ ] Implement training loop with MLX
  - [ ] Add gradient accumulation
  - [ ] Create checkpoint utilities
  - [ ] Build data loading pipeline

#### 6.2 Evaluation and Benchmarking
- **Key Tasks**:
  - [ ] Port evaluation metrics
  - [ ] Create benchmark suite
  - [ ] Compare with PyTorch reference
  - [ ] Profile memory and speed
  - [ ] Validate on ARC tasks

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

### Week 1: Foundation
- Days 1-2: Setup and initialization module
- Days 3-4: RMSNorm implementation
- Day 5: Testing and benchmarking

### Week 2: Core Components
- Days 1-2: SwiGLU activation
- Days 3-4: Sparse embeddings and RoPE
- Day 5: Integration testing

### Week 3: Complex Modules
- Days 1-3: Attention mechanism
- Days 4-5: Initial ACT implementation

### Week 4: Advanced Features
- Days 1-3: Complete ACT mechanism
- Days 4-5: HRM block integration

### Week 5: Model Assembly
- Days 1-2: Full model integration
- Days 3-4: Training infrastructure
- Day 5: Initial validation

### Week 6: Validation and Optimization
- Days 1-2: Comprehensive testing
- Days 3-4: Performance optimization
- Day 5: Documentation and release

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
1. **Initialization** → 2. **RMSNorm** → 3. **SwiGLU** → 4. **Sparse Embeddings** → 5. **RoPE** → 6. **Attention** → 7. **ACT** → 8. **HRM Model** → 9. **Training**

### Key Dependencies
- Attention needs: RoPE, RMSNorm
- ACT needs: All layers and attention
- HRM Model needs: All components
- Training needs: Complete model

### Critical Path
Initialization → RMSNorm → RoPE → Attention → ACT → Model Integration