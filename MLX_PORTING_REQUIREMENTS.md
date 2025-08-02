# MLX Porting Requirements for Hierarchical Reasoning Model (HRM)

## Overview
This document outlines the technical requirements and challenges for porting the Hierarchical Reasoning Model from PyTorch to MLX (Apple's machine learning framework).

**Original HRM Implementation:** Located in sibling directory `../HRM/`
**Target MLX Implementation:** Current directory `~/Documents/MLX/`

## Core Architecture Components

### 1. Attention Mechanism
**Current Implementation:** Flash Attention (`flash_attn_func`)
**MLX Equivalent:** `mlx.core.fast.scaled_dot_product_attention`

**Requirements:**
- Replace Flash Attention calls with MLX's scaled dot product attention
- Adapt tensor shape from PyTorch format to MLX format: `(B, N_heads, Seq_len, Head_dim)`
- Set `mask=None` for non-causal attention (HRM doesn't use causal masking)
- Maintain scaling factor: `1/sqrt(head_dim)`
- Ensure numerical stability matches PyTorch implementation

**Key Code Location:** `../HRM/models/layers.py:130`

### 2. Custom Layers

#### 2.1 RMSNorm (Root Mean Square Normalization)
**Current Implementation:** Custom function in `../HRM/models/layers.py:152-158`
**MLX Equivalent:** Built-in `mlx.nn.RMSNorm`

**Requirements:**
- Verify MLX's RMSNorm matches PyTorch implementation
- Port variance epsilon parameter (default: 1e-5)
- Ensure correct handling of input/output dtypes

#### 2.2 SwiGLU Activation
**Current Implementation:** Custom module in `../HRM/models/layers.py:139-149`
**MLX Equivalent:** Needs custom implementation using MLX ops

**Requirements:**
- Implement gated linear unit pattern: `silu(xW) * xV`
- Port expansion factor logic
- Maintain correct initialization for gate and value projections

#### 2.3 Rotary Position Embeddings (RoPE)
**Current Implementation:** `../HRM/models/layers.py:30-41`
**MLX Equivalent:** Needs custom implementation

**Requirements:**
- Port `apply_rotary_pos_emb` function
- Implement frequency computation with theta parameter (default: 10000.0)
- Support both cached and dynamic computation modes
- Handle proper broadcasting for different tensor shapes

### 3. Custom Initialization
**Current Implementation:** `trunc_normal_init_` in `models/common.py`
**MLX Equivalent:** Needs custom implementation

**Requirements:**
- Port truncated normal initialization matching JAX's behavior
- Support configurable standard deviation and truncation bounds
- Ensure deterministic initialization with proper seeding

### 4. Sparse Embeddings
**Current Implementation:** `CastedSparseEmbedding` with custom optimizer
**MLX Equivalent:** Needs complete reimplementation

**Requirements:**
- Port sparse embedding layer with proper gradient handling
- Implement SignSGD optimizer for sparse parameters
- Remove distributed training logic (initially single-device)
- Maintain int8 storage with float32 computation pattern

### 5. Adaptive Computation Time (ACT) Mechanism
**Current Implementation:** Complex carry state management with Q-learning
**MLX Equivalent:** Needs careful state management porting

**Requirements:**
- Port carry state dataclasses to MLX-compatible structures
- Implement gradient-free forward passes for intermediate steps
- Handle Q-learning exploration during training
- Maintain proper tensor detachment for state management
- Support variable computation steps (up to `halt_max_steps`)

## Data Type Management

### Requirements:
- Forward computation in bfloat16 (configurable)
- Softmax operations in float32 for stability
- Sparse embeddings: int8 storage, float32 computation
- Support for mixed precision throughout the model

### MLX Considerations:
- Verify bfloat16 support on target Apple Silicon
- Implement explicit dtype casting where needed
- Handle MLX's automatic type promotion rules

## Memory Management

### Requirements:
- Port PyTorch buffers to MLX persistent state management
- Handle gradient checkpointing for memory efficiency
- Implement proper tensor copying for carry states
- Optimize for Apple Silicon unified memory architecture

### Key Patterns to Port:
- `nn.Buffer` → MLX equivalent for persistent non-parameter state
- `torch.no_grad()` → MLX gradient context management
- `.detach()` → MLX tensor detachment
- `.clone()` → MLX tensor copying

## Loss Functions

### Requirements:
- Port `stablemax_cross_entropy` with numerical stability
- Implement binary cross-entropy for Q-learning
- Maintain proper gradient scaling
- Support loss accumulation patterns

**Key Code Location:** `models/losses.py`

## Training Pipeline Adaptation

### Single-Device Training (Phase 1):
- Remove all `torch.distributed` dependencies
- Simplify sparse embedding optimizer for single device
- Port learning rate scheduling
- Implement checkpoint saving/loading in MLX format

### Multi-Device Training (Phase 2 - Future):
- Research MLX's distributed training capabilities
- Design new strategy for sparse embedding synchronization
- Implement data parallel training if supported

## Performance Optimizations

### Requirements:
- Leverage MLX's graph compilation where possible
- Optimize memory access patterns for Apple Silicon
- Profile and benchmark against PyTorch baseline
- Implement efficient batch processing

### Key Metrics to Match:
- Training throughput (tokens/second)
- Memory usage per batch
- Convergence behavior
- Numerical accuracy

## Testing Requirements

### Unit Tests:
- Layer-by-layer numerical verification against PyTorch
- Gradient checking for custom operations
- ACT mechanism state management verification
- Loss function numerical stability tests

### Integration Tests:
- Full model forward/backward pass comparison
- Training convergence on small dataset
- Checkpoint compatibility testing
- Performance benchmarking

## Deliverables

1. **Core Layer Implementations**
   - [ ] RMSNorm compatibility
   - [ ] SwiGLU implementation
   - [ ] RoPE implementation
   - [ ] Attention mechanism replacement

2. **Model Architecture**
   - [ ] HRM model class in MLX
   - [ ] ACT carry state management
   - [ ] Hierarchical reasoning modules
   - [ ] Loss function implementations

3. **Training Infrastructure**
   - [ ] Single-device training script
   - [ ] Checkpoint save/load utilities
   - [ ] Data loading pipeline adaptation
   - [ ] Evaluation script

4. **Documentation**
   - [ ] Porting guide with code mappings
   - [ ] Performance comparison report
   - [ ] API differences documentation
   - [ ] Example usage scripts

## Risk Factors

1. **Numerical Differences:** MLX operations may have subtle numerical differences
2. **Missing Operations:** Some PyTorch ops might not have direct MLX equivalents
3. **Performance Gaps:** Initial port may not match PyTorch performance
4. **Debugging Complexity:** Limited debugging tools compared to PyTorch ecosystem

## Success Criteria

- Model achieves equivalent accuracy on validation sets
- Training is numerically stable
- Performance is within 80% of PyTorch baseline
- All core functionality is preserved
- Code is maintainable and well-documented