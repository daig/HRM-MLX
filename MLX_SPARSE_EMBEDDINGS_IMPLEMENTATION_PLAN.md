# MLX Sparse Embeddings Implementation Plan

## Executive Summary

This document outlines the implementation plan for porting HRM's sparse embedding system from PyTorch to MLX. The sparse embedding layer is a critical component that enables efficient training on thousands of puzzle types by only storing and updating embeddings that are actively used. The implementation will maintain the core design principles while adapting to MLX's computation model.

## Architecture Overview

### Current PyTorch Architecture

The sparse embedding system consists of three main components:

1. **CastedSparseEmbedding Layer**
   - Stores a master embedding table in FP32
   - Maintains local buffers for active embeddings in the current batch
   - Supports int8 storage with float32 computation (casting)
   - Zero initialization for stable training

2. **SignSGD Optimizer**
   - Custom optimizer specifically designed for sparse parameters
   - Updates only active embeddings using sign of gradients
   - Includes weight decay for regularization
   - Distributed training support for multi-GPU setups

3. **Gradient Management**
   - Local gradient computation on active embeddings
   - Distributed gradient aggregation across GPUs
   - Sparse scatter updates back to master table

### Key Design Principles

1. **Memory Efficiency**: Only load embeddings that appear in the current batch
2. **Stable Training**: Zero initialization prevents gradient explosion
3. **Sparse Updates**: SignSGD provides consistent updates regardless of gradient magnitude
4. **Type Casting**: Int8 storage with float32/bfloat16 computation for efficiency

## Step-by-Step Implementation Guide

### Phase 1: Core Sparse Embedding Layer

#### 1.1 Basic Structure

```python
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict
import math

class MLXSparseEmbedding(nn.Module):
    """Sparse embedding layer for MLX.
    
    Key differences from PyTorch version:
    - Uses MLX arrays instead of torch tensors
    - No explicit gradient tracking (MLX handles this)
    - Simplified buffer management
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        batch_size: int,
        init_std: float = 0.0,
        dtype: mx.Dtype = mx.float32
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Master embedding table (equivalent to PyTorch buffer)
        # Zero initialization for stability
        if init_std == 0:
            self.weights = mx.zeros((num_embeddings, embedding_dim), dtype=mx.float32)
        else:
            self.weights = self._trunc_normal_init(
                (num_embeddings, embedding_dim), std=init_std
            )
        
        # Local workspace for current batch
        self.local_weights = None
        self.local_ids = None
```

#### 1.2 Initialization Helper

```python
def _trunc_normal_init(self, shape, std=1.0, lower=-2.0, upper=2.0):
    """Truncated normal initialization matching PyTorch version."""
    # MLX implementation of truncated normal
    # This ensures consistent initialization between frameworks
    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    
    # Generate uniform samples and transform
    uniform_samples = mx.random.uniform(
        low=a, high=b, shape=shape
    )
    
    # Apply inverse error function and scaling
    z = (b - a) / 2
    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower ** 2)
    pdf_l = c * math.exp(-0.5 * upper ** 2)
    comp_std = std / math.sqrt(
        1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
    )
    
    # Transform to normal distribution
    result = mx.erfinv(uniform_samples) * sqrt2 * comp_std
    return mx.clip(result, lower * comp_std, upper * comp_std)
```

#### 1.3 Forward Pass Implementation

```python
def __call__(self, inputs: mx.array, training: bool = True) -> mx.array:
    """Forward pass with training/inference modes."""
    if not training:
        # Direct lookup for inference
        return self.weights[inputs].astype(self.dtype)
    
    # Training mode: Extract active embeddings
    # Store IDs for gradient updates
    self.local_ids = inputs
    
    # Gather embeddings for current batch
    self.local_weights = self.weights[inputs]
    
    # Return casted version
    return self.local_weights.astype(self.dtype)
```

### Phase 2: SignSGD Optimizer Implementation

#### 2.1 Optimizer Structure

```python
class SignSGD:
    """Sign-based SGD optimizer for sparse embeddings in MLX.
    
    Key features:
    - Updates parameters using sign of gradients
    - Supports weight decay
    - Efficient for sparse updates
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def update(
        self,
        sparse_embedding: MLXSparseEmbedding,
        gradients: mx.array
    ):
        """Update sparse embeddings using SignSGD."""
        if sparse_embedding.local_ids is None:
            return
        
        # Get unique IDs and aggregate gradients
        unique_ids, inverse_indices = mx.unique(
            sparse_embedding.local_ids, return_inverse=True
        )
        
        # Aggregate gradients for duplicate IDs
        aggregated_grads = self._aggregate_gradients(
            gradients, unique_ids, inverse_indices
        )
        
        # Apply SignSGD update
        self._apply_sign_update(
            sparse_embedding.weights,
            unique_ids,
            aggregated_grads
        )
```

#### 2.2 Gradient Aggregation

```python
def _aggregate_gradients(
    self, 
    gradients: mx.array, 
    unique_ids: mx.array,
    inverse_indices: mx.array
) -> mx.array:
    """Aggregate gradients for duplicate embedding IDs."""
    num_unique = unique_ids.shape[0]
    embedding_dim = gradients.shape[1]
    
    # Initialize aggregated gradients
    aggregated = mx.zeros((num_unique, embedding_dim), dtype=gradients.dtype)
    
    # Sum gradients for each unique ID
    # MLX doesn't have scatter_add, so we'll use a different approach
    for i in range(gradients.shape[0]):
        unique_idx = inverse_indices[i]
        aggregated[unique_idx] += gradients[i]
    
    return aggregated
```

#### 2.3 Sign Update Application

```python
def _apply_sign_update(
    self,
    weights: mx.array,
    ids: mx.array,
    gradients: mx.array
):
    """Apply SignSGD update with weight decay."""
    # Get current weights for active embeddings
    active_weights = weights[ids]
    
    # Apply weight decay: w = w * (1 - lr * wd)
    active_weights *= (1.0 - self.learning_rate * self.weight_decay)
    
    # Apply sign update: w = w - lr * sign(grad)
    active_weights -= self.learning_rate * mx.sign(gradients)
    
    # Update master table (in-place update)
    weights[ids] = active_weights
```

### Phase 3: Integration with HRM Model

#### 3.1 Model Integration

```python
class HRMModelWithSparseEmbeddings(nn.Module):
    """Example integration with HRM model."""
    
    def __init__(self, config):
        super().__init__()
        
        # Sparse puzzle embeddings
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = MLXSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0.0,  # Zero initialization
                dtype=mx.bfloat16 if config.use_bfloat16 else mx.float32
            )
        
        # Other model components...
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
    
    def __call__(self, inputs, puzzle_ids, training=True):
        # Get puzzle embeddings
        if hasattr(self, 'puzzle_emb'):
            puzzle_embeddings = self.puzzle_emb(
                puzzle_ids, training=training
            )
            # Process puzzle embeddings...
        
        # Continue with model forward pass...
```

#### 3.2 Training Loop Integration

```python
def train_step(model, optimizer, sparse_optimizer, batch):
    """Single training step with sparse embeddings."""
    
    def loss_fn(model, batch):
        # Forward pass
        outputs = model(
            batch['inputs'], 
            batch['puzzle_identifiers'],
            training=True
        )
        # Compute loss
        return compute_loss(outputs, batch['labels'])
    
    # Compute gradients
    loss, grads = mx.grad(loss_fn)(model, batch)
    
    # Update main model parameters
    optimizer.update(model, grads)
    
    # Update sparse embeddings separately
    if hasattr(model, 'puzzle_emb'):
        sparse_grads = grads['puzzle_emb']['local_weights']
        sparse_optimizer.update(model.puzzle_emb, sparse_grads)
    
    return loss
```

### Phase 4: Memory Management and Optimization

#### 4.1 Memory-Efficient Gradient Computation

```python
class MemoryEfficientSparseEmbedding(MLXSparseEmbedding):
    """Enhanced version with memory optimizations."""
    
    def __call__(self, inputs: mx.array, training: bool = True) -> mx.array:
        if not training:
            return self.weights[inputs].astype(self.dtype)
        
        # Detach previous local weights to free memory
        if self.local_weights is not None:
            del self.local_weights
        
        # Store only unique IDs to reduce memory
        unique_ids, inverse = mx.unique(inputs, return_inverse=True)
        self.local_ids = unique_ids
        self.inverse_indices = inverse
        
        # Gather unique embeddings only
        unique_embeddings = self.weights[unique_ids]
        
        # Reconstruct full batch embeddings
        self.local_weights = unique_embeddings[inverse]
        
        return self.local_weights.astype(self.dtype)
```

#### 4.2 Int8 Storage Support

```python
class Int8SparseEmbedding(MLXSparseEmbedding):
    """Sparse embeddings with int8 storage for memory efficiency."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store scale factors for int8 quantization
        self.scale = mx.ones((self.num_embeddings,), dtype=mx.float32)
        
        # Convert weights to int8
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Quantize float32 weights to int8."""
        # Compute per-embedding scale factors
        max_vals = mx.max(mx.abs(self.weights), axis=1)
        self.scale = max_vals / 127.0
        
        # Quantize to int8
        self.weights_int8 = mx.round(
            self.weights / self.scale[:, None]
        ).astype(mx.int8)
        
        # Free float32 weights
        del self.weights
    
    def __call__(self, inputs: mx.array, training: bool = True) -> mx.array:
        # Dequantize on the fly
        scales = self.scale[inputs]
        embeddings = self.weights_int8[inputs].astype(mx.float32)
        embeddings *= scales[:, None]
        
        return embeddings.astype(self.dtype)
```

### Phase 5: Testing Strategy

#### 5.1 Unit Tests

```python
def test_sparse_embedding_forward():
    """Test forward pass correctness."""
    embedding = MLXSparseEmbedding(
        num_embeddings=1000,
        embedding_dim=128,
        batch_size=32
    )
    
    # Test training mode
    ids = mx.array([1, 5, 10, 1])  # Note duplicate ID
    output = embedding(ids, training=True)
    assert output.shape == (4, 128)
    
    # Test inference mode
    output_eval = embedding(ids, training=False)
    assert mx.allclose(output, output_eval)

def test_sign_sgd_update():
    """Test SignSGD optimizer."""
    embedding = MLXSparseEmbedding(100, 64, 16)
    optimizer = SignSGD(learning_rate=0.01, weight_decay=0.1)
    
    # Simulate gradients
    ids = mx.array([0, 1, 2])
    embeddings = embedding(ids)
    gradients = mx.random.normal((3, 64))
    
    # Store original weights
    original = embedding.weights[ids].copy()
    
    # Update
    optimizer.update(embedding, gradients)
    
    # Verify update applied
    updated = embedding.weights[ids]
    assert not mx.allclose(original, updated)
```

#### 5.2 Performance Benchmarks

```python
def benchmark_sparse_vs_dense():
    """Compare sparse vs dense embedding performance."""
    import time
    
    num_embeddings = 10000
    embedding_dim = 256
    batch_size = 128
    num_iters = 100
    
    # Sparse embedding
    sparse_emb = MLXSparseEmbedding(
        num_embeddings, embedding_dim, batch_size
    )
    
    # Dense embedding
    dense_emb = nn.Embedding(num_embeddings, embedding_dim)
    
    # Benchmark sparse
    start = time.time()
    for _ in range(num_iters):
        ids = mx.random.randint(0, num_embeddings, (batch_size,))
        _ = sparse_emb(ids)
        mx.eval(_)  # Force computation
    sparse_time = time.time() - start
    
    # Benchmark dense
    start = time.time()
    for _ in range(num_iters):
        ids = mx.random.randint(0, num_embeddings, (batch_size,))
        _ = dense_emb(ids)
        mx.eval(_)
    dense_time = time.time() - start
    
    print(f"Sparse: {sparse_time:.3f}s, Dense: {dense_time:.3f}s")
    print(f"Memory saved: {(1 - batch_size/num_embeddings) * 100:.1f}%")
```

#### 5.3 Correctness Validation

```python
def validate_against_pytorch():
    """Validate MLX implementation against PyTorch reference."""
    import torch
    import numpy as np
    
    # Set same random seed
    np.random.seed(42)
    torch.manual_seed(42)
    mx.random.seed(42)
    
    # Create equivalent embeddings
    n, d = 100, 64
    
    # PyTorch version
    pt_emb = CastedSparseEmbedding(n, d, 16, init_std=0.0, cast_to=torch.float32)
    
    # MLX version
    mlx_emb = MLXSparseEmbedding(n, d, 16, init_std=0.0)
    
    # Copy weights
    mlx_emb.weights = mx.array(pt_emb.weights.numpy())
    
    # Test forward pass
    ids = [1, 5, 10]
    pt_out = pt_emb(torch.tensor(ids)).detach().numpy()
    mlx_out = np.array(mlx_emb(mx.array(ids), training=False))
    
    assert np.allclose(pt_out, mlx_out, rtol=1e-5)
```

### Phase 6: Migration Path

#### 6.1 Single-Device Focus

The initial implementation removes all distributed training logic:

1. **Remove dist.all_gather_into_tensor calls**
2. **Remove world_size parameter**
3. **Simplify gradient aggregation to single-device**
4. **Remove distributed initialization**

#### 6.2 Future Multi-Device Support

Plan for future MLX multi-device support:

```python
class DistributedSparseEmbedding(MLXSparseEmbedding):
    """Placeholder for future distributed support."""
    
    def __init__(self, *args, devices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.devices = devices or [mx.default_device()]
        
        # Shard embeddings across devices
        embeddings_per_device = self.num_embeddings // len(self.devices)
        # Implementation details TBD based on MLX distributed API
```

## Example Usage

### Complete Training Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Model configuration
class Config:
    num_puzzle_identifiers = 5000
    puzzle_emb_ndim = 512
    batch_size = 128
    vocab_size = 10000
    hidden_size = 768
    use_bfloat16 = True

# Initialize model
config = Config()
model = HRMModelWithSparseEmbeddings(config)

# Optimizers
main_optimizer = optim.AdamW(
    learning_rate=1e-4,
    weight_decay=0.1
)

sparse_optimizer = SignSGD(
    learning_rate=1e-2,  # Higher LR for sparse embeddings
    weight_decay=0.01
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Convert batch to MLX arrays
        inputs = mx.array(batch['inputs'])
        puzzle_ids = mx.array(batch['puzzle_identifiers'])
        labels = mx.array(batch['labels'])
        
        # Forward pass
        def loss_fn(model):
            logits = model(inputs, puzzle_ids, training=True)
            return nn.losses.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1)
            )
        
        # Compute gradients
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        
        # Update parameters
        main_optimizer.update(model, grads)
        
        # Update sparse embeddings
        if 'puzzle_emb' in grads:
            sparse_grads = grads['puzzle_emb']['local_weights']
            sparse_optimizer.update(model.puzzle_emb, sparse_grads)
        
        # Evaluate loss
        mx.eval(loss_value)
        print(f"Loss: {loss_value.item():.4f}")
```

## Performance Considerations

### Memory Usage

1. **Sparse Storage**: Only `batch_size / num_embeddings` fraction of memory used
2. **Int8 Quantization**: 4x memory reduction with minimal accuracy loss
3. **Gradient Checkpointing**: Not needed due to sparse updates

### Computation Efficiency

1. **Vectorized Operations**: MLX's efficient array operations
2. **Fused Kernels**: Leverage MLX's kernel fusion capabilities
3. **Lazy Evaluation**: MLX's automatic optimization of computation graphs

### Expected Performance Metrics

- **Memory Reduction**: 90-99% for large embedding tables
- **Training Speed**: Comparable to PyTorch with proper optimization
- **Inference Speed**: Faster due to reduced memory access

## Implementation Timeline

### Week 1: Core Implementation
- Basic MLXSparseEmbedding class
- Forward pass implementation
- Unit tests

### Week 2: Optimizer and Integration
- SignSGD optimizer
- Model integration
- Training loop setup

### Week 3: Optimizations
- Memory optimizations
- Int8 storage support
- Performance benchmarking

### Week 4: Testing and Validation
- Comprehensive testing
- PyTorch validation
- Documentation

## Conclusion

This implementation plan provides a complete roadmap for porting HRM's sparse embedding system to MLX. The design maintains the key innovations while adapting to MLX's computation model. The phased approach ensures each component is properly tested before integration, minimizing risk and ensuring correctness.

Key deliverables:
1. **MLXSparseEmbedding**: Core sparse embedding layer
2. **SignSGD**: Optimized optimizer for sparse parameters
3. **Integration Guide**: How to use with HRM models
4. **Test Suite**: Comprehensive validation
5. **Performance Benchmarks**: Comparison with PyTorch implementation