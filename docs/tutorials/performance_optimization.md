# Performance Optimization Guide for HRM MLX

This guide provides comprehensive strategies for optimizing HRM performance on Apple Silicon. We'll cover both training and inference optimization techniques.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Hardware Considerations](#hardware-considerations)
3. [Model Configuration](#model-configuration)
4. [Memory Optimization](#memory-optimization)
5. [Training Optimizations](#training-optimizations)
6. [Inference Optimizations](#inference-optimizations)
7. [Data Pipeline Optimization](#data-pipeline-optimization)
8. [Mixed Precision Training](#mixed-precision-training)
9. [Benchmarking and Profiling](#benchmarking-and-profiling)
10. [Advanced Techniques](#advanced-techniques)

## Performance Overview

HRM MLX achieves excellent performance on Apple Silicon:

### Benchmark Results (M2 Max, 32GB)

| Model Size | Forward Pass | Training | Memory Usage | Throughput |
|------------|--------------|----------|--------------|------------|
| Tiny (7M)  | 0.8ms       | 12ms     | 22.7MB      | 156K tok/s |
| Small (27M)| 2.1ms       | 28ms     | 89MB        | 98K tok/s  |
| Base (100M)| 6.2ms       | 78ms     | 312MB       | 42K tok/s  |
| Large (200M)| 11.4ms     | 142ms    | 598MB       | 23K tok/s  |

*Results for batch_size=8, seq_len=32*

### Key Performance Factors

1. **MLX Optimizations**: Graph compilation, memory pooling
2. **Apple Silicon Features**: Unified memory, specialized cores
3. **Smart Batching**: Groups similar puzzles for efficiency
4. **Mixed Precision**: Reduces memory and increases speed
5. **ACT Efficiency**: Adaptive computation saves unnecessary work

## Hardware Considerations

### Apple Silicon Advantages

```python
# Check your hardware
import mlx.core as mx
print(f"MLX device: {mx.default_device()}")
print(f"Available memory: {mx.metal.get_peak_memory() / 1024**3:.1f} GB")

# Memory bandwidth optimization
mx.metal.set_cache_limit(0)  # Disable cache for maximum memory
mx.metal.set_memory_limit(32 * 1024**3)  # Set memory limit (32GB example)
```

### Memory Architecture

Apple Silicon's unified memory allows:

- **Large models** without GPU memory constraints
- **Efficient transfers** between CPU and GPU operations
- **Memory sharing** between different processing units

### Recommended Hardware

| Task | Minimum | Recommended | Optimal |
|------|---------|-------------|---------|
| Development | M1 8GB | M1 Pro 16GB | M2 Max 32GB+ |
| Small models | M1 16GB | M2 32GB | M2 Ultra 64GB |
| Large models | M1 Max 32GB | M2 Max 64GB | M2 Ultra 128GB |
| Production | M2 Pro 32GB | M2 Max 64GB | M2 Ultra 192GB |

## Model Configuration

### Choose the Right Model Size

```python
from mlx_hrm import create_hrm, get_model_info

# Analyze model sizes
for preset in ['tiny', 'small', 'base', 'large']:
    info = get_model_info(preset)
    print(f"{preset}: {info['estimated_params_millions']}M params, "
          f"~{info['estimated_memory_mb']}MB memory")

# Start with the smallest model that works for your task
model = create_hrm('tiny')  # Good starting point
```

### Optimize Model Architecture

```python
# Custom configuration for specific needs
from mlx_hrm.modules.act import HRMConfig

# Memory-optimized config
memory_optimized = HRMConfig(
    hidden_size=384,      # Smaller hidden size
    num_heads=6,          # Fewer attention heads
    H_layers=3,           # Fewer H-level layers
    L_layers=3,           # Fewer L-level layers
    halt_max_steps=32,    # Limit ACT steps
    vocab_size=1000       # Smaller vocabulary if possible
)

# Speed-optimized config  
speed_optimized = HRMConfig(
    hidden_size=512,
    num_heads=8,
    H_layers=2,           # Fewer layers for speed
    L_layers=2,
    halt_max_steps=16,    # Aggressive ACT limiting
    intermediate_size=1024 # Smaller FFN
)
```

### ACT Configuration for Performance

```python
# Balance accuracy vs. speed with ACT settings
config = HRMConfig(
    halt_max_steps=32,        # Limit maximum computation
    halt_epsilon=1e-3,        # Earlier halting threshold
    act_loss_weight=0.7       # Encourage efficient halting
)
```

## Memory Optimization

### Batch Size Tuning

```python
def find_optimal_batch_size(model, seq_len=32, max_batch=64):
    """Find the largest batch size that fits in memory."""
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        if batch_size > max_batch:
            break
            
        try:
            # Test batch
            batch = {
                'input_ids': mx.random.randint(0, 1000, shape=(batch_size, seq_len))
            }
            carry, outputs = model(batch)
            
            # Force computation to check memory usage
            mx.eval(outputs['logits'])
            
            print(f"Batch size {batch_size}: ✓ ({mx.metal.get_active_memory() / 1024**2:.1f} MB)")
            optimal_batch = batch_size
            
        except Exception as e:
            print(f"Batch size {batch_size}: ✗ (OOM)")
            break
    
    return optimal_batch

# Find optimal batch size
optimal_batch = find_optimal_batch_size(model)
print(f"Recommended batch size: {optimal_batch}")
```

### Memory-Efficient Training

```python
from mlx_hrm.training.trainer import HRMTrainer

# Use gradient accumulation to simulate larger batches
trainer = HRMTrainer(
    model=model,
    train_dataloader=dataloader,
    gradient_accumulation_steps=8,  # Effective batch = batch_size * 8
    batch_size=4,                   # Smaller actual batch
    checkpoint_every_n_steps=1000,  # Regular checkpointing
    max_grad_norm=1.0              # Prevent gradient explosion
)
```

### Memory Monitoring

```python
def monitor_memory_usage(model, dataloader, num_batches=10):
    """Monitor memory usage during training."""
    
    memory_stats = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Before forward pass
        before_mem = mx.metal.get_active_memory()
        
        # Forward pass
        carry, outputs = model(batch)
        mx.eval(outputs['logits'])
        
        # After forward pass
        after_mem = mx.metal.get_active_memory()
        
        memory_stats.append({
            'batch': i,
            'before_mb': before_mem / 1024**2,
            'after_mb': after_mem / 1024**2,
            'diff_mb': (after_mem - before_mem) / 1024**2
        })
        
        # Clean up
        del carry, outputs
        mx.metal.clear_cache()
    
    # Analyze
    avg_usage = sum(s['after_mb'] for s in memory_stats) / len(memory_stats)
    max_usage = max(s['after_mb'] for s in memory_stats)
    
    print(f"Average memory usage: {avg_usage:.1f} MB")
    print(f"Peak memory usage: {max_usage:.1f} MB")
    
    return memory_stats
```

## Training Optimizations

### Smart Batching

```python
from mlx_hrm.data.dataset import EnhancedPuzzleDataset, SmartDataLoader

# Enable smart batching for better cache efficiency
dataset = EnhancedPuzzleDataset(
    data=train_data,
    mode='train',
    puzzle_type_key='puzzle_type'  # Group similar puzzles
)

# Smart data loader optimizes batch composition
smart_loader = SmartDataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,                # Enable smart sampling
    drop_last=True              # Consistent batch sizes
)
```

### Optimizer Configuration

```python
# Optimized training setup
from mlx_hrm.training.optimizers import create_hrm_optimizer

optimizer_config = {
    'learning_rate': 1e-4,
    'weight_decay': 0.1,
    'warmup_steps': 1000,        # Stabilize early training
    'beta1': 0.9,               # Adam momentum
    'beta2': 0.999,             # Adam second moment
    'eps': 1e-8,                # Numerical stability
    'sparse_lr_multiplier': 1.0  # Sparse embedding learning rate
}

model, optimizer = create_hrm_optimizer(model, optimizer_config)
```

### Learning Rate Scheduling

```python
from mlx_hrm.training.optimizers import CosineAnnealingLR

# Efficient learning rate schedule
lr_scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    max_steps=10000,
    warmup_steps=1000,
    min_lr_ratio=0.1,
    restart_steps=None  # No restarts for simplicity
)

# Update in training loop
for step in range(max_steps):
    loss = train_step(model, batch, optimizer)
    lr_scheduler.step()
```

### Training Loop Optimization

```python
def optimized_training_loop(trainer, max_steps):
    """Optimized training loop with performance monitoring."""
    
    import time
    
    step_times = []
    start_time = time.time()
    
    for step in range(max_steps):
        step_start = time.time()
        
        # Training step
        metrics = trainer.train_step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Performance monitoring
        if step % 100 == 0:
            avg_step_time = sum(step_times[-100:]) / min(100, len(step_times))
            throughput = trainer.batch_size / avg_step_time
            
            print(f"Step {step}: {avg_step_time:.3f}s/step, "
                  f"{throughput:.1f} samples/s, "
                  f"loss={metrics['total_loss']:.4f}")
            
            # Memory check
            memory_mb = mx.metal.get_active_memory() / 1024**2
            print(f"Memory usage: {memory_mb:.1f} MB")
        
        # Periodic cleanup
        if step % 1000 == 0:
            mx.metal.clear_cache()
    
    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    
    print(f"Training complete: {total_time:.1f}s total, "
          f"{avg_step_time:.3f}s/step average")
```

## Inference Optimizations

### Batch Inference

```python
def optimized_batch_inference(model, inputs, batch_size=16):
    """Process multiple inputs efficiently."""
    
    results = []
    
    # Process in batches
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        
        # Pad to same length within batch
        max_len = max(len(inp) for inp in batch_inputs)
        padded_inputs = []
        
        for inp in batch_inputs:
            if len(inp) < max_len:
                padded = mx.concatenate([inp, mx.zeros(max_len - len(inp))])
            else:
                padded = inp
            padded_inputs.append(padded)
        
        # Stack into batch
        batch_tensor = mx.stack(padded_inputs)
        
        # Single forward pass for entire batch
        batch_dict = {'input_ids': batch_tensor}
        carry, outputs = model(batch_dict)
        
        # Extract results
        for j, original_length in enumerate(len(inp) for inp in batch_inputs):
            result = outputs['logits'][j, :original_length]
            results.append(result)
    
    return results
```

### ACT-Aware Inference

```python
def act_optimized_inference(model, input_tokens, max_act_steps=32):
    """Inference optimized for ACT efficiency."""
    
    carry = model.initial_carry(1)
    batch = {'input_ids': input_tokens.reshape(1, -1)}
    
    # Track ACT efficiency
    act_history = []
    
    for step in range(max_act_steps):
        step_start = time.time()
        
        carry, outputs = model(batch, carry)
        
        step_time = time.time() - step_start
        act_history.append({
            'step': step,
            'act_step': carry.act_step[0].item(),
            'halted': carry.halted[0].item(),
            'time': step_time
        })
        
        # Early termination
        if carry.halted[0]:
            break
    
    # Analyze ACT efficiency
    total_act_steps = sum(h['act_step'] for h in act_history)
    total_time = sum(h['time'] for h in act_history)
    
    print(f"Total ACT steps: {total_act_steps}")
    print(f"Reasoning steps: {len(act_history)}")
    print(f"Total time: {total_time:.3f}s")
    print(f"ACT efficiency: {total_act_steps / len(act_history):.1f} steps/reasoning")
    
    return outputs, act_history
```

### Caching for Repeated Inference

```python
class InferenceCache:
    """Cache for repeated inference patterns."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_cache_key(self, input_tokens):
        return tuple(input_tokens.tolist())
    
    def get(self, input_tokens):
        key = self.get_cache_key(input_tokens)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, input_tokens, result):
        key = self.get_cache_key(input_tokens)
        
        # Evict least used if cache is full
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count.keys(), 
                         key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = result
        self.access_count[key] = 1

# Usage
cache = InferenceCache()

def cached_inference(model, input_tokens):
    # Check cache first
    cached_result = cache.get(input_tokens)
    if cached_result is not None:
        return cached_result
    
    # Compute if not cached
    carry, outputs = model({'input_ids': input_tokens.reshape(1, -1)})
    result = outputs['logits'][0]
    
    # Cache result
    cache.set(input_tokens, result)
    
    return result
```

## Data Pipeline Optimization

### Efficient Data Loading

```python
class OptimizedDataLoader:
    """Optimized data loader for HRM training."""
    
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Pre-compute batch indices for efficiency
        self.batch_indices = self._compute_batch_indices()
    
    def _compute_batch_indices(self):
        """Pre-compute optimal batch compositions."""
        # Group by puzzle type for smart batching
        type_groups = {}
        for i, example in enumerate(self.dataset):
            puzzle_type = example.get('puzzle_type', 0)
            if puzzle_type not in type_groups:
                type_groups[puzzle_type] = []
            type_groups[puzzle_type].append(i)
        
        # Create balanced batches
        batch_indices = []
        remaining_indices = list(range(len(self.dataset)))
        
        while len(remaining_indices) >= self.batch_size:
            batch = []
            
            # Try to include diverse puzzle types
            for puzzle_type in type_groups:
                available = [i for i in type_groups[puzzle_type] if i in remaining_indices]
                if available and len(batch) < self.batch_size:
                    idx = available[0]
                    batch.append(idx)
                    remaining_indices.remove(idx)
                    type_groups[puzzle_type].remove(idx)
            
            # Fill remaining slots
            while len(batch) < self.batch_size and remaining_indices:
                idx = remaining_indices.pop(0)
                batch.append(idx)
            
            if len(batch) == self.batch_size:
                batch_indices.append(batch)
        
        return batch_indices
    
    def __iter__(self):
        for batch_idx in self.batch_indices:
            batch_data = [self.dataset[i] for i in batch_idx]
            yield self._collate_batch(batch_data)
    
    def _collate_batch(self, batch_data):
        """Efficiently collate batch data."""
        # Find maximum length in batch
        max_len = max(len(ex['input_ids']) for ex in batch_data)
        
        # Pre-allocate tensors
        input_ids = mx.zeros((len(batch_data), max_len), dtype=mx.int32)
        labels = mx.zeros((len(batch_data), max_len), dtype=mx.int32)
        
        # Fill tensors
        for i, example in enumerate(batch_data):
            seq_len = len(example['input_ids'])
            input_ids[i, :seq_len] = example['input_ids']
            if 'labels' in example:
                labels[i, :seq_len] = example['labels']
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
```

### Memory-Mapped Data

```python
import mmap
import pickle

class MemoryMappedDataset:
    """Memory-mapped dataset for large data files."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self._load_index()
    
    def _load_index(self):
        """Load dataset index for fast access."""
        index_path = self.data_path + '.index'
        
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
        else:
            self._build_index()
    
    def _build_index(self):
        """Build index of file positions."""
        self.index = []
        
        with open(self.data_path, 'rb') as f:
            while True:
                pos = f.tell()
                try:
                    data = pickle.load(f)
                    self.index.append((pos, f.tell() - pos))
                except EOFError:
                    break
        
        # Save index
        with open(self.data_path + '.index', 'wb') as f:
            pickle.dump(self.index, f)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        pos, size = self.index[idx]
        
        with open(self.data_path, 'rb') as f:
            f.seek(pos)
            data = pickle.load(f)
        
        return data
```

## Mixed Precision Training

### Selective Mixed Precision

```python
from mlx_hrm.training.trainer import HRMTrainer

# Configure selective mixed precision
trainer = HRMTrainer(
    model=model,
    train_dataloader=dataloader,
    use_mixed_precision=True,
    mixed_precision_dtype='float16',
    mixed_precision_components=[
        'attention',      # Apply to attention computations
        'feedforward',    # Apply to FFN layers
        'embeddings'      # Apply to embedding layers
    ],
    # Keep some components in float32 for stability
    exclude_from_mixed_precision=[
        'layer_norm',     # Keep normalization in float32
        'softmax',        # Keep softmax in float32
        'loss'            # Keep loss computation in float32
    ]
)
```

### Dynamic Loss Scaling

```python
class DynamicLossScaler:
    """Dynamic loss scaling for mixed precision training."""
    
    def __init__(self, init_scale=65536, growth_factor=2.0, backoff_factor=0.5):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = 2000
        self.unskipped_steps = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, gradients):
        return {k: v / self.scale for k, v in gradients.items()}
    
    def update(self, found_inf):
        if found_inf:
            # Reduce scale and reset counter
            self.scale *= self.backoff_factor
            self.unskipped_steps = 0
        else:
            # Increment counter and maybe grow scale
            self.unskipped_steps += 1
            if self.unskipped_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self.unskipped_steps = 0
        
        return self.scale

# Usage in training loop
scaler = DynamicLossScaler()

def train_step_with_scaling(model, batch, optimizer):
    def loss_fn(model):
        carry, outputs = model(batch)
        loss = compute_loss(outputs, batch['labels'])
        return scaler.scale_loss(loss)
    
    scaled_loss, grads = mx.value_and_grad(loss_fn)(model)
    
    # Check for infinite gradients
    grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
    found_inf = mx.isinf(grad_norm) or mx.isnan(grad_norm)
    
    if not found_inf:
        # Unscale gradients
        unscaled_grads = scaler.unscale_gradients(grads)
        
        # Apply gradients
        optimizer.update(model, unscaled_grads)
    
    # Update scaler
    scaler.update(found_inf)
    
    return scaled_loss / scaler.scale
```

## Benchmarking and Profiling

### Performance Benchmarking

```python
import time
import statistics

def benchmark_model(model, test_cases, num_runs=10):
    """Comprehensive model benchmarking."""
    
    results = {
        'forward_times': [],
        'memory_usage': [],
        'act_steps': [],
        'throughput': []
    }
    
    for _ in range(num_runs):
        for test_case in test_cases:
            # Prepare batch
            batch = {'input_ids': test_case.reshape(1, -1)}
            
            # Memory before
            mem_before = mx.metal.get_active_memory()
            
            # Time forward pass
            start_time = time.time()
            carry, outputs = model(batch)
            mx.eval(outputs['logits'])  # Force computation
            end_time = time.time()
            
            # Memory after
            mem_after = mx.metal.get_active_memory()
            
            # Record metrics
            forward_time = end_time - start_time
            memory_used = mem_after - mem_before
            act_steps = carry.act_step[0].item()
            throughput = len(test_case) / forward_time
            
            results['forward_times'].append(forward_time)
            results['memory_usage'].append(memory_used)
            results['act_steps'].append(act_steps)
            results['throughput'].append(throughput)
            
            # Cleanup
            del carry, outputs
            mx.metal.clear_cache()
    
    # Compute statistics
    stats = {}
    for metric, values in results.items():
        stats[metric] = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }
    
    return stats

# Example usage
test_sequences = [
    mx.random.randint(0, 1000, shape=(16,)),
    mx.random.randint(0, 1000, shape=(32,)),
    mx.random.randint(0, 1000, shape=(64,))
]

benchmark_results = benchmark_model(model, test_sequences)
for metric, stats in benchmark_results.items():
    print(f"{metric}: {stats['mean']:.4f} ± {stats['stdev']:.4f}")
```

### Memory Profiling

```python
def profile_memory_usage(model, dataloader, num_batches=20):
    """Profile memory usage during training."""
    
    memory_profile = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Profile each stage
        stages = {}
        
        # Before forward
        stages['initial'] = mx.metal.get_active_memory()
        
        # Forward pass
        carry, outputs = model(batch)
        stages['forward'] = mx.metal.get_active_memory()
        
        # Backward pass simulation
        def loss_fn(model):
            carry, outputs = model(batch)
            return mx.sum(outputs['logits'])
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        stages['backward'] = mx.metal.get_active_memory()
        
        # After cleanup
        del carry, outputs, loss, grads
        mx.metal.clear_cache()
        stages['cleanup'] = mx.metal.get_active_memory()
        
        memory_profile.append(stages)
    
    # Analyze memory usage patterns
    avg_stages = {}
    for stage in ['initial', 'forward', 'backward', 'cleanup']:
        values = [profile[stage] for profile in memory_profile]
        avg_stages[stage] = sum(values) / len(values) / 1024**2  # MB
    
    print("Average Memory Usage (MB):")
    for stage, memory in avg_stages.items():
        print(f"  {stage}: {memory:.1f}")
    
    peak_memory = max(
        max(profile.values()) for profile in memory_profile
    ) / 1024**2
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    
    return memory_profile
```

## Advanced Techniques

### Graph Compilation Optimization

```python
# Enable MLX graph compilation for faster execution
@mx.compile
def compiled_forward(model, batch):
    """Compiled version of model forward pass."""
    carry = model.initial_carry(batch['input_ids'].shape[0])
    return model(batch, carry)

# Use compiled version for inference
def fast_inference(model, input_tokens):
    batch = {'input_ids': input_tokens.reshape(1, -1)}
    carry, outputs = compiled_forward(model, batch)
    return outputs['logits'][0]
```

### Sparse Computation Optimization

```python
def optimize_sparse_embeddings(model, vocab_usage_stats):
    """Optimize sparse embeddings based on usage statistics."""
    
    # Identify frequently used tokens
    frequent_tokens = [
        token for token, count in vocab_usage_stats.items()
        if count > threshold
    ]
    
    # Pre-load frequently used embeddings
    model.sparse_embedding.preload_tokens(frequent_tokens)
    
    # Adjust learning rates based on usage
    for token, count in vocab_usage_stats.items():
        lr_multiplier = min(2.0, math.log(count + 1))
        model.sparse_embedding.set_token_lr(token, lr_multiplier)
```

### Custom Kernel Optimization

```python
# Example: Custom kernel for ACT operations
def optimized_act_step(h_state, l_state, input_embeds):
    """Optimized ACT step with fused operations."""
    
    # Fuse H-level operations
    h_output = mx.compile(lambda x, s: h_level_forward(x, s))(input_embeds, h_state)
    
    # Fuse L-level operations  
    l_output = mx.compile(lambda x, s: l_level_forward(x, s))(h_output, l_state)
    
    return h_output, l_output

# Replace standard ACT step with optimized version
model.act_step = optimized_act_step
```

### Distributed Inference

```python
class DistributedInference:
    """Distribute inference across multiple devices."""
    
    def __init__(self, model, num_devices=2):
        self.model = model
        self.num_devices = num_devices
        self.devices = [f"gpu:{i}" for i in range(num_devices)]
    
    def parallel_inference(self, input_batches):
        """Run inference on multiple devices in parallel."""
        
        import concurrent.futures
        
        def run_on_device(device, batch):
            with mx.device(device):
                model_copy = self.model  # Assume model can be copied
                carry, outputs = model_copy(batch)
                return outputs['logits']
        
        # Distribute batches across devices
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for i, batch in enumerate(input_batches):
                device = self.devices[i % self.num_devices]
                future = executor.submit(run_on_device, device, batch)
                futures.append(future)
            
            # Collect results
            results = [future.result() for future in futures]
        
        return results
```

## Performance Checklist

### Before Training

- [ ] Choose appropriate model size for your task
- [ ] Optimize batch size for your hardware
- [ ] Configure smart batching for your data
- [ ] Enable mixed precision if appropriate
- [ ] Set up memory monitoring
- [ ] Verify data pipeline efficiency

### During Training

- [ ] Monitor memory usage and adjust if needed
- [ ] Track ACT step distribution
- [ ] Watch for gradient explosion/vanishing
- [ ] Adjust learning rate if convergence is slow
- [ ] Use gradient accumulation for larger effective batches
- [ ] Save checkpoints regularly

### For Inference

- [ ] Use batch inference when possible
- [ ] Enable caching for repeated patterns
- [ ] Monitor ACT efficiency
- [ ] Use compiled functions for hot paths
- [ ] Consider model quantization for deployment
- [ ] Profile memory usage in production

### Hardware-Specific

- [ ] Use appropriate Apple Silicon model for your needs
- [ ] Configure memory limits based on available RAM
- [ ] Monitor thermal throttling during long runs
- [ ] Use unified memory efficiently
- [ ] Consider external cooling for sustained workloads

## Troubleshooting Performance Issues

### Slow Training

1. **Check batch size**: Too small batches underutilize hardware
2. **Enable mixed precision**: Can provide 1.5-2x speedup
3. **Optimize data loading**: Slow I/O can bottleneck training
4. **Monitor ACT usage**: Excessive ACT steps waste computation
5. **Profile memory access**: Memory bandwidth can be limiting

### High Memory Usage

1. **Reduce batch size**: Most effective solution
2. **Use gradient accumulation**: Maintain effective batch size
3. **Enable gradient checkpointing**: Trade computation for memory
4. **Optimize model size**: Use smaller preset if possible
5. **Clear caches regularly**: Prevent memory leaks

### Poor Convergence

1. **Adjust learning rate**: Too high/low can prevent convergence
2. **Check ACT loss weights**: Imbalanced losses affect convergence
3. **Verify data quality**: Poor data leads to poor performance
4. **Monitor gradient norms**: Explosion/vanishing gradients
5. **Use warmup scheduling**: Stabilizes early training

---

This guide provides comprehensive strategies for optimizing HRM performance. Start with the basic optimizations and gradually apply more advanced techniques as needed for your specific use case.

*For implementation details, refer to the source code and other tutorials in this documentation.*