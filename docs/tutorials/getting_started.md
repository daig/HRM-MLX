# Getting Started with HRM MLX

Welcome to the Hierarchical Reasoning Model (HRM) implementation in MLX! This tutorial will guide you through setting up and using HRM for your reasoning tasks.

## Table of Contents

1. [What is HRM?](#what-is-hrm)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding the Architecture](#understanding-the-architecture)
5. [Basic Usage Examples](#basic-usage-examples)
6. [Training Your Own Model](#training-your-own-model)
7. [Working with Checkpoints](#working-with-checkpoints)
8. [Performance Tips](#performance-tips)
9. [Next Steps](#next-steps)

## What is HRM?

The Hierarchical Reasoning Model (HRM) is a novel recurrent neural network architecture designed for complex reasoning tasks. Key features:

- **27M parameters** - Compact yet powerful architecture
- **Hierarchical Structure** - Separate high-level planning (H-level) and low-level computation (L-level)
- **Adaptive Computation Time (ACT)** - Dynamic reasoning steps based on problem complexity
- **Excellent Performance** - Achieves strong results on ARC, Sudoku, and other reasoning benchmarks

### Why MLX?

MLX is Apple's machine learning framework optimized for Apple Silicon. Benefits:

- **Native Apple Silicon Performance** - Optimized for M1/M2/M3 chips
- **Memory Efficiency** - Unified memory architecture support
- **Easy to Use** - Python API similar to PyTorch/JAX
- **Fast Compilation** - Just-in-time compilation for optimal performance

## Installation

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.8+**
- **MLX framework**

### Step 1: Install MLX

```bash
pip install mlx
```

### Step 2: Install HRM MLX

```bash
# Clone the repository
git clone https://github.com/your-org/hrm-mlx.git
cd hrm-mlx

# Install in development mode
pip install -e .
```

### Step 3: Verify Installation

```python
import mlx.core as mx
from mlx_hrm import create_hrm

# Create a simple model
model = create_hrm('tiny')
print(f"Model created with {model.num_parameters:,} parameters")
```

## Quick Start

Here's a minimal example to get you started:

```python
import mlx.core as mx
from mlx_hrm import create_hrm

# 1. Create a model
model = create_hrm('small')  # 27M parameter model from the paper

# 2. Prepare input
input_tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # Batch size 1

# 3. Run inference
batch = {'input_ids': input_tokens}
carry, outputs = model(batch)

# 4. Get predictions
logits = outputs['logits']  # Next token predictions
next_token = mx.argmax(logits[0, -1])  # Most likely next token

print(f"Input: {input_tokens[0].tolist()}")
print(f"Predicted next token: {next_token.item()}")
print(f"ACT steps used: {carry.act_step[0].item()}")
```

## Understanding the Architecture

### Model Presets

HRM MLX comes with several pre-configured model sizes:

| Preset | Parameters | Hidden Size | Heads | Layers | Use Case |
|--------|------------|-------------|-------|---------|----------|
| `tiny` | ~7M | 256 | 4 | 2+2 | Development, testing |
| `small` | ~27M | 512 | 8 | 4+4 | Paper reproduction |
| `base` | ~100M | 768 | 12 | 6+6 | Larger tasks |
| `large` | ~200M | 1024 | 16 | 8+8 | Production use |

### The ACT Mechanism

Adaptive Computation Time allows the model to use more reasoning steps for harder problems:

```python
# Example: Compare ACT usage on different inputs
model = create_hrm('tiny')

# Simple pattern
simple_input = mx.array([[1, 2, 3, 4]])
carry, _ = model({'input_ids': simple_input})
print(f"Simple pattern used {carry.act_step[0].item()} ACT steps")

# Complex pattern  
complex_input = mx.array([[1, 1, 2, 3, 5, 8, 13]])  # Fibonacci
carry, _ = model({'input_ids': complex_input})
print(f"Complex pattern used {carry.act_step[0].item()} ACT steps")
```

### Hierarchical Structure

HRM uses two levels of processing:

- **H-level (Planning)**: High-level reasoning and strategy
- **L-level (Computation)**: Detailed computation and implementation

This separation allows the model to plan before acting, similar to human problem-solving.

## Basic Usage Examples

### 1. Single Sequence Inference

```python
from mlx_hrm import create_hrm

model = create_hrm('small')

# Process a single sequence
tokens = mx.array([10, 20, 30, 40, 50])
logits = model.forward_single(tokens)

print(f"Input shape: {tokens.shape}")
print(f"Output shape: {logits.shape}")
print(f"Next token prediction: {mx.argmax(logits[-1]).item()}")
```

### 2. Text Generation

```python
# Generate text autoregressively
prompt = mx.array([1, 2, 3, 4])

# Basic generation
generated = model.generate(prompt, max_length=20)
print(f"Generated: {generated.tolist()}")

# Generation with temperature control
focused = model.generate(prompt, max_length=20, temperature=0.5)
random = model.generate(prompt, max_length=20, temperature=2.0)

print(f"Focused (temp=0.5): {focused.tolist()}")
print(f"Random (temp=2.0): {random.tolist()}")
```

### 3. Batch Processing

```python
# Process multiple sequences at once
batch = {
    'input_ids': mx.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ])
}

carry, outputs = model(batch)

print(f"Batch size: {batch['input_ids'].shape[0]}")
print(f"ACT steps per sequence: {carry.act_step.tolist()}")
print(f"Halted sequences: {carry.halted.tolist()}")
```

### 4. Model Information

```python
from mlx_hrm import list_available_presets, get_model_info

# List all available presets
presets = list_available_presets()
print(f"Available presets: {presets}")

# Get detailed info for each preset
for preset in presets:
    info = get_model_info(preset)
    print(f"{preset}: {info['estimated_params_millions']}M parameters")
```

## Training Your Own Model

### Basic Training Setup

```python
from mlx_hrm.training.trainer import HRMTrainer
from mlx_hrm.training.act_loss import ACTLossHead
from mlx_hrm.data.dataset import EnhancedPuzzleDataset, SmartDataLoader

# 1. Create model
model = create_hrm('tiny')
loss_head = ACTLossHead(model, loss_type='stablemax', vocab_size=1000)

# 2. Prepare data (mock example)
train_data = [
    {
        'input_ids': mx.random.randint(0, 1000, shape=(32,)),
        'labels': mx.random.randint(0, 1000, shape=(32,)),
        'puzzle_type': 0
    }
    for _ in range(1000)
]

dataset = EnhancedPuzzleDataset(train_data, mode='train')
dataloader = SmartDataLoader(dataset, batch_size=8)

# 3. Create trainer
trainer = HRMTrainer(
    model=loss_head,
    train_dataloader=dataloader,
    optimizer_config={'learning_rate': 1e-4, 'weight_decay': 0.1},
    max_training_steps=1000
)

# 4. Train
trainer.train()
```

For a complete training example, see `examples/training_example.py`.

### Data Format

HRM expects data in this format:

```python
example = {
    'input_ids': mx.array([...]),  # Input token sequence
    'labels': mx.array([...]),     # Target token sequence (for training)
    'puzzle_type': int,            # Optional: puzzle type ID for batching
    'sample_id': int               # Optional: unique sample identifier
}
```

### Loss Functions

HRM supports multiple loss functions:

- **`stablemax`**: Novel stable cross-entropy (recommended)
- **`softmax`**: Standard cross-entropy baseline

```python
# Use stablemax loss (recommended)
loss_head = ACTLossHead(model, loss_type='stablemax')

# Use standard softmax
loss_head = ACTLossHead(model, loss_type='softmax')
```

## Working with Checkpoints

### Saving Checkpoints

```python
from mlx_hrm.utils import save_checkpoint

# Save model with metadata
save_checkpoint(
    model,
    'my_checkpoint.pkl',
    metadata={
        'training_step': 1000,
        'loss': 0.234,
        'dataset': 'ARC-1',
        'accuracy': 0.456
    }
)
```

### Loading Checkpoints

```python
from mlx_hrm import create_hrm_from_checkpoint

# Load model from checkpoint
model = create_hrm_from_checkpoint('my_checkpoint.pkl')

# Or load into existing model
model = create_hrm('small')
model = load_checkpoint(model, 'my_checkpoint.pkl')
```

### Converting PyTorch Checkpoints

```python
# Convert from PyTorch HRM checkpoint
from scripts.convert_checkpoint import convert_pytorch_to_mlx

convert_pytorch_to_mlx(
    pytorch_path='../HRM/checkpoints/model.pt',
    mlx_path='converted_model.pkl'
)

# Load converted model
model = create_hrm_from_checkpoint('converted_model.pkl')
```

## Performance Tips

### 1. Use Appropriate Model Size

- Start with `tiny` for development
- Use `small` for paper reproduction
- Scale up to `base`/`large` for production

### 2. Optimize Batch Size

```python
# Find optimal batch size for your hardware
for batch_size in [1, 2, 4, 8, 16, 32]:
    try:
        # Test batch
        batch = {'input_ids': mx.random.randint(0, 1000, shape=(batch_size, 32))}
        carry, outputs = model(batch)
        print(f"Batch size {batch_size}: OK")
    except Exception as e:
        print(f"Batch size {batch_size}: Failed - {e}")
        break
```

### 3. Use Mixed Precision

```python
# Enable mixed precision training
trainer = HRMTrainer(
    model=model,
    train_dataloader=dataloader,
    use_mixed_precision=True,
    mixed_precision_dtype='float16',
    mixed_precision_components=['attention', 'feedforward']
)
```

### 4. Smart Batching

```python
# Use smart batching for better training efficiency
dataset = EnhancedPuzzleDataset(
    data=train_data,
    mode='train',
    puzzle_type_key='puzzle_type'  # Group similar puzzles
)

dataloader = SmartDataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True  # Enable smart sampling
)
```

### 5. Monitor ACT Usage

```python
# Monitor ACT step distribution during inference
act_steps = []
for batch in test_data:
    carry, _ = model(batch)
    act_steps.extend(carry.act_step.tolist())

print(f"Average ACT steps: {sum(act_steps) / len(act_steps):.1f}")
print(f"Max ACT steps: {max(act_steps)}")
print(f"Min ACT steps: {min(act_steps)}")
```

## Next Steps

### 1. Explore Examples

- `examples/basic_usage.py` - Basic model usage
- `examples/training_example.py` - Complete training pipeline
- `examples/inference_demo.py` - Inference and evaluation

### 2. Read Tutorials

- `docs/tutorials/act_mechanism.md` - Deep dive into ACT
- `docs/tutorials/custom_datasets.md` - Creating custom datasets
- `docs/tutorials/performance_optimization.md` - Advanced optimization

### 3. Experiment with Tasks

Try HRM on different reasoning tasks:

- **ARC puzzles** - Abstract reasoning
- **Sudoku** - Constraint satisfaction
- **Sequence completion** - Pattern recognition
- **Logic puzzles** - Deductive reasoning

### 4. Join the Community

- Report issues on GitHub
- Share your results and use cases
- Contribute improvements and extensions

## Common Issues and Solutions

### Issue: Out of Memory

```python
# Solution: Reduce batch size or use gradient accumulation
trainer = HRMTrainer(
    model=model,
    train_dataloader=dataloader,
    gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
    batch_size=2  # Smaller actual batch size
)
```

### Issue: Slow Training

```python
# Solution: Enable mixed precision and smart batching
trainer = HRMTrainer(
    model=model,
    train_dataloader=smart_dataloader,  # Use SmartDataLoader
    use_mixed_precision=True,
    mixed_precision_dtype='float16'
)
```

### Issue: Poor Convergence

```python
# Solution: Adjust learning rate and use proper scheduler
trainer = HRMTrainer(
    model=model,
    train_dataloader=dataloader,
    optimizer_config={
        'learning_rate': 1e-4,  # Try different learning rates
        'weight_decay': 0.1,
        'warmup_steps': 1000    # Add warmup
    }
)
```

## Conclusion

You now have a solid foundation for using HRM MLX! The key points to remember:

1. **Start Simple** - Use the `tiny` preset for development
2. **Understand ACT** - The adaptive computation is what makes HRM powerful
3. **Use Smart Batching** - Group similar puzzles for better training
4. **Monitor Performance** - Watch ACT usage and training metrics
5. **Experiment** - Try different tasks and configurations

Happy reasoning! 🧠✨

---

*For more advanced topics, check out the other tutorials in the `docs/tutorials/` directory.*