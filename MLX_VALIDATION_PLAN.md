# MLX Validation & Testing Implementation Plan

## Overview
This document provides the detailed implementation plan for validating the MLX HRM implementation against the PyTorch reference. This corresponds to Phase 8 of the master implementation plan.

### Goal
Ensure the MLX implementation matches PyTorch in terms of numerical accuracy, training dynamics, and final performance on benchmark tasks.

### Key Components
1. **Checkpoint Conversion**: Bidirectional PyTorch ↔ MLX conversion
2. **Numerical Validation**: Verify outputs match within tolerance
3. **Performance Benchmarking**: Compare speed and memory usage
4. **Accuracy Validation**: Test on standard benchmarks

## Validation Strategy

### Validation Levels
1. **Component Level**: Individual layers and modules
2. **Model Level**: Full forward/backward passes
3. **Training Level**: Loss curves and convergence
4. **Task Level**: Final accuracy on puzzles

## Implementation Components

### 1. Checkpoint Conversion
**File**: `scripts/convert_checkpoint.py`

```python
#!/usr/bin/env python3
"""Convert checkpoints between PyTorch and MLX formats."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

import torch
import mlx.core as mx

from mlx_hrm.models import create_hrm
from mlx_hrm.utils.checkpoint import save_checkpoint, load_checkpoint


# Parameter name mappings
PYTORCH_TO_MLX_MAPPING = {
    # Embeddings
    'tok_embeddings.weight': 'dense_tok_emb.embedding_weight',
    'sparse_tok_embeddings.embeddings': 'sparse_tok_emb.embeddings',
    
    # Attention layers (example for layer 0)
    'h_blocks.0.self_attn.qkv_proj.weight': 'H_module.blocks.0.self_attn.qkv_proj.weight',
    'h_blocks.0.self_attn.o_proj.weight': 'H_module.blocks.0.self_attn.o_proj.weight',
    'h_blocks.0.mlp.gate_up_proj.weight': 'H_module.blocks.0.mlp.gate_up_proj.weight',
    'h_blocks.0.mlp.down_proj.weight': 'H_module.blocks.0.mlp.down_proj.weight',
    
    # Output heads
    'lm_head.weight': 'lm_head.weight',
    'q_head.weight': 'q_head.weight',
    
    # RoPE (if stored)
    'rope.cos_cached': 'rope.cos_cached',
    'rope.sin_cached': 'rope.sin_cached',
}


def build_full_mapping(config: Dict) -> Dict[str, str]:
    """Build complete parameter mapping based on model config."""
    mapping = {}
    
    # Add base mappings
    mapping.update(PYTORCH_TO_MLX_MAPPING)
    
    # Add layer-specific mappings
    for module in ['H', 'L']:
        n_layers = config[f'{module}_layers']
        
        for i in range(n_layers):
            # Attention
            mapping[f'{module.lower()}_blocks.{i}.self_attn.qkv_proj.weight'] = \
                f'{module}_module.blocks.{i}.self_attn.qkv_proj.weight'
            mapping[f'{module.lower()}_blocks.{i}.self_attn.o_proj.weight'] = \
                f'{module}_module.blocks.{i}.self_attn.o_proj.weight'
            
            # MLP
            mapping[f'{module.lower()}_blocks.{i}.mlp.gate_up_proj.weight'] = \
                f'{module}_module.blocks.{i}.mlp.gate_up_proj.weight'
            mapping[f'{module.lower()}_blocks.{i}.mlp.down_proj.weight'] = \
                f'{module}_module.blocks.{i}.mlp.down_proj.weight'
    
    return mapping


def pytorch_to_mlx(
    pytorch_path: str,
    output_path: str,
    config_path: Optional[str] = None
) -> None:
    """
    Convert PyTorch checkpoint to MLX format.
    
    Args:
        pytorch_path: Path to PyTorch checkpoint
        output_path: Output path for MLX checkpoint
        config_path: Path to model config (if not in checkpoint)
    """
    print(f"Loading PyTorch checkpoint from {pytorch_path}")
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("No config found in checkpoint or provided")
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Build parameter mapping
    mapping = build_full_mapping(config)
    
    # Convert parameters
    mlx_state = {}
    converted_count = 0
    
    for pt_name, pt_tensor in state_dict.items():
        if pt_name in mapping:
            mlx_name = mapping[pt_name]
            
            # Convert tensor
            numpy_array = pt_tensor.detach().numpy()
            mlx_array = mx.array(numpy_array)
            
            # Handle nested structure
            parts = mlx_name.split('.')
            current = mlx_state
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = mlx_array
            converted_count += 1
            
            print(f"  Converted {pt_name} -> {mlx_name}")
        else:
            print(f"  Warning: No mapping for {pt_name}")
    
    print(f"\nConverted {converted_count}/{len(state_dict)} parameters")
    
    # Create MLX model to verify
    from mlx_hrm.modules.act import HRMConfig
    mlx_config = HRMConfig(**config)
    model = create_hrm(mlx_config)
    
    # Save checkpoint
    metadata = {
        'source': 'pytorch',
        'pytorch_checkpoint': pytorch_path,
        'conversion_date': str(Path(pytorch_path).stat().st_mtime)
    }
    
    save_checkpoint(model, output_path, metadata)
    print(f"\nSaved MLX checkpoint to {output_path}")


def mlx_to_pytorch(
    mlx_path: str,
    output_path: str,
    model_only: bool = False
) -> None:
    """
    Convert MLX checkpoint to PyTorch format.
    
    Args:
        mlx_path: Path to MLX checkpoint
        output_path: Output path for PyTorch checkpoint
        model_only: Save only model weights (no optimizer state)
    """
    print(f"Loading MLX checkpoint from {mlx_path}")
    
    # Load MLX checkpoint
    weights, config, metadata = load_checkpoint(mlx_path)
    
    # Build reverse mapping
    mapping = build_full_mapping(config._asdict())
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Convert parameters
    pytorch_state = {}
    converted_count = 0
    
    def flatten_dict(d, parent_key=''):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items
    
    flat_weights = dict(flatten_dict(weights))
    
    for mlx_name, mlx_array in flat_weights.items():
        if mlx_name in reverse_mapping:
            pt_name = reverse_mapping[mlx_name]
            
            # Convert to PyTorch
            numpy_array = np.array(mlx_array)
            pt_tensor = torch.from_numpy(numpy_array)
            
            pytorch_state[pt_name] = pt_tensor
            converted_count += 1
            
            print(f"  Converted {mlx_name} -> {pt_name}")
        else:
            print(f"  Warning: No reverse mapping for {mlx_name}")
    
    print(f"\nConverted {converted_count}/{len(flat_weights)} parameters")
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': pytorch_state,
        'config': config._asdict(),
        'metadata': metadata
    }
    
    # Save
    torch.save(checkpoint, output_path)
    print(f"\nSaved PyTorch checkpoint to {output_path}")


def verify_conversion(
    original_path: str,
    converted_path: str,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that conversion preserved parameter values.
    
    Args:
        original_path: Original checkpoint path
        converted_path: Converted checkpoint path
        tolerance: Maximum allowed difference
        
    Returns:
        True if all parameters match within tolerance
    """
    # Determine formats
    if original_path.endswith('.pt') or original_path.endswith('.pth'):
        # PyTorch -> MLX
        original = torch.load(original_path, map_location='cpu')
        if 'model_state_dict' in original:
            original = original['model_state_dict']
        
        weights, _, _ = load_checkpoint(converted_path)
        
        # Compare each parameter
        # Implementation depends on exact structure
        
    else:
        # MLX -> PyTorch
        weights, _, _ = load_checkpoint(original_path)
        
        converted = torch.load(converted_path, map_location='cpu')
        if 'model_state_dict' in converted:
            converted = converted['model_state_dict']
        
        # Compare each parameter
        # Implementation depends on exact structure
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert checkpoints between PyTorch and MLX')
    parser.add_argument('input', help='Input checkpoint path')
    parser.add_argument('output', help='Output checkpoint path')
    parser.add_argument('--config', help='Model config path (if not in checkpoint)')
    parser.add_argument('--verify', action='store_true', help='Verify conversion')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Verification tolerance')
    
    args = parser.parse_args()
    
    # Determine conversion direction
    if args.input.endswith('.pt') or args.input.endswith('.pth'):
        pytorch_to_mlx(args.input, args.output, args.config)
    else:
        mlx_to_pytorch(args.input, args.output)
    
    # Verify if requested
    if args.verify:
        if verify_conversion(args.input, args.output, args.tolerance):
            print("\nVerification passed!")
        else:
            print("\nVerification failed!")


if __name__ == '__main__':
    main()
```

### 2. Numerical Validation
**File**: `tests/validation/test_numerical_parity.py`

```python
"""Test numerical parity between PyTorch and MLX implementations."""

import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Tuple
import pytest

from mlx_hrm.models import create_hrm as create_mlx_hrm
# Assume PyTorch HRM is available for comparison
from hrm_pytorch import create_hrm as create_pytorch_hrm


class NumericalValidator:
    """Validate numerical outputs between implementations."""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
    
    def compare_tensors(
        self,
        pt_tensor: torch.Tensor,
        mlx_array: mx.array,
        name: str = ""
    ) -> Tuple[bool, Dict]:
        """Compare PyTorch tensor and MLX array."""
        # Convert to numpy for comparison
        pt_numpy = pt_tensor.detach().cpu().numpy()
        mlx_numpy = np.array(mlx_array)
        
        # Compute statistics
        abs_diff = np.abs(pt_numpy - mlx_numpy)
        rel_diff = abs_diff / (np.abs(pt_numpy) + 1e-8)
        
        stats = {
            'name': name,
            'shape': pt_numpy.shape,
            'max_abs_diff': np.max(abs_diff),
            'mean_abs_diff': np.mean(abs_diff),
            'max_rel_diff': np.max(rel_diff),
            'mean_rel_diff': np.mean(rel_diff),
            'matches': np.max(abs_diff) < self.tolerance
        }
        
        return stats['matches'], stats
    
    def validate_forward_pass(
        self,
        pytorch_model,
        mlx_model,
        batch_size: int = 2,
        seq_len: int = 32
    ) -> Dict:
        """Validate forward pass outputs match."""
        # Create identical inputs
        input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len))
        
        # PyTorch forward
        pt_input = torch.tensor(input_ids_np, dtype=torch.long)
        pt_batch = {'input_ids': pt_input}
        
        with torch.no_grad():
            pt_carry = pytorch_model.initial_carry(batch_size)
            pt_new_carry, pt_outputs = pytorch_model(pt_carry, pt_batch)
        
        # MLX forward
        mlx_input = mx.array(input_ids_np)
        mlx_batch = {'input_ids': mlx_input}
        
        mlx_carry = mlx_model.initial_carry(batch_size)
        mlx_new_carry, mlx_outputs = mlx_model(mlx_carry, mlx_batch)
        
        # Compare outputs
        results = {}
        
        # Compare logits
        matches, stats = self.compare_tensors(
            pt_outputs['logits'],
            mlx_outputs['logits'],
            'logits'
        )
        results['logits'] = stats
        
        # Compare Q-values
        matches, stats = self.compare_tensors(
            pt_outputs['q_halt_logits'],
            mlx_outputs['q_halt_logits'],
            'q_halt_logits'
        )
        results['q_halt_logits'] = stats
        
        # Compare carry states
        matches, stats = self.compare_tensors(
            pt_new_carry.inner_carry.z_H,
            mlx_new_carry.inner_carry.z_H,
            'carry_z_H'
        )
        results['carry_z_H'] = stats
        
        return results
    
    def validate_gradients(
        self,
        pytorch_model,
        mlx_model,
        batch_size: int = 2,
        seq_len: int = 32
    ) -> Dict:
        """Validate gradient computation matches."""
        # Create identical inputs
        input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len))
        labels_np = np.random.randint(0, 1000, (batch_size, seq_len))
        
        # PyTorch gradients
        pt_input = torch.tensor(input_ids_np, dtype=torch.long)
        pt_labels = torch.tensor(labels_np, dtype=torch.long)
        pt_batch = {'input_ids': pt_input, 'labels': pt_labels}
        
        pytorch_model.zero_grad()
        pt_carry = pytorch_model.initial_carry(batch_size)
        pt_new_carry, pt_outputs = pytorch_model(pt_carry, pt_batch)
        
        # Simple loss for comparison
        pt_loss = pt_outputs['logits'].mean()
        pt_loss.backward()
        
        # MLX gradients
        mlx_input = mx.array(input_ids_np)
        mlx_labels = mx.array(labels_np)
        mlx_batch = {'input_ids': mlx_input, 'labels': mlx_labels}
        
        def loss_fn(model, carry, batch):
            new_carry, outputs = model(carry, batch)
            return mx.mean(outputs['logits'])
        
        mlx_carry = mlx_model.initial_carry(batch_size)
        mlx_loss, mlx_grads = mx.value_and_grad(loss_fn)(
            mlx_model, mlx_carry, mlx_batch
        )
        
        # Compare gradients
        results = {}
        
        # Example: compare a specific parameter gradient
        # This requires careful parameter matching
        
        return results


def test_component_outputs():
    """Test individual component outputs match."""
    validator = NumericalValidator(tolerance=1e-5)
    
    # Test RMSNorm
    x_np = np.random.randn(2, 32, 512).astype(np.float32)
    
    # PyTorch
    from hrm_pytorch import rms_norm as pt_rms_norm
    pt_x = torch.tensor(x_np)
    pt_out = pt_rms_norm(pt_x, 1e-5)
    
    # MLX
    from mlx_hrm.layers.normalization import rms_norm as mlx_rms_norm
    mlx_x = mx.array(x_np)
    mlx_out = mlx_rms_norm(mlx_x, 1e-5)
    
    matches, stats = validator.compare_tensors(pt_out, mlx_out, 'rms_norm')
    assert matches, f"RMSNorm outputs don't match: {stats}"


def test_model_forward_pass():
    """Test full model forward pass matches."""
    validator = NumericalValidator(tolerance=1e-4)
    
    # Create models with same config
    config = {
        'hidden_size': 128,
        'num_heads': 4,
        'H_layers': 2,
        'L_layers': 2,
        'vocab_size': 1000,
        'seq_len': 32,
        'batch_size': 2
    }
    
    pytorch_model = create_pytorch_hrm(config)
    mlx_model = create_mlx_hrm(config)
    
    # Load same weights
    # ... checkpoint loading ...
    
    # Validate
    results = validator.validate_forward_pass(pytorch_model, mlx_model)
    
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Max abs diff: {stats['max_abs_diff']:.6f}")
        print(f"  Mean abs diff: {stats['mean_abs_diff']:.6f}")
        assert stats['matches'], f"{name} outputs don't match"


def test_training_dynamics():
    """Test that training dynamics match over multiple steps."""
    # Create identical models
    # Run several training steps
    # Compare loss curves
    pass
```

### 3. Performance Benchmarking
**File**: `benchmarks/benchmark_full_model.py`

```python
"""Benchmark MLX HRM performance."""

import time
import argparse
from typing import Dict, List
import mlx.core as mx
import numpy as np

from mlx_hrm.models import create_hrm
from mlx_hrm.training.utils import compute_model_flops


class PerformanceBenchmark:
    """Comprehensive performance benchmarking."""
    
    def __init__(self, warmup_steps: int = 10, benchmark_steps: int = 100):
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps
    
    def benchmark_forward(
        self,
        model,
        batch_size: int,
        seq_len: int
    ) -> Dict:
        """Benchmark forward pass performance."""
        # Create dummy data
        batch = {
            'input_ids': mx.random.randint(0, 1000, (batch_size, seq_len))
        }
        
        carry = model.initial_carry(batch_size)
        
        # Warmup
        for _ in range(self.warmup_steps):
            carry, outputs = model(carry, batch)
        mx.eval(outputs['logits'])  # Force evaluation
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(self.benchmark_steps):
            carry, outputs = model(carry, batch)
        mx.eval(outputs['logits'])  # Force evaluation
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        total_tokens = batch_size * seq_len * self.benchmark_steps
        throughput = total_tokens / elapsed
        time_per_step = elapsed / self.benchmark_steps
        
        # Estimate FLOPs
        flops_per_step = compute_model_flops(model, batch_size, seq_len)
        tflops = (flops_per_step * self.benchmark_steps) / elapsed / 1e12
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_time': elapsed,
            'time_per_step': time_per_step,
            'throughput_tokens_per_sec': throughput,
            'estimated_tflops': tflops
        }
    
    def benchmark_training_step(
        self,
        model,
        batch_size: int,
        seq_len: int
    ) -> Dict:
        """Benchmark training step (forward + backward)."""
        from mlx_hrm.training.act_loss import ACTLossHead
        
        # Wrap with loss head
        loss_model = ACTLossHead(model)
        
        # Create dummy data
        batch = {
            'input_ids': mx.random.randint(0, 1000, (batch_size, seq_len)),
            'labels': mx.random.randint(0, 1000, (batch_size, seq_len))
        }
        
        def loss_fn(model, carry, batch):
            new_carry, loss, _, _ = model(carry, batch)
            return loss
        
        # Warmup
        for _ in range(self.warmup_steps):
            carry = loss_model.initial_carry(batch_size)
            loss, grads = mx.value_and_grad(loss_fn)(loss_model, carry, batch)
        mx.eval(loss)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(self.benchmark_steps):
            carry = loss_model.initial_carry(batch_size)
            loss, grads = mx.value_and_grad(loss_fn)(loss_model, carry, batch)
        mx.eval(loss)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_time': elapsed,
            'time_per_step': elapsed / self.benchmark_steps,
            'throughput_tokens_per_sec': (batch_size * seq_len * self.benchmark_steps) / elapsed
        }
    
    def benchmark_memory_usage(
        self,
        model,
        batch_size: int,
        seq_len: int
    ) -> Dict:
        """Benchmark memory usage."""
        # Note: MLX memory profiling is different from PyTorch
        # This is a simplified version
        
        # Create batch
        batch = {
            'input_ids': mx.random.randint(0, 1000, (batch_size, seq_len))
        }
        
        # Forward pass
        carry = model.initial_carry(batch_size)
        carry, outputs = model(carry, batch)
        mx.eval(outputs['logits'])
        
        # Estimate memory
        # Count parameters
        param_count = sum(p.size for p in model.parameters().values())
        param_memory_mb = param_count * 4 / 1024 / 1024  # float32
        
        # Estimate activation memory (rough)
        hidden_size = model.config.hidden_size
        num_layers = model.config.H_layers + model.config.L_layers
        activation_memory_mb = (
            batch_size * seq_len * hidden_size * num_layers * 4 / 1024 / 1024
        )
        
        return {
            'param_count': param_count,
            'param_memory_mb': param_memory_mb,
            'estimated_activation_memory_mb': activation_memory_mb,
            'estimated_total_memory_mb': param_memory_mb + activation_memory_mb
        }


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on different model sizes."""
    benchmark = PerformanceBenchmark()
    
    configs = [
        ('tiny', 2, 128),
        ('small', 32, 512),
        ('base', 16, 1024)
    ]
    
    results = []
    
    for model_size, batch_size, seq_len in configs:
        print(f"\nBenchmarking {model_size} model...")
        
        model = create_hrm(model_size)
        
        # Forward pass
        fwd_results = benchmark.benchmark_forward(model, batch_size, seq_len)
        print(f"  Forward throughput: {fwd_results['throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Estimated TFLOPS: {fwd_results['estimated_tflops']:.2f}")
        
        # Training step
        train_results = benchmark.benchmark_training_step(model, batch_size, seq_len)
        print(f"  Training throughput: {train_results['throughput_tokens_per_sec']:.1f} tokens/sec")
        
        # Memory
        mem_results = benchmark.benchmark_memory_usage(model, batch_size, seq_len)
        print(f"  Parameter memory: {mem_results['param_memory_mb']:.1f} MB")
        print(f"  Total memory: {mem_results['estimated_total_memory_mb']:.1f} MB")
        
        results.append({
            'model_size': model_size,
            'forward': fwd_results,
            'training': train_results,
            'memory': mem_results
        })
    
    return results


if __name__ == '__main__':
    results = run_comprehensive_benchmark()
    
    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
```

### 4. Accuracy Validation
**File**: `tests/validation/test_puzzle_accuracy.py`

```python
"""Validate model accuracy on puzzle benchmarks."""

import mlx.core as mx
import numpy as np
from typing import Dict, List
from pathlib import Path

from mlx_hrm.models import create_hrm
from mlx_hrm.data.dataset import PuzzleDataset, DataLoader


class AccuracyValidator:
    """Validate model accuracy on benchmark tasks."""
    
    def __init__(self, model, device='gpu'):
        self.model = model
        self.device = device
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        puzzle_type: str,
        batch_size: int = 32
    ) -> Dict:
        """Evaluate model on a puzzle dataset."""
        # Load dataset
        dataset = PuzzleDataset(dataset_path, puzzle_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation metrics
        total_correct = 0
        total_sequences = 0
        per_puzzle_accuracy = {}
        
        self.model.eval()
        
        for batch in dataloader:
            batch_size = batch['input_ids'].shape[0]
            carry = self.model.initial_carry(batch_size)
            
            # Forward pass
            new_carry, outputs = self.model(carry, batch)
            
            # Get predictions
            predictions = mx.argmax(outputs['logits'], axis=-1)
            labels = batch['labels']
            
            # Compute accuracy
            mask = labels != -100  # Ignore padding
            correct = (predictions == labels) & mask
            
            # Sequence-level accuracy
            seq_correct = mx.all(correct | ~mask, axis=1)
            total_correct += mx.sum(seq_correct).item()
            total_sequences += batch_size
            
            # Per-puzzle accuracy
            if 'puzzle_ids' in batch:
                puzzle_ids = batch['puzzle_ids']
                for i in range(batch_size):
                    pid = puzzle_ids[i].item()
                    if pid not in per_puzzle_accuracy:
                        per_puzzle_accuracy[pid] = {'correct': 0, 'total': 0}
                    
                    per_puzzle_accuracy[pid]['total'] += 1
                    if seq_correct[i]:
                        per_puzzle_accuracy[pid]['correct'] += 1
        
        # Compute final metrics
        overall_accuracy = total_correct / total_sequences
        
        # Per-puzzle accuracies
        puzzle_accuracies = {}
        for pid, stats in per_puzzle_accuracy.items():
            puzzle_accuracies[pid] = stats['correct'] / stats['total']
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_sequences': total_sequences,
            'total_correct': total_correct,
            'per_puzzle_accuracy': puzzle_accuracies,
            'mean_puzzle_accuracy': np.mean(list(puzzle_accuracies.values()))
        }
    
    def validate_against_baseline(
        self,
        dataset_path: str,
        puzzle_type: str,
        expected_accuracy: float,
        tolerance: float = 0.02
    ) -> bool:
        """Validate that model meets expected accuracy."""
        results = self.evaluate_dataset(dataset_path, puzzle_type)
        
        print(f"\nResults on {puzzle_type}:")
        print(f"  Overall accuracy: {results['overall_accuracy']:.4f}")
        print(f"  Expected accuracy: {expected_accuracy:.4f}")
        print(f"  Difference: {abs(results['overall_accuracy'] - expected_accuracy):.4f}")
        
        return abs(results['overall_accuracy'] - expected_accuracy) < tolerance


def test_arc_accuracy():
    """Test accuracy on ARC-1 dataset."""
    model = create_hrm('small')
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_small_arc.mlx'
    model.load_checkpoint(checkpoint_path)
    
    validator = AccuracyValidator(model)
    
    # Expected accuracy from paper: ~42%
    success = validator.validate_against_baseline(
        'data/arc-1',
        'arc',
        expected_accuracy=0.42,
        tolerance=0.02
    )
    
    assert success, "ARC accuracy below expected"


def test_sudoku_accuracy():
    """Test accuracy on Sudoku dataset."""
    model = create_hrm('small')
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_small_sudoku.mlx'
    model.load_checkpoint(checkpoint_path)
    
    validator = AccuracyValidator(model)
    
    # Expected accuracy from paper: ~98%
    success = validator.validate_against_baseline(
        'data/sudoku-hard',
        'sudoku',
        expected_accuracy=0.98,
        tolerance=0.01
    )
    
    assert success, "Sudoku accuracy below expected"


def test_maze_accuracy():
    """Test accuracy on maze navigation."""
    model = create_hrm('small')
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_small_maze.mlx'
    model.load_checkpoint(checkpoint_path)
    
    validator = AccuracyValidator(model)
    
    # Expected accuracy from paper: ~95%
    success = validator.validate_against_baseline(
        'data/maze-medium',
        'maze',
        expected_accuracy=0.95,
        tolerance=0.02
    )
    
    assert success, "Maze accuracy below expected"
```

## Validation Pipeline

### Step 1: Component Validation
1. Test each layer type in isolation
2. Compare outputs with PyTorch
3. Verify gradients match

### Step 2: Model Validation
1. Convert a trained PyTorch checkpoint
2. Run inference on test examples
3. Compare outputs numerically

### Step 3: Training Validation
1. Train small model from scratch
2. Compare loss curves with PyTorch
3. Verify convergence behavior

### Step 4: Benchmark Validation
1. Run on standard benchmarks
2. Compare accuracy with published results
3. Ensure within expected range

## Success Criteria

### Numerical Accuracy
- ✅ Forward pass: < 1e-4 relative error
- ✅ Gradients: < 1e-3 relative error
- ✅ Loss values: < 1e-4 absolute error

### Performance Targets
- ✅ Inference: ≥ 80% of PyTorch speed
- ✅ Training: ≥ 70% of PyTorch speed
- ✅ Memory: ≤ 110% of PyTorch usage

### Task Accuracy
- ✅ ARC-1: 40-44% (paper: 42%)
- ✅ Sudoku Hard: 96-99% (paper: 98%)
- ✅ Maze Medium: 93-97% (paper: 95%)

## Implementation Timeline

### Day 1: Checkpoint Conversion (4-5 hours)
- [ ] Implement bidirectional converter
- [ ] Test on small model
- [ ] Verify parameter mapping
- [ ] Handle edge cases

### Day 2: Numerical Validation (4-5 hours)
- [ ] Component-level tests
- [ ] Full model comparison
- [ ] Gradient validation
- [ ] Statistical analysis

### Day 3: Performance & Accuracy (4-5 hours)
- [ ] Run benchmarks
- [ ] Test on puzzle datasets
- [ ] Compare with baselines
- [ ] Generate report

## Deliverables

1. **Conversion Script**: Reliable checkpoint converter
2. **Validation Suite**: Comprehensive numerical tests
3. **Benchmark Results**: Performance comparison
4. **Accuracy Report**: Results on standard benchmarks
5. **Validation Certificate**: Summary of all tests passed

## Next Steps

After validation:
1. Move to Phase 9: Documentation & Examples
2. Prepare for public release
3. Create migration guide for PyTorch users