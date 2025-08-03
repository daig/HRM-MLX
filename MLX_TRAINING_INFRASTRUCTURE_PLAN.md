# MLX Training Infrastructure Implementation Plan

## Overview
This document provides the detailed implementation plan for building a complete training infrastructure for HRM in MLX. This corresponds to Phase 7 of the master implementation plan.

### Goal
Create a robust, efficient training pipeline that includes custom optimizers, training loops, data loading, and checkpoint management - all optimized for MLX and Apple Silicon.

### Key Components
1. **Adam-atan2 Optimizer**: Special optimizer for sparse embeddings
2. **Training Loop**: Efficient training with MLX optimizations
3. **Data Pipeline**: Fast data loading for puzzle datasets
4. **Training Utilities**: Learning rate scheduling, gradient clipping, etc.

## Architecture Overview

### Training Pipeline Flow
```
DataLoader
    ↓ (batches)
HRM Model with ACTLossHead
    ↓ (loss, gradients)
Optimizer (Adam for dense, Adam-atan2 for sparse)
    ↓ (parameter updates)
Metrics Tracking & Checkpointing
    ↓ (saved models)
Validation Loop
```

## Implementation Components

### 1. Adam-atan2 Optimizer
**File**: `src/mlx_hrm/training/optimizers.py`

```python
"""Custom optimizers for HRM training."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
import math


class AdamAtan2(nn.Module):
    """
    Adam-atan2 optimizer for sparse embeddings.
    
    This optimizer uses atan2 for adaptive learning rates and is
    specifically designed for sparse embedding updates in HRM.
    
    Key features:
    - Adaptive learning rate based on gradient history
    - Special handling for sparse updates
    - Weight decay support
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        atan2_scale: float = 1.0
    ):
        """
        Initialize Adam-atan2 optimizer.
        
        Args:
            learning_rate: Base learning rate
            betas: Coefficients for computing running averages
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient
            atan2_scale: Scaling factor for atan2 computation
        """
        super().__init__()
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.atan2_scale = atan2_scale
        
        # State will be stored per parameter
        self.state = {}
        self.step = 0
    
    def init_state(self, param_shape: Tuple) -> Dict[str, mx.array]:
        """Initialize optimizer state for a parameter."""
        return {
            'm': mx.zeros(param_shape),  # First moment
            'v': mx.zeros(param_shape),  # Second moment
            'step': mx.array(0)
        }
    
    def update_single(
        self,
        param: mx.array,
        grad: mx.array,
        state: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Update a single parameter.
        
        Args:
            param: Parameter to update
            grad: Gradient for the parameter
            state: Optimizer state for this parameter
            
        Returns:
            Updated parameter and state
        """
        # Increment step
        step = state['step'] + 1
        state['step'] = step
        
        # Get moments
        m = state['m']
        v = state['v']
        
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** step)
        v_hat = v / (1 - self.beta2 ** step)
        
        # Adam-atan2 update
        # Use atan2 to compute adaptive learning rate
        angle = mx.arctan2(m_hat, mx.sqrt(v_hat) + self.eps)
        update = self.atan2_scale * angle
        
        # Apply weight decay
        if self.weight_decay > 0:
            param = param * (1 - self.lr * self.weight_decay)
        
        # Apply update
        param = param - self.lr * update
        
        # Update state
        state['m'] = m
        state['v'] = v
        
        return param, state
    
    def update(
        self,
        model: nn.Module,
        gradients: Dict[str, mx.array]
    ) -> nn.Module:
        """
        Update all model parameters.
        
        Args:
            model: Model to update
            gradients: Dictionary of gradients
            
        Returns:
            Updated model
        """
        self.step += 1
        
        # Get model parameters
        params = model.parameters()
        
        # Update each parameter
        for name, param in params.items():
            if name in gradients:
                grad = gradients[name]
                
                # Initialize state if needed
                if name not in self.state:
                    self.state[name] = self.init_state(param.shape)
                
                # Update parameter
                new_param, new_state = self.update_single(
                    param, grad, self.state[name]
                )
                
                # Update in model
                params[name] = new_param
                self.state[name] = new_state
        
        return model


class SparseAwareOptimizer:
    """
    Wrapper that uses different optimizers for sparse and dense parameters.
    
    This allows using Adam-atan2 for sparse embeddings while using
    standard Adam for other parameters.
    """
    
    def __init__(
        self,
        dense_optimizer: nn.Module,
        sparse_optimizer: nn.Module,
        sparse_param_names: List[str]
    ):
        """
        Initialize sparse-aware optimizer.
        
        Args:
            dense_optimizer: Optimizer for dense parameters
            sparse_optimizer: Optimizer for sparse parameters
            sparse_param_names: Names of sparse parameters
        """
        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.sparse_param_names = set(sparse_param_names)
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> nn.Module:
        """Update model with appropriate optimizer for each parameter."""
        # Split gradients
        dense_grads = {}
        sparse_grads = {}
        
        for name, grad in gradients.items():
            if any(sparse_name in name for sparse_name in self.sparse_param_names):
                sparse_grads[name] = grad
            else:
                dense_grads[name] = grad
        
        # Update with respective optimizers
        if dense_grads:
            model = self.dense_optimizer.update(model, dense_grads)
        if sparse_grads:
            model = self.sparse_optimizer.update(model, sparse_grads)
        
        return model
```

### 2. Training Loop
**File**: `src/mlx_hrm/training/trainer.py`

```python
"""Main training loop for HRM."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Callable, Any
from pathlib import Path
import time
import json

from ..models import HRM
from .act_loss import ACTLossHead
from .metrics import MetricsTracker
from .optimizers import AdamAtan2, SparseAwareOptimizer
from ..utils.checkpoint import save_checkpoint


class HRMTrainer:
    """
    Training loop for HRM with MLX.
    
    Handles:
    - Training and validation loops
    - Gradient accumulation
    - Mixed precision training
    - Checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        model: HRM,
        train_dataloader: Any,
        val_dataloader: Optional[Any] = None,
        optimizer_config: Optional[Dict] = None,
        loss_type: str = 'stablemax',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every_n_steps: int = 1000,
        eval_every_n_steps: int = 500,
        early_stopping_patience: Optional[int] = None,
        use_mixed_precision: bool = True,
        log_every_n_steps: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: HRM model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer_config: Configuration for optimizers
            loss_type: 'stablemax' or 'softmax'
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every_n_steps: Checkpoint frequency
            eval_every_n_steps: Evaluation frequency
            early_stopping_patience: Steps for early stopping
            use_mixed_precision: Whether to use mixed precision
            log_every_n_steps: Logging frequency
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Wrap model with loss head
        self.loss_model = ACTLossHead(model, loss_type=loss_type)
        
        # Create optimizers
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Training settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        
        # Evaluation
        self.eval_every_n_steps = eval_every_n_steps
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # State
        self.global_step = 0
        self.epoch = 0
    
    def _create_optimizer(self, config: Optional[Dict]) -> Any:
        """Create optimizer based on configuration."""
        if config is None:
            config = {
                'dense_lr': 1e-4,
                'sparse_lr': 1e-4,
                'weight_decay': 0.1,
                'sparse_weight_decay': 0.1
            }
        
        # Dense optimizer (standard Adam)
        dense_opt = mx.optimizers.Adam(
            learning_rate=config.get('dense_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.1)
        )
        
        # Sparse optimizer (Adam-atan2)
        sparse_opt = AdamAtan2(
            learning_rate=config.get('sparse_lr', 1e-4),
            weight_decay=config.get('sparse_weight_decay', 0.1)
        )
        
        # Combined optimizer
        return SparseAwareOptimizer(
            dense_optimizer=dense_opt,
            sparse_optimizer=sparse_opt,
            sparse_param_names=['sparse_tok_emb']
        )
    
    def _training_step(self, batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict]:
        """Single training step."""
        # Get initial carry
        batch_size = batch['input_ids'].shape[0]
        carry = self.loss_model.initial_carry(batch_size)
        
        # Forward pass
        def loss_fn(model, carry, batch):
            new_carry, loss, metrics, _ = self.loss_model(carry, batch)
            return loss, (new_carry, metrics)
        
        # Compute loss and gradients
        (loss, (new_carry, metrics)), grads = mx.value_and_grad(
            loss_fn, has_aux=True
        )(self.loss_model, carry, batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        return loss, grads, metrics
    
    def _clip_gradients(self, grads: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Clip gradients by global norm."""
        if self.max_grad_norm is None:
            return grads
        
        # Compute global norm
        total_norm = 0.0
        for grad in grads.values():
            total_norm += mx.sum(grad ** 2).item()
        total_norm = math.sqrt(total_norm)
        
        # Clip if needed
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            grads = {k: v * scale for k, v in grads.items()}
        
        return grads
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        accumulated_grads = None
        accumulated_loss = 0.0
        
        for step, batch in enumerate(self.train_dataloader):
            # Training step
            loss, grads, metrics = self._training_step(batch)
            accumulated_loss += loss.item()
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                for k in accumulated_grads:
                    accumulated_grads[k] += grads[k]
            
            # Update metrics
            self.train_metrics.update(metrics)
            
            # Update weights if accumulated enough
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                accumulated_grads = self._clip_gradients(accumulated_grads)
                
                # Update parameters
                self.model = self.optimizer.update(self.model, accumulated_grads)
                
                # Reset accumulation
                accumulated_grads = None
                accumulated_loss = 0.0
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    self._log_metrics('train', self.train_metrics.get_current())
                
                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_every_n_steps == 0:
                    val_loss = self.evaluate()
                    if self._check_early_stopping(val_loss):
                        return False  # Stop training
                
                # Checkpointing
                if self.checkpoint_dir and self.global_step % self.checkpoint_every_n_steps == 0:
                    self._save_checkpoint()
        
        return True  # Continue training
    
    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with mx.no_grad():
            for batch in self.val_dataloader:
                batch_size = batch['input_ids'].shape[0]
                carry = self.loss_model.initial_carry(batch_size)
                
                # Forward pass
                new_carry, loss, metrics, _ = self.loss_model(carry, batch)
                
                total_loss += loss.item()
                self.val_metrics.update(metrics)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self._log_metrics('val', self.val_metrics.get_current())
        
        self.model.train()
        return avg_loss
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if should stop training early."""
        if self.early_stopping_patience is None:
            return False
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model
            if self.checkpoint_dir:
                self._save_checkpoint('best_model')
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint."""
        if name is None:
            name = f'checkpoint_step_{self.global_step}'
        
        checkpoint_path = self.checkpoint_dir / name
        
        metadata = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.model.config._asdict()
        }
        
        save_checkpoint(self.model, str(checkpoint_path), metadata)
    
    def _log_metrics(self, phase: str, metrics: Dict[str, float]):
        """Log metrics to console."""
        metric_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"[{phase}] Step {self.global_step}: {metric_str}")
    
    def train(self, num_epochs: int):
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            continue_training = self.train_epoch()
            
            if not continue_training:
                print("Early stopping triggered")
                break
            
            # End of epoch evaluation
            if self.val_dataloader:
                val_loss = self.evaluate()
                print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")
        
        print("Training completed")
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint('final_model')
```

### 3. Data Pipeline
**File**: `src/mlx_hrm/data/dataset.py`

```python
"""Data loading utilities for HRM training."""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Iterator
from pathlib import Path
import json
import random


class PuzzleDataset:
    """
    Dataset for loading puzzle data.
    
    Supports:
    - ARC (Abstract Reasoning Corpus)
    - Sudoku
    - Maze navigation
    - Custom puzzle formats
    """
    
    def __init__(
        self,
        data_path: str,
        puzzle_type: str,
        max_seq_len: int = 512,
        augmentation: bool = True
    ):
        """
        Initialize puzzle dataset.
        
        Args:
            data_path: Path to dataset
            puzzle_type: Type of puzzle ('arc', 'sudoku', 'maze')
            max_seq_len: Maximum sequence length
            augmentation: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.puzzle_type = puzzle_type
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation
        
        # Load data
        self.examples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load puzzle data from disk."""
        examples = []
        
        if self.puzzle_type == 'arc':
            # Load ARC format
            for json_file in self.data_path.glob('*.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    examples.extend(self._process_arc_data(data))
        
        elif self.puzzle_type == 'sudoku':
            # Load Sudoku format
            for txt_file in self.data_path.glob('*.txt'):
                with open(txt_file, 'r') as f:
                    for line in f:
                        examples.append(self._process_sudoku_line(line.strip()))
        
        elif self.puzzle_type == 'maze':
            # Load maze format
            for npz_file in self.data_path.glob('*.npz'):
                data = np.load(npz_file)
                examples.extend(self._process_maze_data(data))
        
        else:
            raise ValueError(f"Unknown puzzle type: {self.puzzle_type}")
        
        return examples
    
    def _process_arc_data(self, data: Dict) -> List[Dict]:
        """Process ARC JSON data."""
        examples = []
        
        for task in data.get('train', []) + data.get('test', []):
            input_grid = task['input']
            output_grid = task['output']
            
            # Convert to tokens
            input_tokens = self._grid_to_tokens(input_grid)
            output_tokens = self._grid_to_tokens(output_grid)
            
            # Create example
            example = {
                'input_ids': input_tokens,
                'labels': output_tokens,
                'puzzle_id': data.get('id', 0)
            }
            examples.append(example)
        
        return examples
    
    def _grid_to_tokens(self, grid: List[List[int]]) -> List[int]:
        """Convert 2D grid to token sequence."""
        # Flatten grid with special tokens
        tokens = [self.BOG_TOKEN]  # Beginning of grid
        for row in grid:
            tokens.extend(row)
            tokens.append(self.EOL_TOKEN)  # End of line
        tokens.append(self.EOG_TOKEN)  # End of grid
        return tokens
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Apply augmentation if enabled
        if self.augmentation:
            example = self._augment_example(example)
        
        # Pad to max length
        input_ids = self._pad_sequence(example['input_ids'], self.max_seq_len)
        labels = self._pad_sequence(example['labels'], self.max_seq_len)
        
        return {
            'input_ids': mx.array(input_ids),
            'labels': mx.array(labels),
            'puzzle_id': mx.array(example.get('puzzle_id', 0))
        }
    
    def _pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        """Pad sequence to maximum length."""
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [self.PAD_TOKEN] * (max_len - len(seq))
        return seq
    
    def _augment_example(self, example: Dict) -> Dict:
        """Apply data augmentation."""
        # Implement puzzle-specific augmentation
        # e.g., rotation, reflection for grids
        return example


class DataLoader:
    """
    Efficient data loader for MLX.
    
    Features:
    - Batching
    - Shuffling
    - Prefetching
    - Memory pinning
    """
    
    def __init__(
        self,
        dataset: PuzzleDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2
    ):
        """
        Initialize data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of loading workers
            prefetch_factor: Number of batches to prefetch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # Collect batch data
            batch_data = {
                'input_ids': [],
                'labels': [],
                'puzzle_ids': []
            }
            
            for idx in batch_indices:
                example = self.dataset[idx]
                for key in batch_data:
                    if key in example:
                        batch_data[key].append(example[key])
            
            # Stack into batch
            batch = {}
            for key, values in batch_data.items():
                if values:
                    batch[key] = mx.stack(values)
            
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
```

### 4. Training Utilities
**File**: `src/mlx_hrm/training/utils.py`

```python
"""Training utilities for HRM."""

import mlx.core as mx
from typing import Dict, Optional, Callable
import math


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    
    Gradually decreases learning rate following a cosine curve.
    """
    
    def __init__(
        self,
        optimizer: Any,
        T_max: int,
        eta_min: float = 0,
        warmup_steps: int = 0
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            warmup_steps: Linear warmup steps
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self.optimizer.lr = lr
        return lr


def create_optimizer_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_patterns: List[str] = ['bias', 'norm']
) -> Dict[str, List[str]]:
    """
    Create parameter groups for different weight decay.
    
    Args:
        model: Model to get parameters from
        weight_decay: Default weight decay
        no_decay_patterns: Patterns for parameters without decay
        
    Returns:
        Dictionary mapping group names to parameter names
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.parameters().items():
        if any(pattern in name for pattern in no_decay_patterns):
            no_decay_params.append(name)
        else:
            decay_params.append(name)
    
    return {
        'decay': decay_params,
        'no_decay': no_decay_params
    }


def compute_model_flops(
    model: nn.Module,
    batch_size: int,
    seq_len: int
) -> float:
    """
    Estimate FLOPs for model forward pass.
    
    Args:
        model: HRM model
        batch_size: Batch size
        seq_len: Sequence length
        
    Returns:
        Estimated FLOPs
    """
    config = model.config
    
    # Attention FLOPs: 4 * batch * seq_len^2 * hidden_size
    attention_flops = 4 * batch_size * seq_len * seq_len * config.hidden_size
    
    # FFN FLOPs: 2 * batch * seq_len * hidden * expansion
    ffn_flops = 2 * batch_size * seq_len * config.hidden_size * config.hidden_size * config.expansion
    
    # Total layers
    total_layers = (config.H_layers * config.H_cycles + 
                   config.L_layers * config.L_cycles)
    
    # ACT steps (average)
    avg_act_steps = config.halt_max_steps / 2
    
    total_flops = (attention_flops + ffn_flops) * total_layers * avg_act_steps
    
    return total_flops
```

## Testing Strategy

### 1. Optimizer Tests
**File**: `tests/unit/test_optimizers.py`

```python
def test_adam_atan2():
    """Test Adam-atan2 optimizer."""
    # Create simple model
    model = nn.Linear(10, 5)
    optimizer = AdamAtan2(learning_rate=0.01)
    
    # Compute gradients
    x = mx.random.normal((32, 10))
    loss = mx.mean(model(x) ** 2)
    grads = mx.grad(loss)(model)
    
    # Update
    model = optimizer.update(model, grads)
    
    # Check parameters changed
    assert not mx.allclose(model.weight, original_weight)
```

### 2. Training Tests
**File**: `tests/integration/test_training.py`

```python
def test_training_loop():
    """Test complete training loop."""
    # Create small model and dataset
    model = create_hrm('tiny')
    dataset = PuzzleDataset('test_data', 'arc')
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Create trainer
    trainer = HRMTrainer(
        model=model,
        train_dataloader=dataloader,
        checkpoint_dir='test_checkpoints'
    )
    
    # Train for one epoch
    trainer.train(num_epochs=1)
    
    # Check checkpoint exists
    assert Path('test_checkpoints/final_model').exists()
```

## Performance Considerations

### 1. Data Loading
- Prefetch next batch while training
- Use memory pinning for faster GPU transfer
- Parallel data augmentation

### 2. Training Efficiency
- Gradient accumulation for large effective batch sizes
- Mixed precision training by default
- Efficient checkpoint saving (incremental)

### 3. Memory Management
- Clear gradients after optimizer step
- Periodic garbage collection
- Monitor memory usage

## Implementation Timeline

### Days 1-2: Optimizers (6-8 hours)
- [ ] Implement Adam-atan2 optimizer
- [ ] Create sparse-aware optimizer wrapper
- [ ] Add learning rate scheduling
- [ ] Test optimizer convergence

### Days 2-3: Training Loop (6-8 hours)
- [ ] Implement main trainer class
- [ ] Add gradient accumulation
- [ ] Create evaluation loop
- [ ] Add checkpointing

### Days 3-4: Data Pipeline (4-6 hours)
- [ ] Implement puzzle dataset
- [ ] Create efficient data loader
- [ ] Add data augmentation
- [ ] Test data loading speed

## Success Criteria

1. **Functionality**
   - ✅ Training runs without errors
   - ✅ Checkpoints save/load correctly
   - ✅ Metrics tracked properly
   - ✅ Early stopping works

2. **Performance**
   - ✅ Training throughput > 1000 tokens/sec
   - ✅ Efficient memory usage
   - ✅ Data loading not a bottleneck
   - ✅ Scales to large batches

3. **Robustness**
   - ✅ Handles interruptions gracefully
   - ✅ Reproducible training
   - ✅ Good error messages
   - ✅ Stable convergence

## Next Steps

After completing training infrastructure:
1. Move to Phase 8: Validation & Testing
2. Run training on real datasets
3. Compare with PyTorch implementation