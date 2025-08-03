"""Main training loop for HRM."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from typing import Dict, Optional, Any, Iterator
from pathlib import Path
import time
import json
import math

from .act_loss import ACTLossHead
from .metrics import MetricsTracker
from .optimizers import create_hrm_optimizer, CosineAnnealingLR, SparseAwareOptimizer
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
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Iterator[Dict[str, mx.array]],
        val_dataloader: Optional[Iterator[Dict[str, mx.array]]] = None,
        optimizer_config: Optional[Dict] = None,
        loss_type: str = 'stablemax',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every_n_steps: int = 1000,
        eval_every_n_steps: int = 500,
        early_stopping_patience: Optional[int] = None,
        use_mixed_precision: bool = False,
        mixed_precision_dtype: str = 'float16',
        mixed_precision_components: Optional[list] = None,
        log_every_n_steps: int = 10,
        max_training_steps: Optional[int] = None,
        warmup_steps: int = 10000
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
            use_mixed_precision: Whether to use mixed precision (disabled by default)
            mixed_precision_dtype: Dtype for mixed precision ('float16' or 'bfloat16')
            mixed_precision_components: List of components to use mixed precision on
            log_every_n_steps: Logging frequency
            max_training_steps: Maximum training steps for LR scheduling
            warmup_steps: Linear warmup steps
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Wrap model with loss head
        self.loss_model = ACTLossHead(model, loss_type=loss_type)
        
        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Create learning rate scheduler
        if max_training_steps:
            self.lr_scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=max_training_steps,
                warmup_steps=warmup_steps
            )
        else:
            self.lr_scheduler = None
        
        # Training settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision
        
        # Validate and set precision dtype
        try:
            self.mixed_precision_dtype = getattr(mx, mixed_precision_dtype)
        except AttributeError:
            raise ValueError(f"Invalid mixed precision dtype: {mixed_precision_dtype}. "
                           f"Must be one of: float16, bfloat16")
        
        # Configure which batch keys to cast to mixed precision
        if mixed_precision_components is None:
            # Default: cast only input tokens, keep labels/puzzle_id in original dtype for stability
            mixed_precision_components = ['input_ids']
        self.mixed_precision_keys = set(mixed_precision_components)
        
        # Print mixed precision status
        if self.use_mixed_precision:
            print(f"Mixed precision enabled: {mixed_precision_dtype}")
            print(f"  Keys to cast: {sorted(self.mixed_precision_keys)}")
            print(f"  Other keys remain float32 for stability")
        else:
            print("Mixed precision disabled (using float32 throughout)")
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        
        # Timing
        self.start_time = None
        self.epoch_start_time = None
    
    def _create_optimizer(self, config: Optional[Dict]) -> SparseAwareOptimizer:
        """Create optimizer based on configuration."""
        if config is None:
            config = {
                'dense_lr': 1e-4,
                'sparse_lr': 1e-4,
                'dense_weight_decay': 0.1,
                'sparse_weight_decay': 0.1
            }
        
        return create_hrm_optimizer(
            dense_lr=config.get('dense_lr', 1e-4),
            sparse_lr=config.get('sparse_lr', 1e-4),
            dense_weight_decay=config.get('dense_weight_decay', 0.1),
            sparse_weight_decay=config.get('sparse_weight_decay', 0.1)
        )
    
    def _prepare_batch_for_mixed_precision(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Prepare batch with appropriate dtypes for mixed precision.
        
        Strategy:
        - Cast configured input keys to mixed precision for speed/memory
        - Keep labels, targets, and unconfigured keys in original dtype for stability
        - Only cast actual MLX arrays, skip other data types
        """
        if not self.use_mixed_precision:
            return batch
        
        mixed_precision_batch = {}
        for key, value in batch.items():
            # Only cast MLX arrays that are in our configured set
            if (key in self.mixed_precision_keys and 
                isinstance(value, mx.array) and 
                value.dtype in [mx.int32, mx.int64, mx.float32]):  # Only cast from these dtypes
                try:
                    mixed_precision_batch[key] = value.astype(self.mixed_precision_dtype)
                except Exception as e:
                    # If casting fails, keep original and warn
                    print(f"Warning: Failed to cast {key} to {self.mixed_precision_dtype}: {e}")
                    mixed_precision_batch[key] = value
            else:
                # Keep original dtype for labels, non-arrays, or unconfigured keys
                mixed_precision_batch[key] = value
        
        return mixed_precision_batch
    
    def _cast_gradients_to_float32(self, grads):
        """
        Cast all gradients to float32 for optimizer stability.
        
        This is crucial for mixed precision training - gradients should always
        be accumulated and applied in float32 for numerical stability.
        Uses tree_map to handle nested gradient structures.
        """
        if not self.use_mixed_precision:
            return grads
        
        def cast_to_float32(x):
            # Only cast actual MLX arrays, skip dict/nested structures
            if isinstance(x, mx.array):
                return x.astype(mx.float32)
            return x
        
        return mlx.utils.tree_map(cast_to_float32, grads)
    
    def _training_step(self, batch: Dict[str, mx.array]) -> tuple[mx.array, Dict[str, mx.array], Dict]:
        """
        Single training step with optional mixed precision support.
        
        Strategy:
        - Cast input tensors to mixed precision for forward pass
        - Keep loss computation in float32 for stability  
        - Cast gradients back to float32 for optimizer updates
        - Avoid mixed precision for sensitive components (ACT, sparse embeddings)
        """
        # Prepare batch with appropriate dtypes
        processed_batch = self._prepare_batch_for_mixed_precision(batch)
        
        # Get initial carry
        batch_size = processed_batch['input_ids'].shape[0]
        carry = self.loss_model.initial_carry(batch_size)
        
        # Forward pass using ACTLossHead (matches PyTorch reference pattern)
        def loss_fn(processed_batch):
            # Let ACTLossHead handle all loss computation - clean separation like PyTorch
            new_carry, total_loss, metrics, outputs = self.loss_model(carry, processed_batch)
            
            # Ensure loss is in float32 for numerical stability
            if hasattr(total_loss, 'astype'):
                total_loss = total_loss.astype(mx.float32)
            
            return total_loss, (new_carry, metrics)
        
        # Compute gradients for the full loss model (since it contains trainable parameters)
        loss_and_grad_fn = nn.value_and_grad(self.loss_model, loss_fn)
        value_result, grads = loss_and_grad_fn(processed_batch)
        loss, (new_carry, metrics) = value_result
        
        # Cast gradients to float32 for stability (critical for mixed precision)
        grads = self._cast_gradients_to_float32(grads)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Scale gradients too (use tree_map for nested structures)
        def scale_grad(x):
            # Only scale actual MLX arrays, skip dict/nested structures
            if isinstance(x, mx.array):
                return x / self.gradient_accumulation_steps
            return x
        grads = mlx.utils.tree_map(scale_grad, grads)
        
        return loss, grads, metrics
    
    def _clip_gradients(self, grads):
        """Clip gradients by global norm using tree operations."""
        if self.max_grad_norm is None:
            return grads
        
        # Compute global norm using tree operations
        def norm_sq(x):
            # Only process actual MLX arrays
            if isinstance(x, mx.array):
                return mx.sum(x ** 2)
            return mx.array(0.0)
        
        norm_sq_values = mlx.utils.tree_map(norm_sq, grads)
        total_norm_sq = mlx.utils.tree_reduce(lambda x, y: x + y, norm_sq_values, mx.array(0.0))
        total_norm = mx.sqrt(total_norm_sq).item()
        
        # Clip if needed
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            grads = mlx.utils.tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, grads)
            print(f"Gradients clipped: {total_norm:.4f} -> {self.max_grad_norm}")
        
        return grads
    
    def train_epoch(self) -> bool:
        """Train for one epoch. Returns True to continue training, False to stop."""
        self.epoch_start_time = time.time()
        accumulated_grads = {}
        accumulated_loss = 0.0
        accumulation_count = 0
        
        for step, batch in enumerate(self.train_dataloader):
            # Training step
            try:
                loss, grads, metrics = self._training_step(batch)
                accumulated_loss += loss.item()
                accumulation_count += 1
                
                # Accumulate gradients
                if not accumulated_grads:
                    accumulated_grads = grads
                else:
                    for k in accumulated_grads:
                        if k in grads:
                            accumulated_grads[k] += grads[k]
                
                # Update metrics
                self.train_metrics.update(metrics)
                
                # Update weights if accumulated enough
                if accumulation_count >= self.gradient_accumulation_steps:
                    # Clip gradients
                    accumulated_grads = self._clip_gradients(accumulated_grads)
                    
                    # Update parameters
                    model_params = self.loss_model.model.parameters()
                    updated_params = self.optimizer.update(model_params, accumulated_grads)
                    self.loss_model.model.update(updated_params)
                    
                    # Update learning rate
                    if self.lr_scheduler:
                        self.lr_scheduler.step(self.global_step)
                    
                    # Reset accumulation
                    accumulated_grads = {}
                    accumulated_loss = 0.0
                    accumulation_count = 0
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_every_n_steps == 0:
                        self._log_metrics('train', self.train_metrics.get_current())
                    
                    # Evaluation
                    if (self.val_dataloader and 
                        self.global_step % self.eval_every_n_steps == 0):
                        val_loss = self.evaluate()
                        if self._check_early_stopping(val_loss):
                            return False  # Stop training
                    
                    # Checkpointing
                    if (self.checkpoint_dir and 
                        self.global_step % self.checkpoint_every_n_steps == 0):
                        self._save_checkpoint()
                
            except Exception as e:
                print(f"Training step failed: {e}")
                # Continue training on individual step failures
                continue
        
        return True  # Continue training
    
    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        print("Running evaluation...")
        total_loss = 0.0
        num_batches = 0
        
        # Clear validation metrics
        self.val_metrics.reset()
        
        for batch in self.val_dataloader:
            try:
                # Prepare batch with appropriate dtypes for mixed precision
                processed_batch = self._prepare_batch_for_mixed_precision(batch)
                
                batch_size = processed_batch['input_ids'].shape[0]
                carry = self.loss_model.initial_carry(batch_size)
                
                # Forward pass without gradients
                new_carry, loss, metrics, _ = self.loss_model(carry, processed_batch)
                
                # Ensure loss is in float32 for accumulation
                if hasattr(loss, 'astype'):
                    loss = loss.astype(mx.float32)
                
                total_loss += loss.item()
                self.val_metrics.update(metrics)
                num_batches += 1
                
                # Limit validation to reasonable number of batches
                if num_batches >= 100:  # Don't run forever
                    break
                    
            except Exception as e:
                print(f"Validation step failed: {e}")
                continue
        
        if num_batches == 0:
            print("Warning: No successful validation batches")
            return float('inf')
        
        avg_loss = total_loss / num_batches
        self._log_metrics('val', self.val_metrics.get_current())
        
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
        
        should_stop = self.patience_counter >= self.early_stopping_patience
        if should_stop:
            print(f"Early stopping triggered after {self.patience_counter} steps without improvement")
        
        return should_stop
    
    def _save_checkpoint(self, name: Optional[str] = None):
        """Save model checkpoint."""
        if name is None:
            name = f'checkpoint_step_{self.global_step}'
        
        checkpoint_path = self.checkpoint_dir / name
        
        # Get current learning rates
        current_lrs = "N/A"
        if isinstance(self.optimizer, SparseAwareOptimizer):
            current_lrs = f"dense: {self.optimizer.dense_optimizer.lr:.2e}, sparse: {self.optimizer.sparse_optimizer.lr:.2e}"
        
        metadata = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'current_learning_rates': current_lrs,
            'patience_counter': self.patience_counter
        }
        
        # Add model config if available
        if hasattr(self.model, 'config'):
            metadata['config'] = self.model.config._asdict()
        
        try:
            save_checkpoint(self.model, str(checkpoint_path), metadata)
            print(f"Checkpoint saved: {name}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _log_metrics(self, phase: str, metrics: Dict[str, float]):
        """Log metrics to console."""
        if not metrics:
            return
        
        # Format metrics nicely
        metric_strs = []
        for k, v in metrics.items():
            if k == 'count':
                continue  # Don't log count separately
            if 'loss' in k:
                metric_strs.append(f'{k}: {v:.4f}')
            elif 'accuracy' in k:
                metric_strs.append(f'{k}: {v:.3f}')
            else:
                metric_strs.append(f'{k}: {v:.2f}')
        
        # Calculate time info
        time_info = ""
        if phase == 'train' and self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            time_info = f" | epoch_time: {epoch_time:.1f}s"
        
        # Current learning rate info
        lr_info = ""
        if isinstance(self.optimizer, SparseAwareOptimizer):
            lr_info = f" | lr: {self.optimizer.dense_optimizer.lr:.2e}"
        
        metric_str = ' | '.join(metric_strs)
        print(f"[{phase}] Step {self.global_step}: {metric_str}{lr_info}{time_info}")
    
    def train(self, num_epochs: int):
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.num_parameters():,}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Max gradient norm: {self.max_grad_norm}")
        
        self.start_time = time.time()
        
        try:
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
                
                # End of epoch metrics logging
                self.train_metrics.log_epoch(epoch + 1)
                if self.val_dataloader:
                    self.val_metrics.log_epoch(epoch + 1)
            
            # Training completed
            total_time = time.time() - self.start_time
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Total steps: {self.global_step}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            
            # Save final checkpoint
            if self.checkpoint_dir:
                self._save_checkpoint('final_model')
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if self.checkpoint_dir:
                self._save_checkpoint('interrupted_model')
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            if self.checkpoint_dir:
                self._save_checkpoint('error_model')
            raise
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from ..utils.checkpoint import load_checkpoint
        
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        try:
            weights, config, metadata = load_checkpoint(checkpoint_path)
            
            # Load model weights
            self.model.update(weights)
            self.loss_model.model.update(weights)
            
            # Restore training state
            if metadata:
                self.global_step = metadata.get('global_step', 0)
                self.epoch = metadata.get('epoch', 0)
                self.best_val_loss = metadata.get('best_val_loss', float('inf'))
                self.patience_counter = metadata.get('patience_counter', 0)
            
            print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
            
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            raise


def create_trainer(
    model: nn.Module,
    train_dataloader: Iterator,
    val_dataloader: Optional[Iterator] = None,
    **kwargs
) -> HRMTrainer:
    """
    Create HRM trainer with sensible defaults.
    
    Args:
        model: HRM model to train
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured HRMTrainer instance
    """
    # Default configuration
    defaults = {
        'optimizer_config': {
            'dense_lr': 1e-4,
            'sparse_lr': 1e-4,
            'dense_weight_decay': 0.1,
            'sparse_weight_decay': 0.1
        },
        'loss_type': 'stablemax',
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'checkpoint_every_n_steps': 1000,
        'eval_every_n_steps': 500,
        'log_every_n_steps': 10,
        'use_mixed_precision': False,  # Disabled by default for stability/reproducibility
        'mixed_precision_dtype': 'float16',
        'mixed_precision_components': ['input_ids'],  # Only cast input tokens to mixed precision
        'warmup_steps': 10000
    }
    
    # Merge with provided kwargs
    config = {**defaults, **kwargs}
    
    return HRMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **config
    )