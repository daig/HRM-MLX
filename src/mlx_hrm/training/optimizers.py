"""Custom optimizers for HRM training."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Optional, Tuple, List, Any, Union
import math


class AdamAtan2:
    """
    Adam-atan2 optimizer for main model parameters.
    
    This optimizer replaces the traditional Adam division with atan2,
    providing better numerical stability and scale invariance.
    
    Key innovation: update = atan2(m_hat, sqrt(v_hat)) instead of m_hat / sqrt(v_hat)
    
    Based on the adam-atan2 package used in the original HRM implementation.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.1,
        atan2_scale: float = 1.0
    ):
        """
        Initialize Adam-atan2 optimizer.
        
        Args:
            learning_rate: Base learning rate
            betas: Coefficients for computing running averages (β1, β2)
            weight_decay: Weight decay coefficient (decoupled)
            atan2_scale: Scaling factor for atan2 computation
        """
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.atan2_scale = atan2_scale
        
        # Global state
        self.step_count = 0
        self.state = {}
    
    def init_param_state(self, param_shape: Tuple[int, ...]) -> Dict[str, mx.array]:
        """Initialize optimizer state for a parameter."""
        return {
            'exp_avg': mx.zeros(param_shape),      # First moment estimate
            'exp_avg_sq': mx.zeros(param_shape),   # Second moment estimate
        }
    
    def update_param(
        self,
        param: mx.array,
        grad: mx.array,
        param_name: str
    ) -> mx.array:
        """
        Update a single parameter using Adam-atan2.
        
        Args:
            param: Parameter to update
            grad: Gradient for the parameter
            param_name: Name of parameter (for state tracking)
            
        Returns:
            Updated parameter
        """
        # Initialize state if needed
        if param_name not in self.state:
            self.state[param_name] = self.init_param_state(param.shape)
        
        state = self.state[param_name]
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        # Update biased first moment estimate
        exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate  
        exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        bias_correct1 = 1 - self.beta1 ** self.step_count
        bias_correct2 = 1 - self.beta2 ** self.step_count
        
        # Corrected first and second moments
        m_hat = exp_avg / bias_correct1
        v_hat = exp_avg_sq / bias_correct2
        
        # Adam-atan2 update: key innovation
        # Instead of m_hat / (sqrt(v_hat) + eps), use atan2(m_hat, sqrt(v_hat))
        denominator = mx.sqrt(v_hat)
        update = mx.arctan2(m_hat, denominator) * self.atan2_scale
        
        # Apply weight decay (decoupled)
        if self.weight_decay > 0:
            param = param * (1 - self.lr * self.weight_decay)
        
        # Apply parameter update
        param = param - self.lr * update
        
        # Update state
        state['exp_avg'] = exp_avg
        state['exp_avg_sq'] = exp_avg_sq
        
        return param
    
    def update(self, model_params: Dict[str, mx.array], gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Update all model parameters.
        
        Args:
            model_params: Dictionary of model parameters
            gradients: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        self.step_count += 1
        
        updated_params = {}
        for param_name, param in model_params.items():
            if param_name in gradients:
                updated_params[param_name] = self.update_param(
                    param, gradients[param_name], param_name
                )
            else:
                updated_params[param_name] = param
        
        return updated_params


class SignSGD:
    """
    Sign-SGD optimizer for sparse embeddings.
    
    This optimizer is specifically designed for sparse embedding parameters
    and uses the sign of gradients for updates, providing robustness for
    sparse and high-dimensional parameters.
    
    Based on the CastedSparseEmbeddingSignSGD implementation in HRM.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1
    ):
        """
        Initialize Sign-SGD optimizer.
        
        Args:
            learning_rate: Learning rate for sparse parameters
            weight_decay: Weight decay coefficient (decoupled)
        """
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        # No momentum state needed for Sign-SGD
        self.step_count = 0
    
    def update_param(
        self,
        param: mx.array,
        grad: mx.array
    ) -> mx.array:
        """
        Update sparse parameter using Sign-SGD.
        
        Args:
            param: Sparse parameter to update
            grad: Gradient for the parameter
            
        Returns:
            Updated parameter
        """
        # Apply weight decay (decoupled)
        if self.weight_decay > 0:
            param = param * (1 - self.lr * self.weight_decay)
        
        # Sign-SGD update: w = w - lr * sign(grad)
        param = param - self.lr * mx.sign(grad)
        
        return param
    
    def update(self, model_params: Dict[str, mx.array], gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Update sparse parameters.
        
        Args:
            model_params: Dictionary of sparse parameters
            gradients: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        self.step_count += 1
        
        updated_params = {}
        for param_name, param in model_params.items():
            if param_name in gradients:
                updated_params[param_name] = self.update_param(
                    param, gradients[param_name]
                )
            else:
                updated_params[param_name] = param
        
        return updated_params


class SparseAwareOptimizer:
    """
    Wrapper that uses different optimizers for sparse and dense parameters.
    
    This replicates the dual-optimizer approach used in HRM:
    - Adam-atan2 for dense model parameters (attention, FFN, etc.)
    - Sign-SGD for sparse embedding parameters
    
    Based on the HRM training setup in pretrain.py.
    """
    
    def __init__(
        self,
        dense_optimizer: AdamAtan2,
        sparse_optimizer: SignSGD,
        sparse_param_patterns: List[str]
    ):
        """
        Initialize sparse-aware optimizer.
        
        Args:
            dense_optimizer: Adam-atan2 optimizer for dense parameters
            sparse_optimizer: Sign-SGD optimizer for sparse parameters  
            sparse_param_patterns: Patterns to identify sparse parameters
        """
        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.sparse_param_patterns = sparse_param_patterns
    
    def _split_parameters(
        self, 
        model_params: Dict[str, mx.array], 
        gradients: Dict[str, mx.array]
    ) -> Tuple[Dict[str, mx.array], Dict[str, mx.array], Dict[str, mx.array], Dict[str, mx.array]]:
        """Split parameters and gradients into dense and sparse groups."""
        dense_params = {}
        sparse_params = {}
        dense_grads = {}
        sparse_grads = {}
        
        for param_name in model_params:
            param = model_params[param_name]
            
            # Check if this is a sparse parameter
            is_sparse = any(pattern in param_name for pattern in self.sparse_param_patterns)
            
            if is_sparse:
                sparse_params[param_name] = param
                if param_name in gradients:
                    sparse_grads[param_name] = gradients[param_name]
            else:
                dense_params[param_name] = param
                if param_name in gradients:
                    dense_grads[param_name] = gradients[param_name]
        
        return dense_params, sparse_params, dense_grads, sparse_grads
    
    def update(self, model_params: Dict[str, mx.array], gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Update all model parameters with appropriate optimizers.
        
        Args:
            model_params: Dictionary of all model parameters
            gradients: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        # Split parameters by type
        dense_params, sparse_params, dense_grads, sparse_grads = self._split_parameters(
            model_params, gradients
        )
        
        # Update each group with appropriate optimizer
        updated_params = {}
        
        if dense_params:
            updated_dense = self.dense_optimizer.update(dense_params, dense_grads)
            updated_params.update(updated_dense)
        
        if sparse_params:
            updated_sparse = self.sparse_optimizer.update(sparse_params, sparse_grads)
            updated_params.update(updated_sparse)
        
        return updated_params
    
    @property
    def step_count(self) -> int:
        """Get current step count (from dense optimizer)."""
        return self.dense_optimizer.step_count


def create_hrm_optimizer(
    dense_lr: float = 1e-4,
    sparse_lr: float = 1e-4,
    dense_weight_decay: float = 0.1,
    sparse_weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.999)
) -> SparseAwareOptimizer:
    """
    Create the standard HRM optimizer setup.
    
    This replicates the exact optimizer configuration used in the original
    HRM training with separate learning rates and weight decay for dense
    and sparse parameters.
    
    Args:
        dense_lr: Learning rate for dense parameters (Adam-atan2)
        sparse_lr: Learning rate for sparse embeddings (Sign-SGD)
        dense_weight_decay: Weight decay for dense parameters
        sparse_weight_decay: Weight decay for sparse parameters
        betas: Adam momentum coefficients
        
    Returns:
        Configured sparse-aware optimizer
    """
    # Dense optimizer: Adam-atan2 for main model
    dense_optimizer = AdamAtan2(
        learning_rate=dense_lr,
        betas=betas,
        weight_decay=dense_weight_decay
    )
    
    # Sparse optimizer: Sign-SGD for embeddings
    sparse_optimizer = SignSGD(
        learning_rate=sparse_lr,
        weight_decay=sparse_weight_decay
    )
    
    # Patterns to identify sparse parameters
    # Based on HRM's sparse embedding parameter names
    sparse_patterns = [
        'sparse_tok_emb',  # Main sparse embedding layer
        'puzzle_emb',      # Alternative naming
    ]
    
    return SparseAwareOptimizer(
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        sparse_param_patterns=sparse_patterns
    )


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Implements the learning rate schedule used in HRM training:
    - Linear warmup for initial steps
    - Cosine annealing decay
    """
    
    def __init__(
        self,
        optimizer: Union[AdamAtan2, SignSGD, SparseAwareOptimizer],
        T_max: int,
        eta_min: float = 0.0,
        warmup_steps: int = 10000,
        last_step: int = -1
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of training steps
            eta_min: Minimum learning rate
            warmup_steps: Number of linear warmup steps
            last_step: Last step for resuming
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.last_step = last_step
        
        # Store base learning rates
        if isinstance(optimizer, SparseAwareOptimizer):
            self.base_dense_lr = optimizer.dense_optimizer.lr
            self.base_sparse_lr = optimizer.sparse_optimizer.lr
        elif isinstance(optimizer, AdamAtan2):
            self.base_lr = optimizer.lr
        elif isinstance(optimizer, SignSGD):
            self.base_lr = optimizer.lr
        else:
            raise ValueError(f"Unsupported optimizer type: {type(optimizer)}")
    
    def get_lr(self, step: int) -> Union[float, Tuple[float, float]]:
        """Compute learning rate for current step."""
        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.T_max - self.warmup_steps)
            warmup_factor = self.eta_min + (1.0 - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        if isinstance(self.optimizer, SparseAwareOptimizer):
            dense_lr = self.base_dense_lr * warmup_factor
            sparse_lr = self.base_sparse_lr * warmup_factor
            return dense_lr, sparse_lr
        else:
            return self.base_lr * warmup_factor
    
    def step(self, step: Optional[int] = None):
        """Update learning rate."""
        if step is None:
            step = self.last_step + 1
        
        self.last_step = step
        
        if isinstance(self.optimizer, SparseAwareOptimizer):
            dense_lr, sparse_lr = self.get_lr(step)
            self.optimizer.dense_optimizer.lr = dense_lr
            self.optimizer.sparse_optimizer.lr = sparse_lr
        else:
            lr = self.get_lr(step)
            self.optimizer.lr = lr