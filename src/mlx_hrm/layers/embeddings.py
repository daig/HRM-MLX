"""
Sparse embedding layer and SignSGD optimizer for MLX HRM implementation.

This module implements:
1. CastedSparseEmbedding: Memory-efficient sparse embedding layer
2. SignSGD: Custom optimizer for sparse parameters

Key features:
- Sparse embeddings only store and update embeddings that are actually used
- SignSGD optimizer for stable training with sparse gradients
- Zero initialization with truncated normal for stable training
- Type casting support for mixed precision training
"""

from typing import Optional, Union, List, Tuple, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

from .initialization import EmbeddingTruncNormal


class CastedSparseEmbedding(nn.Module):
    """Sparse embedding layer optimized for puzzle-specific parameters.
    
    This is a key innovation in HRM: instead of using dense embeddings for all
    puzzle types, we use sparse embeddings that only activate for puzzles in
    the current batch. This dramatically reduces memory usage and allows
    scaling to thousands of puzzle types.
    
    During training:
    - Only embeddings for puzzles in the batch are loaded
    - Gradients are computed locally for batch-specific embeddings
    - Custom optimizer handles the sparse updates efficiently
    
    During inference:
    - Direct lookup from the full embedding table
    
    Note: MLX implementation differs from PyTorch in gradient handling.
    MLX uses a functional approach where gradients are computed via grad()
    rather than storing them on tensors.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float = 0.02,
        cast_to: Optional[mx.Dtype] = None
    ):
        """Initialize sparse embedding layer.
        
        Args:
            num_embeddings: Size of the embedding vocabulary
            embedding_dim: Dimension of each embedding vector
            batch_size: Maximum batch size (for local workspace allocation)
            init_std: Standard deviation for weight initialization
            cast_to: Target dtype for computation (e.g., mx.bfloat16)
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cast_to = cast_to or mx.float32
        
        # Master weights: Full embedding table
        # We use a custom embedding layer with truncated normal initialization
        self._embedding = EmbeddingTruncNormal(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            std=init_std
        )
        
        # For compatibility with SignSGD optimizer
        # Store reference to master weights
        self._weights = self._embedding.weight
        
        # Local workspace for current batch (only used during training)
        # These will be populated during forward pass
        self._local_weights: Optional[mx.array] = None
        self._local_ids: Optional[mx.array] = None
        
    @property
    def weight(self) -> mx.array:
        """Access to the master embedding table."""
        return self._weights
    
    def __call__(self, inputs: mx.array) -> mx.array:
        """Look up embeddings for the given puzzle IDs.
        
        Args:
            inputs: Puzzle identifiers [batch_size]
            
        Returns:
            Embeddings for the puzzles [batch_size, embedding_dim]
        """
        # During training, we need to track which embeddings are used
        # for sparse gradient updates
        if self.training:
            # Store the IDs for later use by the optimizer
            self._local_ids = inputs
            
            # Get embeddings for current batch
            embeddings = self._embedding(inputs)
            
            # Store local copy for gradient computation
            # In MLX, we don't need explicit gradient tracking like PyTorch
            self._local_weights = embeddings
            
            # Cast to target dtype if specified
            if self.cast_to != embeddings.dtype:
                embeddings = embeddings.astype(self.cast_to)
                
            return embeddings
        else:
            # Inference mode: Direct lookup
            embeddings = self._embedding(inputs)
            
            # Cast to target dtype if specified
            if self.cast_to != embeddings.dtype:
                embeddings = embeddings.astype(self.cast_to)
                
            return embeddings
    
    def get_sparse_gradients(self) -> Tuple[mx.array, mx.array]:
        """Get the IDs and gradients for sparse update.
        
        Returns:
            Tuple of (embedding_ids, gradients) for the current batch
        """
        if self._local_ids is None:
            raise ValueError("No forward pass has been performed yet")
        
        return self._local_ids, self._local_weights


@dataclass
class SignSGDState:
    """State for SignSGD optimizer."""
    # No momentum or other state needed for basic SignSGD
    step: int = 0


class SignSGD:
    """Sign-based SGD optimizer for sparse embeddings.
    
    Sign-SGD is particularly effective for sparse embeddings because:
    1. It's robust to gradient magnitude variations
    2. Updates are ±1 * learning_rate, providing consistent updates
    3. Works well with sparse gradients (most embeddings have zero gradient)
    
    This implementation focuses on single-device training.
    Multi-device support can be added later.
    """
    
    def __init__(
        self,
        learning_rate: Union[float, mx.array] = 1e-3,
        weight_decay: float = 1e-2
    ):
        """Initialize SignSGD optimizer.
        
        Args:
            learning_rate: Learning rate for updates
            weight_decay: L2 regularization strength
        """
        if isinstance(learning_rate, float):
            self._learning_rate = mx.array(learning_rate)
        else:
            self._learning_rate = learning_rate
            
        self.weight_decay = weight_decay
        self.state: Dict[int, SignSGDState] = {}
    
    def update_sparse_embedding(
        self,
        embedding: CastedSparseEmbedding,
        gradients: mx.array,
        embedding_ids: mx.array
    ) -> None:
        """Update sparse embeddings using SignSGD.
        
        Args:
            embedding: The sparse embedding layer to update
            gradients: Gradients for embeddings in the current batch
            embedding_ids: IDs of embeddings that have gradients
        """
        # Get or create state
        # Use embedding module ID since weight arrays are immutable in MLX
        param_id = id(embedding)
        if param_id not in self.state:
            self.state[param_id] = SignSGDState()
        
        state = self.state[param_id]
        state.step += 1
        
        # Apply SignSGD update with weight decay
        # Update rule: w = w * (1 - lr * wd) - lr * sign(grad)
        
        # Get current weights for the embeddings that have gradients
        current_weights = embedding.weight[embedding_ids]
        
        # Apply weight decay
        if self.weight_decay > 0:
            current_weights = current_weights * (1.0 - self._learning_rate * self.weight_decay)
        
        # Apply sign-based gradient update
        updates = current_weights - self._learning_rate * mx.sign(gradients)
        
        # Update only the used embeddings in the master table
        # In MLX, we need to create a new array with the updates
        # MLX doesn't support in-place updates, so we'll create a new array
        new_weights = mx.array(embedding.weight)  # Create a copy
        for i, idx in enumerate(embedding_ids.tolist()):
            new_weights[idx] = updates[i]
        
        # Update the embedding weights
        embedding._weights = new_weights
        embedding._embedding.weight = new_weights
    
    def update(
        self,
        model: nn.Module,
        gradients: Dict[str, mx.array]
    ) -> None:
        """Update all sparse embeddings in the model.
        
        This method finds all CastedSparseEmbedding layers in the model
        and updates them using the computed gradients.
        
        Args:
            model: The model containing sparse embeddings
            gradients: Dictionary mapping parameter names to gradients
        """
        # Find all sparse embedding layers
        for name, module in model.named_modules():
            if isinstance(module, CastedSparseEmbedding):
                # Get the embedding IDs and local weights from the forward pass
                embedding_ids, local_weights = module.get_sparse_gradients()
                
                # Find the gradient for this embedding's local weights
                # The gradient key should match the parameter name
                grad_key = None
                for key in gradients:
                    if name in key and "local_weights" in key:
                        grad_key = key
                        break
                
                if grad_key and grad_key in gradients:
                    gradient = gradients[grad_key]
                    self.update_sparse_embedding(module, gradient, embedding_ids)


def create_sparse_embedding_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2
) -> Tuple[SignSGD, List[mx.array]]:
    """Create SignSGD optimizer and get parameters for sparse embeddings.
    
    This is a helper function that:
    1. Creates a SignSGD optimizer
    2. Finds all sparse embedding parameters
    3. Returns optimizer and list of parameters to track
    
    Args:
        model: The model containing sparse embeddings
        learning_rate: Learning rate for SignSGD
        weight_decay: L2 regularization strength
        
    Returns:
        Tuple of (optimizer, sparse_parameters)
    """
    optimizer = SignSGD(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Find all sparse embedding parameters
    sparse_params = []
    for name, module in model.named_modules():
        if isinstance(module, CastedSparseEmbedding):
            # We track the local weights for gradient computation
            # The actual weight update happens through the optimizer
            if module._local_weights is not None:
                sparse_params.append(module._local_weights)
    
    return optimizer, sparse_params