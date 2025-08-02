"""
Demonstration of sparse embeddings with SignSGD optimizer.

This example shows how to:
1. Create a sparse embedding layer
2. Use it in a simple model
3. Train with SignSGD optimizer
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_hrm.layers import CastedSparseEmbedding, SignSGD


def create_toy_dataset(num_puzzles: int = 10, seq_length: int = 8, num_samples: int = 100):
    """Create a toy dataset with puzzle IDs and targets."""
    # Random puzzle IDs
    puzzle_ids = mx.random.randint(0, num_puzzles, shape=(num_samples,))
    
    # Random sequences for each sample
    sequences = mx.random.normal(shape=(num_samples, seq_length, 16))
    
    # Simple targets (sum of sequence values per puzzle)
    targets = mx.sum(sequences, axis=(1, 2))
    
    return puzzle_ids, sequences, targets


class SparseEmbeddingModel(nn.Module):
    """Simple model using sparse embeddings."""
    
    def __init__(self, num_puzzles: int, embedding_dim: int, hidden_dim: int, batch_size: int):
        super().__init__()
        
        # Sparse embedding layer for puzzle-specific parameters
        self.puzzle_embedding = CastedSparseEmbedding(
            num_embeddings=num_puzzles,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            init_std=0.02,
            cast_to=mx.float32  # Can use mx.bfloat16 for efficiency
        )
        
        # Regular layers
        self.input_proj = nn.Linear(16, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim + embedding_dim, 1)
        
    def __call__(self, puzzle_ids: mx.array, sequences: mx.array):
        # Get puzzle-specific embeddings
        puzzle_emb = self.puzzle_embedding(puzzle_ids)  # [batch_size, embedding_dim]
        
        # Process sequences
        seq_features = self.input_proj(sequences)  # [batch_size, seq_length, hidden_dim]
        seq_features = mx.mean(seq_features, axis=1)  # [batch_size, hidden_dim]
        
        # Combine with puzzle embeddings
        combined = mx.concatenate([seq_features, puzzle_emb], axis=-1)
        
        # Output prediction
        output = self.output_proj(combined)
        return mx.squeeze(output, axis=-1)


def train_step(model: nn.Module, sparse_optimizer: SignSGD, dense_optimizer: optim.Optimizer,
               puzzle_ids: mx.array, sequences: mx.array, targets: mx.array):
    """Single training step with separate optimizers for sparse and dense parameters."""
    
    def loss_fn(model, puzzle_ids, sequences, targets):
        predictions = model(puzzle_ids, sequences)
        return mx.mean((predictions - targets) ** 2)
    
    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model, puzzle_ids, sequences, targets)
    
    # Update dense parameters with regular optimizer
    dense_optimizer.update(model, grads)
    
    # Update sparse embeddings with SignSGD
    # The sparse optimizer will find and update only the sparse embedding layers
    sparse_optimizer.update(model, grads)
    
    return loss


def main():
    # Configuration
    num_puzzles = 20
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 16
    num_epochs = 10
    
    # Create model
    model = SparseEmbeddingModel(
        num_puzzles=num_puzzles,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size
    )
    
    # Create optimizers
    # SignSGD for sparse embeddings
    sparse_optimizer = SignSGD(learning_rate=0.01, weight_decay=0.01)
    
    # Regular Adam for dense parameters
    dense_params = [p for name, p in model.parameters().items() 
                   if 'puzzle_embedding' not in name]
    dense_optimizer = optim.Adam(learning_rate=0.001)
    
    # Generate toy data
    puzzle_ids, sequences, targets = create_toy_dataset(num_puzzles=num_puzzles)
    
    # Training loop
    print("Training sparse embedding model...")
    model.train()
    
    for epoch in range(num_epochs):
        # Simple batch iteration (in practice, use proper data loading)
        total_loss = 0.0
        num_batches = len(puzzle_ids) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_puzzle_ids = puzzle_ids[start_idx:end_idx]
            batch_sequences = sequences[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            loss = train_step(
                model, sparse_optimizer, dense_optimizer,
                batch_puzzle_ids, batch_sequences, batch_targets
            )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Demonstrate inference
    print("\nInference mode:")
    model.eval()
    
    # Test on a few samples
    test_puzzle_ids = mx.array([0, 5, 10])
    test_sequences = mx.random.normal(shape=(3, 8, 16))
    
    predictions = model(test_puzzle_ids, test_sequences)
    print(f"Predictions for puzzles {test_puzzle_ids.tolist()}: {predictions.tolist()}")
    
    # Show that only used embeddings were updated
    print(f"\nSparse optimizer state entries: {len(sparse_optimizer.state)}")
    print("(Should be 1, tracking the puzzle embedding layer)")


if __name__ == "__main__":
    main()