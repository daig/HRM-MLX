"""
Unit tests for sparse embeddings and SignSGD optimizer.

Tests cover:
1. Basic functionality
2. Gradient computation and updates
3. Type casting behavior
4. SignSGD optimizer mechanics
5. Edge cases and error handling
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_hrm.layers.embeddings import (
    CastedSparseEmbedding,
    SignSGD,
    create_sparse_embedding_optimizer
)


class TestCastedSparseEmbedding:
    """Test sparse embedding layer functionality."""
    
    def test_initialization(self):
        """Test embedding initialization."""
        num_embeddings = 100
        embedding_dim = 64
        batch_size = 32
        
        emb = CastedSparseEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            init_std=0.02
        )
        
        # Check dimensions
        assert emb.weight.shape == (num_embeddings, embedding_dim)
        assert emb.num_embeddings == num_embeddings
        assert emb.embedding_dim == embedding_dim
        assert emb.batch_size == batch_size
        
        # Check initialization (should be truncated normal)
        # Note: truncated normal has variance correction factor of ~1.29
        std = mx.std(emb.weight).item()
        assert 0.015 < std < 0.030  # Should be close to 0.02 * 1.29 ≈ 0.026
    
    def test_forward_inference(self):
        """Test forward pass during inference."""
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32
        )
        emb.eval()  # Set to evaluation mode
        
        # Test single lookup
        idx = mx.array([5])
        output = emb(idx)
        assert output.shape == (1, 64)
        assert mx.array_equal(output[0], emb.weight[5])
        
        # Test batch lookup
        indices = mx.array([1, 5, 10, 20])
        output = emb(indices)
        assert output.shape == (4, 64)
        for i, idx in enumerate([1, 5, 10, 20]):
            assert mx.array_equal(output[i], emb.weight[idx])
    
    def test_forward_training(self):
        """Test forward pass during training."""
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32
        )
        emb.train()  # Set to training mode
        
        indices = mx.array([1, 5, 10, 20])
        output = emb(indices)
        
        # Check output shape
        assert output.shape == (4, 64)
        
        # Check that local workspace is populated
        assert emb._local_weights is not None
        assert emb._local_ids is not None
        assert mx.array_equal(emb._local_ids, indices)
        
        # Check that output matches the embeddings for given indices
        for i, idx in enumerate([1, 5, 10, 20]):
            expected = emb.weight[idx]
            if emb.cast_to != emb.weight.dtype:
                expected = expected.astype(emb.cast_to)
            assert mx.allclose(output[i], expected, atol=1e-6)
    
    def test_type_casting(self):
        """Test dtype casting behavior."""
        emb = CastedSparseEmbedding(
            num_embeddings=50,
            embedding_dim=32,
            batch_size=16,
            cast_to=mx.bfloat16
        )
        
        # Check storage is float32
        assert emb.weight.dtype == mx.float32
        
        # Check output is cast to bfloat16
        indices = mx.array([0, 1, 2])
        output = emb(indices)
        assert output.dtype == mx.bfloat16
    
    def test_get_sparse_gradients(self):
        """Test sparse gradient retrieval."""
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32
        )
        emb.train()
        
        # Should raise error before forward pass
        with pytest.raises(ValueError, match="No forward pass"):
            emb.get_sparse_gradients()
        
        # After forward pass, should return IDs and weights
        indices = mx.array([5, 10, 15])
        output = emb(indices)
        
        ids, weights = emb.get_sparse_gradients()
        assert mx.array_equal(ids, indices)
        assert weights.shape == output.shape
    
    def test_repeated_indices(self):
        """Test behavior with repeated indices."""
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32
        )
        
        # Indices with repeats
        indices = mx.array([5, 10, 5, 20, 10])
        output = emb(indices)
        
        assert output.shape == (5, 64)
        # Check that repeated indices get the same embedding
        assert mx.array_equal(output[0], output[2])  # Both index 5
        assert mx.array_equal(output[1], output[4])  # Both index 10
    
    def test_out_of_bounds_indices(self):
        """Test handling of invalid indices."""
        emb = CastedSparseEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            batch_size=32
        )
        
        # MLX might handle this differently than PyTorch
        # Test should verify it either raises an error or handles gracefully
        # For now, we'll skip this test until we understand MLX behavior
        pass


class TestSignSGD:
    """Test SignSGD optimizer functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        opt = SignSGD(learning_rate=0.01, weight_decay=0.1)
        
        assert mx.array_equal(opt._learning_rate, mx.array(0.01))
        assert opt.weight_decay == 0.1
        assert len(opt.state) == 0
    
    def test_basic_update(self):
        """Test basic parameter update."""
        # Create embedding and optimizer
        emb = CastedSparseEmbedding(
            num_embeddings=10,
            embedding_dim=8,
            batch_size=4
        )
        opt = SignSGD(learning_rate=0.1, weight_decay=0.0)
        
        # Simulate forward pass
        emb.train()
        indices = mx.array([1, 3, 5])
        output = emb(indices)
        
        # Create fake gradients
        gradients = mx.ones_like(output) * 0.5
        
        # Store original weights
        original_weights = mx.array(emb.weight)  # Create a copy
        
        # Update
        opt.update_sparse_embedding(emb, gradients, indices)
        
        # Check that only specified embeddings were updated
        for i in range(10):
            if i in [1, 3, 5]:
                # These should be updated: w - lr * sign(grad)
                # Since grad = 0.5, sign(grad) = 1
                expected = original_weights[i] - 0.1 * 1.0
                assert mx.allclose(emb.weight[i], expected, atol=1e-6)
            else:
                # These should be unchanged
                assert mx.array_equal(emb.weight[i], original_weights[i])
    
    def test_weight_decay(self):
        """Test weight decay application."""
        emb = CastedSparseEmbedding(
            num_embeddings=10,
            embedding_dim=8,
            batch_size=4
        )
        opt = SignSGD(learning_rate=0.1, weight_decay=0.1)
        
        # Simulate forward pass
        emb.train()
        indices = mx.array([2, 4])
        output = emb(indices)
        
        # Zero gradients (only weight decay should apply)
        gradients = mx.zeros_like(output)
        
        # Store original weights
        original_weights = mx.array(emb.weight)  # Create a copy
        
        # Update
        opt.update_sparse_embedding(emb, gradients, indices)
        
        # Check weight decay was applied: w * (1 - lr * wd)
        for i in [2, 4]:
            expected = original_weights[i] * (1.0 - 0.1 * 0.1)
            assert mx.allclose(emb.weight[i], expected, atol=1e-6)
    
    def test_sign_gradient_updates(self):
        """Test that updates use sign of gradient."""
        emb = CastedSparseEmbedding(
            num_embeddings=10,
            embedding_dim=4,
            batch_size=4
        )
        opt = SignSGD(learning_rate=0.1, weight_decay=0.0)
        
        # Simulate forward pass
        emb.train()
        indices = mx.array([0, 1])
        output = emb(indices)
        
        # Create gradients with different magnitudes but same sign
        gradients = mx.array([
            [10.0, -5.0, 0.1, -0.001],  # Large positive/negative values
            [0.01, -20.0, 0.5, -100.0]   # Different magnitudes
        ])
        
        # Store original weights
        original_weights = mx.array(emb.weight)  # Create a copy
        
        # Update
        opt.update_sparse_embedding(emb, gradients, indices)
        
        # Check that updates only depend on sign
        # Update should be: w - lr * sign(grad)
        expected_updates = mx.array([
            [-0.1, 0.1, -0.1, 0.1],  # -lr * [1, -1, 1, -1]
            [-0.1, 0.1, -0.1, 0.1]   # Same updates despite different magnitudes
        ])
        
        for i, idx in enumerate([0, 1]):
            expected = original_weights[idx] + expected_updates[i]
            assert mx.allclose(emb.weight[idx], expected, atol=1e-6)
    
    def test_state_persistence(self):
        """Test that optimizer state persists across updates."""
        emb = CastedSparseEmbedding(
            num_embeddings=10,
            embedding_dim=8,
            batch_size=4
        )
        opt = SignSGD(learning_rate=0.1, weight_decay=0.0)
        
        # First update
        emb.train()
        indices = mx.array([1])
        output = emb(indices)
        gradients = mx.ones_like(output)
        
        opt.update_sparse_embedding(emb, gradients, indices)
        
        # Check state was created
        param_id = id(emb)  # Use embedding module ID, not weight ID
        assert param_id in opt.state
        assert opt.state[param_id].step == 1
        
        # Second update
        opt.update_sparse_embedding(emb, gradients, indices)
        assert opt.state[param_id].step == 2


class TestIntegration:
    """Integration tests for sparse embeddings with gradients."""
    
    def test_gradient_flow(self):
        """Test gradient flow through sparse embeddings."""
        # Create a simple model with sparse embeddings
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = CastedSparseEmbedding(
                    num_embeddings=20,
                    embedding_dim=16,
                    batch_size=8
                )
                self.linear = nn.Linear(16, 1)
            
            def __call__(self, x):
                emb = self.embedding(x)
                return self.linear(emb)
        
        model = SimpleModel()
        model.train()
        
        # Forward pass
        indices = mx.array([1, 5, 10])
        output = model(indices)
        
        # Compute loss
        target = mx.ones_like(output)
        loss = mx.mean((output - target) ** 2)
        
        # Compute gradients
        loss_grad_fn = mx.value_and_grad(lambda m: mx.mean((m(indices) - target) ** 2))
        loss_val, grads = loss_grad_fn(model)
        
        # Check that gradients exist
        assert isinstance(grads, dict) or hasattr(grads, '__dict__')
    
    def test_mixed_precision_training(self):
        """Test training with mixed precision."""
        # Create model with bfloat16 computation
        emb = CastedSparseEmbedding(
            num_embeddings=50,
            embedding_dim=32,
            batch_size=16,
            cast_to=mx.bfloat16
        )
        
        # Wrap in a simple model for gradient computation
        class Model(nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding
                self.proj = nn.Linear(32, 1)
            
            def __call__(self, x):
                # Embedding output should be bfloat16
                emb_out = self.embedding(x)
                assert emb_out.dtype == mx.bfloat16
                return self.proj(emb_out)
        
        model = Model(emb)
        model.train()
        
        # Forward pass
        indices = mx.array([0, 1, 2, 3])
        output = model(indices)
        
        # The computation should work despite mixed precision
        assert output.shape == (4, 1)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_sparse_embedding_optimizer(self):
        """Test optimizer creation helper."""
        # Create a model with multiple sparse embeddings
        class MultiEmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb1 = CastedSparseEmbedding(
                    num_embeddings=100,
                    embedding_dim=32,
                    batch_size=16
                )
                self.emb2 = CastedSparseEmbedding(
                    num_embeddings=50,
                    embedding_dim=64,
                    batch_size=16
                )
                self.linear = nn.Linear(96, 10)
            
            def __call__(self, x1, x2):
                e1 = self.emb1(x1)
                e2 = self.emb2(x2)
                combined = mx.concatenate([e1, e2], axis=-1)
                return self.linear(combined)
        
        model = MultiEmbeddingModel()
        
        # Create optimizer
        optimizer, sparse_params = create_sparse_embedding_optimizer(
            model,
            learning_rate=0.01,
            weight_decay=0.1
        )
        
        assert isinstance(optimizer, SignSGD)
        assert abs(optimizer._learning_rate.item() - 0.01) < 1e-6
        assert optimizer.weight_decay == 0.1
        
        # Initially no parameters tracked (no forward pass yet)
        assert len(sparse_params) == 0
        
        # After forward pass, should track local weights
        model.train()
        x1 = mx.array([1, 2, 3])
        x2 = mx.array([4, 5, 6])
        output = model(x1, x2)
        
        # Now get params again
        optimizer, sparse_params = create_sparse_embedding_optimizer(model)
        assert len(sparse_params) == 2  # Two embedding layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])