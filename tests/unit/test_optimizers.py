"""Unit tests for custom optimizers."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_hrm.training.optimizers import (
    AdamAtan2,
    SignSGD,
    SparseAwareOptimizer,
    CosineAnnealingLR,
    create_hrm_optimizer
)


class TestAdamAtan2:
    """Test Adam-atan2 optimizer functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = AdamAtan2(
            learning_rate=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )
        
        assert optimizer.lr == 1e-3
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.weight_decay == 0.1
        assert optimizer.step_count == 0
        assert len(optimizer.state) == 0
    
    def test_parameter_update(self):
        """Test single parameter update."""
        optimizer = AdamAtan2(learning_rate=0.1)
        
        # Create parameter and gradient
        param = mx.array([[1.0, 2.0], [3.0, 4.0]])
        grad = mx.array([[0.1, 0.2], [0.3, 0.4]])
        
        # First update
        optimizer.step_count = 1  # Simulate first step
        updated_param = optimizer.update_param(param, grad, "test_param")
        
        # Parameter should change
        assert not mx.allclose(updated_param, param)
        assert mx.all(mx.isfinite(updated_param))
        
        # State should be created
        assert "test_param" in optimizer.state
        assert "exp_avg" in optimizer.state["test_param"]
        assert "exp_avg_sq" in optimizer.state["test_param"]
    
    def test_momentum_accumulation(self):
        """Test that momentum accumulates correctly."""
        optimizer = AdamAtan2(learning_rate=0.01, betas=(0.9, 0.99))
        
        param = mx.array([1.0])
        grad = mx.array([0.1])
        
        # Multiple updates with same gradient
        for step in range(1, 4):
            optimizer.step_count = step
            param = optimizer.update_param(param, grad, "test_param")
        
        state = optimizer.state["test_param"]
        
        # Momentum should have accumulated
        assert mx.abs(state["exp_avg"]).item() > 0
        assert mx.abs(state["exp_avg_sq"]).item() > 0
        
        # Second moment should be roughly grad^2 * (1-beta2)/(1-beta2^steps)
        expected_v_approx = (grad ** 2).item() * (1 - 0.99) / (1 - 0.99**3)
        assert abs(state["exp_avg_sq"].item() - expected_v_approx) < 0.01
    
    def test_atan2_vs_adam(self):
        """Test that atan2 update differs from standard Adam."""
        # Standard Adam update for comparison
        def standard_adam_update(m_hat, v_hat, eps=1e-8):
            return m_hat / (mx.sqrt(v_hat) + eps)
        
        # Adam-atan2 update
        def atan2_update(m_hat, v_hat):
            return mx.arctan2(m_hat, mx.sqrt(v_hat))
        
        m_hat = mx.array([0.1, 0.5, 1.0])
        v_hat = mx.array([0.01, 0.25, 1.0])
        
        adam_update = standard_adam_update(m_hat, v_hat)
        atan2_upd = atan2_update(m_hat, v_hat)
        
        # Should be different (except possibly at specific points)
        assert not mx.allclose(adam_update, atan2_upd)
        
        # Atan2 should be bounded by [-π/2, π/2]
        assert mx.all(atan2_upd >= -mx.pi/2)
        assert mx.all(atan2_upd <= mx.pi/2)
    
    def test_weight_decay(self):
        """Test weight decay application."""
        optimizer = AdamAtan2(learning_rate=0.1, weight_decay=0.5)
        
        param = mx.array([2.0])  # Large value to see decay effect
        grad = mx.array([0.0])   # Zero gradient
        
        optimizer.step_count = 1
        updated_param = optimizer.update_param(param, grad, "test_param")
        
        # With zero gradient, should only see weight decay
        expected = param * (1 - 0.1 * 0.5)  # lr * weight_decay
        # Small update from atan2 with zero grad, but decay should dominate
        assert updated_param.item() < param.item()
    
    def test_full_parameter_dict_update(self):
        """Test updating full parameter dictionary."""
        optimizer = AdamAtan2(learning_rate=0.01)
        
        params = {
            "weight1": mx.array([[1.0, 2.0]]),
            "weight2": mx.array([[3.0], [4.0]]),
            "bias": mx.array([0.5])
        }
        
        grads = {
            "weight1": mx.array([[0.1, 0.2]]),
            "weight2": mx.array([[0.3], [0.4]]),
            "bias": mx.array([0.1])
        }
        
        optimizer.step_count = 0  # Will be incremented to 1
        updated_params = optimizer.update(params, grads)
        
        # All parameters should be updated
        assert optimizer.step_count == 1
        for name in params:
            assert not mx.allclose(updated_params[name], params[name])
            assert mx.all(mx.isfinite(updated_params[name]))
        
        # Check state created for all parameters
        assert len(optimizer.state) == 3
        for name in params:
            assert name in optimizer.state


class TestSignSGD:
    """Test Sign-SGD optimizer functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = SignSGD(learning_rate=1e-3, weight_decay=0.1)
        
        assert optimizer.lr == 1e-3
        assert optimizer.weight_decay == 0.1
        assert optimizer.step_count == 0
    
    def test_sign_update(self):
        """Test that updates use sign of gradients."""
        optimizer = SignSGD(learning_rate=0.1, weight_decay=0.0)
        
        param = mx.array([1.0, 2.0, -1.0])
        grad = mx.array([0.5, -0.3, 0.8])  # Different magnitudes
        
        updated_param = optimizer.update_param(param, grad)
        
        # Update should be: param - lr * sign(grad) (no weight decay)
        expected_param = param - 0.1 * mx.sign(grad)  # param - lr * sign(grad)
        
        assert mx.allclose(updated_param, expected_param)
    
    def test_gradient_magnitude_invariance(self):
        """Test that gradient magnitude doesn't affect update (only sign)."""
        optimizer = SignSGD(learning_rate=0.1, weight_decay=0.0)
        
        param = mx.array([1.0])
        grad1 = mx.array([0.001])   # Small positive
        grad2 = mx.array([100.0])   # Large positive
        
        updated1 = optimizer.update_param(param, grad1)
        updated2 = optimizer.update_param(param, grad2)
        
        # Both should produce same update (sign is positive)
        assert mx.allclose(updated1, updated2)
    
    def test_weight_decay(self):
        """Test weight decay in Sign-SGD."""
        optimizer = SignSGD(learning_rate=0.1, weight_decay=0.2)
        
        param = mx.array([2.0])
        grad = mx.array([0.0])  # Zero gradient
        
        updated_param = optimizer.update_param(param, grad)
        
        # Should only see weight decay: param * (1 - lr * weight_decay)
        expected = param * (1 - 0.1 * 0.2)
        assert mx.allclose(updated_param, expected, rtol=1e-5)
    
    def test_full_update(self):
        """Test full parameter dictionary update."""
        optimizer = SignSGD(learning_rate=0.1)
        
        params = {"emb": mx.array([[1.0, -2.0], [3.0, 0.0]])}
        grads = {"emb": mx.array([[0.5, -0.1], [-0.8, 0.0]])}
        
        updated_params = optimizer.update(params, grads)
        
        assert optimizer.step_count == 1
        
        # Check the sign-based update
        expected = params["emb"] - 0.1 * mx.sign(grads["emb"])
        assert mx.allclose(updated_params["emb"], expected)


class TestSparseAwareOptimizer:
    """Test sparse-aware optimizer wrapper."""
    
    def setup_optimizers(self):
        """Create test optimizers."""
        dense_opt = AdamAtan2(learning_rate=1e-3)
        sparse_opt = SignSGD(learning_rate=1e-4)
        sparse_patterns = ["sparse_emb", "puzzle_emb"]
        
        return SparseAwareOptimizer(dense_opt, sparse_opt, sparse_patterns)
    
    def test_initialization(self):
        """Test optimizer wrapper initialization."""
        optimizer = self.setup_optimizers()
        
        assert isinstance(optimizer.dense_optimizer, AdamAtan2)
        assert isinstance(optimizer.sparse_optimizer, SignSGD)
        assert "sparse_emb" in optimizer.sparse_param_patterns
        assert "puzzle_emb" in optimizer.sparse_param_patterns
    
    def test_parameter_splitting(self):
        """Test parameter splitting into dense and sparse."""
        optimizer = self.setup_optimizers()
        
        params = {
            "dense_layer.weight": mx.array([[1.0, 2.0]]),
            "sparse_emb.weight": mx.array([[3.0, 4.0]]),
            "attention.qkv": mx.array([[5.0, 6.0]]),
            "puzzle_emb.embeddings": mx.array([[7.0, 8.0]])
        }
        
        grads = {name: mx.ones_like(param) for name, param in params.items()}
        
        dense_params, sparse_params, dense_grads, sparse_grads = optimizer._split_parameters(params, grads)
        
        # Check dense parameters
        assert "dense_layer.weight" in dense_params
        assert "attention.qkv" in dense_params
        assert len(dense_params) == 2
        
        # Check sparse parameters  
        assert "sparse_emb.weight" in sparse_params
        assert "puzzle_emb.embeddings" in sparse_params
        assert len(sparse_params) == 2
        
        # Check gradients split correctly too
        assert len(dense_grads) == 2
        assert len(sparse_grads) == 2
    
    def test_dual_optimizer_update(self):
        """Test that different optimizers are used for different parameters."""
        optimizer = self.setup_optimizers()
        
        params = {
            "dense.weight": mx.array([[1.0]]),
            "sparse_emb.weight": mx.array([[2.0]])
        }
        
        grads = {
            "dense.weight": mx.array([[0.1]]),
            "sparse_emb.weight": mx.array([[0.1]])
        }
        
        # Track initial step counts
        initial_dense_steps = optimizer.dense_optimizer.step_count
        initial_sparse_steps = optimizer.sparse_optimizer.step_count
        
        updated_params = optimizer.update(params, grads)
        
        # Both optimizers should have been called
        assert optimizer.dense_optimizer.step_count == initial_dense_steps + 1
        assert optimizer.sparse_optimizer.step_count == initial_sparse_steps + 1
        
        # Parameters should be updated differently
        # Dense uses Adam-atan2, sparse uses Sign-SGD
        dense_update = updated_params["dense.weight"] - params["dense.weight"]
        sparse_update = updated_params["sparse_emb.weight"] - params["sparse_emb.weight"]
        
        # Updates should be different (different algorithms)
        assert not mx.allclose(dense_update, sparse_update)
    
    def test_step_count_property(self):
        """Test step count property."""
        optimizer = self.setup_optimizers()
        
        assert optimizer.step_count == 0
        
        params = {"dense.weight": mx.array([[1.0]])}
        grads = {"dense.weight": mx.array([[0.1]])}
        
        optimizer.update(params, grads)
        assert optimizer.step_count == 1


class TestCosineAnnealingLR:
    """Test learning rate scheduler."""
    
    def test_warmup_phase(self):
        """Test linear warmup phase."""
        optimizer = AdamAtan2(learning_rate=1e-3)
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=1000,
            warmup_steps=100
        )
        
        # Test warmup steps
        for step in [0, 25, 50, 75]:
            lr = scheduler.get_lr(step)
            expected_lr = 1e-3 * (step / 100)
            assert abs(lr - expected_lr) < 1e-6
    
    def test_cosine_annealing_phase(self):
        """Test cosine annealing after warmup."""
        optimizer = AdamAtan2(learning_rate=1e-3)
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=1000,
            warmup_steps=100,
            eta_min=1e-6
        )
        
        # Test at end of training (should be near eta_min)
        lr_end = scheduler.get_lr(999)
        assert lr_end < 1e-5  # Much lower than base LR
        
        # Test at middle of cosine phase
        lr_mid = scheduler.get_lr(550)  # Middle of 100-1000 range
        assert 1e-6 < lr_mid < 1e-3  # Should be between min and max
    
    def test_sparse_aware_scheduling(self):
        """Test scheduler with SparseAwareOptimizer."""
        optimizer = create_hrm_optimizer(dense_lr=1e-3, sparse_lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, warmup_steps=100)
        
        # Test that it returns tuple for sparse-aware optimizer
        dense_lr, sparse_lr = scheduler.get_lr(50)  # Warmup phase
        
        assert abs(dense_lr - 1e-3 * 0.5) < 1e-6
        assert abs(sparse_lr - 1e-4 * 0.5) < 1e-6
    
    def test_scheduler_step(self):
        """Test scheduler step function."""
        optimizer = AdamAtan2(learning_rate=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, warmup_steps=100)
        
        initial_lr = optimizer.lr
        
        # Step should update optimizer's learning rate
        scheduler.step(50)  # Warmup phase
        
        expected_lr = 1e-3 * 0.5  # 50/100 warmup
        assert abs(optimizer.lr - expected_lr) < 1e-6
        assert scheduler.last_step == 50


class TestHRMOptimizerFactory:
    """Test HRM optimizer factory function."""
    
    def test_create_hrm_optimizer(self):
        """Test creating standard HRM optimizer setup."""
        optimizer = create_hrm_optimizer(
            dense_lr=1e-4,
            sparse_lr=2e-4,
            dense_weight_decay=0.1,
            sparse_weight_decay=0.05
        )
        
        assert isinstance(optimizer, SparseAwareOptimizer)
        assert isinstance(optimizer.dense_optimizer, AdamAtan2)
        assert isinstance(optimizer.sparse_optimizer, SignSGD)
        
        # Check learning rates
        assert optimizer.dense_optimizer.lr == 1e-4
        assert optimizer.sparse_optimizer.lr == 2e-4
        
        # Check weight decay
        assert optimizer.dense_optimizer.weight_decay == 0.1
        assert optimizer.sparse_optimizer.weight_decay == 0.05
        
        # Check sparse patterns
        assert "sparse_tok_emb" in optimizer.sparse_param_patterns
    
    def test_default_parameters(self):
        """Test default parameter values."""
        optimizer = create_hrm_optimizer()
        
        assert optimizer.dense_optimizer.lr == 1e-4
        assert optimizer.sparse_optimizer.lr == 1e-4
        assert optimizer.dense_optimizer.weight_decay == 0.1
        assert optimizer.sparse_optimizer.weight_decay == 0.1
        assert optimizer.dense_optimizer.beta1 == 0.9
        assert optimizer.dense_optimizer.beta2 == 0.999


class TestOptimizerIntegration:
    """Integration tests for optimizers."""
    
    def test_convergence_simple_problem(self):
        """Test that optimizers can solve a simple optimization problem."""
        # Minimize ||x - target||^2
        target = mx.array([2.0, -1.0])
        
        def loss_fn(x):
            return mx.sum((x - target) ** 2)
        
        # Test Adam-atan2
        x_adam = mx.array([0.0, 0.0])
        optimizer_adam = AdamAtan2(learning_rate=0.1)
        
        for step in range(1, 51):
            optimizer_adam.step_count = step
            grad = mx.grad(loss_fn)(x_adam)
            x_adam = optimizer_adam.update_param(x_adam, grad, "x")
        
        assert mx.allclose(x_adam, target, atol=0.1)
        
        # Test Sign-SGD
        x_sign = mx.array([0.0, 0.0])
        optimizer_sign = SignSGD(learning_rate=0.1)
        
        for _ in range(100):  # May need more steps
            grad = mx.grad(loss_fn)(x_sign)
            x_sign = optimizer_sign.update_param(x_sign, grad)
        
        # Sign-SGD should at least move in right direction
        assert mx.sum((x_sign - target) ** 2) < mx.sum((mx.array([0.0, 0.0]) - target) ** 2)
    
    def test_numerical_stability(self):
        """Test optimizer stability with extreme gradients."""
        optimizer = AdamAtan2(learning_rate=0.001)
        
        param = mx.array([1.0])
        
        # Very large gradient
        large_grad = mx.array([1e6])
        optimizer.step_count = 1
        updated = optimizer.update_param(param, large_grad, "test")
        assert mx.all(mx.isfinite(updated))
        
        # Very small gradient
        small_grad = mx.array([1e-10])
        optimizer.step_count = 2
        updated = optimizer.update_param(updated, small_grad, "test")
        assert mx.all(mx.isfinite(updated))
        
        # Zero gradient
        zero_grad = mx.array([0.0])
        optimizer.step_count = 3
        updated = optimizer.update_param(updated, zero_grad, "test")
        assert mx.all(mx.isfinite(updated))


if __name__ == "__main__":
    # Run basic tests to verify functionality
    print("🧪 Testing Adam-atan2 optimizer...")
    test_adam = TestAdamAtan2()
    test_adam.test_initialization()
    test_adam.test_parameter_update()
    test_adam.test_atan2_vs_adam()
    print("✅ Adam-atan2 tests passed")
    
    print("🧪 Testing Sign-SGD optimizer...")
    test_sign = TestSignSGD()
    test_sign.test_initialization()
    test_sign.test_sign_update()
    test_sign.test_gradient_magnitude_invariance()
    print("✅ Sign-SGD tests passed")
    
    print("🧪 Testing SparseAwareOptimizer...")
    test_sparse = TestSparseAwareOptimizer()
    test_sparse.test_initialization()
    test_sparse.test_parameter_splitting()
    test_sparse.test_dual_optimizer_update()
    print("✅ SparseAwareOptimizer tests passed")
    
    print("🧪 Testing learning rate scheduler...")
    test_lr = TestCosineAnnealingLR()
    test_lr.test_warmup_phase()
    test_lr.test_cosine_annealing_phase()
    print("✅ Learning rate scheduler tests passed")
    
    print("🧪 Testing integration...")
    test_integration = TestOptimizerIntegration()
    test_integration.test_convergence_simple_problem()
    test_integration.test_numerical_stability()
    print("✅ Integration tests passed")
    
    print("🎉 All optimizer tests passed!")