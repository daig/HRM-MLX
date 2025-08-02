"""Unit tests for RMSNorm implementations."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_hrm.layers.normalization import (
    rms_norm, 
    RMSNorm, 
    RMSNormCompatible,
    create_rms_norm
)


class TestRMSNormFunctional:
    """Test the functional RMSNorm implementation."""
    
    def test_shape_preservation(self):
        """Test that RMSNorm preserves input shapes."""
        test_shapes = [
            (32, 768),           # 2D input
            (4, 128, 768),       # 3D input (batch, seq, hidden)
            (2, 8, 64, 768),     # 4D input
            (1, 512),            # Small batch
        ]
        
        for shape in test_shapes:
            x = mx.random.normal(shape)
            output = rms_norm(x)
            assert output.shape == shape, f"Shape mismatch for {shape}"
    
    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        shape = (2, 10, 64)
        
        # Test float32
        x_f32 = mx.random.normal(shape, dtype=mx.float32)
        output_f32 = rms_norm(x_f32)
        assert output_f32.dtype == mx.float32
        
        # Test float16
        x_f16 = mx.random.normal(shape, dtype=mx.float16)
        output_f16 = rms_norm(x_f16)
        assert output_f16.dtype == mx.float16
        
        # Test bfloat16 if supported
        if hasattr(mx, 'bfloat16'):
            x_bf16 = mx.random.normal(shape, dtype=mx.bfloat16)
            output_bf16 = rms_norm(x_bf16)
            assert output_bf16.dtype == mx.bfloat16
    
    def test_normalization_property(self):
        """Test that output has approximately unit RMS."""
        batch_size, seq_len, hidden_dim = 2, 10, 64
        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        
        output = rms_norm(x)
        
        # Compute RMS of output
        rms = mx.sqrt(mx.mean(mx.square(output), axis=-1))
        
        # Should be close to 1
        expected = mx.ones_like(rms)
        assert mx.allclose(rms, expected, atol=1e-3), "RMS not close to 1"
    
    def test_epsilon_effect(self):
        """Test that epsilon prevents division by zero."""
        # Create input with very small values
        x = mx.full((2, 10, 64), 1e-10)
        
        # Should not produce NaN or Inf
        output = rms_norm(x, variance_epsilon=1e-5)
        assert not mx.any(mx.isnan(output)), "Output contains NaN"
        assert not mx.any(mx.isinf(output)), "Output contains Inf"
    
    def test_zero_input(self):
        """Test behavior with zero input."""
        x = mx.zeros((2, 10, 64))
        output = rms_norm(x, variance_epsilon=1e-5)
        
        # Should be all zeros (0 / sqrt(eps) * rsqrt(eps) = 0)
        assert mx.allclose(output, mx.zeros_like(output)), "Zero input should produce zero output"


class TestRMSNormModule:
    """Test the module-based RMSNorm implementation."""
    
    def test_with_scale_parameter(self):
        """Test RMSNorm with learnable scale parameter."""
        hidden_dim = 64
        norm = RMSNorm(hidden_dim, use_scale=True)
        
        x = mx.random.normal((2, 10, hidden_dim))
        output = norm(x)
        
        assert output.shape == x.shape
        # Check that internal norm exists
        assert norm._norm is not None
    
    def test_without_scale_parameter(self):
        """Test RMSNorm without learnable scale parameter."""
        hidden_dim = 64
        norm = RMSNorm(hidden_dim, use_scale=False)
        
        x = mx.random.normal((2, 10, hidden_dim))
        output = norm(x)
        
        assert output.shape == x.shape
        # Should use functional implementation
        assert norm._norm is None
        
        # Should match functional implementation
        expected = rms_norm(x, norm.eps)
        assert mx.allclose(output, expected, atol=1e-6)
    
    def test_compatible_module(self):
        """Test RMSNormCompatible matches functional behavior."""
        norm = RMSNormCompatible(eps=1e-5)
        
        x = mx.random.normal((2, 10, 64))
        output = norm(x)
        
        # Should match functional implementation exactly
        expected = rms_norm(x, variance_epsilon=1e-5)
        assert mx.array_equal(output, expected)


class TestRMSNormFactory:
    """Test the factory function for creating RMSNorm variants."""
    
    def test_functional_only(self):
        """Test functional_only returns None."""
        norm = create_rms_norm(functional_only=True)
        assert norm is None
    
    def test_with_scale(self):
        """Test creation with scale parameter."""
        norm = create_rms_norm(dims=64, use_scale=True)
        assert isinstance(norm, RMSNorm)
        assert norm.use_scale is True
    
    def test_without_scale(self):
        """Test creation without scale parameter."""
        norm = create_rms_norm(use_scale=False)
        assert isinstance(norm, RMSNormCompatible)
    
    def test_missing_dims_error(self):
        """Test error when dims not specified with use_scale=True."""
        with pytest.raises(ValueError, match="dims must be specified"):
            create_rms_norm(use_scale=True)


class TestNumericalStability:
    """Test numerical stability in edge cases."""
    
    def test_large_values(self):
        """Test RMSNorm with very large values."""
        x = mx.full((2, 10, 64), 1e8)
        output = rms_norm(x)
        
        assert not mx.any(mx.isnan(output)), "Large values produced NaN"
        assert not mx.any(mx.isinf(output)), "Large values produced Inf"
        
        # Should normalize to approximately 1
        rms = mx.sqrt(mx.mean(mx.square(output), axis=-1))
        assert mx.allclose(rms, mx.ones_like(rms), atol=1e-3)
    
    def test_small_values(self):
        """Test RMSNorm with very small values."""
        x = mx.full((2, 10, 64), 1e-8)
        output = rms_norm(x)
        
        assert not mx.any(mx.isnan(output)), "Small values produced NaN"
        assert not mx.any(mx.isinf(output)), "Small values produced Inf"
    
    def test_mixed_values(self):
        """Test RMSNorm with mixed large and small values."""
        x = mx.random.normal((2, 10, 64))
        # Add some extreme values
        x[0, 0, :10] = 1e8
        x[0, 1, :10] = 1e-8
        
        output = rms_norm(x)
        assert not mx.any(mx.isnan(output)), "Mixed values produced NaN"
        assert not mx.any(mx.isinf(output)), "Mixed values produced Inf"
    
    def test_different_epsilon_values(self):
        """Test stability with different epsilon values."""
        x = mx.random.normal((2, 10, 64)) * 0.01  # Small scale input
        
        epsilons = [1e-8, 1e-6, 1e-5, 1e-3]
        for eps in epsilons:
            output = rms_norm(x, variance_epsilon=eps)
            assert not mx.any(mx.isnan(output)), f"Epsilon {eps} produced NaN"
            assert not mx.any(mx.isinf(output)), f"Epsilon {eps} produced Inf"


class TestGradientFlow:
    """Test gradient flow through RMSNorm."""
    
    def test_gradient_functional(self):
        """Test gradients through functional RMSNorm."""
        def loss_fn(x):
            normalized = rms_norm(x)
            return mx.sum(normalized)
        
        x = mx.random.normal((2, 10, 64))
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(x)
        
        assert grad.shape == x.shape
        assert not mx.any(mx.isnan(grad)), "Gradient contains NaN"
        assert not mx.any(mx.isinf(grad)), "Gradient contains Inf"
    
    def test_gradient_module_with_scale(self):
        """Test gradients through module RMSNorm with scale."""
        hidden_dim = 64
        norm = RMSNorm(hidden_dim, use_scale=True)
        
        def loss_fn(x):
            return mx.sum(norm(x))
        
        x = mx.random.normal((2, 10, hidden_dim))
        
        # Get gradients w.r.t. input
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(x)
        
        assert grad.shape == x.shape
        assert not mx.any(mx.isnan(grad)), "Gradient contains NaN"
        
        # Get gradients w.r.t. parameters
        loss, grads = mx.value_and_grad(loss_fn)(x)
        assert not mx.isnan(loss), "Loss is NaN"
    
    def test_gradient_module_without_scale(self):
        """Test gradients through module RMSNorm without scale."""
        norm = RMSNormCompatible(eps=1e-5)
        
        def loss_fn(x):
            return mx.sum(norm(x))
        
        x = mx.random.normal((2, 10, 64))
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(x)
        
        assert grad.shape == x.shape
        assert not mx.any(mx.isnan(grad)), "Gradient contains NaN"


class TestEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_single_element(self):
        """Test with single element tensors."""
        x = mx.array([[[1.0]]])  # Shape (1, 1, 1)
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        x = mx.random.normal((1, 128, 512))
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_very_long_sequence(self):
        """Test with very long sequences."""
        x = mx.random.normal((2, 2048, 64))
        output = rms_norm(x)
        assert output.shape == x.shape
        
        # Check normalization still works
        rms = mx.sqrt(mx.mean(mx.square(output), axis=-1))
        assert mx.allclose(rms, mx.ones_like(rms), atol=1e-3)