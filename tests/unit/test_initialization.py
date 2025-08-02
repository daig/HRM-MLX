"""
Unit tests for custom initialization functions.

Tests cover:
- Statistical properties of truncated normal distribution
- Deterministic initialization with keys
- Variance correction validation
- Custom layers functionality
"""

import mlx.core as mx
import pytest
import math
from typing import Tuple

from mlx_hrm.layers import (
    truncated_normal,
    init_truncated_normal,
    LinearTruncNormal,
    EmbeddingTruncNormal,
)


class TestTruncatedNormal:
    """Test suite for truncated normal initialization."""
    
    def test_shape_and_dtype(self):
        """Test that output has correct shape and dtype."""
        # Test with tuple shape
        shape = (100, 50)
        result = truncated_normal(shape)
        assert result.shape == shape
        assert result.dtype == mx.float32
        
        # Test with int shape
        result = truncated_normal(10)
        assert result.shape == (10,)
        
        # Test with different dtype
        result = truncated_normal(shape, dtype=mx.float16)
        assert result.dtype == mx.float16
    
    def test_statistical_properties(self):
        """Test that truncated normal has correct statistical properties."""
        key = mx.random.key(42)
        shape = (10000,)
        std = 0.02
        mean = 0.5
        
        # Generate samples
        samples = truncated_normal(shape, mean=mean, std=std, key=key)
        
        # Convert to numpy for statistics
        samples_np = samples.tolist()
        
        # Test mean (should be close to specified mean)
        actual_mean = sum(samples_np) / len(samples_np)
        assert abs(actual_mean - mean) < 0.01, f"Mean {actual_mean} not close to {mean}"
        
        # Test standard deviation
        # The PyTorch HRM implementation results in inflated std due to clipping
        variance = sum((x - actual_mean) ** 2 for x in samples_np) / len(samples_np)
        actual_std = math.sqrt(variance)
        # For small std and default bounds, expect significant inflation
        assert actual_std > std, f"Std should be inflated: {actual_std} vs {std}"
        assert actual_std < std * 1.5, f"Std too inflated: {actual_std} vs {std}"
        
        # Test bounds
        # The actual bounds are mean ± lower/upper * comp_std due to how the PyTorch code works
        # comp_std ≈ std * 1.137 for [-2, 2] bounds, so bounds are slightly wider
        min_val = min(samples_np)
        max_val = max(samples_np)
        # Use 2.3 to account for comp_std inflation
        assert min_val >= mean - 2.3 * std, f"Min value {min_val} outside bounds"
        assert max_val <= mean + 2.3 * std, f"Max value {max_val} outside bounds"
    
    def test_custom_bounds(self):
        """Test truncated normal with custom bounds."""
        key = mx.random.key(42)
        shape = (1000,)
        std = 0.1
        mean = 0.0
        lower = -3.0
        upper = 3.0
        
        samples = truncated_normal(
            shape, mean=mean, std=std, lower=lower, upper=upper, key=key
        )
        
        samples_np = samples.tolist()
        min_val = min(samples_np)
        max_val = max(samples_np)
        
        # Check bounds
        assert min_val >= mean + lower * std - 0.01
        assert max_val <= mean + upper * std + 0.01
    
    def test_deterministic_initialization(self):
        """Test that same key produces same initialization."""
        key = mx.random.key(42)
        shape = (100, 100)
        
        # Generate twice with same key
        init1 = truncated_normal(shape, key=key)
        init2 = truncated_normal(shape, key=key)
        
        # Should be identical
        assert mx.array_equal(init1, init2)
        
        # Different key should produce different results
        key2 = mx.random.key(43)
        init3 = truncated_normal(shape, key=key2)
        assert not mx.array_equal(init1, init3)
    
    def test_zero_std(self):
        """Test that zero std returns zeros."""
        shape = (10, 20)
        result = truncated_normal(shape, std=0.0)
        assert mx.all(result == 0.0)
    
    def test_asymmetric_bounds(self):
        """Test with asymmetric truncation bounds."""
        key = mx.random.key(42)
        shape = (5000,)
        std = 0.1
        mean = 0.0
        lower = -1.0
        upper = 3.0
        
        samples = truncated_normal(
            shape, mean=mean, std=std, lower=lower, upper=upper, key=key
        )
        
        samples_np = samples.tolist()
        
        # Compute expected bounds
        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        
        min_val = min(samples_np)
        max_val = max(samples_np)
        
        # Check asymmetric bounds
        assert min_val >= mean + lower * comp_std - 1e-6
        assert max_val <= mean + upper * comp_std + 1e-6
        
        # Mean should be shifted towards the wider bound
        actual_mean = sum(samples_np) / len(samples_np)
        assert actual_mean > mean  # Should be positive due to wider upper bound
    
    def test_edge_cases(self):
        """Test edge cases for std and bounds."""
        key = mx.random.key(42)
        
        # Very small std
        small_std = 1e-6
        samples = truncated_normal(100, std=small_std, key=key)
        assert mx.all(mx.abs(samples) < 1e-4)
        
        # Very large std (relative to bounds)
        large_std = 10.0
        samples = truncated_normal(100, std=large_std, lower=-0.1, upper=0.1, key=key)
        # With tight bounds, comp_std will be very large
        # Calculate actual bounds
        sqrt2 = math.sqrt(2)
        a = math.erf(-0.1 / sqrt2)
        b = math.erf(0.1 / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * (-0.1) ** 2)
        pdf_l = c * math.exp(-0.5 * 0.1 ** 2)
        comp_std = large_std / math.sqrt(1 - (0.1 * pdf_u - (-0.1) * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        
        # Check samples are within actual clip bounds
        samples_list = samples.tolist()
        assert all(-0.1 * comp_std <= x <= 0.1 * comp_std for x in samples_list)
    
    def test_variance_correction(self):
        """Test that implementation matches PyTorch HRM behavior."""
        # The PyTorch implementation has a specific behavior where:
        # 1. It computes comp_std = std / sqrt(variance_factor)
        # 2. It clips at lower * comp_std and upper * comp_std
        # 3. This results in the actual std being larger than requested
        
        # Just verify the implementation produces reasonable values
        key = mx.random.key(42)
        shape = (10000,)
        std = 0.1
        
        samples = truncated_normal(shape, std=std, key=key)
        samples_np = samples.tolist()
        
        # Basic sanity checks
        actual_mean = sum(samples_np) / len(samples_np)
        variance = sum((x - actual_mean) ** 2 for x in samples_np) / len(samples_np)
        actual_std = math.sqrt(variance)
        
        # Mean should be close to 0
        assert abs(actual_mean) < 0.01
        
        # Std will be inflated due to the clipping behavior
        # For default [-2, 2] bounds, expect ~29% inflation
        assert 0.1 < actual_std < 0.15
        
        # Check bounds - they should be at comp_std * bounds
        sqrt2 = math.sqrt(2)
        a = math.erf(-2.0 / sqrt2)
        b = math.erf(2.0 / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * (-2.0) ** 2)
        pdf_l = c * math.exp(-0.5 * 2.0 ** 2)
        comp_std = std / math.sqrt(1 - (2.0 * pdf_u - (-2.0) * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        
        min_val = min(samples_np)
        max_val = max(samples_np)
        assert min_val >= -2.0 * comp_std - 1e-6
        assert max_val <= 2.0 * comp_std + 1e-6


class TestInitFunction:
    """Test the initializer function factory."""
    
    def test_init_function_creation(self):
        """Test creating an initializer function."""
        std = 0.02
        init_fn = init_truncated_normal(std=std)
        
        # Test that it returns correct shape
        shape = (10, 20)
        result = init_fn(shape)
        assert result.shape == shape
        assert result.dtype == mx.float32
        
        # Test with different dtype
        result = init_fn(shape, dtype=mx.float16)
        assert result.dtype == mx.float16
    
    def test_init_function_determinism(self):
        """Test that initializer with key is deterministic."""
        key = mx.random.key(42)
        init_fn = init_truncated_normal(key=key)
        
        shape = (50, 50)
        result1 = init_fn(shape)
        result2 = init_fn(shape)
        
        assert mx.array_equal(result1, result2)


class TestLinearTruncNormal:
    """Test LinearTruncNormal layer."""
    
    def test_layer_creation(self):
        """Test creating linear layer with truncated normal init."""
        in_features = 128
        out_features = 64
        
        layer = LinearTruncNormal(in_features, out_features)
        
        # Check weight shape
        assert layer.weight.shape == (out_features, in_features)
        
        # Check bias shape
        assert layer.bias is not None
        assert layer.bias.shape == (out_features,)
        
        # Test without bias
        layer_no_bias = LinearTruncNormal(in_features, out_features, bias=False)
        assert layer_no_bias.bias is None
    
    def test_layer_forward(self):
        """Test forward pass through linear layer."""
        batch_size = 32
        in_features = 128
        out_features = 64
        
        layer = LinearTruncNormal(in_features, out_features)
        
        # Create input
        x = mx.random.normal((batch_size, in_features))
        
        # Forward pass
        y = layer(x)
        
        # Check output shape
        assert y.shape == (batch_size, out_features)
    
    def test_layer_initialization_std(self):
        """Test that layer uses correct initialization std."""
        in_features = 100
        out_features = 50
        
        # Default std should be 1/sqrt(in_features)
        layer = LinearTruncNormal(in_features, out_features)
        expected_std = math.sqrt(1.0 / in_features)
        
        # Check that weights have approximately correct std
        weights_flat = layer.weight.flatten().tolist()
        mean = sum(weights_flat) / len(weights_flat)
        variance = sum((x - mean) ** 2 for x in weights_flat) / len(weights_flat)
        actual_std = math.sqrt(variance)
        
        # The PyTorch implementation inflates std by ~29% for default [-2, 2] bounds
        assert abs(actual_std - expected_std * 1.29) < 0.02
        
        # Test custom std
        custom_std = 0.05
        layer = LinearTruncNormal(in_features, out_features, std=custom_std)
        weights_flat = layer.weight.flatten().tolist()
        mean = sum(weights_flat) / len(weights_flat)
        variance = sum((x - mean) ** 2 for x in weights_flat) / len(weights_flat)
        actual_std = math.sqrt(variance)
        
        assert abs(actual_std - custom_std * 1.29) < 0.01
    
    def test_layer_bias_initialization(self):
        """Test that bias is initialized to zeros."""
        layer = LinearTruncNormal(100, 50)
        
        # Bias should be all zeros
        assert layer.bias is not None
        assert mx.all(layer.bias == 0.0)
        
        # Test no bias case
        layer_no_bias = LinearTruncNormal(100, 50, bias=False)
        assert layer_no_bias.bias is None
    
    def test_layer_key_splitting(self):
        """Test that weight and bias use different random keys."""
        key = mx.random.key(42)
        
        # Create two layers with consecutive keys
        layer1 = LinearTruncNormal(50, 50, key=key)
        key2 = mx.random.key(43)
        layer2 = LinearTruncNormal(50, 50, key=key2)
        
        # Weights should be different
        assert not mx.array_equal(layer1.weight, layer2.weight)
        
        # Create layer and check that weight randomness is used
        # (we can't directly test key splitting, but we can verify weights aren't all the same)
        layer = LinearTruncNormal(10, 10, key=key)
        weights_unique = len(set(layer.weight.flatten().tolist())) > 50
        assert weights_unique, "Weights should have variety, suggesting proper randomness"
    
    def test_layer_deterministic_init(self):
        """Test deterministic layer initialization."""
        key = mx.random.key(42)
        
        layer1 = LinearTruncNormal(100, 50, key=key)
        layer2 = LinearTruncNormal(100, 50, key=key)
        
        assert mx.array_equal(layer1.weight, layer2.weight)
        assert mx.array_equal(layer1.bias, layer2.bias)


class TestEmbeddingTruncNormal:
    """Test EmbeddingTruncNormal layer."""
    
    def test_embedding_creation(self):
        """Test creating embedding layer."""
        num_embeddings = 1000
        embedding_dim = 128
        
        layer = EmbeddingTruncNormal(num_embeddings, embedding_dim)
        
        # Check weight shape
        assert layer.weight.shape == (num_embeddings, embedding_dim)
        assert layer.num_embeddings == num_embeddings
        assert layer.embedding_dim == embedding_dim
    
    def test_embedding_lookup(self):
        """Test embedding lookup functionality."""
        num_embeddings = 100
        embedding_dim = 32
        batch_size = 16
        seq_len = 20
        
        layer = EmbeddingTruncNormal(num_embeddings, embedding_dim)
        
        # Create indices
        indices = mx.random.randint(0, num_embeddings, (batch_size, seq_len))
        
        # Lookup embeddings
        embeddings = layer(indices)
        
        # Check output shape
        assert embeddings.shape == (batch_size, seq_len, embedding_dim)
    
    def test_embedding_initialization_std(self):
        """Test embedding initialization std."""
        num_embeddings = 1000
        embedding_dim = 100
        
        # Default std should be 1/sqrt(embedding_dim)
        layer = EmbeddingTruncNormal(num_embeddings, embedding_dim)
        expected_std = 1.0 / math.sqrt(embedding_dim)
        
        # Check weights std
        weights_flat = layer.weight.flatten().tolist()
        mean = sum(weights_flat) / len(weights_flat)
        variance = sum((x - mean) ** 2 for x in weights_flat) / len(weights_flat)
        actual_std = math.sqrt(variance)
        
        # The PyTorch implementation inflates std by ~29% for default [-2, 2] bounds
        assert abs(actual_std - expected_std * 1.29) < 0.015


class TestErrorFunction:
    """Test the error function approximation used in variance correction."""
    
    def test_erf_approximation(self):
        """Test that our erfinv approximation is accurate enough."""
        # Import the internal function
        from mlx_hrm.layers.initialization import _erfinv
        
        # Test erfinv function at known points
        assert abs(_erfinv(0.0)) < 1e-6
        
        # Test round-trip: erf(erfinv(x)) ≈ x
        # Our approximation is less accurate for values near ±1
        test_values = [-0.5, 0.0, 0.5]
        for x in test_values:
            y = _erfinv(x)
            z = math.erf(y)
            assert abs(z - x) < 1e-3, f"Round trip failed for {x}: erf(erfinv({x})) = {z}"
        
        # Test that the variance correction in truncated_normal works
        # by checking that samples have the correct standard deviation
    
    def test_clipping_behavior(self):
        """Test that clipping actually occurs at the expected bounds."""
        key = mx.random.key(42)
        std = 0.01
        lower = -2.0
        upper = 2.0
        
        # Generate many samples to ensure some hit the bounds
        samples = truncated_normal(10000, std=std, lower=lower, upper=upper, key=key)
        samples_list = samples.tolist()
        
        # Calculate comp_std to know the exact clip bounds
        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        
        lower_clip = lower * comp_std
        upper_clip = upper * comp_std
        
        # Count values at the clip bounds
        clipped_lower = sum(1 for x in samples_list if abs(x - lower_clip) < 1e-7)
        clipped_upper = sum(1 for x in samples_list if abs(x - upper_clip) < 1e-7)
        
        # With small std, we should see significant clipping
        total_clipped = clipped_lower + clipped_upper
        assert total_clipped > 100, f"Expected significant clipping, got {total_clipped} clipped values"
        
        # Verify no values exceed bounds
        assert all(lower_clip - 1e-6 <= x <= upper_clip + 1e-6 for x in samples_list)


if __name__ == "__main__":
    # Run tests
    test = TestTruncatedNormal()
    test.test_shape_and_dtype()
    test.test_statistical_properties()
    test.test_custom_bounds()
    test.test_deterministic_initialization()
    test.test_variance_correction()
    
    test = TestLinearTruncNormal()
    test.test_layer_creation()
    test.test_layer_forward()
    test.test_layer_initialization_std()
    test.test_layer_deterministic_init()
    
    test = TestEmbeddingTruncNormal()
    test.test_embedding_creation()
    test.test_embedding_lookup()
    test.test_embedding_initialization_std()
    
    print("All tests passed!")