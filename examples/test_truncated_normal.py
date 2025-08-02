#!/usr/bin/env python3
"""
Demonstrate JAX-compliant truncated normal initialization.

This script shows that our implementation correctly applies variance correction
to achieve the target standard deviation after truncation.
"""

import mlx.core as mx
from mlx_hrm.layers import truncated_normal, LinearTruncNormal, EmbeddingTruncNormal
import math


def test_variance_correction():
    """Test that variance correction works across different truncation bounds."""
    print("Testing variance correction for truncated normal distribution")
    print("=" * 60)
    
    key = mx.random.key(42)
    n_samples = 50000
    target_std = 0.1
    
    bounds_configs = [
        ("Tight", -1.0, 1.0),
        ("Standard", -2.0, 2.0),
        ("Wide", -3.0, 3.0),
    ]
    
    for name, lower, upper in bounds_configs:
        samples = truncated_normal(
            n_samples, 
            std=target_std, 
            lower=lower, 
            upper=upper, 
            key=key
        )
        
        # Compute statistics
        samples_list = samples.tolist()
        mean = sum(samples_list) / len(samples_list)
        variance = sum((x - mean) ** 2 for x in samples_list) / len(samples_list)
        actual_std = math.sqrt(variance)
        
        print(f"\n{name} bounds [{lower}, {upper}]:")
        print(f"  Target std: {target_std:.6f}")
        print(f"  Actual std: {actual_std:.6f}")
        print(f"  Error: {abs(actual_std - target_std):.6f} ({abs(actual_std - target_std)/target_std*100:.2f}%)")
        print(f"  Min: {min(samples_list):.6f}, Max: {max(samples_list):.6f}")


def test_deterministic_initialization():
    """Test deterministic initialization with keys."""
    print("\n\nTesting deterministic initialization")
    print("=" * 60)
    
    key = mx.random.key(42)
    
    # Create two models with same key
    model1 = LinearTruncNormal(100, 50, key=key)
    model2 = LinearTruncNormal(100, 50, key=key)
    
    # Check weights are identical
    weights_equal = mx.array_equal(model1.weight, model2.weight)
    print(f"Weights equal with same key: {weights_equal}")
    
    # Create model with different key
    key2 = mx.random.key(43)
    model3 = LinearTruncNormal(100, 50, key=key2)
    
    weights_different = not mx.array_equal(model1.weight, model3.weight)
    print(f"Weights different with different key: {weights_different}")


def test_layer_initialization():
    """Test custom layers with truncated normal initialization."""
    print("\n\nTesting layer initialization")
    print("=" * 60)
    
    # Linear layer
    linear = LinearTruncNormal(512, 256)
    print(f"\nLinear layer (512 → 256):")
    print(f"  Weight shape: {linear.weight.shape}")
    print(f"  Bias shape: {linear.bias.shape if linear.bias is not None else 'None'}")
    
    # Check initialization std
    weights_flat = linear.weight.flatten().tolist()
    mean = sum(weights_flat) / len(weights_flat)
    variance = sum((x - mean) ** 2 for x in weights_flat) / len(weights_flat)
    actual_std = math.sqrt(variance)
    expected_std = math.sqrt(1.0 / 512)  # LeCun initialization
    
    print(f"  Expected std: {expected_std:.6f}")
    print(f"  Actual std: {actual_std:.6f}")
    
    # Embedding layer  
    embedding = EmbeddingTruncNormal(10000, 128)
    print(f"\nEmbedding layer (10000 × 128):")
    print(f"  Weight shape: {embedding.weight.shape}")
    
    # Test embedding lookup
    indices = mx.array([0, 10, 100, 1000])
    embeddings = embedding(indices)
    print(f"  Lookup shape for indices {indices.tolist()}: {embeddings.shape}")


def main():
    """Run all tests."""
    print("JAX-Compliant Truncated Normal Initialization Tests")
    print("=" * 60)
    
    test_variance_correction()
    test_deterministic_initialization()
    test_layer_initialization()
    
    print("\n\nAll tests completed successfully!")


if __name__ == "__main__":
    main()