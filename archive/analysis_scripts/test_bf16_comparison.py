#!/usr/bin/env python3
"""
Compare BF16 accumulation behavior between PyTorch and MLX.
This tests whether the large error we see in MLX BF16 accumulation 
is also present in PyTorch, or if it's MLX-specific.
"""

def test_mlx_bf16():
    """Test BF16 accumulation in MLX"""
    try:
        import mlx.core as mx
        
        # Same test pattern as in our mixed precision test
        values = mx.array([1.0, -0.99] * 50)
        expected_sum = 0.5
        
        values_fp32 = values.astype(mx.float32)
        values_bf16 = values.astype(mx.bfloat16)
        
        sum_fp32 = mx.sum(values_fp32)
        sum_bf16 = mx.sum(values_bf16).astype(mx.float32)
        
        print("MLX BF16 accumulation test:")
        print(f"Expected: {expected_sum:.6f}")
        print(f"FP32 sum: {sum_fp32:.6f}")
        print(f"BF16 sum: {sum_bf16:.6f}")
        
        sum_diff = mx.abs(sum_fp32 - sum_bf16)
        relative_diff = sum_diff / (mx.abs(sum_fp32) + 1e-8)
        
        print(f"FP32 vs BF16 difference: {sum_diff:.6f}")
        print(f"Relative difference: {relative_diff:.6f}")
        print(f"Relative diff percentage: {relative_diff*100:.2f}%")
        
        return float(relative_diff)
        
    except ImportError:
        print("MLX not available")
        return None

def test_pytorch_bf16():
    """Test BF16 accumulation in PyTorch"""
    try:
        import torch
        
        # Same test pattern
        values = torch.tensor([1.0, -0.99] * 50)
        expected_sum = 0.5
        
        values_fp32 = values.to(torch.float32)
        values_bf16 = values.to(torch.bfloat16)
        
        sum_fp32 = torch.sum(values_fp32)
        sum_bf16 = torch.sum(values_bf16).to(torch.float32)
        
        print("\nPyTorch BF16 accumulation test:")
        print(f"Expected: {expected_sum:.6f}")
        print(f"FP32 sum: {sum_fp32:.6f}")
        print(f"BF16 sum: {sum_bf16:.6f}")
        
        sum_diff = torch.abs(sum_fp32 - sum_bf16)
        relative_diff = sum_diff / (torch.abs(sum_fp32) + 1e-8)
        
        print(f"FP32 vs BF16 difference: {sum_diff:.6f}")
        print(f"Relative difference: {relative_diff:.6f}")
        print(f"Relative diff percentage: {relative_diff*100:.2f}%")
        
        return float(relative_diff)
        
    except ImportError:
        print("PyTorch not available")
        return None

def test_numerical_analysis():
    """Analyze why this pattern might be problematic for BF16"""
    print("\nNumerical analysis:")
    print("Pattern: [1.0, -0.99] * 50 = 100 values")
    print("Expected sum: (1.0 + (-0.99)) * 50 = 0.01 * 50 = 0.5")
    print()
    print("BF16 characteristics:")
    print("- 8-bit exponent, 7-bit mantissa")
    print("- Can represent 1.0 exactly")
    print("- 0.99 may not be representable exactly in BF16")
    print()
    
    # Check what 0.99 becomes in BF16
    try:
        import mlx.core as mx
        orig = mx.array(0.99, dtype=mx.float32)
        bf16_val = orig.astype(mx.bfloat16).astype(mx.float32)
        error = mx.abs(orig - bf16_val)
        print(f"0.99 in FP32: {orig}")
        print(f"0.99 in BF16->FP32: {bf16_val}")
        print(f"Representation error: {error}")
        print(f"Error per operation: {error} * 50 = {error * 50}")
    except ImportError:
        pass

if __name__ == "__main__":
    print("Comparing BF16 accumulation behavior between PyTorch and MLX")
    print("=" * 60)
    
    mlx_error = test_mlx_bf16()
    pytorch_error = test_pytorch_bf16()
    test_numerical_analysis()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if mlx_error is not None:
        print(f"MLX relative error: {mlx_error*100:.2f}%")
    if pytorch_error is not None:  
        print(f"PyTorch relative error: {pytorch_error*100:.2f}%")
        
    if mlx_error is not None and pytorch_error is not None:
        if abs(mlx_error - pytorch_error) < 0.01:  # Within 1%
            print("✅ Both frameworks show similar BF16 accumulation behavior")
            print("This suggests the MLX test failure reflects genuine BF16 limitations")
        else:
            print("❌ Frameworks show different BF16 behavior - may indicate MLX issue")