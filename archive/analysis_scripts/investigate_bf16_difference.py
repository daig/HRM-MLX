#!/usr/bin/env python3
"""
Deep investigation into why MLX and PyTorch show different BF16 accumulation behavior.
This could indicate a critical implementation difference.
"""

def detailed_analysis():
    """Detailed step-by-step analysis of BF16 behavior"""
    print("=== DETAILED BF16 ACCUMULATION ANALYSIS ===")
    
    # Test both frameworks with detailed logging
    try:
        import mlx.core as mx
        print("\n1. MLX BF16 REPRESENTATION TEST:")
        
        # Test individual value representations
        val1_f32 = mx.array(1.0, dtype=mx.float32)
        val2_f32 = mx.array(-0.99, dtype=mx.float32)
        
        val1_bf16 = val1_f32.astype(mx.bfloat16)
        val2_bf16 = val2_f32.astype(mx.bfloat16)
        
        print(f"  1.0 FP32: {val1_f32}")
        print(f"  1.0 BF16: {val1_bf16.astype(mx.float32)}")
        print(f"  -0.99 FP32: {val2_f32}")
        print(f"  -0.99 BF16: {val2_bf16.astype(mx.float32)}")
        
        # Test the pair sum
        pair_f32 = val1_f32 + val2_f32
        pair_bf16 = val1_bf16 + val2_bf16
        
        print(f"  Pair sum FP32: {pair_f32}")
        print(f"  Pair sum BF16: {pair_bf16.astype(mx.float32)}")
        
        # Test different accumulation strategies
        print("\n2. MLX ACCUMULATION STRATEGIES:")
        
        # Strategy 1: Cast to BF16 then sum
        values = mx.array([1.0, -0.99] * 50)
        values_bf16 = values.astype(mx.bfloat16)
        sum_bf16_v1 = mx.sum(values_bf16)
        print(f"  Strategy 1 (cast then sum): {sum_bf16_v1.astype(mx.float32)}")
        
        # Strategy 2: Individual pair sums then accumulate
        pairs = values.reshape(50, 2)
        pairs_bf16 = pairs.astype(mx.bfloat16)
        pair_sums = mx.sum(pairs_bf16, axis=1)
        sum_bf16_v2 = mx.sum(pair_sums)
        print(f"  Strategy 2 (pair sums): {sum_bf16_v2.astype(mx.float32)}")
        
        # Strategy 3: Sequential accumulation
        running_sum = mx.array(0.0, dtype=mx.bfloat16)
        for i in range(50):
            running_sum = running_sum + mx.array(1.0, dtype=mx.bfloat16) + mx.array(-0.99, dtype=mx.bfloat16)
        print(f"  Strategy 3 (sequential): {running_sum.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")
    
    try:
        import torch
        print("\n3. PYTORCH BF16 REPRESENTATION TEST:")
        
        # Test individual value representations
        val1_f32 = torch.tensor(1.0, dtype=torch.float32)
        val2_f32 = torch.tensor(-0.99, dtype=torch.float32)
        
        val1_bf16 = val1_f32.to(torch.bfloat16)
        val2_bf16 = val2_f32.to(torch.bfloat16)
        
        print(f"  1.0 FP32: {val1_f32}")
        print(f"  1.0 BF16: {val1_bf16.to(torch.float32)}")
        print(f"  -0.99 FP32: {val2_f32}")
        print(f"  -0.99 BF16: {val2_bf16.to(torch.float32)}")
        
        # Test the pair sum
        pair_f32 = val1_f32 + val2_f32
        pair_bf16 = val1_bf16 + val2_bf16
        
        print(f"  Pair sum FP32: {pair_f32}")
        print(f"  Pair sum BF16: {pair_bf16.to(torch.float32)}")
        
        # Test different accumulation strategies
        print("\n4. PYTORCH ACCUMULATION STRATEGIES:")
        
        # Strategy 1: Cast to BF16 then sum
        values = torch.tensor([1.0, -0.99] * 50)
        values_bf16 = values.to(torch.bfloat16)
        sum_bf16_v1 = torch.sum(values_bf16)
        print(f"  Strategy 1 (cast then sum): {sum_bf16_v1.to(torch.float32)}")
        
        # Strategy 2: Individual pair sums then accumulate
        pairs = values.reshape(50, 2)
        pairs_bf16 = pairs.to(torch.bfloat16)
        pair_sums = torch.sum(pairs_bf16, dim=1)
        sum_bf16_v2 = torch.sum(pair_sums)
        print(f"  Strategy 2 (pair sums): {sum_bf16_v2.to(torch.float32)}")
        
        # Strategy 3: Sequential accumulation
        running_sum = torch.tensor(0.0, dtype=torch.bfloat16)
        for i in range(50):
            running_sum = running_sum + torch.tensor(1.0, dtype=torch.bfloat16) + torch.tensor(-0.99, dtype=torch.bfloat16)
        print(f"  Strategy 3 (sequential): {running_sum.to(torch.float32)}")
        
    except ImportError:
        print("PyTorch not available")

def test_intermediate_precision():
    """Test if frameworks use different intermediate precisions"""
    print("\n=== INTERMEDIATE PRECISION ANALYSIS ===")
    
    try:
        import mlx.core as mx
        print("\n5. MLX INTERMEDIATE PRECISION:")
        
        # Test with smaller numbers to see accumulation behavior
        small_values = mx.array([0.01] * 100)  # Should sum to 1.0
        small_bf16 = small_values.astype(mx.bfloat16)
        small_sum = mx.sum(small_bf16)
        print(f"  100 * 0.01 in BF16: {small_sum.astype(mx.float32)} (expected: 1.0)")
        
        # Test powers of 2 (should be exact in BF16)
        powers_values = mx.array([0.125, -0.125] * 50)  # Should sum to 0.0
        powers_bf16 = powers_values.astype(mx.bfloat16)
        powers_sum = mx.sum(powers_bf16)
        print(f"  50 * (0.125 - 0.125) in BF16: {powers_sum.astype(mx.float32)} (expected: 0.0)")
        
    except ImportError:
        pass
        
    try:
        import torch
        print("\n6. PYTORCH INTERMEDIATE PRECISION:")
        
        # Test with smaller numbers
        small_values = torch.tensor([0.01] * 100)
        small_bf16 = small_values.to(torch.bfloat16)
        small_sum = torch.sum(small_bf16)
        print(f"  100 * 0.01 in BF16: {small_sum.to(torch.float32)} (expected: 1.0)")
        
        # Test powers of 2
        powers_values = torch.tensor([0.125, -0.125] * 50)
        powers_bf16 = powers_values.to(torch.bfloat16)
        powers_sum = torch.sum(powers_bf16)
        print(f"  50 * (0.125 - 0.125) in BF16: {powers_sum.to(torch.float32)} (expected: 0.0)")
        
    except ImportError:
        pass

def test_raw_binary_representation():
    """Check if the frameworks interpret BF16 bits differently"""
    print("\n=== BINARY REPRESENTATION ANALYSIS ===")
    
    try:
        import mlx.core as mx
        import numpy as np
        
        print("\n7. MLX BINARY REPRESENTATION:")
        
        # Create specific BF16 values and check their binary representation
        test_val = mx.array(-0.99, dtype=mx.float32)
        test_bf16 = test_val.astype(mx.bfloat16)
        test_back = test_bf16.astype(mx.float32)
        
        print(f"  -0.99 FP32: {test_val}")
        print(f"  -0.99 BF16->FP32: {test_back}")
        print(f"  Difference: {mx.abs(test_val - test_back)}")
        
        # Convert to numpy to examine bits
        np_val = np.array(float(test_back))
        print(f"  As numpy: {np_val}")
        print(f"  Hex: {np_val.view(np.uint32):08x}")
        
    except ImportError:
        pass
    
    try:
        import torch
        import numpy as np
        
        print("\n8. PYTORCH BINARY REPRESENTATION:")
        
        # Same test with PyTorch
        test_val = torch.tensor(-0.99, dtype=torch.float32)
        test_bf16 = test_val.to(torch.bfloat16)
        test_back = test_bf16.to(torch.float32)
        
        print(f"  -0.99 FP32: {test_val}")
        print(f"  -0.99 BF16->FP32: {test_back}")
        print(f"  Difference: {torch.abs(test_val - test_back)}")
        
        # Convert to numpy to examine bits
        np_val = test_back.numpy()
        print(f"  As numpy: {np_val}")
        print(f"  Hex: {np_val.view(np.uint32):08x}")
        
    except ImportError:
        pass

def test_accumulation_order():
    """Test if accumulation order affects results"""
    print("\n=== ACCUMULATION ORDER ANALYSIS ===")
    
    try:
        import mlx.core as mx
        print("\n9. MLX ACCUMULATION ORDER:")
        
        values = mx.array([1.0, -0.99] * 50)
        values_bf16 = values.astype(mx.bfloat16)
        
        # Forward order
        sum_forward = mx.sum(values_bf16)
        print(f"  Forward sum: {sum_forward.astype(mx.float32)}")
        
        # Reverse order
        sum_reverse = mx.sum(values_bf16[::-1])
        print(f"  Reverse sum: {sum_reverse.astype(mx.float32)}")
        
        # Grouped differently
        reshaped = values_bf16.reshape(50, 2)
        sum_grouped = mx.sum(mx.sum(reshaped, axis=1))
        print(f"  Grouped sum: {sum_grouped.astype(mx.float32)}")
        
    except ImportError:
        pass
        
    try:
        import torch
        print("\n10. PYTORCH ACCUMULATION ORDER:")
        
        values = torch.tensor([1.0, -0.99] * 50)
        values_bf16 = values.to(torch.bfloat16)
        
        # Forward order
        sum_forward = torch.sum(values_bf16)
        print(f"  Forward sum: {sum_forward.to(torch.float32)}")
        
        # Reverse order
        sum_reverse = torch.sum(torch.flip(values_bf16, [0]))
        print(f"  Reverse sum: {sum_reverse.to(torch.float32)}")
        
        # Grouped differently
        reshaped = values_bf16.reshape(50, 2)
        sum_grouped = torch.sum(torch.sum(reshaped, dim=1))
        print(f"  Grouped sum: {sum_grouped.to(torch.float32)}")
        
    except ImportError:
        pass

if __name__ == "__main__":
    print("CRITICAL INVESTIGATION: MLX vs PyTorch BF16 Differences")
    print("=" * 70)
    
    detailed_analysis()
    test_intermediate_precision()
    test_raw_binary_representation()
    test_accumulation_order()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE - Check output above for differences")