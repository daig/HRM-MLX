#!/usr/bin/env python3
"""
Deep dive into MLX's reduction algorithm to understand why it gives
such wildly different results based on grouping.
"""

def test_minimal_reproduction():
    """Test with minimal case to isolate the issue"""
    print("=== MINIMAL REPRODUCTION ===")
    
    try:
        import mlx.core as mx
        
        # Start with just 4 values to see the pattern
        values = [1.0, -0.99, 1.0, -0.99]
        arr = mx.array(values, dtype=mx.bfloat16)
        
        print(f"Values: {values}")
        print(f"BF16 representation: {arr.astype(mx.float32)}")
        
        # Different ways to sum the same 4 values
        direct = mx.sum(arr)
        pairs = mx.sum(arr.reshape(2, 2), axis=1)
        pair_sum = mx.sum(pairs)
        left_to_right = ((arr[0] + arr[1]) + arr[2]) + arr[3]
        right_to_left = arr[0] + (arr[1] + (arr[2] + arr[3]))
        
        print(f"\nSummation strategies:")
        print(f"  Direct sum: {direct.astype(mx.float32)}")
        print(f"  Pair sums: {pairs.astype(mx.float32)} -> {pair_sum.astype(mx.float32)}")
        print(f"  Left-to-right: {left_to_right.astype(mx.float32)}")
        print(f"  Right-to-left: {right_to_left.astype(mx.float32)}")
        
        # If direct != pair_sum, something is very wrong
        if abs(float(direct) - float(pair_sum)) > 1e-10:
            print(f"🚨 ALERT: Direct sum != Pair sum! Difference: {abs(float(direct) - float(pair_sum))}")
        
    except ImportError:
        print("MLX not available")

def test_accumulation_order():
    """Test if MLX respects accumulation order consistently"""
    print("\n=== ACCUMULATION ORDER TEST ===")
    
    try:
        import mlx.core as mx
        
        # Test with 8 values to see if order matters
        values = [1.0, -0.99] * 4
        arr = mx.array(values, dtype=mx.bfloat16)
        
        print(f"Original array: {arr.astype(mx.float32)}")
        
        # Forward and reverse order
        forward = mx.sum(arr)
        reverse = mx.sum(arr[::-1])
        
        print(f"Forward sum: {forward.astype(mx.float32)}")
        print(f"Reverse sum: {reverse.astype(mx.float32)}")
        
        if abs(float(forward) - float(reverse)) > 1e-10:
            print(f"🚨 ALERT: Forward != Reverse! This suggests MLX uses order-dependent algorithm")
            print(f"Difference: {abs(float(forward) - float(reverse))}")
        else:
            print("✅ Forward and reverse give same result")
            
        # Test different permutations
        import numpy as np
        np.random.seed(42)
        indices = np.random.permutation(len(values))
        shuffled = arr[mx.array(indices)]
        shuffled_sum = mx.sum(shuffled)
        
        print(f"Shuffled indices: {indices}")
        print(f"Shuffled sum: {shuffled_sum.astype(mx.float32)}")
        
        if abs(float(forward) - float(shuffled_sum)) > 1e-10:
            print(f"🚨 ALERT: Order matters! Shuffled gives different result")
            print(f"Difference: {abs(float(forward) - float(shuffled_sum))}")
        
    except ImportError:
        print("MLX not available")

def test_chunk_sizes():
    """Test how different chunk sizes affect the result"""
    print("\n=== CHUNK SIZE EFFECT ===")
    
    try:
        import mlx.core as mx
        
        # Our problematic pattern
        values = [1.0, -0.99] * 20  # Smaller for easier debugging
        arr = mx.array(values, dtype=mx.bfloat16)
        
        print(f"Testing {len(values)} values: [1.0, -0.99] * 20")
        
        # Different chunk sizes
        chunk_sizes = [1, 2, 4, 5, 8, 10, 20, 40]
        
        for chunk_size in chunk_sizes:
            if chunk_size <= len(values):
                chunks = []
                for i in range(0, len(values), chunk_size):
                    chunk = mx.sum(arr[i:i+chunk_size])
                    chunks.append(chunk)
                
                if len(chunks) > 1:
                    result = mx.sum(mx.array(chunks))
                else:
                    result = chunks[0]
                
                print(f"  Chunk size {chunk_size:2d}: {result.astype(mx.float32)}")
        
        # Compare with direct sum
        direct = mx.sum(arr)
        print(f"  Direct sum:    {direct.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")

def test_data_type_behavior():
    """Test if the issue is specific to bfloat16"""
    print("\n=== DATA TYPE BEHAVIOR ===")
    
    try:
        import mlx.core as mx
        
        values = [1.0, -0.99] * 10
        dtypes = [mx.float16, mx.bfloat16, mx.float32]
        
        for dtype in dtypes:
            arr = mx.array(values, dtype=dtype)
            
            # Direct vs grouped
            direct = mx.sum(arr)
            grouped = mx.sum(mx.sum(arr.reshape(10, 2), axis=1))
            
            diff = abs(float(direct) - float(grouped))
            
            print(f"{str(dtype):15s}: direct={direct.astype(mx.float32):10.6f}, "
                  f"grouped={grouped.astype(mx.float32):10.6f}, diff={diff:.8f}")
            
            if diff > 1e-10:
                print(f"  🚨 {dtype} shows inconsistency!")
        
    except ImportError:
        print("MLX not available")

def test_metal_kernel_hypothesis():
    """Test hypothesis about Metal kernel behavior"""
    print("\n=== METAL KERNEL HYPOTHESIS ===")
    
    try:
        import mlx.core as mx
        
        # Test if the issue is related to GPU threading/SIMD
        print("Testing if MLX's Metal implementation has thread-dependent summation...")
        
        # Small array that fits in one thread group
        small_arr = mx.array([1.0, -0.99] * 4, dtype=mx.bfloat16)
        
        # Large array that requires multiple thread groups
        large_arr = mx.array([1.0, -0.99] * 500, dtype=mx.bfloat16)
        
        print(f"Small array (8 elements):")
        small_direct = mx.sum(small_arr)
        small_grouped = mx.sum(mx.sum(small_arr.reshape(4, 2), axis=1))
        print(f"  Direct: {small_direct.astype(mx.float32)}")
        print(f"  Grouped: {small_grouped.astype(mx.float32)}")
        print(f"  Diff: {abs(float(small_direct) - float(small_grouped))}")
        
        print(f"\nLarge array (1000 elements):")
        large_direct = mx.sum(large_arr)
        large_grouped = mx.sum(mx.sum(large_arr.reshape(500, 2), axis=1))
        print(f"  Direct: {large_direct.astype(mx.float32)}")
        print(f"  Grouped: {large_grouped.astype(mx.float32)}")
        print(f"  Diff: {abs(float(large_direct) - float(large_grouped))}")
        
        # Test if the difference scales with array size
        sizes = [10, 50, 100, 500]
        print(f"\nScaling test:")
        for size in sizes:
            arr = mx.array([1.0, -0.99] * size, dtype=mx.bfloat16)
            direct = mx.sum(arr)
            grouped = mx.sum(mx.sum(arr.reshape(size, 2), axis=1))
            diff = abs(float(direct) - float(grouped))
            expected = size * 0.01  # Theoretical result
            print(f"  Size {size*2:4d}: direct={direct.astype(mx.float32):8.5f}, "
                  f"grouped={grouped.astype(mx.float32):8.5f}, diff={diff:.6f}")
        
    except ImportError:
        print("MLX not available")

def investigate_precision_accumulation():
    """Investigate if MLX uses different precision for accumulation"""
    print("\n=== PRECISION ACCUMULATION INVESTIGATION ===")
    
    try:
        import mlx.core as mx
        
        # Test with values that reveal precision handling
        print("Testing accumulation precision...")
        
        # Case 1: Values where BF16 precision matters
        test_val = -0.99
        bf16_val = mx.array(test_val, dtype=mx.bfloat16).astype(mx.float32)
        print(f"Original -0.99 becomes: {bf16_val} in BF16")
        print(f"Error per value: {abs(test_val - float(bf16_val))}")
        
        # Case 2: Cumulative error analysis
        for n in [5, 10, 25, 50]:
            arr = mx.array([1.0, -0.99] * n, dtype=mx.bfloat16)
            
            # Different strategies
            direct = mx.sum(arr)
            
            # Manual step-by-step to see intermediate values
            acc = mx.array(0.0, dtype=mx.bfloat16)
            for i in range(len(arr)):
                acc = acc + arr[i]
            manual = acc
            
            # Grouped pairs
            pairs = arr.reshape(n, 2)
            pair_sums = mx.sum(pairs, axis=1)
            grouped = mx.sum(pair_sums)
            
            print(f"\n{n} pairs ({n*2} values):")
            print(f"  Direct:   {direct.astype(mx.float32)}")
            print(f"  Manual:   {manual.astype(mx.float32)}")
            print(f"  Grouped:  {grouped.astype(mx.float32)}")
            
            # Check if direct == manual (should be true for sequential algorithm)
            if abs(float(direct) - float(manual)) < 1e-10:
                print(f"  ✅ Direct matches manual (sequential algorithm)")
            else:
                print(f"  🚨 Direct != Manual! MLX uses non-sequential algorithm")
                print(f"     Difference: {abs(float(direct) - float(manual))}")
        
    except ImportError:
        print("MLX not available")

if __name__ == "__main__":
    print("DEBUGGING MLX REDUCTION ALGORITHM")
    print("=" * 50)
    
    test_minimal_reproduction()
    test_accumulation_order()
    test_chunk_sizes()
    test_data_type_behavior()
    test_metal_kernel_hypothesis()
    investigate_precision_accumulation()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")