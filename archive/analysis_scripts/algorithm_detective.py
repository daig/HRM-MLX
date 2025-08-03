#!/usr/bin/env python3
"""
Detective work to figure out which summation algorithm each framework uses.
We'll test various patterns to see if we can identify the algorithm.
"""

def test_reduction_patterns():
    """Test various patterns to identify the summation algorithm"""
    print("=== ALGORITHM DETECTION ===")
    
    # Test 1: Sequential vs Tree patterns
    # Different algorithms show different error patterns
    patterns = [
        ("Small powers", [0.125] * 8),  # Should be exact in BF16
        ("Large + Small", [1000.0] + [0.01] * 100),  # Tests precision loss patterns
        ("Alternating large", [1.0, -1.0] * 50),  # Should be exactly 0
        ("Binary tree friendly", [1.0] * 16),  # Powers of 2 for tree reduction
        ("Sequential friendly", [0.1] * 10),  # Simple sequential pattern
    ]
    
    for name, values in patterns:
        print(f"\n{name} pattern:")
        
        try:
            import mlx.core as mx
            mlx_arr = mx.array(values, dtype=mx.bfloat16)
            mlx_sum = mx.sum(mlx_arr).astype(mx.float32)
            print(f"  MLX: {mlx_sum}")
        except ImportError:
            print("  MLX not available")
            
        try:
            import torch
            torch_arr = torch.tensor(values, dtype=torch.bfloat16)
            torch_sum = torch.sum(torch_arr).to(torch.float32)
            print(f"  PyTorch: {torch_sum}")
        except ImportError:
            print("  PyTorch not available")

def test_associativity():
    """Test if the frameworks respect associativity differently"""
    print("\n=== ASSOCIATIVITY TEST ===")
    
    # Test: (a + b) + c vs a + (b + c)
    # Different algorithms might group operations differently
    values = [1.0, -0.99, 0.5]  # Simple 3-element case
    
    try:
        import mlx.core as mx
        arr = mx.array(values, dtype=mx.bfloat16)
        
        # Left associative: ((a + b) + c)
        left = mx.sum(arr[:2]) + arr[2]
        
        # Right associative: (a + (b + c))
        right = arr[0] + mx.sum(arr[1:])
        
        # Vector sum
        vector = mx.sum(arr)
        
        print(f"MLX:")
        print(f"  Left: {left.astype(mx.float32)}")
        print(f"  Right: {right.astype(mx.float32)}")
        print(f"  Vector: {vector.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")
        
    try:
        import torch
        arr = torch.tensor(values, dtype=torch.bfloat16)
        
        # Left associative
        left = torch.sum(arr[:2]) + arr[2]
        
        # Right associative  
        right = arr[0] + torch.sum(arr[1:])
        
        # Vector sum
        vector = torch.sum(arr)
        
        print(f"PyTorch:")
        print(f"  Left: {left.to(torch.float32)}")
        print(f"  Right: {right.to(torch.float32)}")
        print(f"  Vector: {vector.to(torch.float32)}")
        
    except ImportError:
        print("PyTorch not available")

def test_tree_vs_sequential():
    """Test patterns that reveal tree vs sequential reduction"""
    print("\n=== TREE VS SEQUENTIAL DETECTION ===")
    
    # Tree reduction tends to be more accurate for large arrays
    # Sequential can accumulate more error
    
    # Power of 2 sizes (tree-friendly)
    for size in [8, 16, 32]:
        values = [0.01] * size  # Should sum to size * 0.01
        expected = size * 0.01
        
        print(f"\nSize {size} (expected: {expected}):")
        
        try:
            import mlx.core as mx
            mlx_arr = mx.array(values, dtype=mx.bfloat16)
            mlx_sum = mx.sum(mlx_arr).astype(mx.float32)
            mlx_error = abs(float(mlx_sum) - expected)
            print(f"  MLX: {mlx_sum} (error: {mlx_error:.6f})")
        except ImportError:
            print("  MLX not available")
            
        try:
            import torch
            torch_arr = torch.tensor(values, dtype=torch.bfloat16)
            torch_sum = torch.sum(torch_arr).to(torch.float32)
            torch_error = abs(float(torch_sum) - expected)
            print(f"  PyTorch: {torch_sum} (error: {torch_error:.6f})")
        except ImportError:
            print("  PyTorch not available")

def test_intermediate_precision():
    """Test if frameworks use different intermediate precisions"""
    print("\n=== INTERMEDIATE PRECISION TEST ===")
    
    # Values that would show if intermediate computations use higher precision
    values = [1e-5] * 1000  # Very small values, should sum to 0.01
    expected = 0.01
    
    print(f"1000 tiny values (expected: {expected}):")
    
    try:
        import mlx.core as mx
        mlx_arr = mx.array(values, dtype=mx.bfloat16)
        mlx_sum = mx.sum(mlx_arr).astype(mx.float32)
        mlx_error = abs(float(mlx_sum) - expected)
        print(f"  MLX: {mlx_sum} (error: {mlx_error:.6f})")
    except ImportError:
        print("  MLX not available")
        
    try:
        import torch
        torch_arr = torch.tensor(values, dtype=torch.bfloat16)
        torch_sum = torch.sum(torch_arr).to(torch.float32)
        torch_error = abs(float(torch_sum) - expected)
        print(f"  PyTorch: {torch_sum} (error: {torch_error:.6f})")
    except ImportError:
        print("  PyTorch not available")

def analyze_specific_case():
    """Analyze our specific failing case in detail"""
    print("\n=== SPECIFIC CASE ANALYSIS ===")
    
    # Our problematic case: [1.0, -0.99] * 50
    values = [1.0, -0.99] * 50
    expected = 0.5
    
    try:
        import mlx.core as mx
        
        # Different grouping strategies
        arr = mx.array(values, dtype=mx.bfloat16)
        
        # Strategy 1: Direct sum (what our test does)
        direct = mx.sum(arr)
        
        # Strategy 2: Reshape and sum in groups
        reshaped = arr.reshape(50, 2)
        grouped = mx.sum(mx.sum(reshaped, axis=1))
        
        # Strategy 3: Manual chunking 
        chunks = []
        for i in range(0, len(values), 10):
            chunk = mx.sum(arr[i:i+10])
            chunks.append(chunk)
        chunked = mx.sum(mx.array(chunks))
        
        print(f"MLX strategies:")
        print(f"  Direct: {direct.astype(mx.float32)}")
        print(f"  Grouped: {grouped.astype(mx.float32)}")
        print(f"  Chunked: {chunked.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")
        
    try:
        import torch
        
        # Same strategies with PyTorch
        arr = torch.tensor(values, dtype=torch.bfloat16)
        
        # Strategy 1: Direct sum
        direct = torch.sum(arr)
        
        # Strategy 2: Reshape and sum in groups
        reshaped = arr.reshape(50, 2)
        grouped = torch.sum(torch.sum(reshaped, dim=1))
        
        # Strategy 3: Manual chunking
        chunks = []
        for i in range(0, len(values), 10):
            chunk = torch.sum(arr[i:i+10])
            chunks.append(chunk)
        chunked = torch.sum(torch.stack(chunks))
        
        print(f"PyTorch strategies:")
        print(f"  Direct: {direct.to(torch.float32)}")
        print(f"  Grouped: {grouped.to(torch.float32)}")
        print(f"  Chunked: {chunked.to(torch.float32)}")
        
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    print("ALGORITHM DETECTIVE: Finding Sum Implementation Differences")
    print("=" * 60)
    
    test_reduction_patterns()
    test_associativity()
    test_tree_vs_sequential()
    test_intermediate_precision()
    analyze_specific_case()
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS:")
    print("- Look for patterns in the results above")
    print("- Tree reduction typically shows lower error for large arrays")
    print("- Sequential reduction accumulates error linearly")
    print("- Different associativity groupings reveal algorithm structure")