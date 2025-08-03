#!/usr/bin/env python3
"""
Isolate the exact difference in BF16 summation between MLX and PyTorch
"""

def test_minimal_case():
    """Test with minimal case to isolate the difference"""
    print("=== MINIMAL CASE ANALYSIS ===")
    
    # Test with just a few values first
    for n in [2, 4, 8, 16]:
        print(f"\nTesting with {n} pairs ([1.0, -0.99] * {n}):")
        
        try:
            import mlx.core as mx
            values = mx.array([1.0, -0.99] * n)
            values_bf16 = values.astype(mx.bfloat16)
            mlx_sum = mx.sum(values_bf16).astype(mx.float32)
            print(f"  MLX sum: {mlx_sum}")
        except ImportError:
            print("  MLX not available")
            
        try:
            import torch
            values = torch.tensor([1.0, -0.99] * n)
            values_bf16 = values.to(torch.bfloat16)
            torch_sum = torch.sum(values_bf16).to(torch.float32)
            print(f"  PyTorch sum: {torch_sum}")
        except ImportError:
            print("  PyTorch not available")

def test_manual_accumulation():
    """Manually accumulate to see step-by-step differences"""
    print("\n=== MANUAL ACCUMULATION ANALYSIS ===")
    
    try:
        import mlx.core as mx
        print("\nMLX manual accumulation:")
        
        # Start with 0
        acc = mx.array(0.0, dtype=mx.bfloat16)
        values = [1.0, -0.99] * 4  # Just 4 pairs for debugging
        
        for i, val in enumerate(values):
            new_val = mx.array(val, dtype=mx.bfloat16)
            acc = acc + new_val
            print(f"  Step {i+1}: + {val} -> {acc.astype(mx.float32)}")
            
        print(f"  Final MLX manual: {acc.astype(mx.float32)}")
        
        # Compare with vectorized sum
        vec_values = mx.array(values, dtype=mx.bfloat16)
        vec_sum = mx.sum(vec_values)
        print(f"  Final MLX vectorized: {vec_sum.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")
        
    try:
        import torch
        print("\nPyTorch manual accumulation:")
        
        # Start with 0
        acc = torch.tensor(0.0, dtype=torch.bfloat16)
        values = [1.0, -0.99] * 4  # Same 4 pairs
        
        for i, val in enumerate(values):
            new_val = torch.tensor(val, dtype=torch.bfloat16)
            acc = acc + new_val
            print(f"  Step {i+1}: + {val} -> {acc.to(torch.float32)}")
            
        print(f"  Final PyTorch manual: {acc.to(torch.float32)}")
        
        # Compare with vectorized sum
        vec_values = torch.tensor(values, dtype=torch.bfloat16)
        vec_sum = torch.sum(vec_values)
        print(f"  Final PyTorch vectorized: {vec_sum.to(torch.float32)}")
        
    except ImportError:
        print("PyTorch not available")

def test_pair_accumulation():
    """Test pair-wise accumulation pattern"""
    print("\n=== PAIR ACCUMULATION ANALYSIS ===") 
    
    try:
        import mlx.core as mx
        print("\nMLX pair accumulation:")
        
        # Create pairs and sum them first
        n_pairs = 50
        pairs = mx.array([[1.0, -0.99]] * n_pairs, dtype=mx.bfloat16)
        print(f"  Pairs shape: {pairs.shape}")
        
        # Sum each pair
        pair_sums = mx.sum(pairs, axis=1)
        print(f"  First few pair sums: {pair_sums[:5].astype(mx.float32)}")
        
        # Sum all pair sums
        total = mx.sum(pair_sums)
        print(f"  MLX pair-wise total: {total.astype(mx.float32)}")
        
        # Compare with flattened approach
        flat = pairs.flatten()
        flat_sum = mx.sum(flat)
        print(f"  MLX flattened total: {flat_sum.astype(mx.float32)}")
        
    except ImportError:
        print("MLX not available")
        
    try:
        import torch
        print("\nPyTorch pair accumulation:")
        
        # Same test with PyTorch
        n_pairs = 50
        pairs = torch.tensor([[1.0, -0.99]] * n_pairs, dtype=torch.bfloat16)
        print(f"  Pairs shape: {pairs.shape}")
        
        # Sum each pair
        pair_sums = torch.sum(pairs, dim=1)
        print(f"  First few pair sums: {pair_sums[:5].to(torch.float32)}")
        
        # Sum all pair sums
        total = torch.sum(pair_sums)
        print(f"  PyTorch pair-wise total: {total.to(torch.float32)}")
        
        # Compare with flattened approach
        flat = pairs.flatten()
        flat_sum = torch.sum(flat)
        print(f"  PyTorch flattened total: {flat_sum.to(torch.float32)}")
        
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    print("ISOLATING BF16 SUMMATION DIFFERENCES")
    print("=" * 50)
    
    test_minimal_case()
    test_manual_accumulation()
    test_pair_accumulation()
    
    print("\n" + "=" * 50)
    print("KEY FINDING: Check if MLX and PyTorch show different results")
    print("for the SAME accumulation pattern with BF16 data")