#!/usr/bin/env python3
"""
Test the hypothesis that PyTorch promotes to FP32 during sum operations
while MLX does pure BF16 accumulation.
"""

def test_promotion_hypothesis():
    """Test if PyTorch promotes to FP32 internally during BF16 sum"""
    print("=== TESTING PROMOTION HYPOTHESIS ===")
    
    # Test case that would show dramatic difference between FP32 and BF16 accumulation
    values = [1.0, -0.99] * 50
    expected_fp32 = 0.5  # Theoretical result
    
    try:
        import torch
        
        print("PyTorch behavior:")
        
        # Test 1: Pure BF16 sum (what we think PyTorch does)
        torch_bf16 = torch.tensor(values, dtype=torch.bfloat16)
        torch_result = torch.sum(torch_bf16)
        print(f"  BF16 sum result: {torch_result.to(torch.float32)}")
        
        # Test 2: Manual FP32 promotion (to see if this matches PyTorch)
        torch_fp32_promoted = torch.sum(torch_bf16.to(torch.float32))
        print(f"  FP32 promoted:   {torch_fp32_promoted}")
        
        # Test 3: Pure FP32 sum for reference
        torch_fp32_direct = torch.sum(torch.tensor(values, dtype=torch.float32))
        print(f"  FP32 direct:     {torch_fp32_direct}")
        
        # Check if PyTorch BF16 sum matches FP32 promotion
        diff_promoted = abs(float(torch_result) - float(torch_fp32_promoted))
        diff_direct = abs(float(torch_result) - float(torch_fp32_direct))
        
        print(f"  Difference vs FP32 promoted: {diff_promoted:.8f}")
        print(f"  Difference vs FP32 direct:   {diff_direct:.8f}")
        
        if diff_promoted < 1e-6:
            print("  🎯 HYPOTHESIS CONFIRMED: PyTorch promotes to FP32 during sum!")
        elif diff_direct < 1e-6:
            print("  🎯 PyTorch matches pure FP32 computation!")
        else:
            print("  ❓ PyTorch uses a different strategy")
            
    except ImportError:
        print("PyTorch not available")
    
    try:
        import mlx.core as mx
        
        print("\nMLX behavior:")
        
        # Test 1: Pure BF16 sum (what MLX actually does)
        mlx_bf16 = mx.array(values, dtype=mx.bfloat16)
        mlx_result = mx.sum(mlx_bf16)
        print(f"  BF16 sum result: {mlx_result.astype(mx.float32)}")
        
        # Test 2: Manual FP32 promotion
        mlx_fp32_promoted = mx.sum(mlx_bf16.astype(mx.float32))
        print(f"  FP32 promoted:   {mlx_fp32_promoted}")
        
        # Test 3: Pure FP32 sum for reference
        mlx_fp32_direct = mx.sum(mx.array(values, dtype=mx.float32))
        print(f"  FP32 direct:     {mlx_fp32_direct}")
        
        # Check MLX behavior
        diff_promoted = abs(float(mlx_result) - float(mlx_fp32_promoted))
        diff_direct = abs(float(mlx_result) - float(mlx_fp32_direct))
        
        print(f"  Difference vs FP32 promoted: {diff_promoted:.8f}")
        print(f"  Difference vs FP32 direct:   {diff_direct:.8f}")
        
        if diff_promoted < 1e-6:
            print("  ❓ MLX also promotes to FP32 (unexpected)")
        elif diff_direct < 1e-6:
            print("  ❓ MLX matches pure FP32 (unexpected)")
        else:
            print("  ✅ CONFIRMED: MLX does pure BF16 accumulation")
            
    except ImportError:
        print("MLX not available")

def test_different_patterns():
    """Test different patterns to confirm the promotion hypothesis"""
    print("\n=== TESTING DIFFERENT PATTERNS ===")
    
    patterns = [
        ("Small alternating", [0.1, -0.09] * 10),
        ("Large alternating", [10.0, -9.9] * 10), 
        ("Tiny values", [1e-4] * 1000),
        ("Mixed scale", [1000.0] + [0.001] * 1000),
    ]
    
    for name, values in patterns:
        print(f"\n{name}:")
        
        try:
            import torch
            torch_bf16 = torch.tensor(values, dtype=torch.bfloat16)
            torch_result = torch.sum(torch_bf16)
            torch_fp32 = torch.sum(torch_bf16.to(torch.float32))
            torch_diff = abs(float(torch_result) - float(torch_fp32))
            print(f"  PyTorch BF16 vs FP32 promoted: {torch_diff:.8f}")
        except ImportError:
            pass
            
        try:
            import mlx.core as mx
            mlx_bf16 = mx.array(values, dtype=mx.bfloat16)
            mlx_result = mx.sum(mlx_bf16)
            mlx_fp32 = mx.sum(mlx_bf16.astype(mx.float32))
            mlx_diff = abs(float(mlx_result) - float(mlx_fp32))
            print(f"  MLX BF16 vs FP32 promoted:     {mlx_diff:.8f}")
        except ImportError:
            pass

def test_manual_bf16_accumulation():
    """Manually implement BF16 accumulation to match MLX behavior"""
    print("\n=== MANUAL BF16 ACCUMULATION TEST ===")
    
    try:
        import mlx.core as mx
        
        values = [1.0, -0.99] * 10
        arr = mx.array(values, dtype=mx.bfloat16)
        
        print("Comparing MLX sum with manual BF16 accumulation:")
        
        # MLX sum
        mlx_sum = mx.sum(arr)
        print(f"  MLX sum:            {mlx_sum.astype(mx.float32)}")
        
        # Manual BF16 accumulation (pure BF16)
        acc_bf16 = mx.array(0.0, dtype=mx.bfloat16)
        for val in arr:
            acc_bf16 = acc_bf16 + val
        print(f"  Manual BF16 accum:  {acc_bf16.astype(mx.float32)}")
        
        # Manual FP32 accumulation 
        acc_fp32 = mx.array(0.0, dtype=mx.float32)
        for val in arr:
            acc_fp32 = acc_fp32 + val.astype(mx.float32)
        print(f"  Manual FP32 accum:  {acc_fp32}")
        
        # Check which one MLX matches
        diff_bf16 = abs(float(mlx_sum) - float(acc_bf16))
        diff_fp32 = abs(float(mlx_sum) - float(acc_fp32))
        
        print(f"  MLX vs manual BF16: {diff_bf16:.8f}")
        print(f"  MLX vs manual FP32: {diff_fp32:.8f}")
        
        if diff_bf16 < diff_fp32:
            print("  ➜ MLX uses BF16-like accumulation")
        else:
            print("  ➜ MLX uses FP32-like accumulation")
            
    except ImportError:
        print("MLX not available")

def test_pytorch_documentation():
    """Check if we can infer PyTorch's behavior from simple tests"""
    print("\n=== PYTORCH BEHAVIOR INFERENCE ===")
    
    try:
        import torch
        
        # Simple test that would be very different between BF16 and FP32 accumulation
        values = [1.0, -0.99] * 5  # Small case for clarity
        
        print("PyTorch sum behavior analysis:")
        
        # Test 1: BF16 sum
        bf16_tensor = torch.tensor(values, dtype=torch.bfloat16)
        bf16_sum = torch.sum(bf16_tensor)
        print(f"  BF16 sum: {bf16_sum.to(torch.float32)}")
        
        # Test 2: What if we force pure BF16 accumulation?
        # This should match MLX if MLX does pure BF16
        manual_bf16_acc = torch.tensor(0.0, dtype=torch.bfloat16)
        for val in bf16_tensor:
            manual_bf16_acc = manual_bf16_acc + val
        print(f"  Manual BF16: {manual_bf16_acc.to(torch.float32)}")
        
        # Test 3: FP32 promoted sum
        fp32_promoted = torch.sum(bf16_tensor.to(torch.float32)).to(torch.bfloat16)
        print(f"  FP32 promoted: {fp32_promoted.to(torch.float32)}")
        
        # Compare
        diff_manual = abs(float(bf16_sum) - float(manual_bf16_acc))
        diff_promoted = abs(float(bf16_sum) - float(fp32_promoted))
        
        print(f"  PyTorch vs manual BF16: {diff_manual:.8f}")
        print(f"  PyTorch vs FP32 promoted: {diff_promoted:.8f}")
        
        if diff_manual < 1e-8:
            print("  ➜ PyTorch does pure BF16 accumulation")
        elif diff_promoted < 1e-8:
            print("  ➜ PyTorch promotes to FP32 during sum!")
        else:
            print("  ➜ PyTorch uses unknown strategy")
            
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    print("TESTING PROMOTION HYPOTHESIS")
    print("=" * 50)
    print("Hypothesis: PyTorch promotes BF16→FP32 during sum, MLX uses pure BF16")
    print("=" * 50)
    
    test_promotion_hypothesis()
    test_different_patterns()
    test_manual_bf16_accumulation()
    test_pytorch_documentation()
    
    print("\n" + "=" * 50)
    print("CONCLUSION: Check results above to confirm/refute hypothesis")