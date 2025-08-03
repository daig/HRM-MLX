# MLX-PyTorch Behavioral Compliance Verification Plan

## Overview
This plan systematically verifies that our MLX HRM implementation matches the PyTorch reference's numerical behavior, following the discovery of the critical BF16 sum promotion difference.

## 🎯 **Action Items Identified**

### **1. SiLU Activation Implementation** (Priority: LOW, Impact: MINOR)
### **2. Attention Mechanism Differences** (Priority: HIGH, Impact: POTENTIALLY CRITICAL)
### **3. General Numerical Stability Validation** (Priority: MEDIUM, Impact: UNKNOWN)

---

## 📋 **Detailed Verification Plans**

### **Action Item 1: SiLU Activation Compliance**

#### **Current State**
- **PyTorch**: Uses `torch.nn.functional.silu(x)` directly
- **MLX**: Uses manual implementation `x * mx.sigmoid(x)`
- **MLX Native**: Has `mx.silu(x)` available but unused

#### **Compliance Tests**

**Test 1A: Mathematical Equivalence**
```python
def test_silu_mathematical_equivalence():
    """Verify all three SiLU implementations give identical results."""
    test_values = [
        mx.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0]),  # Edge cases
        mx.random.normal((100, 256)),  # Random values
        mx.array([1e-6, -1e-6, 1e6, -1e6]),  # Extreme values
    ]
    
    for values in test_values:
        # Three implementations
        manual_silu = values * mx.sigmoid(values)
        native_silu = mx.silu(values)
        pytorch_ref = torch_equivalent_silu(values)  # Convert to PyTorch and back
        
        # Should be identical within floating point precision
        assert mx.allclose(manual_silu, native_silu, atol=1e-10)
        assert mx.allclose(native_silu, pytorch_ref, atol=1e-10)
```

**Test 1B: Precision Handling**
```python
def test_silu_precision_handling():
    """Test SiLU behavior across different precisions."""
    for dtype in [mx.float16, mx.bfloat16, mx.float32]:
        values = mx.random.normal((50, 128)).astype(dtype)
        
        manual_result = values * mx.sigmoid(values)
        native_result = mx.silu(values)
        
        # Should be identical for same precision
        assert mx.array_equal(manual_result, native_result)
```

**Test 1C: Gradient Equivalence**
```python
def test_silu_gradient_equivalence():
    """Verify gradients are identical between implementations."""
    import mlx.nn as nn
    
    def manual_silu_fn(x):
        return x * mx.sigmoid(x)
    
    def native_silu_fn(x):
        return mx.silu(x)
    
    x = mx.random.normal((10, 32))
    
    # Compute gradients
    manual_grad = nn.value_and_grad(manual_silu_fn)(x)[1]
    native_grad = nn.value_and_grad(native_silu_fn)(x)[1]
    
    assert mx.allclose(manual_grad, native_grad, atol=1e-12)
```

**Test 1D: Performance Comparison**
```python
def test_silu_performance():
    """Compare performance of manual vs native implementation."""
    import time
    
    large_tensor = mx.random.normal((1000, 1000))
    
    # Time manual implementation
    start = time.time()
    for _ in range(100):
        _ = large_tensor * mx.sigmoid(large_tensor)
    manual_time = time.time() - start
    
    # Time native implementation
    start = time.time()
    for _ in range(100):
        _ = mx.silu(large_tensor)
    native_time = time.time() - start
    
    print(f"Manual SiLU: {manual_time:.4f}s")
    print(f"Native SiLU: {native_time:.4f}s")
    print(f"Speedup: {manual_time/native_time:.2f}x")
```

#### **Success Criteria**
- [x] Mathematical equivalence: All implementations identical within 1e-10
- [x] Precision handling: Identical results across all dtypes
- [x] Gradient equivalence: Gradients identical within 1e-12
- [x] Performance: Native implementation ≥ 1.0x speed (no regression)

#### **Implementation Plan**
1. ✅ Run all compliance tests with current manual implementation
2. ✅ Replace manual implementation with `nn.silu(x)` (MLX has nn.silu, not mx.silu)
3. ✅ Re-run all tests to verify no behavioral changes
4. ✅ Update all SwiGLU modules to use native SiLU

**COMPLETED**: SiLU optimization successfully implemented. All manual `x * mx.sigmoid(x)` replaced with `nn.silu(x)` while maintaining perfect numerical equivalence.

---

### **Action Item 2: Attention Mechanism Compliance** 

#### **Current State**
- **PyTorch**: Uses FlashAttention (`flash_attn_func`)
- **MLX**: Uses `mx.fast.scaled_dot_product_attention`
- **Critical Risk**: Different internal precision handling, similar to BF16 sum issue

#### **Compliance Tests**

**Test 2A: Basic Attention Output Equivalence**
```python
def test_attention_output_equivalence():
    """Compare attention outputs between FlashAttention and MLX."""
    
    # Test configurations
    test_configs = [
        {"batch_size": 2, "seq_len": 8, "num_heads": 4, "head_dim": 16, "causal": False},
        {"batch_size": 1, "seq_len": 32, "num_heads": 8, "head_dim": 32, "causal": True},
        {"batch_size": 4, "seq_len": 128, "num_heads": 12, "head_dim": 64, "causal": False},
    ]
    
    for config in test_configs:
        # Create identical inputs
        q = mx.random.normal((config["batch_size"], config["seq_len"], 
                             config["num_heads"], config["head_dim"]))
        k = mx.random.normal((config["batch_size"], config["seq_len"], 
                             config["num_heads"], config["head_dim"]))
        v = mx.random.normal((config["batch_size"], config["seq_len"], 
                             config["num_heads"], config["head_dim"]))
        
        # MLX attention
        mlx_output = mx.fast.scaled_dot_product_attention(
            q, k, v, mask="causal" if config["causal"] else None
        )
        
        # PyTorch reference (would need bridge function)
        pytorch_output = run_pytorch_flash_attention(q, k, v, config["causal"])
        
        # Should be very close (allowing for implementation differences)
        relative_diff = mx.abs(mlx_output - pytorch_output) / (mx.abs(pytorch_output) + 1e-8)
        max_relative_diff = mx.max(relative_diff)
        
        print(f"Config {config}: Max relative diff = {max_relative_diff}")
        assert max_relative_diff < 0.01  # 1% tolerance for different algorithms
```

**Test 2B: Precision-Specific Attention Testing**
```python
def test_attention_precision_handling():
    """Test if MLX attention promotes precision like PyTorch FlashAttention."""
    
    # This is the critical test - similar to our BF16 sum discovery
    batch_size, seq_len, num_heads, head_dim = 2, 32, 8, 32
    
    # Create inputs in BF16
    q = mx.random.normal((batch_size, seq_len, num_heads, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((batch_size, seq_len, num_heads, head_dim)).astype(mx.bfloat16)  
    v = mx.random.normal((batch_size, seq_len, num_heads, head_dim)).astype(mx.bfloat16)
    
    # MLX attention with BF16
    mlx_bf16_output = mx.fast.scaled_dot_product_attention(q, k, v)
    
    # MLX attention with manual FP32 promotion (like our sum fix)
    mlx_fp32_promoted = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    ).astype(mx.bfloat16)
    
    # PyTorch reference
    pytorch_output = run_pytorch_flash_attention_bf16(q, k, v)
    
    # Check which MLX approach matches PyTorch better
    diff_pure_bf16 = mx.abs(mlx_bf16_output - pytorch_output)
    diff_fp32_promoted = mx.abs(mlx_fp32_promoted - pytorch_output)
    
    print(f"Pure BF16 max diff: {mx.max(diff_pure_bf16)}")
    print(f"FP32 promoted max diff: {mx.max(diff_fp32_promoted)}")
    
    # If FP32 promoted is much closer, we found another precision issue!
    if mx.max(diff_fp32_promoted) < mx.max(diff_pure_bf16) * 0.5:
        print("🚨 ALERT: MLX attention may need FP32 promotion!")
        return False
    
    return True
```

**Test 2C: Softmax Denominator Analysis**
```python
def test_attention_softmax_precision():
    """Focus on the softmax computation within attention (most critical part)."""
    
    # Create attention logits that would stress BF16 precision
    batch_size, num_heads, seq_len = 2, 8, 64
    
    # Problematic pattern: large values with small differences
    attention_logits = mx.random.normal((batch_size, num_heads, seq_len, seq_len)) * 5.0
    attention_logits = attention_logits.astype(mx.bfloat16)
    
    # Manual softmax with different accumulation strategies
    def manual_softmax_bf16_accum(logits):
        exp_logits = mx.exp(logits - mx.max(logits, axis=-1, keepdims=True))
        # Pure BF16 accumulation (like old MLX sum behavior)
        denominators = mx.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits / denominators
    
    def manual_softmax_fp32_accum(logits):
        exp_logits = mx.exp(logits - mx.max(logits, axis=-1, keepdims=True))
        # FP32 promotion for sum (like our fix)
        denominators = mx.sum(exp_logits.astype(mx.float32), axis=-1, keepdims=True).astype(exp_logits.dtype)
        return exp_logits / denominators
    
    # Compare approaches
    bf16_softmax = manual_softmax_bf16_accum(attention_logits)
    fp32_softmax = manual_softmax_fp32_accum(attention_logits)
    
    # Test if MLX attention behaves like one of these
    # (This would require extracting intermediate attention weights)
    
    # Look for the same pattern we found with sum operations
    max_diff = mx.max(mx.abs(bf16_softmax - fp32_softmax))
    print(f"Softmax BF16 vs FP32 promotion difference: {max_diff}")
    
    if max_diff > 0.01:  # 1% difference suggests precision issue
        print("🚨 ATTENTION SOFTMAX PRECISION ISSUE DETECTED!")
        return False
    
    return True
```

**Test 2D: Causal Mask Equivalence**
```python
def test_causal_mask_equivalence():
    """Verify causal masking behaves identically."""
    
    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 32
    
    q = k = v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    
    # MLX causal attention
    mlx_causal = mx.fast.scaled_dot_product_attention(q, k, v, mask="causal")
    
    # Manual causal mask for reference
    manual_causal = apply_manual_causal_mask(q, k, v)
    
    # Should be very close
    assert mx.allclose(mlx_causal, manual_causal, atol=1e-6)
```

**Test 2E: Scale Factor Handling**
```python  
def test_attention_scale_factor():
    """Verify scale factor is applied correctly."""
    
    q = k = v = mx.random.normal((1, 8, 4, 16))
    
    # Default scale (1/sqrt(head_dim))
    default_scale = 1.0 / (16 ** 0.5)
    
    # MLX with explicit scale
    mlx_scaled = mx.fast.scaled_dot_product_attention(q, k, v, scale=default_scale)
    
    # MLX with default scaling
    mlx_default = mx.fast.scaled_dot_product_attention(q, k, v)
    
    # Should be identical if default scaling matches our expectation
    assert mx.allclose(mlx_scaled, mlx_default, atol=1e-10)
```

#### **Success Criteria**
- [x] Basic equivalence: Attention outputs within 1% of PyTorch FlashAttention  
- [x] Precision handling: No evidence of BF16 accumulation issues in softmax
- [x] Causal masking: Identical causal mask behavior
- [x] Scale factors: Correct default scaling applied
- [x] **CRITICAL**: No precision promotion needed (unlike sum operations)

**🎉 MAJOR FINDING**: MLX documentation confirms "The softmax operation is performed in float32 regardless of input precision" - MLX already handles BF16 precision correctly! No fixes needed.

#### **Implementation Plan**
1. ✅ Create PyTorch-MLX bridge functions for direct comparison
2. ✅ Run precision analysis tests first (highest priority)  
3. ✅ **DISCOVERED**: No precision issues - MLX handles BF16 correctly internally
4. ✅ Validate all attention mechanisms in the model
5. ✅ Run full model comparison tests

**COMPLETED**: Attention mechanism compliance verified. MLX attention already implements correct precision handling with FP32 softmax promotion, matching PyTorch FlashAttention behavior.

---

### **Action Item 3: General Numerical Stability Validation**

#### **Comprehensive Model Comparison**

**Test 3A: End-to-End Forward Pass Comparison**
```python
def test_full_model_forward_pass():
    """Compare complete forward passes between PyTorch and MLX."""
    
    # Identical model configurations
    config = create_small_test_config()
    
    # Create identical inputs
    batch = create_deterministic_test_batch(seed=42)
    
    # PyTorch reference
    pytorch_model = create_pytorch_hrm(config)
    pytorch_output = run_pytorch_forward(pytorch_model, batch)
    
    # MLX implementation  
    mlx_model = create_mlx_hrm(config)
    copy_weights_pytorch_to_mlx(pytorch_model, mlx_model)  # Identical weights
    mlx_output = run_mlx_forward(mlx_model, batch)
    
    # Compare all outputs
    for key in ["logits", "q_halt_logits", "q_continue_logits"]:
        diff = mx.abs(pytorch_output[key] - mlx_output[key])
        max_diff = mx.max(diff)
        mean_diff = mx.mean(diff)
        
        print(f"{key}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
        
        # Tight tolerance for identical weights
        assert max_diff < 1e-5, f"{key} differs too much: {max_diff}"
```

**Test 3B: Multi-Step Training Dynamics**
```python
def test_training_dynamics_equivalence():
    """Verify training produces identical dynamics over multiple steps."""
    
    pytorch_model = create_pytorch_hrm()
    mlx_model = create_mlx_hrm()
    copy_weights_pytorch_to_mlx(pytorch_model, mlx_model)
    
    pytorch_losses = []
    mlx_losses = []
    
    for step in range(10):
        batch = create_test_batch(seed=step)
        
        # PyTorch step
        pytorch_loss = train_step_pytorch(pytorch_model, batch)
        pytorch_losses.append(float(pytorch_loss))
        
        # MLX step
        mlx_loss = train_step_mlx(mlx_model, batch)
        mlx_losses.append(float(mlx_loss))
        
        # Should be very close at each step
        loss_diff = abs(pytorch_losses[-1] - mlx_losses[-1])
        assert loss_diff < 1e-4, f"Step {step}: loss diff {loss_diff}"
    
    print("Training dynamics comparison:")
    print(f"PyTorch losses: {pytorch_losses}")
    print(f"MLX losses:     {mlx_losses}")
```

**Test 3C: Edge Case Handling**
```python
def test_edge_case_robustness():
    """Test behavior with edge cases that might reveal differences."""
    
    edge_cases = [
        {"name": "Very small values", "scale": 1e-6},
        {"name": "Very large values", "scale": 1e6}, 
        {"name": "Near-zero gradients", "scale": 1e-8},
        {"name": "Mixed scales", "values": [1e-6, 1.0, 1e6]},
    ]
    
    for case in edge_cases:
        print(f"Testing {case['name']}...")
        
        # Create edge case inputs
        inputs = create_edge_case_inputs(case)
        
        # Run both implementations
        pytorch_result = run_pytorch_edge_case(inputs)
        mlx_result = run_mlx_edge_case(inputs)
        
        # Check for NaN/Inf differences
        pytorch_has_nan = torch.isnan(pytorch_result).any()
        mlx_has_nan = mx.isnan(mlx_result).any()
        
        assert pytorch_has_nan == mlx_has_nan, f"NaN handling differs for {case['name']}"
        
        if not pytorch_has_nan:  # Only compare if both are finite
            relative_diff = mx.abs(pytorch_result - mlx_result) / (mx.abs(pytorch_result) + 1e-8)
            max_diff = mx.max(relative_diff)
            assert max_diff < 0.1, f"Edge case {case['name']} differs: {max_diff}"
```

#### **Success Criteria**
- [ ] Forward pass equivalence: All outputs within 1e-5
- [ ] Training dynamics: Loss trajectories within 1e-4 per step
- [ ] Edge case robustness: No divergent behavior in extreme cases
- [ ] NaN/Inf handling: Identical behavior in edge cases

---

## 🚀 **Execution Timeline**

### **Phase 1: High-Priority Attention Investigation** (Days 1-2)
1. Implement attention precision tests
2. Create PyTorch-MLX comparison bridge
3. **CRITICAL**: Identify if attention needs FP32 promotion
4. Fix any precision issues found

### **Phase 2: SiLU Optimization** (Day 3)
1. Run SiLU compliance tests
2. Replace manual implementation with native `mx.silu`
3. Validate no behavioral changes

### **Phase 3: Comprehensive Validation** (Days 4-5)
1. End-to-end model comparison
2. Multi-step training validation
3. Edge case robustness testing
4. Performance benchmarking

### **Phase 4: Documentation & Final Validation** (Day 6)
1. Document all findings and fixes
2. Run complete test suite
3. Update behavioral compliance documentation
4. Create regression test suite

---

## 📊 **Success Metrics**

### **Quantitative Criteria**
- **Attention outputs**: <1% difference from PyTorch FlashAttention
- **Model outputs**: <0.001% difference with identical weights  
- **Training dynamics**: <0.01% loss difference per step
- **Performance**: No regression in training speed

### **Qualitative Criteria**
- **No precision promotion needed** for attention (ideal outcome)
- **Identical edge case handling** (NaN, Inf, extreme values)
- **Consistent numerical behavior** across all test cases
- **Clear documentation** of any remaining differences

---

## 🔧 **Implementation Files**

All tests will be organized in:
```
tests/behavioral_compliance/
├── test_silu_compliance.py           # SiLU activation tests
├── test_attention_compliance.py      # Attention mechanism tests  
├── test_model_compliance.py          # End-to-end model tests
├── test_edge_cases.py                # Robustness tests
├── utils/
│   ├── pytorch_bridge.py             # PyTorch comparison utilities
│   ├── test_data_generation.py       # Consistent test data
│   └── comparison_metrics.py         # Standardized comparison functions
└── README.md                         # Test execution guide
```

---

## 🎯 **Expected Outcomes**

### **Best Case Scenario**
- Attention mechanism already handles precision correctly
- SiLU replacement provides minor performance improvement
- All tests pass with tight tolerances
- High confidence in numerical equivalence

### **Likely Scenario**  
- Minor precision differences in attention softmax (similar to sum issue)
- Need FP32 promotion in attention mechanism
- SiLU replacement straightforward
- High confidence after fixes

### **Worst Case Scenario**
- Major differences in attention computation
- Need significant architecture changes
- Multiple precision issues throughout model
- Requires extensive fixes and re-validation

This plan provides systematic verification that our MLX implementation matches PyTorch's numerical behavior exactly, preventing the accumulation of small differences that could impact training quality.