#!/usr/bin/env python3
"""Analyze stablemax numerical stability vs performance trade-offs."""

import mlx.core as mx
import numpy as np
import time

print('=== Stablemax Numerical Stability Analysis ===')

# Test data that might cause numerical issues
test_cases = {
    'normal': np.random.randn(4, 32, 5000).astype(np.float32),
    'large_values': np.random.randn(4, 32, 5000).astype(np.float32) * 10,
    'extreme_values': np.random.randn(4, 32, 5000).astype(np.float32) * 50,
}

def stablemax_fp32(logits):
    """Current FP32 implementation."""
    x = mx.array(logits, dtype=mx.float32)
    s_x = mx.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
    log_probs = mx.log(s_x) - mx.log(mx.sum(s_x, axis=-1, keepdims=True))
    return log_probs

def stablemax_fp64_partial(logits):
    """FP64 only for log computation, rest in FP32."""
    x = mx.array(logits, dtype=mx.float32)
    s_x = mx.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
    
    # Only the log computation in FP64
    s_x_fp64 = s_x.astype(mx.float64)
    log_s_x = mx.log(s_x_fp64)
    log_sum = mx.log(mx.sum(s_x_fp64, axis=-1, keepdims=True))
    log_probs_fp64 = log_s_x - log_sum
    
    return log_probs_fp64.astype(mx.float32)

def stablemax_fp64_full(logits):
    """Full FP64 implementation."""
    x = mx.array(logits, dtype=mx.float64)
    s_x = mx.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
    log_probs = mx.log(s_x) - mx.log(mx.sum(s_x, axis=-1, keepdims=True))
    return log_probs.astype(mx.float32)

def stablemax_improved_fp32(logits):
    """Improved FP32 with better numerical stability tricks."""
    x = mx.array(logits, dtype=mx.float32)
    
    # Subtract max for numerical stability (like standard logsoftmax)
    x_max = mx.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    
    s_x = mx.where(x_shifted < 0, 1.0 / (1.0 - x_shifted + 1e-30), x_shifted + 1.0)
    log_probs = mx.log(s_x) - mx.log(mx.sum(s_x, axis=-1, keepdims=True))
    return log_probs

mx.set_default_device(mx.cpu)

for case_name, test_data in test_cases.items():
    print(f'\n=== {case_name.upper()} VALUES ===')
    
    # Get reference implementation (FP64 full)
    ref_output = stablemax_fp64_full(test_data)
    
    implementations = {
        'FP32 Original': stablemax_fp32,
        'FP64 Partial (log only)': stablemax_fp64_partial, 
        'FP32 Improved': stablemax_improved_fp32,
    }
    
    for impl_name, impl_func in implementations.items():
        try:
            # Compute output
            start = time.time()
            output = impl_func(test_data)
            elapsed = time.time() - start
            
            # Measure difference from reference
            diff = mx.max(mx.abs(output - ref_output))
            rel_diff = mx.max(mx.abs((output - ref_output) / (mx.abs(ref_output) + 1e-10)))
            
            print(f'{impl_name:20}: {elapsed*1000:6.2f}ms, max_diff={float(diff):.2e}, rel_diff={float(rel_diff):.2e}')
            
        except Exception as e:
            print(f'{impl_name:20}: ERROR - {e}')

print('\n=== Performance vs Quality Trade-off Analysis ===')

# Focus on the most realistic case
test_data = test_cases['large_values']
print(f'Test data shape: {test_data.shape}')
print(f'Value range: [{test_data.min():.2f}, {test_data.max():.2f}]')

# Benchmark all approaches
approaches = {
    'FP32 Only': stablemax_fp32,
    'FP64 Log Only': stablemax_fp64_partial,
    'FP64 Full': stablemax_fp64_full,
    'FP32 Max-Shifted': stablemax_improved_fp32,
}

ref_output = stablemax_fp64_full(test_data)

print('\nMethod                 | Time (ms) | Max Error | Rel Error | Recommendation')
print('-' * 80)

for method_name, method_func in approaches.items():
    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        output = method_func(test_data)
        mx.eval(output)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    
    # Accuracy
    if method_name != 'FP64 Full':
        diff = mx.max(mx.abs(output - ref_output))
        rel_diff = mx.max(mx.abs((output - ref_output) / (mx.abs(ref_output) + 1e-10)))
        max_err = f'{float(diff):.1e}'
        rel_err = f'{float(rel_diff):.1e}'
    else:
        max_err = 'Reference'
        rel_err = 'Reference'
    
    # Recommendation
    if avg_time < 2 and (max_err == 'Reference' or 'e-0' in max_err or 'e-1' in max_err):
        rec = '✅ Recommended'
    elif avg_time < 5:
        rec = '⚠️  Acceptable'
    else:
        rec = '❌ Too slow'
    
    print(f'{method_name:22} | {avg_time:8.2f} | {max_err:>9} | {rel_err:>9} | {rec}')

print('\n=== Memory Usage Analysis ===')
# Estimate memory usage for different approaches
vocab_size = 5000
batch_size, seq_len = 4, 32
total_elements = batch_size * seq_len * vocab_size

print(f'Total logits elements: {total_elements:,}')
print(f'FP32 memory: {total_elements * 4 / 1024**2:.1f} MB')
print(f'FP64 memory: {total_elements * 8 / 1024**2:.1f} MB')
print(f'Additional memory for FP64: {total_elements * 4 / 1024**2:.1f} MB')

print('\n=== Training Impact Simulation ===')
# Estimate impact on overall training
forward_pass_time = 50  # ms (typical estimate)
stablemax_fp32_time = 0.9  # from benchmark
stablemax_fp64_time = 1.9  # from benchmark

fp32_total = forward_pass_time + stablemax_fp32_time
fp64_total = forward_pass_time + stablemax_fp64_time

print(f'Forward pass without loss: {forward_pass_time:.1f}ms')
print(f'With FP32 stablemax: {fp32_total:.1f}ms (total)')
print(f'With FP64 stablemax: {fp64_total:.1f}ms (total)')
print(f'Slowdown: {fp64_total/fp32_total:.1%}')

# Per epoch impact
steps_per_epoch = 1000
epoch_time_fp32 = steps_per_epoch * fp32_total / 1000  # seconds
epoch_time_fp64 = steps_per_epoch * fp64_total / 1000  # seconds
additional_time = epoch_time_fp64 - epoch_time_fp32

print(f'\nPer epoch ({steps_per_epoch} steps):')
print(f'FP32: {epoch_time_fp32:.1f}s')
print(f'FP64: {epoch_time_fp64:.1f}s')
print(f'Additional time: {additional_time:.1f}s ({additional_time/60:.1f} minutes)')