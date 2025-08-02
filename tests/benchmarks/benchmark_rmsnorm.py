"""Benchmark RMSNorm performance."""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_hrm.layers.normalization import rms_norm, RMSNorm, RMSNormCompatible


def benchmark_forward_pass(name, norm_fn, x, warmup=10, iterations=100):
    """Benchmark forward pass performance."""
    # Warmup
    for _ in range(warmup):
        output = norm_fn(x)
        mx.eval(output)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        output = norm_fn(x)
        mx.eval(output)
    elapsed = time.time() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    throughput = x.size / (elapsed / iterations) / 1e9  # GB/s
    
    print(f"{name:30} | {ms_per_iter:8.3f} ms/iter | {throughput:8.2f} GB/s")
    return ms_per_iter


def benchmark_backward_pass(name, norm_module, x, warmup=10, iterations=100):
    """Benchmark backward pass performance."""
    def loss_fn(x):
        return mx.sum(norm_module(x))
    
    # Warmup
    for _ in range(warmup):
        loss, grads = mx.value_and_grad(loss_fn)(x)
        mx.eval(loss, grads)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        loss, grads = mx.value_and_grad(loss_fn)(x)
        mx.eval(loss, grads)
    elapsed = time.time() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    print(f"{name:30} | {ms_per_iter:8.3f} ms/iter")
    return ms_per_iter


def main():
    """Run RMSNorm benchmarks."""
    print("RMSNorm Performance Benchmarks")
    print("=" * 70)
    
    # Test configurations
    configs = [
        ("Small", (32, 512, 768)),     # Typical small transformer
        ("Medium", (64, 128, 1024)),   # Medium config
        ("Large", (16, 1024, 2048)),   # Large sequence/hidden
        ("XLarge", (8, 2048, 4096)),   # Very large config
    ]
    
    for config_name, (batch, seq_len, hidden_dim) in configs:
        print(f"\nConfiguration: {config_name} - Shape: {(batch, seq_len, hidden_dim)}")
        print("-" * 70)
        
        # Create input data
        x = mx.random.normal((batch, seq_len, hidden_dim), dtype=mx.float32)
        
        # Create norm variants
        functional_norm = lambda x: rms_norm(x, variance_epsilon=1e-5)
        compatible_norm = RMSNormCompatible(eps=1e-5)
        module_no_scale = RMSNorm(hidden_dim, eps=1e-5, use_scale=False)
        module_with_scale = RMSNorm(hidden_dim, eps=1e-5, use_scale=True)
        builtin_norm = nn.RMSNorm(hidden_dim, eps=1e-5)
        
        print("\nForward Pass:")
        print("Method                        | Time (ms)   | Throughput")
        print("-" * 70)
        
        # Benchmark forward passes
        benchmark_forward_pass("Functional", functional_norm, x)
        benchmark_forward_pass("Compatible Module", compatible_norm, x)
        benchmark_forward_pass("Module (no scale)", module_no_scale, x)
        benchmark_forward_pass("Module (with scale)", module_with_scale, x)
        benchmark_forward_pass("MLX Built-in", builtin_norm, x)
        
        print("\nBackward Pass:")
        print("Method                        | Time (ms)")
        print("-" * 70)
        
        # Benchmark backward passes (only for modules)
        benchmark_backward_pass("Compatible Module", compatible_norm, x)
        benchmark_backward_pass("Module (no scale)", module_no_scale, x)
        benchmark_backward_pass("Module (with scale)", module_with_scale, x)
        benchmark_backward_pass("MLX Built-in", builtin_norm, x)
    
    # Memory usage comparison
    print("\n" + "=" * 70)
    print("Memory Usage Comparison")
    print("=" * 70)
    
    batch, seq_len, hidden_dim = 32, 512, 768
    x = mx.random.normal((batch, seq_len, hidden_dim), dtype=mx.float32)
    
    # Calculate theoretical memory usage
    input_memory = x.nbytes / (1024 * 1024)  # MB
    print(f"\nInput tensor memory: {input_memory:.2f} MB")
    
    # Check parameter counts
    module_with_scale = RMSNorm(hidden_dim, eps=1e-5, use_scale=True)
    builtin_norm = nn.RMSNorm(hidden_dim, eps=1e-5)
    
    if hasattr(module_with_scale._norm, 'parameters'):
        scale_params = sum(p.size for p in module_with_scale._norm.parameters().values())
        print(f"Module with scale parameters: {scale_params} parameters")
    
    if hasattr(builtin_norm, 'parameters'):
        builtin_params = sum(p.size for p in builtin_norm.parameters().values())
        print(f"Built-in RMSNorm parameters: {builtin_params} parameters")
    
    # Test different dtypes
    print("\n" + "=" * 70)
    print("Data Type Performance Comparison")
    print("=" * 70)
    
    batch, seq_len, hidden_dim = 32, 512, 768
    dtypes = [mx.float32, mx.float16]
    if hasattr(mx, 'bfloat16'):
        dtypes.append(mx.bfloat16)
    
    for dtype in dtypes:
        print(f"\nData type: {dtype}")
        x = mx.random.normal((batch, seq_len, hidden_dim), dtype=dtype)
        
        # Only benchmark functional version for simplicity
        ms_time = benchmark_forward_pass("Functional RMSNorm", 
                                       lambda x: rms_norm(x), x, 
                                       warmup=20, iterations=200)


if __name__ == "__main__":
    main()