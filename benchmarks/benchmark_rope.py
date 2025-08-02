"""
Performance benchmarks for RoPE implementation.

Measures initialization time, forward pass time, and memory usage.
"""

import time
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mlx_hrm.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


def benchmark_rope_initialization():
    """Benchmark RoPE initialization at various scales."""
    print("\n=== RoPE Initialization Benchmark ===")
    
    configs = [
        ("Small", 64, 512),
        ("Medium", 128, 2048),
        ("Large", 256, 8192),
        ("XLarge", 512, 16384),
    ]
    
    for name, dim, max_pos in configs:
        # Warmup
        _ = RotaryEmbedding(dim, max_pos)
        
        # Time initialization
        start = time.perf_counter()
        rope = RotaryEmbedding(dim, max_pos)
        # Force computation
        mx.eval(rope.cos_cached)
        mx.eval(rope.sin_cached)
        end = time.perf_counter()
        
        init_time = (end - start) * 1000  # Convert to ms
        cache_size_mb = (rope.cos_cached.nbytes + rope.sin_cached.nbytes) / (1024 * 1024)
        
        print(f"{name:8} (dim={dim:3}, max_pos={max_pos:5}): "
              f"{init_time:6.2f} ms, cache: {cache_size_mb:6.2f} MB")


def benchmark_rope_forward():
    """Benchmark RoPE forward pass (getting cos/sin values)."""
    print("\n=== RoPE Forward Pass Benchmark ===")
    
    dim = 128
    max_pos = 2048
    rope = RotaryEmbedding(dim, max_pos)
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    for seq_len in seq_lengths:
        # Warmup
        for _ in range(10):
            cos, sin = rope(seq_len)
            mx.eval(cos)
            mx.eval(sin)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            cos, sin = rope(seq_len)
            mx.eval(cos)
            mx.eval(sin)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"Seq length {seq_len:4}: {avg_time:6.3f} ms/iter")


def benchmark_apply_rope():
    """Benchmark applying RoPE to tensors."""
    print("\n=== Apply RoPE Benchmark ===")
    
    configs = [
        ("Small", 8, 512, 8, 64),
        ("Medium", 16, 1024, 12, 64),
        ("Large", 32, 2048, 16, 128),
        ("XLarge", 64, 4096, 20, 128),
    ]
    
    for name, batch, seq_len, num_heads, head_dim in configs:
        # Setup
        rope = RotaryEmbedding(head_dim, seq_len)
        cos, sin = rope(seq_len)
        
        # Create input tensors
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        
        # Warmup
        for _ in range(10):
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            mx.eval(q_rot)
            mx.eval(k_rot)
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            mx.eval(q_rot)
            mx.eval(k_rot)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        throughput = (batch * seq_len * num_heads * head_dim * 2) / (avg_time / 1000) / 1e9  # GFLOPS
        
        print(f"{name:8} (B={batch:2}, L={seq_len:4}, H={num_heads:2}, D={head_dim:3}): "
              f"{avg_time:6.2f} ms/iter, {throughput:6.2f} GFLOPS")


def benchmark_dtype_performance():
    """Benchmark performance with different dtypes."""
    print("\n=== Dtype Performance Benchmark ===")
    
    batch, seq_len, num_heads, head_dim = 16, 1024, 12, 64
    rope = RotaryEmbedding(head_dim, seq_len)
    
    for dtype in [mx.float16, mx.float32]:
        # Get cos/sin in specified dtype
        cos, sin = rope(seq_len, dtype=dtype)
        
        # Create tensors in specified dtype
        q = mx.random.normal((batch, seq_len, num_heads, head_dim), dtype=dtype)
        k = mx.random.normal((batch, seq_len, num_heads, head_dim), dtype=dtype)
        
        # Warmup
        for _ in range(10):
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            mx.eval(q_rot)
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            mx.eval(q_rot)
            mx.eval(k_rot)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"{str(dtype):8}: {avg_time:6.2f} ms/iter")


def benchmark_cache_extension():
    """Benchmark cache extension performance."""
    print("\n=== Cache Extension Benchmark ===")
    
    initial_max = 1024
    rope = RotaryEmbedding(128, initial_max)
    
    extensions = [2048, 4096, 8192, 16384]
    
    for new_max in extensions:
        start = time.perf_counter()
        rope.extend_cache(new_max)
        mx.eval(rope.cos_cached)
        mx.eval(rope.sin_cached)
        end = time.perf_counter()
        
        extend_time = (end - start) * 1000
        cache_size_mb = (rope.cos_cached.nbytes + rope.sin_cached.nbytes) / (1024 * 1024)
        
        print(f"Extend to {new_max:5}: {extend_time:6.2f} ms, total cache: {cache_size_mb:6.2f} MB")


def main():
    """Run all benchmarks."""
    print("MLX RoPE Performance Benchmarks")
    print("==============================")
    
    # Set random seed for reproducibility
    mx.random.seed(42)
    
    # Run benchmarks
    benchmark_rope_initialization()
    benchmark_rope_forward()
    benchmark_apply_rope()
    benchmark_dtype_performance()
    benchmark_cache_extension()
    
    print("\nBenchmarks complete!")


if __name__ == "__main__":
    main()