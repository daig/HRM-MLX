"""
Benchmark script for Attention module.

This script measures the performance of the MLX attention implementation
across various configurations and compares with theoretical expectations.
"""

import time
import argparse
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_hrm.modules.attention import Attention
from mlx_hrm.modules.rope import RotaryEmbedding


def benchmark_attention_forward(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: mx.Dtype = mx.float32,
    use_rope: bool = False,
    causal: bool = False,
    num_warmup: int = 10,
    num_runs: int = 100
) -> dict:
    """Benchmark attention forward pass."""
    
    head_dim = hidden_size // num_heads
    
    # Create attention module
    attn = Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        causal=causal
    )
    
    # Create RoPE if needed
    rope = None
    cos_sin = None
    if use_rope:
        rope = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max(512, seq_len),
            base=10000.0
        )
        cos, sin = rope(seq_len, dtype=dtype)
        cos_sin = (cos, sin)
    
    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size), dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        _ = attn(cos_sin, x)
    mx.eval(x)
    
    # Time execution
    start_time = time.time()
    for _ in range(num_runs):
        output = attn(cos_sin, x)
        mx.eval(output)
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / num_runs
    
    # Calculate FLOPS
    # Attention FLOPS ≈ 4 * batch * seq_len^2 * hidden_size (simplified)
    flops = 4 * batch_size * seq_len * seq_len * hidden_size
    gflops = (flops * num_runs) / (elapsed * 1e9)
    
    # Calculate memory
    def count_params(params):
        count = 0
        for name, value in params.items():
            if isinstance(value, mx.array):
                count += value.size
            elif isinstance(value, dict):
                count += count_params(value)
        return count
    
    param_count = count_params(attn.parameters())
    param_memory = param_count * 4  # float32
    
    return {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "gqa_ratio": num_heads // num_kv_heads,
            "dtype": str(dtype),
            "use_rope": use_rope,
            "causal": causal
        },
        "performance": {
            "total_time": elapsed,
            "avg_time_per_forward": avg_time,
            "throughput_samples_per_sec": (batch_size * num_runs) / elapsed,
            "estimated_gflops": gflops
        },
        "memory": {
            "param_count": param_count,
            "param_memory_mb": param_memory / (1024 * 1024)
        }
    }


def benchmark_attention_configurations():
    """Benchmark various attention configurations."""
    
    configs = [
        # (batch, seq_len, hidden, n_heads, n_kv_heads, name)
        (32, 128, 768, 12, 12, "Base (BERT-like)"),
        (32, 256, 768, 12, 12, "Medium sequence"),
        (16, 512, 768, 12, 12, "Long sequence"),
        (32, 128, 768, 12, 3, "GQA 4:1"),
        (32, 128, 768, 12, 2, "GQA 6:1"),
        (32, 128, 768, 12, 1, "GQA 12:1 (MQA)"),
        (8, 512, 1024, 16, 16, "Large model"),
        (8, 512, 1024, 16, 4, "Large model GQA 4:1"),
    ]
    
    print("=" * 80)
    print("MLX Attention Benchmarks")
    print("=" * 80)
    
    results = []
    
    for batch, seq_len, hidden, n_heads, n_kv_heads, name in configs:
        print(f"\nBenchmarking: {name}")
        print(f"Config: batch={batch}, seq={seq_len}, hidden={hidden}, "
              f"heads={n_heads}, kv_heads={n_kv_heads}")
        
        # Benchmark without RoPE
        result = benchmark_attention_forward(
            batch, seq_len, hidden, n_heads, n_kv_heads,
            dtype=mx.float32, use_rope=False, causal=False
        )
        
        print(f"Time per forward: {result['performance']['avg_time_per_forward']*1000:.2f} ms")
        print(f"Throughput: {result['performance']['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Estimated GFLOPS: {result['performance']['estimated_gflops']:.1f}")
        
        results.append((name, result))
    
    return results


def benchmark_dtype_comparison():
    """Compare performance across different data types."""
    
    print("\n" + "=" * 80)
    print("Data Type Comparison")
    print("=" * 80)
    
    batch, seq_len, hidden = 16, 256, 768
    n_heads = 12
    
    dtypes = [mx.float32, mx.float16]
    if hasattr(mx, 'bfloat16'):
        dtypes.append(mx.bfloat16)
    
    for dtype in dtypes:
        print(f"\nBenchmarking with {dtype}")
        
        result = benchmark_attention_forward(
            batch, seq_len, hidden, n_heads, n_heads,
            dtype=dtype, use_rope=True, causal=False
        )
        
        print(f"Time per forward: {result['performance']['avg_time_per_forward']*1000:.2f} ms")
        print(f"Throughput: {result['performance']['throughput_samples_per_sec']:.1f} samples/sec")


def benchmark_causal_comparison():
    """Compare causal vs non-causal attention."""
    
    print("\n" + "=" * 80)
    print("Causal vs Non-Causal Comparison")
    print("=" * 80)
    
    batch, seq_len, hidden = 16, 256, 768
    n_heads = 12
    
    for causal in [False, True]:
        print(f"\n{'Causal' if causal else 'Non-causal'} attention")
        
        result = benchmark_attention_forward(
            batch, seq_len, hidden, n_heads, n_heads,
            dtype=mx.float32, use_rope=True, causal=causal
        )
        
        print(f"Time per forward: {result['performance']['avg_time_per_forward']*1000:.2f} ms")
        print(f"Throughput: {result['performance']['throughput_samples_per_sec']:.1f} samples/sec")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX Attention")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--dtype", action="store_true", help="Run dtype comparison")
    parser.add_argument("--causal", action="store_true", help="Run causal comparison")
    args = parser.parse_args()
    
    if args.full or not any([args.dtype, args.causal]):
        benchmark_attention_configurations()
    
    if args.dtype or args.full:
        benchmark_dtype_comparison()
    
    if args.causal or args.full:
        benchmark_causal_comparison()
    
    print("\n" + "=" * 80)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()