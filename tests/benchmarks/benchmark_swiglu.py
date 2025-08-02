"""Benchmark SwiGLU performance."""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_hrm.layers.activations import SwiGLU, SwiGLUFactory, _find_multiple


def benchmark_forward_pass(name, module, x, warmup=10, iterations=100):
    """Benchmark forward pass performance."""
    # Warmup
    for _ in range(warmup):
        output = module(x)
        mx.eval(output)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        output = module(x)
        mx.eval(output)
    elapsed = time.time() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    
    # Calculate approximate FLOPS
    if hasattr(module, 'intermediate_dim'):
        intermediate_dim = module.intermediate_dim
        batch_size = x.shape[0] if x.ndim >= 2 else 1
        seq_len = x.shape[-2] if x.ndim >= 2 else 1
        hidden_dim = x.shape[-1]
        
        # FLOPs calculation for SwiGLU
        flops_per_token = (
            2 * hidden_dim * intermediate_dim * 2 +  # gate_up projection
            2 * intermediate_dim * hidden_dim +      # down projection
            intermediate_dim * 3                     # sigmoid + 2 multiplies
        )
        total_flops = flops_per_token * batch_size * seq_len * iterations
        gflops = total_flops / elapsed / 1e9
        
        print(f"{name:35} | {ms_per_iter:8.3f} ms/iter | {gflops:8.1f} GFLOPS")
    else:
        print(f"{name:35} | {ms_per_iter:8.3f} ms/iter")
    
    return ms_per_iter


def benchmark_backward_pass(name, module, x, warmup=10, iterations=100):
    """Benchmark backward pass performance."""
    def loss_fn(x):
        return mx.sum(module(x) ** 2)
    
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
    print(f"{name:35} | {ms_per_iter:8.3f} ms/iter")
    return ms_per_iter


def create_mlp_baseline(hidden_dim, expansion=4.0):
    """Create standard MLP for comparison."""
    intermediate_dim = int(hidden_dim * expansion)
    return nn.Sequential(
        nn.Linear(hidden_dim, intermediate_dim, bias=False),
        nn.GELU(),
        nn.Linear(intermediate_dim, hidden_dim, bias=False)
    )


def main():
    """Run SwiGLU benchmarks."""
    print("SwiGLU Performance Benchmarks")
    print("=" * 80)
    
    # Test configurations
    configs = [
        ("Small", (32, 512, 768), 4.0),      # Standard transformer
        ("Medium", (16, 1024, 1024), 4.0),   # Larger model
        ("Large", (8, 2048, 2048), 4.0),     # Very large
        ("High Expansion", (32, 512, 768), 8.0),  # Higher expansion
    ]
    
    for config_name, (batch, seq_len, hidden_dim), expansion in configs:
        print(f"\nConfiguration: {config_name}")
        print(f"Shape: ({batch}, {seq_len}, {hidden_dim}), Expansion: {expansion}")
        print("-" * 80)
        
        # Create input data
        x = mx.random.normal((batch, seq_len, hidden_dim), dtype=mx.float32)
        
        # Create models
        swiglu_default = SwiGLU(hidden_dim, expansion=expansion)
        swiglu_efficient = SwiGLUFactory.create_efficient(hidden_dim)
        mlp_baseline = create_mlp_baseline(hidden_dim, expansion)
        
        # Show dimensions
        print(f"\nSwiGLU intermediate dim: {swiglu_default.intermediate_dim}")
        expected_inter = _find_multiple(int(round(expansion * hidden_dim * 2 / 3)), 256)
        print(f"Expected intermediate: {expected_inter}")
        print(f"MLP intermediate dim: {int(hidden_dim * expansion)}")
        
        print("\nForward Pass:")
        print("Method                              | Time (ms)   | Performance")
        print("-" * 80)
        
        # Benchmark forward passes
        benchmark_forward_pass("SwiGLU (default)", swiglu_default, x)
        benchmark_forward_pass("SwiGLU (efficient)", swiglu_efficient, x)
        benchmark_forward_pass("MLP + GELU (baseline)", mlp_baseline, x)
        
        print("\nBackward Pass:")
        print("Method                              | Time (ms)")
        print("-" * 80)
        
        # Benchmark backward passes
        benchmark_backward_pass("SwiGLU (default)", swiglu_default, x)
        benchmark_backward_pass("SwiGLU (efficient)", swiglu_efficient, x)
        benchmark_backward_pass("MLP + GELU (baseline)", mlp_baseline, x)
    
    # Memory usage comparison
    print("\n" + "=" * 80)
    print("Memory Usage Analysis")
    print("=" * 80)
    
    hidden_dim = 768
    expansion = 4.0
    
    # Create models
    swiglu = SwiGLU(hidden_dim, expansion)
    mlp = create_mlp_baseline(hidden_dim, expansion)
    
    # Calculate parameter counts
    swiglu_params = (
        swiglu.gate_up_proj.weight.size +
        swiglu.down_proj.weight.size
    )
    
    mlp_params = sum(
        layer.weight.size for layer in mlp.layers 
        if hasattr(layer, 'weight')
    )
    
    print(f"\nHidden dimension: {hidden_dim}, Expansion: {expansion}")
    print(f"SwiGLU parameters: {swiglu_params:,}")
    print(f"SwiGLU intermediate: {swiglu.intermediate_dim}")
    print(f"MLP parameters: {mlp_params:,}")
    print(f"MLP intermediate: {int(hidden_dim * expansion)}")
    print(f"Parameter ratio (SwiGLU/MLP): {swiglu_params/mlp_params:.3f}")
    
    # Activation function comparison
    print("\n" + "=" * 80)
    print("Activation Function Comparison")
    print("=" * 80)
    
    x = mx.random.normal((1000,))
    
    print("\nTiming different activation functions (1000 elements, 1000 iterations):")
    
    # SiLU/Swish
    start = time.time()
    for _ in range(1000):
        y = x * mx.sigmoid(x)
        mx.eval(y)
    silu_time = (time.time() - start) * 1000
    print(f"SiLU (x * sigmoid(x)): {silu_time:.3f} ms")
    
    # GELU
    start = time.time()
    for _ in range(1000):
        y = nn.gelu(x)
        mx.eval(y)
    gelu_time = (time.time() - start) * 1000
    print(f"GELU: {gelu_time:.3f} ms")
    
    # ReLU
    start = time.time()
    for _ in range(1000):
        y = nn.relu(x)
        mx.eval(y)
    relu_time = (time.time() - start) * 1000
    print(f"ReLU: {relu_time:.3f} ms")
    
    print(f"\nSiLU vs GELU ratio: {silu_time/gelu_time:.2f}x")
    print(f"SiLU vs ReLU ratio: {silu_time/relu_time:.2f}x")
    
    # Different dtypes
    print("\n" + "=" * 80)
    print("Data Type Performance")
    print("=" * 80)
    
    batch, seq_len, hidden_dim = 32, 512, 768
    swiglu = SwiGLU(hidden_dim)
    
    dtypes = [mx.float32, mx.float16]
    if hasattr(mx, 'bfloat16'):
        dtypes.append(mx.bfloat16)
    
    print("\nForward pass with different dtypes:")
    for dtype in dtypes:
        x = mx.random.normal((batch, seq_len, hidden_dim), dtype=dtype)
        ms_time = benchmark_forward_pass(f"SwiGLU ({dtype})", swiglu, x, 
                                       warmup=20, iterations=200)


if __name__ == "__main__":
    main()