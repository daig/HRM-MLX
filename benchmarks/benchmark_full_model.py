#!/usr/bin/env python3
"""
Comprehensive performance benchmarking for MLX HRM implementation.

This script benchmarks various aspects of HRM performance including:
- Forward pass throughput
- Training step throughput  
- Memory usage
- Inference latency
- Scaling with different model sizes and sequence lengths

Usage:
    python benchmarks/benchmark_full_model.py --model tiny
    python benchmarks/benchmark_full_model.py --all-models --save-results
    python benchmarks/benchmark_full_model.py --custom --batch-size 8 --seq-len 512
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import mlx.core as mx

# Add source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mlx_hrm.models.factory import create_hrm
from mlx_hrm.configs.model_presets import get_preset_config, list_presets
from mlx_hrm.training.act_loss import ACTLossHead
from mlx_hrm.modules.act import HRMConfig


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for HRM models.
    
    Features:
    - Forward pass benchmarking
    - Training step benchmarking (forward + backward)
    - Memory usage estimation
    - Scaling analysis across different configurations
    - Detailed performance profiling
    """
    
    def __init__(
        self, 
        warmup_steps: int = 10, 
        benchmark_steps: int = 100,
        enable_profiling: bool = False
    ):
        """
        Initialize benchmark suite.
        
        Args:
            warmup_steps: Number of warmup iterations
            benchmark_steps: Number of benchmark iterations
            enable_profiling: Enable detailed profiling (not yet implemented in MLX)
        """
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps
        self.enable_profiling = enable_profiling
        
    def _create_test_batch(
        self, 
        batch_size: int, 
        seq_len: int, 
        vocab_size: int,
        include_labels: bool = False
    ) -> Dict[str, mx.array]:
        """Create test batch with appropriate data distribution."""
        batch = {
            'input_ids': mx.random.randint(0, vocab_size, (batch_size, seq_len))
        }
        
        if include_labels:
            # Labels with some padding tokens (-100)
            labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            # Add some padding (10% of tokens)
            mask = mx.random.uniform(0, 1, (batch_size, seq_len)) < 0.1
            labels = mx.where(mask, -100, labels)
            batch['labels'] = labels
            
        return batch
    
    def benchmark_forward_pass(
        self,
        model,
        batch_size: int,
        seq_len: int,
        vocab_size: int = 1000
    ) -> Dict[str, float]:
        """
        Benchmark forward pass performance.
        
        Returns:
            Performance metrics including throughput and latency
        """
        print(f"  Benchmarking forward pass: batch_size={batch_size}, seq_len={seq_len}")
        
        # Create test data
        batch = self._create_test_batch(batch_size, seq_len, vocab_size)
        carry = model.initial_carry(batch_size)
        
        # Warmup
        for _ in range(self.warmup_steps):
            carry, outputs = model(carry, batch)
            mx.eval([outputs['logits'], outputs['q_halt_logits']])
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(self.benchmark_steps):
            carry, outputs = model(carry, batch)
            mx.eval([outputs['logits'], outputs['q_halt_logits']])
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        total_tokens = batch_size * seq_len * self.benchmark_steps
        throughput_tokens_per_sec = total_tokens / elapsed
        latency_per_step = elapsed / self.benchmark_steps * 1000  # ms
        
        # Estimate FLOPs (rough approximation)
        # Transformer FLOPs ≈ 6 * N * seq_len where N is parameter count
        def count_parameters(param_dict):
            total = 0
            for v in param_dict.values():
                if isinstance(v, dict):
                    total += count_parameters(v)
                elif hasattr(v, 'size'):
                    total += v.size
            return total
        
        param_count = count_parameters(model.parameters())
        flops_per_step = 6 * param_count * seq_len
        estimated_tflops = (flops_per_step * self.benchmark_steps) / elapsed / 1e12
        
        return {
            'throughput_tokens_per_sec': throughput_tokens_per_sec,
            'latency_ms_per_step': latency_per_step,
            'estimated_tflops': estimated_tflops,
            'total_time_sec': elapsed,
            'steps_per_sec': self.benchmark_steps / elapsed,
            'tokens_per_step': batch_size * seq_len,
            'param_count': param_count,
        }
    
    def benchmark_training_step(
        self,
        model,
        batch_size: int,
        seq_len: int,
        vocab_size: int = 1000,
        loss_type: str = 'stablemax'
    ) -> Dict[str, float]:
        """
        Benchmark training step (forward + backward).
        
        Returns:
            Training performance metrics
        """
        print(f"  Benchmarking training step: batch_size={batch_size}, seq_len={seq_len}")
        
        # Wrap with loss head
        loss_model = ACTLossHead(model, loss_type=loss_type)
        
        # Create test data with labels
        batch = self._create_test_batch(batch_size, seq_len, vocab_size, include_labels=True)
        
        def loss_fn(model_params):
            """Loss function for gradient computation."""
            loss_model.model.update(model_params)
            carry = loss_model.initial_carry(batch_size)
            new_carry, loss, metrics, _ = loss_model(carry, batch)
            return loss
        
        # Warmup
        for _ in range(self.warmup_steps):
            carry = loss_model.initial_carry(batch_size)
            loss, grads = mx.value_and_grad(loss_fn)(loss_model.model.parameters())
            mx.eval([loss])
        
        # Benchmark
        start_time = time.time()
        
        total_loss = 0.0
        for _ in range(self.benchmark_steps):
            carry = loss_model.initial_carry(batch_size)
            loss, grads = mx.value_and_grad(loss_fn)(loss_model.model.parameters())
            mx.eval([loss])
            total_loss += float(loss)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        total_tokens = batch_size * seq_len * self.benchmark_steps
        throughput_tokens_per_sec = total_tokens / elapsed
        latency_per_step = elapsed / self.benchmark_steps * 1000  # ms
        avg_loss = total_loss / self.benchmark_steps
        
        return {
            'throughput_tokens_per_sec': throughput_tokens_per_sec,
            'latency_ms_per_step': latency_per_step,
            'total_time_sec': elapsed,
            'steps_per_sec': self.benchmark_steps / elapsed,
            'tokens_per_step': batch_size * seq_len,
            'avg_loss': avg_loss,
        }
    
    def benchmark_memory_usage(
        self,
        model,
        batch_size: int,
        seq_len: int,
        vocab_size: int = 1000
    ) -> Dict[str, float]:
        """
        Estimate memory usage for model and activations.
        
        Note: MLX doesn't have the same memory profiling as PyTorch,
        so this provides rough estimates based on model architecture.
        """
        print(f"  Estimating memory usage: batch_size={batch_size}, seq_len={seq_len}")
        
        config = model.config
        
        # Parameter memory
        def count_parameters(param_dict):
            total = 0
            for v in param_dict.values():
                if isinstance(v, dict):
                    total += count_parameters(v)
                elif hasattr(v, 'size'):
                    total += v.size
            return total
        
        param_count = count_parameters(model.parameters())
        param_memory_mb = param_count * 4 / 1024 / 1024  # Assuming float32
        
        # Activation memory (rough estimate)
        hidden_size = config.hidden_size
        num_layers = config.H_layers + config.L_layers
        
        # Attention activations: Q, K, V, output
        attention_memory = batch_size * seq_len * hidden_size * 4 * num_layers
        
        # MLP activations (with expansion factor)
        mlp_intermediate_size = int(hidden_size * config.expansion * 2 / 3)
        mlp_memory = batch_size * seq_len * mlp_intermediate_size * 2 * num_layers
        
        # Embeddings and output
        embedding_memory = batch_size * seq_len * hidden_size * 2  # input + output
        
        # ACT carry states (H and L level)
        carry_memory = batch_size * seq_len * hidden_size * 2
        
        total_activation_memory = attention_memory + mlp_memory + embedding_memory + carry_memory
        activation_memory_mb = total_activation_memory * 4 / 1024 / 1024  # float32
        
        # Total memory estimate
        total_memory_mb = param_memory_mb + activation_memory_mb
        
        # Memory efficiency metrics
        memory_per_token = total_memory_mb / (batch_size * seq_len)
        
        return {
            'param_memory_mb': param_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'total_memory_mb': total_memory_mb,
            'memory_per_token_mb': memory_per_token,
            'param_count': param_count,
            'model_size_mb': param_memory_mb,  # Model file size estimate
        }
    
    def benchmark_inference_latency(
        self,
        model,
        seq_len: int,
        vocab_size: int = 1000,
        num_trials: int = 50
    ) -> Dict[str, float]:
        """
        Benchmark single-sequence inference latency.
        
        This measures the latency for processing a single sequence,
        which is important for real-time applications.
        """
        print(f"  Benchmarking inference latency: seq_len={seq_len}")
        
        # Single sequence
        batch_size = 1
        batch = self._create_test_batch(batch_size, seq_len, vocab_size)
        
        # Measure multiple trials
        latencies = []
        
        for trial in range(num_trials):
            carry = model.initial_carry(batch_size)
            
            start_time = time.time()
            carry, outputs = model(carry, batch)
            mx.eval([outputs['logits']])
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Statistics
        latencies = np.array(latencies)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'seq_len': seq_len,
            'num_trials': num_trials,
        }
    
    def benchmark_scaling(
        self,
        model_preset: str,
        batch_sizes: List[int] = None,
        seq_lens: List[int] = None
    ) -> Dict[str, Dict]:
        """
        Benchmark scaling behavior across different batch sizes and sequence lengths.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        if seq_lens is None:
            seq_lens = [64, 128, 256, 512]
        
        print(f"\nBenchmarking scaling for {model_preset} model:")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Sequence lengths: {seq_lens}")
        
        scaling_results = {
            'model_preset': model_preset,
            'batch_scaling': {},
            'sequence_scaling': {},
        }
        
        config = get_preset_config(model_preset)
        model = create_hrm(config)
        
        # Batch size scaling (fixed seq_len=256)
        print("\n  Testing batch size scaling...")
        for batch_size in batch_sizes:
            print(f"    Batch size {batch_size}...")
            try:
                forward_metrics = self.benchmark_forward_pass(model, batch_size, 256)
                training_metrics = self.benchmark_training_step(model, batch_size, 256)
                memory_metrics = self.benchmark_memory_usage(model, batch_size, 256)
                
                scaling_results['batch_scaling'][batch_size] = {
                    'forward': forward_metrics,
                    'training': training_metrics,
                    'memory': memory_metrics,
                }
            except Exception as e:
                print(f"      Failed: {e}")
                scaling_results['batch_scaling'][batch_size] = {'error': str(e)}
        
        # Sequence length scaling (fixed batch_size=4)
        print("\n  Testing sequence length scaling...")
        for seq_len in seq_lens:
            print(f"    Sequence length {seq_len}...")
            try:
                forward_metrics = self.benchmark_forward_pass(model, 4, seq_len)
                training_metrics = self.benchmark_training_step(model, 4, seq_len)
                memory_metrics = self.benchmark_memory_usage(model, 4, seq_len)
                inference_metrics = self.benchmark_inference_latency(model, seq_len)
                
                scaling_results['sequence_scaling'][seq_len] = {
                    'forward': forward_metrics,
                    'training': training_metrics,
                    'memory': memory_metrics,
                    'inference': inference_metrics,
                }
            except Exception as e:
                print(f"      Failed: {e}")
                scaling_results['sequence_scaling'][seq_len] = {'error': str(e)}
        
        return scaling_results
    
    def benchmark_model_preset(
        self,
        preset: str,
        batch_size: int = 8,
        seq_len: int = 256
    ) -> Dict[str, any]:
        """
        Comprehensive benchmark for a single model preset.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking {preset} model:")
        print(f"{'='*60}")
        
        try:
            config = get_preset_config(preset)
            model = create_hrm(config)
            
            print(f"Model configuration:")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Heads: {config.num_heads}")
            print(f"  H-layers: {config.H_layers}, L-layers: {config.L_layers}")
            print(f"  H-cycles: {config.H_cycles}, L-cycles: {config.L_cycles}")
            
            # Run all benchmarks
            results = {
                'model_preset': preset,
                'config': config._asdict(),
                'benchmark_config': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'warmup_steps': self.warmup_steps,
                    'benchmark_steps': self.benchmark_steps,
                },
            }
            
            # Forward pass
            print(f"\nForward pass benchmark:")
            forward_results = self.benchmark_forward_pass(model, batch_size, seq_len)
            results['forward'] = forward_results
            
            print(f"  Throughput: {forward_results['throughput_tokens_per_sec']:.1f} tokens/sec")
            print(f"  Latency: {forward_results['latency_ms_per_step']:.2f} ms/step")
            print(f"  Estimated TFLOPS: {forward_results['estimated_tflops']:.3f}")
            
            # Training step
            print(f"\nTraining step benchmark:")
            training_results = self.benchmark_training_step(model, batch_size, seq_len)
            results['training'] = training_results
            
            print(f"  Throughput: {training_results['throughput_tokens_per_sec']:.1f} tokens/sec")
            print(f"  Latency: {training_results['latency_ms_per_step']:.2f} ms/step")
            print(f"  Average loss: {training_results['avg_loss']:.4f}")
            
            # Memory usage
            print(f"\nMemory usage estimate:")
            memory_results = self.benchmark_memory_usage(model, batch_size, seq_len)
            results['memory'] = memory_results
            
            print(f"  Model size: {memory_results['model_size_mb']:.1f} MB")
            print(f"  Total memory: {memory_results['total_memory_mb']:.1f} MB")
            print(f"  Memory per token: {memory_results['memory_per_token_mb']:.3f} MB")
            
            # Inference latency
            print(f"\nInference latency (single sequence):")
            inference_results = self.benchmark_inference_latency(model, seq_len)
            results['inference'] = inference_results
            
            print(f"  Mean latency: {inference_results['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {inference_results['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency: {inference_results['p99_latency_ms']:.2f} ms")
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed for {preset}: {e}")
            return {
                'model_preset': preset,
                'error': str(e),
                'benchmark_config': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                }
            }


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on all model presets."""
    benchmark = PerformanceBenchmark(warmup_steps=5, benchmark_steps=50)
    
    presets = list_presets()
    print(f"Running comprehensive benchmarks on {len(presets)} model presets...")
    print(f"Available presets: {presets}")
    
    all_results = []
    
    for preset in presets:
        try:
            results = benchmark.benchmark_model_preset(preset)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Failed to benchmark {preset}: {e}")
            all_results.append({
                'model_preset': preset,
                'error': str(e)
            })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark MLX HRM performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a specific model
  python benchmark_full_model.py --model tiny
  
  # Benchmark all models
  python benchmark_full_model.py --all-models
  
  # Custom configuration
  python benchmark_full_model.py --custom --batch-size 4 --seq-len 512
  
  # Scaling analysis
  python benchmark_full_model.py --scaling tiny
  
  # Save results to file
  python benchmark_full_model.py --all-models --save-results results.json
        """
    )
    
    parser.add_argument(
        '--model',
        choices=list_presets(),
        help='Model preset to benchmark'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Benchmark all available model presets'
    )
    
    parser.add_argument(
        '--scaling',
        metavar='PRESET',
        help='Run scaling analysis for specified model preset'
    )
    
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Use custom batch size and sequence length'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for benchmarking (default: 8)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=256,
        help='Sequence length for benchmarking (default: 256)'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=10,
        help='Number of warmup steps (default: 10)'
    )
    
    parser.add_argument(
        '--benchmark-steps',
        type=int,
        default=100,
        help='Number of benchmark steps (default: 100)'
    )
    
    parser.add_argument(
        '--save-results',
        metavar='FILE',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.model or args.all_models or args.scaling or args.custom):
        parser.error('Must specify --model, --all-models, --scaling, or --custom')
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps
    )
    
    results = []
    
    if args.all_models:
        print("Running comprehensive benchmark on all models...")
        results = run_comprehensive_benchmark()
        
    elif args.model:
        print(f"Benchmarking {args.model} model...")
        result = benchmark.benchmark_model_preset(
            args.model, 
            args.batch_size, 
            args.seq_len
        )
        results = [result]
        
    elif args.scaling:
        print(f"Running scaling analysis for {args.scaling} model...")
        scaling_result = benchmark.benchmark_scaling(args.scaling)
        results = [scaling_result]
        
    elif args.custom:
        # Use tiny model for custom benchmark
        print(f"Running custom benchmark (batch_size={args.batch_size}, seq_len={args.seq_len})...")
        result = benchmark.benchmark_model_preset(
            'tiny', 
            args.batch_size, 
            args.seq_len
        )
        results = [result]
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.save_results}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['model_preset']}: {result['error']}")
        else:
            preset = result['model_preset']
            if 'forward' in result:
                forward = result['forward']
                memory = result['memory']
                print(f"✅ {preset}:")
                print(f"    Forward throughput: {forward['throughput_tokens_per_sec']:.1f} tokens/sec")
                print(f"    Model size: {memory['model_size_mb']:.1f} MB")
                print(f"    Total memory: {memory['total_memory_mb']:.1f} MB")
            elif 'batch_scaling' in result:
                print(f"✅ {preset} scaling: {len(result['batch_scaling'])} batch sizes, {len(result['sequence_scaling'])} sequence lengths")
    
    print(f"\n🎉 Benchmark completed!")


if __name__ == '__main__':
    main()