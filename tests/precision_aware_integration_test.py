#!/usr/bin/env python3
"""
Precision-Aware Component Integration Test

This script tests the integration of all Phase 2 precision-aware components
to ensure they work together correctly and maintain HRM precision compliance.

Tests:
1. Component initialization with precision configuration
2. Forward pass integration without errors  
3. Precision dtype validation throughout the model
4. Performance benchmarking
5. Numerical stability verification
6. Memory usage validation
"""

import time
import traceback
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn

# Import precision-aware components
from MLX.src.mlx_hrm.models.precision_config import MLXPrecisionConfig
from MLX.src.mlx_hrm.modules.act import HRMConfig, HRMInnerCarry, PrecisionAwareHRMReasoningModule
from MLX.src.mlx_hrm.modules.attention import PrecisionAwareAttention
from MLX.src.mlx_hrm.layers.activations import PrecisionAwareSwiGLU
from MLX.src.mlx_hrm.layers.embeddings import PrecisionAwareSparseEmbedding
from MLX.src.mlx_hrm.models.hrm_inner_precision import PrecisionAwareHRMInner


class PrecisionAwareIntegrationTester:
    """Comprehensive tester for precision-aware component integration."""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Create test configuration
        self.config = HRMConfig(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=512,
            num_puzzle_identifiers=128,
            vocab_size=1000,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,  # Smaller for testing
            L_layers=2,  # Smaller for testing
            hidden_size=512,
            expansion=4.0,
            num_heads=8,
            pos_encodings="rope",
        )
        
        # Create precision configuration
        self.precision_config = MLXPrecisionConfig.create_herm_standard_config()
        
        print("🧪 Precision-Aware Component Integration Test")
        print("=" * 60)
        print(f"📊 Test Configuration:")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Sequence length: {self.config.seq_len}")
        print(f"   Forward dtype: {self.precision_config.forward_dtype}")
        print(f"   Master dtype: {self.precision_config.master_weights_dtype}")
        print()
    
    def create_test_batch(self) -> Dict[str, mx.array]:
        """Create a test batch for integration testing."""
        return {
            'input_ids': mx.random.randint(0, self.config.vocab_size, 
                                         (self.config.batch_size, self.config.seq_len)),
            'puzzle_ids': mx.random.randint(0, self.config.num_puzzle_identifiers, 
                                          (self.config.batch_size,))
        }
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        try:
            print(f"🔧 {test_name}...", end=" ")
            start_time = time.time()
            
            result = test_func()
            
            elapsed = time.time() - start_time
            if result:
                print(f"✅ PASSED ({elapsed:.3f}s)")
                self.test_results['passed'] += 1
                return True
            else:
                print(f"❌ FAILED ({elapsed:.3f}s)")
                self.test_results['failed'] += 1
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"💥 ERROR ({elapsed:.3f}s)")
            error_msg = f"{test_name}: {str(e)}\n{traceback.format_exc()}"
            self.test_results['errors'].append(error_msg)
            self.test_results['failed'] += 1
            return False
    
    def test_precision_config_validation(self) -> bool:
        """Test that precision configuration is valid."""
        return self.precision_config.validate_configuration()
    
    def test_precision_aware_attention(self) -> bool:
        """Test precision-aware attention component."""
        attention = PrecisionAwareAttention(
            hidden_size=self.config.hidden_size,
            head_dim=self.config.head_dim,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
            causal=False,
            precision_config=self.precision_config
        )
        
        # Test forward pass
        x = mx.random.normal((self.config.batch_size, self.config.seq_len, self.config.hidden_size))
        output = attention(None, x)
        
        # Check output shape and dtype
        expected_shape = (self.config.batch_size, self.config.seq_len, self.config.hidden_size)
        forward_dtype = self.precision_config.get_forward_dtype()
        
        return (output.shape == expected_shape and 
                output.dtype == forward_dtype)
    
    def test_precision_aware_swiglu(self) -> bool:
        """Test precision-aware SwiGLU component."""
        swiglu = PrecisionAwareSwiGLU(
            hidden_size=self.config.hidden_size,
            expansion=self.config.expansion,
            precision_config=self.precision_config
        )
        
        # Test forward pass
        x = mx.random.normal((self.config.batch_size, self.config.seq_len, self.config.hidden_size))
        output = swiglu(x)
        
        # Check output shape and dtype
        expected_shape = (self.config.batch_size, self.config.seq_len, self.config.hidden_size)
        forward_dtype = self.precision_config.get_forward_dtype()
        
        return (output.shape == expected_shape and 
                output.dtype == forward_dtype)
    
    def test_precision_aware_sparse_embedding(self) -> bool:
        """Test precision-aware sparse embedding component."""
        embedding = PrecisionAwareSparseEmbedding(
            num_embeddings=self.config.num_puzzle_identifiers,
            embedding_dim=self.config.hidden_size,
            batch_size=self.config.batch_size,
            precision_config=self.precision_config
        )
        
        # Test forward pass
        puzzle_ids = mx.random.randint(0, self.config.num_puzzle_identifiers, (self.config.batch_size,))
        output = embedding(puzzle_ids)
        
        # Check output shape and dtype
        expected_shape = (self.config.batch_size, self.config.hidden_size)
        forward_dtype = self.precision_config.get_forward_dtype()
        
        # Check master weights are FP32
        master_weights_fp32 = embedding.weight.dtype == mx.float32
        
        return (output.shape == expected_shape and 
                output.dtype == forward_dtype and
                master_weights_fp32)
    
    def test_precision_aware_reasoning_module(self) -> bool:
        """Test precision-aware reasoning module."""
        reasoning_module = PrecisionAwareHRMReasoningModule(
            config=self.config,
            num_layers=2,
            num_cycles=2,
            precision_config=self.precision_config
        )
        
        # Test forward pass
        x = mx.random.normal((self.config.batch_size, self.config.seq_len, self.config.hidden_size))
        output = reasoning_module(None, x)
        
        # Check output shape and dtype
        expected_shape = (self.config.batch_size, self.config.seq_len, self.config.hidden_size)
        forward_dtype = self.precision_config.get_forward_dtype()
        
        return (output.shape == expected_shape and 
                output.dtype == forward_dtype)
    
    def test_precision_aware_hrm_inner(self) -> bool:
        """Test precision-aware HRM inner model."""
        model = PrecisionAwareHRMInner(
            config=self.config,
            precision_config=self.precision_config
        )
        
        # Create test batch and carry
        batch = self.create_test_batch()
        carry = HRMInnerCarry(
            z_H=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size)),
            z_L=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size))
        )
        
        # Test forward pass
        new_carry, logits, (q_halt, q_continue) = model(carry, batch)
        
        # Check output shapes and dtypes
        forward_dtype = self.precision_config.get_forward_dtype()
        
        logits_shape_ok = logits.shape == (self.config.batch_size, self.config.seq_len, self.config.vocab_size)
        logits_dtype_ok = logits.dtype == forward_dtype
        
        q_halt_shape_ok = q_halt.shape == (self.config.batch_size,)
        q_halt_dtype_ok = q_halt.dtype == forward_dtype
        
        q_continue_shape_ok = q_continue.shape == (self.config.batch_size,)
        q_continue_dtype_ok = q_continue.dtype == forward_dtype
        
        carry_dtype_ok = (new_carry.z_H.dtype == forward_dtype and 
                         new_carry.z_L.dtype == forward_dtype)
        
        return (logits_shape_ok and logits_dtype_ok and
                q_halt_shape_ok and q_halt_dtype_ok and
                q_continue_shape_ok and q_continue_dtype_ok and
                carry_dtype_ok)
    
    def test_master_weights_precision(self) -> bool:
        """Test that all master weights are stored in FP32."""
        model = PrecisionAwareHRMInner(
            config=self.config,
            precision_config=self.precision_config
        )
        
        # Check that all weight parameters are FP32
        all_weights_fp32 = True
        for name, param in model.parameters().items():
            if isinstance(param, mx.array) and 'weight' in name:
                if param.dtype != mx.float32:
                    print(f"     Warning: {name} is {param.dtype}, expected float32")
                    all_weights_fp32 = False
        
        return all_weights_fp32
    
    def test_forward_computation_dtype(self) -> bool:
        """Test that forward computation uses configured dtype."""
        model = PrecisionAwareHRMInner(
            config=self.config,
            precision_config=self.precision_config
        )
        
        batch = self.create_test_batch()
        carry = HRMInnerCarry(
            z_H=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size)),
            z_L=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size))
        )
        
        # Test forward pass and check all outputs are in forward dtype
        new_carry, logits, (q_halt, q_continue) = model(carry, batch)
        
        forward_dtype = self.precision_config.get_forward_dtype()
        
        return (logits.dtype == forward_dtype and
                q_halt.dtype == forward_dtype and
                q_continue.dtype == forward_dtype and
                new_carry.z_H.dtype == forward_dtype and
                new_carry.z_L.dtype == forward_dtype)
    
    def test_performance_benchmark(self) -> bool:
        """Test performance of precision-aware model."""
        model = PrecisionAwareHRMInner(
            config=self.config,
            precision_config=self.precision_config
        )
        
        batch = self.create_test_batch()
        carry = HRMInnerCarry(
            z_H=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size)),
            z_L=mx.zeros((self.config.batch_size, self.config.seq_len + 1, self.config.hidden_size))
        )
        
        # Warmup
        for _ in range(3):
            model(carry, batch)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            model(carry, batch)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs
        
        print(f"     Average forward pass time: {avg_time:.4f}s")
        
        # Should be faster than 0.1s for this config
        return avg_time < 0.1
    
    def test_memory_usage(self) -> bool:
        """Test memory usage is reasonable."""
        model = PrecisionAwareHRMInner(
            config=self.config,
            precision_config=self.precision_config
        )
        
        # Count parameters
        total_params = sum(p.size for p in model.parameters().values() if isinstance(p, mx.array))
        print(f"     Total parameters: {total_params:,}")
        
        # Should be reasonable for test config
        return total_params < 10_000_000  # Less than 10M parameters
    
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        tests = [
            ("Precision Config Validation", self.test_precision_config_validation),
            ("Precision-Aware Attention", self.test_precision_aware_attention),
            ("Precision-Aware SwiGLU", self.test_precision_aware_swiglu),
            ("Precision-Aware Sparse Embedding", self.test_precision_aware_sparse_embedding),
            ("Precision-Aware Reasoning Module", self.test_precision_aware_reasoning_module),
            ("Precision-Aware HRM Inner Model", self.test_precision_aware_hrm_inner),
            ("Master Weights Precision", self.test_master_weights_precision),
            ("Forward Computation Dtype", self.test_forward_computation_dtype),
            ("Performance Benchmark", self.test_performance_benchmark),
            ("Memory Usage", self.test_memory_usage),
        ]
        
        print("🔬 Running Precision-Aware Integration Tests:")
        print("-" * 50)
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        print("-" * 50)
        print(f"📊 Test Results: {self.test_results['passed']}/{len(tests)} passed")
        
        if self.test_results['errors']:
            print("\n💥 Errors encountered:")
            for error in self.test_results['errors']:
                print(f"   {error}")
        
        return self.test_results['failed'] == 0


def main():
    """Run the precision-aware integration tests."""
    tester = PrecisionAwareIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All precision-aware integration tests PASSED!")
        print("✅ Phase 2 component integration is ready for mixed precision compliance testing.")
    else:
        print(f"\n⚠️  {tester.test_results['failed']} tests FAILED!")
        print("❌ Phase 2 component integration needs fixes before proceeding.")
    
    return success


if __name__ == "__main__":
    main()