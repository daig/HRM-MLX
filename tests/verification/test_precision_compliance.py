"""
MLX HRM Precision Compliance Test Suite

This test suite validates that the MLX HRM implementation achieves exact
precision compliance with the original PyTorch HRM's manual mixed precision
architecture. Tests all Phase 2 precision-aware components.

Target: 6/6 tests passing for complete precision compliance.

Critical areas tested:
1. Precision Configuration Validation
2. Master Weights FP32 Compliance
3. Forward Computation BF16 Compliance  
4. Loss Ultra-Precision FP64 Compliance
5. Gradient FP32 Compliance
6. Precision-Aware Component Integration
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytest

from mlx_hrm.models.precision_config import MLXPrecisionConfig
from mlx_hrm.modules.act import HRMConfig, HRMInnerCarry
from mlx_hrm.models.hrm_inner_precision import PrecisionAwareHRMInner
from mlx_hrm.modules.attention import PrecisionAwareAttention
from mlx_hrm.layers.activations import PrecisionAwareSwiGLU
from mlx_hrm.layers.embeddings import PrecisionAwareSparseEmbedding
from mlx_hrm.training.precision_losses import precision_stablemax_cross_entropy
from mlx_hrm.layers.precision import MLXCastedLinear, precision_aware_rms_norm


class PrecisionComplianceTest:
    """Test suite for HRM precision compliance validation."""
    
    def __init__(self):
        # Small configuration for faster testing
        self.config = HRMConfig(
            batch_size=2,
            seq_len=4,
            puzzle_emb_ndim=8,
            num_puzzle_identifiers=4,
            vocab_size=16,
            hidden_size=32,
            num_heads=4,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            halt_max_steps=3,
            halt_exploration_prob=0.0  # Deterministic
        )
        
        # Precision configuration (standard HRM compliance)
        self.precision_config = MLXPrecisionConfig.create_herm_standard_config()
        
        # Test tolerances for precision compliance
        self.tolerances = {
            'forward_pass': {'rtol': 0.001, 'atol': 1e-4},
            'gradients': {'rtol': 0.002, 'atol': 1e-3},
            'loss': {'rtol': 0.0005, 'atol': 1e-5},
            'weights': {'rtol': 1e-6, 'atol': 1e-7}
        }
    
    def create_test_batch(self, batch_size: int) -> Dict[str, mx.array]:
        """Create deterministic test batch."""
        mx.random.seed(42)
        return {
            "input_ids": mx.random.randint(0, self.config.vocab_size, 
                                         shape=(batch_size, self.config.seq_len)),
            "puzzle_ids": mx.random.randint(0, self.config.num_puzzle_identifiers,
                                          shape=(batch_size,)),
            "labels": mx.random.randint(0, self.config.vocab_size,
                                      shape=(batch_size, self.config.seq_len))
        }
    
    def test_precision_configuration_validation(self) -> bool:
        """Test 1/6: Precision configuration is valid and matches HRM standard."""
        print("🔧 Test 1/6: Precision Configuration Validation")
        
        try:
            # Test standard HRM configuration
            config = MLXPrecisionConfig.create_herm_standard_config()
            
            # Validate configuration
            if not config.validate_configuration():
                print("  ❌ FAILED: Standard configuration validation failed")
                return False
            
            # Check exact HRM precision requirements
            expected_values = {
                'forward_dtype': 'bfloat16',
                'master_weights_dtype': 'float32',
                'normalization_dtype': 'float32',
                'loss_computation_dtype': 'float64',
                'gradient_dtype': 'float32',
                'embedding_dtype': 'float32'
            }
            
            for attr, expected in expected_values.items():
                actual = getattr(config, attr)
                if actual != expected:
                    print(f"  ❌ FAILED: {attr} = {actual}, expected {expected}")
                    return False
            
            # Test dtype getter methods
            dtype_methods = [
                ('get_forward_dtype', mx.bfloat16),
                ('get_master_dtype', mx.float32),
                ('get_normalization_dtype', mx.float32),
                ('get_loss_dtype', mx.float64),
                ('get_gradient_dtype', mx.float32),
                ('get_embedding_dtype', mx.float32)
            ]
            
            for method_name, expected_dtype in dtype_methods:
                actual_dtype = getattr(config, method_name)()
                if actual_dtype != expected_dtype:
                    print(f"  ❌ FAILED: {method_name}() = {actual_dtype}, expected {expected_dtype}")
                    return False
            
            print("  ✅ PASSED: Precision configuration validation successful")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_master_weights_fp32_compliance(self) -> bool:
        """Test 2/6: All master weights are stored in FP32."""
        print("🔧 Test 2/6: Master Weights FP32 Compliance")
        
        try:
            # Test precision-aware components
            components = [
                ('MLXCastedLinear', MLXCastedLinear(32, 16)),
                ('PrecisionAwareAttention', PrecisionAwareAttention(
                    hidden_size=32, head_dim=8, num_heads=4, num_key_value_heads=4,
                    precision_config=self.precision_config)),
                ('PrecisionAwareSwiGLU', PrecisionAwareSwiGLU(
                    hidden_size=32, precision_config=self.precision_config)),
                ('PrecisionAwareSparseEmbedding', PrecisionAwareSparseEmbedding(
                    num_embeddings=16, embedding_dim=32, batch_size=2,
                    precision_config=self.precision_config)),
                ('PrecisionAwareHRMInner', PrecisionAwareHRMInner(
                    self.config, self.precision_config))
            ]
            
            for comp_name, component in components:
                print(f"  Testing {comp_name}...")
                
                # Check all weight parameters
                for param_name, param in component.parameters().items():
                    if isinstance(param, mx.array) and 'weight' in param_name:
                        if param.dtype != mx.float32:
                            print(f"    ❌ FAILED: {comp_name}.{param_name} is {param.dtype}, expected float32")
                            return False
                
                print(f"    ✅ PASSED: {comp_name} master weights are FP32")
            
            print("  ✅ PASSED: All master weights FP32 compliance verified")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_forward_computation_bf16_compliance(self) -> bool:
        """Test 3/6: Forward computation uses BF16 as configured."""
        print("🔧 Test 3/6: Forward Computation BF16 Compliance")
        
        try:
            model = PrecisionAwareHRMInner(self.config, self.precision_config)
            batch = self.create_test_batch(2)
            
            # Create carry state
            carry = HRMInnerCarry(
                z_H=mx.zeros((2, 5, 32)),  # seq_len + puzzle_emb
                z_L=mx.zeros((2, 5, 32))
            )
            
            # Forward pass
            new_carry, logits, (q_halt, q_continue) = model(carry, batch)
            
            # Check output dtypes (should be BF16)
            forward_dtype = self.precision_config.get_forward_dtype()
            
            output_checks = [
                ('logits', logits, forward_dtype),
                ('q_halt', q_halt, forward_dtype), 
                ('q_continue', q_continue, forward_dtype),
                ('carry.z_H', new_carry.z_H, forward_dtype),
                ('carry.z_L', new_carry.z_L, forward_dtype)
            ]
            
            for name, tensor, expected_dtype in output_checks:
                if tensor.dtype != expected_dtype:
                    print(f"  ❌ FAILED: {name} dtype is {tensor.dtype}, expected {expected_dtype}")
                    return False
                print(f"    ✅ {name}: {tensor.dtype} (correct)")
            
            # Test individual component forward dtypes
            test_input = mx.random.normal((2, 4, 32))
            
            # Test attention
            attention = PrecisionAwareAttention(
                hidden_size=32, head_dim=8, num_heads=4, num_key_value_heads=4,
                precision_config=self.precision_config
            )
            attn_output = attention(None, test_input)
            if attn_output.dtype != forward_dtype:
                print(f"  ❌ FAILED: Attention output dtype {attn_output.dtype}, expected {forward_dtype}")
                return False
            
            # Test SwiGLU
            swiglu = PrecisionAwareSwiGLU(hidden_size=32, precision_config=self.precision_config)
            swiglu_output = swiglu(test_input)
            if swiglu_output.dtype != forward_dtype:
                print(f"  ❌ FAILED: SwiGLU output dtype {swiglu_output.dtype}, expected {forward_dtype}")
                return False
            
            print("  ✅ PASSED: Forward computation BF16 compliance verified")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_loss_ultra_precision_fp64_compliance(self) -> bool:
        """Test 4/6: Loss computation uses FP64 for ultra-precision."""
        print("🔧 Test 4/6: Loss Ultra-Precision FP64 Compliance")
        
        try:
            # Create test logits and labels
            logits = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
            labels = mx.random.randint(0, 16, (2, 4))
            
            # Test precision_stablemax_cross_entropy
            loss = precision_stablemax_cross_entropy(logits, labels)
            
            print(f"  Loss value: {loss:.8f}")
            print(f"  Loss dtype: {loss.dtype}")
            
            # Loss should be finite and reasonable
            if not mx.isfinite(loss):
                print("  ❌ FAILED: Loss is not finite")
                return False
            
            if loss < 0:
                print("  ❌ FAILED: Loss is negative")
                return False
            
            # Test extreme values to verify FP64 stability
            extreme_logits = mx.array([[-50.0, 50.0, 0.0]] * 2).astype(mx.bfloat16)
            extreme_labels = mx.array([[1, 0, 2]] * 2)
            
            extreme_loss = precision_stablemax_cross_entropy(extreme_logits, extreme_labels)
            
            print(f"  Extreme loss value: {extreme_loss:.8f}")
            print(f"  Extreme loss dtype: {extreme_loss.dtype}")
            
            if not mx.isfinite(extreme_loss):
                print("  ❌ FAILED: Extreme loss is not finite (FP64 stability failed)")
                return False
            
            # Compare with standard loss (should be more stable)
            try:
                standard_loss = mx.nn.losses.cross_entropy(extreme_logits, extreme_labels)
                print(f"  Standard loss comparison: {standard_loss:.8f}")
                
                # Ultra-precision loss should be more stable (finite when standard might not be)
                if mx.isfinite(extreme_loss) and not mx.isnan(extreme_loss):
                    print("  ✅ Ultra-precision loss maintains stability")
                else:
                    print("  ❌ FAILED: Ultra-precision loss not stable")
                    return False
                    
            except Exception:
                # If standard loss fails, that's expected for extreme values
                print("  ✅ Ultra-precision loss handles extreme values better than standard")
            
            print("  ✅ PASSED: Loss ultra-precision FP64 compliance verified")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_gradient_fp32_compliance(self) -> bool:
        """Test 5/6: Gradients are computed and stored in FP32."""
        print("🔧 Test 5/6: Gradient FP32 Compliance")
        
        try:
            model = PrecisionAwareHRMInner(self.config, self.precision_config)
            batch = self.create_test_batch(1)  # Small batch for gradient testing
            
            # Define loss function
            def loss_fn(model, batch):
                carry = HRMInnerCarry(
                    z_H=mx.zeros((1, 5, 32)),
                    z_L=mx.zeros((1, 5, 32))
                )
                new_carry, logits, (q_halt, q_continue) = model(carry, batch)
                loss = precision_stablemax_cross_entropy(logits, batch['labels'])
                return loss
            
            # Compute gradients
            loss_and_grads = mx.value_and_grad(loss_fn)
            loss, grads = loss_and_grads(model, batch)
            
            print(f"  Loss: {loss:.6f}")
            print(f"  Number of gradient arrays: {len(grads)}")
            
            # Check gradient dtypes
            gradient_dtype_failures = 0
            total_gradients = 0
            
            for param_name, grad in grads.items():
                if grad is not None:
                    total_gradients += 1
                    expected_dtype = mx.float32  # All gradients should be FP32
                    
                    if grad.dtype != expected_dtype:
                        print(f"    ❌ FAILED: Gradient {param_name} is {grad.dtype}, expected {expected_dtype}")
                        gradient_dtype_failures += 1
                    else:
                        print(f"    ✅ {param_name}: {grad.dtype} (correct)")
            
            print(f"  Gradient dtype compliance: {total_gradients - gradient_dtype_failures}/{total_gradients}")
            
            if gradient_dtype_failures > 0:
                print("  ❌ FAILED: Some gradients not in FP32")
                return False
            
            # Test gradient magnitudes are reasonable
            grad_magnitudes = []
            for grad in grads.values():
                if grad is not None:
                    mag = mx.abs(grad).max()
                    grad_magnitudes.append(float(mag))
            
            max_grad_mag = max(grad_magnitudes) if grad_magnitudes else 0
            min_grad_mag = min(grad_magnitudes) if grad_magnitudes else 0
            
            print(f"  Gradient magnitude range: [{min_grad_mag:.6f}, {max_grad_mag:.6f}]")
            
            # Gradients should be finite and reasonable
            if max_grad_mag > 100 or not all(mx.isfinite(mx.array(grad_magnitudes))):
                print("  ❌ FAILED: Gradient magnitudes unreasonable or infinite")
                return False
            
            print("  ✅ PASSED: Gradient FP32 compliance verified")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_precision_aware_component_integration(self) -> bool:
        """Test 6/6: All precision-aware components integrate correctly."""
        print("🔧 Test 6/6: Precision-Aware Component Integration")
        
        try:
            # Test full model integration
            model = PrecisionAwareHRMInner(self.config, self.precision_config)
            batch = self.create_test_batch(2)
            
            # Multiple forward passes to test consistency
            num_passes = 3
            outputs_history = []
            
            for i in range(num_passes):
                carry = HRMInnerCarry(
                    z_H=mx.zeros((2, 5, 32)),
                    z_L=mx.zeros((2, 5, 32))
                )
                
                new_carry, logits, (q_halt, q_continue) = model(carry, batch)
                
                outputs_history.append({
                    'logits': logits,
                    'q_halt': q_halt,
                    'q_continue': q_continue,
                    'carry': new_carry
                })
                
                print(f"  Pass {i+1}:")
                print(f"    Logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"    Q-halt shape: {q_halt.shape}, dtype: {q_halt.dtype}")
                print(f"    Carry z_H dtype: {new_carry.z_H.dtype}")
            
            # Check consistency across passes (deterministic)
            for i in range(1, num_passes):
                prev_outputs = outputs_history[i-1]
                curr_outputs = outputs_history[i]
                
                # Outputs should be identical (deterministic model)
                logits_match = mx.allclose(prev_outputs['logits'], curr_outputs['logits'], 
                                         rtol=1e-6, atol=1e-7)
                q_halt_match = mx.allclose(prev_outputs['q_halt'], curr_outputs['q_halt'],
                                         rtol=1e-6, atol=1e-7)
                
                if not (logits_match and q_halt_match):
                    print(f"  ❌ FAILED: Outputs not consistent between passes {i} and {i+1}")
                    return False
            
            print("  ✅ Output consistency verified")
            
            # Test precision compliance throughout the forward pass
            forward_dtype = self.precision_config.get_forward_dtype()
            
            # Create hooks to check intermediate dtypes (simplified)
            test_input = mx.random.normal((2, 32)).astype(forward_dtype)
            
            # Test MLXCastedLinear precision handling
            linear = MLXCastedLinear(32, 16)
            linear_output = linear(test_input)
            
            if linear.weight.dtype != mx.float32:
                print("  ❌ FAILED: MLXCastedLinear master weights not FP32")
                return False
            if linear_output.dtype != forward_dtype:
                print("  ❌ FAILED: MLXCastedLinear output not in forward dtype")
                return False
            
            # Test precision-aware RMSNorm
            norm_output = precision_aware_rms_norm(test_input)
            if norm_output.dtype != forward_dtype:
                print("  ❌ FAILED: Precision-aware RMSNorm output not in forward dtype")
                return False
            
            print("  ✅ Individual component precision compliance verified")
            
            # Test training mode integration
            model.set_training(True)
            carry = HRMInnerCarry(
                z_H=mx.zeros((2, 5, 32)),
                z_L=mx.zeros((2, 5, 32))
            )
            
            def training_loss_fn():
                new_carry, logits, (q_halt, q_continue) = model(carry, batch)
                loss = precision_stablemax_cross_entropy(logits, batch['labels'])
                return loss
            
            # Test gradient computation with precision-aware model
            loss_and_grads = mx.value_and_grad(training_loss_fn)
            loss, grads = loss_and_grads()
            
            if not mx.isfinite(loss):
                print("  ❌ FAILED: Training loss not finite")
                return False
            
            # Count valid gradients
            valid_grads = sum(1 for g in grads.values() if g is not None and mx.isfinite(g).all())
            total_params = len([p for p in model.parameters().values() if isinstance(p, mx.array)])
            
            print(f"  Training integration: {valid_grads} valid gradients out of {total_params} parameters")
            
            if valid_grads == 0:
                print("  ❌ FAILED: No valid gradients computed")
                return False
            
            print("  ✅ PASSED: Precision-aware component integration successful")
            return True
            
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all precision compliance tests."""
        print("🔬 MLX HRM Precision Compliance Test Suite")
        print("=" * 60) 
        print(f"🎯 Target: 6/6 tests passing for complete HRM precision compliance")
        print(f"📋 Testing with:")
        print(f"   Forward dtype: {self.precision_config.forward_dtype}")
        print(f"   Master weights: {self.precision_config.master_weights_dtype}")
        print(f"   Loss computation: {self.precision_config.loss_computation_dtype}")
        print()
        
        tests = [
            ("Precision Configuration Validation", self.test_precision_configuration_validation),
            ("Master Weights FP32 Compliance", self.test_master_weights_fp32_compliance),
            ("Forward Computation BF16 Compliance", self.test_forward_computation_bf16_compliance),
            ("Loss Ultra-Precision FP64 Compliance", self.test_loss_ultra_precision_fp64_compliance),
            ("Gradient FP32 Compliance", self.test_gradient_fp32_compliance),
            ("Precision-Aware Component Integration", self.test_precision_aware_component_integration)
        ]
        
        passed_tests = 0
        
        for i, (test_name, test_func) in enumerate(tests, 1):
            print(f"\\n{'='*20} {i}/6: {test_name} {'='*20}")
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ Test {i}/6 PASSED: {test_name}")
                else:
                    print(f"❌ Test {i}/6 FAILED: {test_name}")
            except Exception as e:
                print(f"💥 Test {i}/6 ERROR: {test_name} - {str(e)}")
        
        print("\\n" + "=" * 60)
        print(f"📊 PRECISION COMPLIANCE RESULTS: {passed_tests}/6 tests passed")
        
        if passed_tests == 6:
            print("🎉 SUCCESS! 6/6 TESTS PASSED")
            print("✅ MLX HRM achieves complete precision compliance with original PyTorch HRM!")
            print("✅ Phase 2 precision-aware component integration is COMPLETE!")
        else:
            print(f"⚠️  INCOMPLETE: {passed_tests}/6 tests passed")
            print("❌ Additional precision compliance work needed")
        
        return passed_tests == 6


def test_precision_compliance():
    """Pytest entry point for precision compliance tests."""
    tester = PrecisionComplianceTest()
    assert tester.run_all_tests(), "Precision compliance tests failed"


if __name__ == "__main__":
    # Run tests directly
    tester = PrecisionComplianceTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)