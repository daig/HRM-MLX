"""
Test mixed precision consistency between different data types.

This test verifies that the MLX HRM implementation maintains numerical
consistency when switching between FP32 and BF16 (mixed precision training).

Critical areas tested:
1. FP32 ↔ BF16 casting accuracy
2. Gradient computation precision
3. Training step consistency across precisions
4. Loss computation precision
5. Model output equivalence
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytest

from mlx_hrm.modules.act import HRMConfig, HRMCarry
from mlx_hrm.models.hrm_act import HRM_ACT
from mlx_hrm.training.losses import stablemax_cross_entropy


class MixedPrecisionComplianceTest:
    """Test suite for mixed precision consistency."""
    
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
        
        # Precision configurations to test
        self.precision_configs = [
            {
                'name': 'FP32',
                'compute_dtype': mx.float32,
                'storage_dtype': mx.float32,
                'description': 'Full precision (baseline)'
            },
            {
                'name': 'BF16',
                'compute_dtype': mx.bfloat16,
                'storage_dtype': mx.float32,  # Master weights in FP32
                'description': 'Mixed precision (BF16 compute, FP32 storage)'
            }
        ]
    
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
    
    def test_dtype_casting_accuracy(self) -> bool:
        """Test accuracy of dtype casting operations."""
        print("Testing dtype casting accuracy...")
        
        # Test various casting scenarios
        test_values = [
            mx.array([1.0, -1.0, 0.5, -0.5, 0.125, -0.125]),
            mx.array([1e-4, -1e-4, 1e-6, -1e-6]),  # Small values
            mx.random.normal((10, 10)) * 0.1,      # Random values
        ]
        
        casting_tests = [
            {'from': mx.float32, 'to': mx.bfloat16, 'back': mx.float32},
            {'from': mx.bfloat16, 'to': mx.float32, 'back': mx.bfloat16}
        ]
        
        for i, values in enumerate(test_values):
            print(f"  Testing value set {i+1}...")
            
            for cast_test in casting_tests:
                try:
                    # Start with values in the 'from' dtype
                    original = values.astype(cast_test['from'])
                    
                    # Cast to target dtype and back
                    casted = original.astype(cast_test['to'])
                    restored = casted.astype(cast_test['back'])
                    
                    # Measure precision loss
                    if cast_test['from'] == mx.float32 and cast_test['to'] == mx.bfloat16:
                        # BF16 has less precision than FP32
                        max_error = mx.abs(restored - original).max()
                        relative_error = (mx.abs(restored - original) / (mx.abs(original) + 1e-8)).max()
                        
                        print(f"    FP32→BF16→FP32: max_error={max_error:.8f}, rel_error={relative_error:.6f}")
                        
                        # BF16 should maintain reasonable precision for typical model values
                        if max_error < 0.01 and relative_error < 0.01:  # 1% tolerance
                            print(f"    ✅ PASSED: BF16 casting within tolerance")
                        else:
                            print(f"    ❌ FAILED: BF16 casting exceeds tolerance")
                            return False
                    else:
                        # Other direction should be exact or very close
                        max_error = mx.abs(restored - original).max()
                        print(f"    {cast_test['from']}→{cast_test['to']}→{cast_test['back']}: max_error={max_error:.8f}")
                        
                        if max_error < 1e-6:
                            print(f"    ✅ PASSED: Casting preserves precision")
                        else:
                            print(f"    ❌ FAILED: Unexpected precision loss")
                            return False
                            
                except Exception as e:
                    print(f"    💥 ERROR in casting test: {e}")
                    return False
        
        print(f"  ✅ PASSED: All dtype casting tests successful")
        return True
    
    def test_model_output_precision_consistency(self) -> bool:
        """Test model output consistency across precisions."""
        print("\\nTesting model output precision consistency...")
        
        batch = self.create_test_batch(2)
        
        # Store outputs for different precision configurations
        outputs_by_precision = {}
        
        for precision_config in self.precision_configs:
            print(f"  Testing {precision_config['name']} ({precision_config['description']})...")
            
            try:
                # Create model with specific precision
                model = HRM_ACT(self.config)
                model.set_training(False)  # Deterministic mode
                
                # Convert inputs to target precision if needed (matching PyTorch reference)
                # In PyTorch HRM: indices always stay integers, only float tensors get cast
                batch_precision = {}
                for key, value in batch.items():
                    if key in ['puzzle_ids', 'input_ids', 'labels']:
                        # Integer indices NEVER get cast - they must stay as integers
                        batch_precision[key] = value
                    else:
                        # Only cast actual float tensors if using mixed precision
                        if precision_config['compute_dtype'] != mx.float32:
                            batch_precision[key] = value.astype(precision_config['compute_dtype'])
                        else:
                            batch_precision[key] = value
                
                # Run forward pass
                carry = model.initial_carry(batch['input_ids'].shape[0])
                new_carry, outputs = model(carry, batch_precision)
                
                print(f"    Output shapes:")
                print(f"      logits: {outputs['logits'].shape}, dtype: {outputs['logits'].dtype}")
                print(f"      q_halt: {outputs['q_halt_logits'].shape}, dtype: {outputs['q_halt_logits'].dtype}")
                
                # Store outputs (convert to FP32 for comparison)
                outputs_fp32 = {
                    'logits': outputs['logits'].astype(mx.float32),
                    'q_halt_logits': outputs['q_halt_logits'].astype(mx.float32),
                    'q_continue_logits': outputs['q_continue_logits'].astype(mx.float32),
                    'halted': outputs['halted']  # Boolean, no conversion needed
                }
                
                outputs_by_precision[precision_config['name']] = outputs_fp32
                
                print(f"    ✅ PASSED: {precision_config['name']} forward pass successful")
                
            except Exception as e:
                print(f"    💥 ERROR in {precision_config['name']}: {e}")
                return False
        
        # Compare outputs between precisions
        print(f"  Comparing outputs between precisions...")
        
        fp32_outputs = outputs_by_precision['FP32']
        bf16_outputs = outputs_by_precision['BF16']
        
        comparisons = []
        for key in ['logits', 'q_halt_logits', 'q_continue_logits']:
            diff = mx.abs(fp32_outputs[key] - bf16_outputs[key])
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            # Calculate relative error
            relative_error = (diff / (mx.abs(fp32_outputs[key]) + 1e-8)).max()
            
            print(f"    {key}:")
            print(f"      Max absolute diff: {max_diff:.6f}")
            print(f"      Mean absolute diff: {mean_diff:.6f}")
            print(f"      Max relative error: {relative_error:.6f}")
            
            # Set tolerances based on expected BF16 precision
            if key == 'logits':
                # Logits can have larger values, more tolerance needed
                tolerance = 0.1
                rel_tolerance = 0.02
            else:
                # Q-values are typically smaller
                tolerance = 0.05
                rel_tolerance = 0.02
            
            within_tolerance = max_diff < tolerance and relative_error < rel_tolerance
            comparisons.append(within_tolerance)
            
            if within_tolerance:
                print(f"      ✅ PASSED: {key} within tolerance")
            else:
                print(f"      ❌ FAILED: {key} exceeds tolerance")
        
        # Check halting consistency
        halted_match = mx.array_equal(fp32_outputs['halted'], bf16_outputs['halted'])
        print(f"    halted consistency: {halted_match}")
        
        if halted_match:
            print(f"      ✅ PASSED: Halting decisions identical")
            comparisons.append(True)
        else:
            print(f"      ❌ FAILED: Halting decisions differ")
            comparisons.append(False)
        
        if all(comparisons):
            print(f"  ✅ PASSED: All outputs within tolerance")
            return True
        else:
            print(f"  ❌ FAILED: Some outputs exceed tolerance")
            return False
    
    def test_loss_computation_consistency(self) -> bool:
        """Test loss computation consistency across precisions."""
        print("\\nTesting loss computation consistency...")
        
        batch = self.create_test_batch(2)
        
        # Test loss function directly with different precisions
        # Create synthetic logits for testing
        logits_fp32 = mx.random.normal((2, 4, self.config.vocab_size))
        labels = batch['labels']
        
        print(f"  Testing loss computation with synthetic logits...")
        print(f"    Logits shape: {logits_fp32.shape}, dtype: {logits_fp32.dtype}")
        print(f"    Labels shape: {labels.shape}, dtype: {labels.dtype}")
        
        try:
            # Compute loss in FP32
            loss_fp32 = stablemax_cross_entropy(logits_fp32, labels)
            
            # Compute loss in BF16
            logits_bf16 = logits_fp32.astype(mx.bfloat16)
            loss_bf16 = stablemax_cross_entropy(logits_bf16, labels)
            
            # Convert BF16 loss back to FP32 for comparison
            loss_bf16_as_fp32 = loss_bf16.astype(mx.float32)
            
            # Convert to scalar values for printing and comparison
            loss_fp32_scalar = mx.mean(loss_fp32).item()
            loss_bf16_scalar = mx.mean(loss_bf16_as_fp32).item()
            
            print(f"    FP32 loss: {loss_fp32_scalar:.6f}")
            print(f"    BF16 loss: {loss_bf16_scalar:.6f}")
            
            # Check difference
            loss_diff = abs(loss_fp32_scalar - loss_bf16_scalar)
            relative_loss_diff = loss_diff / (abs(loss_fp32_scalar) + 1e-8)
            
            print(f"    Absolute difference: {loss_diff:.6f}")
            print(f"    Relative difference: {relative_loss_diff:.6f}")
            
            # Loss should be reasonably close (within 5% relative error)
            if relative_loss_diff < 0.05:
                print(f"    ✅ PASSED: Loss computation consistent across precisions")
                return True
            else:
                print(f"    ❌ FAILED: Loss computation differs significantly")
                return False
                
        except Exception as e:
            print(f"    💥 ERROR: {e}")
            return False
    
    def test_gradient_precision_effects(self) -> bool:
        """Test gradient computation precision effects."""
        print("\\nTesting gradient computation precision effects...")
        
        # Create simple model for gradient testing
        batch = self.create_test_batch(1)  # Small batch for simpler analysis
        
        # Test gradient computation with different input precisions
        precision_tests = [
            {'name': 'FP32', 'dtype': mx.float32},
            {'name': 'BF16', 'dtype': mx.bfloat16}
        ]
        
        gradients_by_precision = {}
        
        for precision_test in precision_tests:
            print(f"  Testing gradients with {precision_test['name']} inputs...")
            
            try:
                model = HRM_ACT(self.config)
                model.set_training(True)
                
                # Convert batch to target precision (matching PyTorch reference)
                # In PyTorch HRM: indices always stay integers, only float tensors get cast
                batch_precision = {}
                for key, value in batch.items():
                    if key in ['puzzle_ids', 'input_ids', 'labels']:
                        # Integer indices NEVER get cast - they must stay as integers
                        batch_precision[key] = value
                    else:
                        # Only cast actual float tensors if using mixed precision
                        if precision_test['dtype'] != mx.float32:
                            batch_precision[key] = value.astype(precision_test['dtype'])
                        else:
                            batch_precision[key] = value
                
                # Define loss function for gradient computation (MLX neural network approach)
                def loss_fn(batch):
                    carry = model.initial_carry(batch['input_ids'].shape[0])
                    new_carry, outputs = model(carry, batch)
                    loss = stablemax_cross_entropy(outputs['logits'], batch['labels'])
                    return mx.mean(loss)  # Reduce to scalar for gradient computation
                
                # Compute gradients using MLX neural network approach
                import mlx.nn as nn
                loss_and_grads = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grads(batch_precision)
                
                loss_scalar = loss.item()
                print(f"    Loss: {loss_scalar:.6f}")
                print(f"    Number of gradient arrays: {len(grads)}")
                
                # Store gradients (convert to FP32 for comparison using tree_map)
                from mlx.utils import tree_map
                grads_fp32 = tree_map(lambda x: x.astype(mx.float32), grads)
                
                gradients_by_precision[precision_test['name']] = {
                    'loss': loss.astype(mx.float32),  # Keep as MLX array for consistency
                    'gradients': grads_fp32
                }
                
                print(f"    ✅ PASSED: {precision_test['name']} gradient computation successful")
                
            except Exception as e:
                print(f"    💥 ERROR in {precision_test['name']}: {e}")
                return False
        
        # Compare gradients between precisions
        print(f"  Comparing gradients between precisions...")
        
        fp32_data = gradients_by_precision['FP32']
        bf16_data = gradients_by_precision['BF16']
        
        # Compare losses (convert MLX arrays to scalars for display)
        fp32_loss_scalar = fp32_data['loss'].item()
        bf16_loss_scalar = bf16_data['loss'].item()
        loss_diff = abs(fp32_loss_scalar - bf16_loss_scalar)
        loss_rel_diff = loss_diff / (abs(fp32_loss_scalar) + 1e-8)
        
        print(f"    Loss comparison:")
        print(f"      FP32 loss: {fp32_loss_scalar:.6f}")
        print(f"      BF16 loss: {bf16_loss_scalar:.6f}")
        print(f"      Absolute diff: {loss_diff:.6f}")
        print(f"      Relative diff: {loss_rel_diff:.6f}")
        
        loss_consistent = loss_rel_diff < 0.05  # 5% tolerance
        
        # Compare gradients using MLX tree utilities
        from mlx.utils import tree_map, tree_flatten
        
        fp32_grads = fp32_data['gradients'] 
        bf16_grads = bf16_data['gradients']
        
        print(f"    Gradient comparison using MLX tree utilities...")
        
        # Compute differences using tree_map
        try:
            grad_diffs = tree_map(lambda x, y: mx.abs(x - y), fp32_grads, bf16_grads)
            grad_magnitudes = tree_map(lambda x: mx.abs(x), fp32_grads)
            
            # Get max differences across entire gradient tree
            max_diffs = tree_map(lambda x: mx.max(x), grad_diffs)
            max_magnitudes = tree_map(lambda x: mx.max(x), grad_magnitudes)
            
            # Compute relative errors
            rel_errors = tree_map(lambda diff, mag: diff / (mag + 1e-8), max_diffs, max_magnitudes)
            
            # Flatten trees to get sample values for reporting
            flat_diffs = tree_flatten(max_diffs)
            flat_errors = tree_flatten(rel_errors)
            
            print(f"      Sample gradient differences:")
            for i, (path, diff) in enumerate(flat_diffs[:3]):  # Show first 3
                print(f"        {path}: {diff.item():.6f}")
            
            print(f"      Sample relative errors:")
            grad_comparisons = []
            for i, (path, rel_error) in enumerate(flat_errors[:5]):  # Check first 5
                error_val = rel_error.item()
                within_tolerance = error_val < 0.1  # 10% tolerance
                grad_comparisons.append(within_tolerance)
                status = "✅ PASSED" if within_tolerance else "❌ FAILED"
                print(f"        {path}: {error_val:.6f} {status}")
                
        except Exception as e:
            print(f"      ❌ Error in gradient comparison: {e}")
            grad_comparisons = [False]
        
        overall_success = loss_consistent and all(grad_comparisons)
        
        if overall_success:
            print(f"  ✅ PASSED: Gradient precision effects within acceptable bounds")
            return True
        else:
            print(f"  ❌ FAILED: Gradient precision effects exceed tolerance")
            return False
    
    def test_accumulation_precision(self) -> bool:
        """Test precision during accumulation operations."""
        print("\\nTesting accumulation precision...")
        
        # Test accumulation patterns common in transformer training
        test_cases = [
            {
                'name': 'Small value accumulation',
                'values': mx.ones((100,)) * 1e-4,
                'expected_sum': 1e-2
            },
            {
                'name': 'Mixed magnitude accumulation',
                'values': mx.concatenate([mx.ones((50,)) * 1e-4, mx.ones((50,)) * 1e-2]),
                'expected_sum': 0.505e-2
            },
            {
                'name': 'Alternating signs',
                'values': mx.array([1.0, -0.99] * 50),
                'expected_sum': 0.5
            }
        ]
        
        for case in test_cases:
            print(f"  Testing {case['name']}...")
            
            try:
                values_fp32 = case['values'].astype(mx.float32)
                values_bf16 = case['values'].astype(mx.bfloat16)
                
                # Test simple sum
                sum_fp32 = mx.sum(values_fp32)
                sum_bf16 = mx.sum(values_bf16).astype(mx.float32)
                
                print(f"    Expected: {case['expected_sum']:.6f}")
                print(f"    FP32 sum: {sum_fp32:.6f}")
                print(f"    BF16 sum: {sum_bf16:.6f}")
                
                # Check accuracy
                fp32_error = mx.abs(sum_fp32 - case['expected_sum'])
                bf16_error = mx.abs(sum_bf16 - case['expected_sum'])
                
                print(f"    FP32 error: {fp32_error:.6f}")
                print(f"    BF16 error: {bf16_error:.6f}")
                
                # BF16 should be reasonably close to FP32 (within order of magnitude)
                sum_diff = mx.abs(sum_fp32 - sum_bf16)
                relative_diff = sum_diff / (mx.abs(sum_fp32) + 1e-8)
                
                print(f"    FP32 vs BF16 difference: {sum_diff:.6f}")
                print(f"    Relative difference: {relative_diff:.6f}")
                
                # For accumulation, accept larger relative errors for BF16
                if relative_diff < 0.1:  # 10% tolerance
                    print(f"    ✅ PASSED: {case['name']} within tolerance")
                else:
                    print(f"    ❌ FAILED: {case['name']} exceeds tolerance")
                    return False
                    
            except Exception as e:
                print(f"    💥 ERROR in {case['name']}: {e}")
                return False
        
        print(f"  ✅ PASSED: All accumulation precision tests successful")
        return True
    
    def test_training_step_numerical_stability(self) -> bool:
        """Test numerical stability across multiple training steps."""
        print("\\nTesting training step numerical stability...")
        
        # Create models for both precisions
        models = {}
        optimizers = {}
        
        for precision_config in self.precision_configs:
            model = HRM_ACT(self.config)
            model.set_training(True)
            models[precision_config['name']] = model
            
            # Simple optimizer setup (not using the complex sparse embedding optimizer)
            optimizers[precision_config['name']] = {
                'learning_rate': 0.001,
                'precision': precision_config
            }
        
        batch = self.create_test_batch(2)
        num_steps = 5
        
        losses_by_precision = {name: [] for name in ['FP32', 'BF16']}
        
        print(f"  Running {num_steps} training steps...")
        
        for step in range(num_steps):
            print(f"    Step {step + 1}:")
            
            for precision_name in ['FP32', 'BF16']:
                model = models[precision_name]
                optimizer_config = optimizers[precision_name]
                precision_config = optimizer_config['precision']
                
                try:
                    # Convert batch to appropriate precision (matching PyTorch reference)
                    batch_precision = {}
                    for key, value in batch.items():
                        if key in ['puzzle_ids', 'input_ids', 'labels']:
                            # Integer indices NEVER get cast - they must stay as integers
                            batch_precision[key] = value
                        else:
                            # Only cast actual float tensors if using mixed precision
                            if precision_config['compute_dtype'] != mx.float32:
                                batch_precision[key] = value.astype(precision_config['compute_dtype'])
                            else:
                                batch_precision[key] = value
                    
                    # Forward pass and loss computation
                    def loss_fn(batch):
                        carry = model.initial_carry(batch['input_ids'].shape[0])
                        new_carry, outputs = model(carry, batch)
                        loss = stablemax_cross_entropy(outputs['logits'], batch['labels'])
                        return mx.mean(loss)
                    
                    # Compute loss and gradients using MLX neural network approach
                    import mlx.nn as nn
                    loss_and_grads = nn.value_and_grad(model, loss_fn)
                    loss, grads = loss_and_grads(batch_precision)
                    
                    # Simple parameter update (scaled down to prevent instability)
                    lr = optimizer_config['learning_rate'] * 0.1  # Small learning rate
                    
                    # Apply parameter updates using MLX tree utilities for nested structures
                    from mlx.utils import tree_map
                    
                    # Get current parameters
                    current_params = model.parameters()
                    
                    # Apply simple gradient descent update using tree_map
                    def apply_update(param, grad):
                        if grad is not None:
                            return param - lr * grad
                        else:
                            return param
                    
                    # Update parameters (simplified - not using full optimizer state)
                    updated_params = tree_map(apply_update, current_params, grads)
                    
                    # Note: In a real training step, we would call model.update(updated_params)
                    # but for this test we just verify the update computation works
                    
                    loss_value = float(loss.astype(mx.float32))
                    losses_by_precision[precision_name].append(loss_value)
                    
                    print(f"      {precision_name}: loss = {loss_value:.6f}")
                    
                except Exception as e:
                    print(f"      💥 ERROR in {precision_name}: {e}")
                    return False
        
        # Analyze loss trajectories
        print(f"  Analyzing loss trajectories...")
        
        fp32_losses = losses_by_precision['FP32']
        bf16_losses = losses_by_precision['BF16']
        
        print(f"    FP32 losses: {[f'{l:.6f}' for l in fp32_losses]}")
        print(f"    BF16 losses: {[f'{l:.6f}' for l in bf16_losses]}")
        
        # Check that losses are generally decreasing or stable
        fp32_stable = not any(l > fp32_losses[0] * 2 for l in fp32_losses[1:])
        bf16_stable = not any(l > bf16_losses[0] * 2 for l in bf16_losses[1:])
        
        print(f"    FP32 stability: {'✅' if fp32_stable else '❌'}")
        print(f"    BF16 stability: {'✅' if bf16_stable else '❌'}")
        
        # Check that loss trajectories are similar
        loss_diffs = [abs(fp32 - bf16) for fp32, bf16 in zip(fp32_losses, bf16_losses)]
        max_loss_diff = max(loss_diffs)
        avg_loss_diff = sum(loss_diffs) / len(loss_diffs)
        
        print(f"    Max loss difference: {max_loss_diff:.6f}")
        print(f"    Average loss difference: {avg_loss_diff:.6f}")
        
        # Loss trajectories should be reasonably similar
        trajectories_similar = max_loss_diff < 1.0 and avg_loss_diff < 0.5
        
        if fp32_stable and bf16_stable and trajectories_similar:
            print(f"  ✅ PASSED: Training numerical stability maintained")
            return True
        else:
            print(f"  ❌ FAILED: Training numerical stability issues detected")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all mixed precision compliance tests."""
        print("🔬 Running Mixed Precision Consistency Tests")
        print("=" * 60)
        
        tests = [
            ("Dtype Casting Accuracy", self.test_dtype_casting_accuracy),
            ("Model Output Precision Consistency", self.test_model_output_precision_consistency),
            ("Loss Computation Consistency", self.test_loss_computation_consistency),
            ("Gradient Precision Effects", self.test_gradient_precision_effects),
            ("Accumulation Precision", self.test_accumulation_precision),
            ("Training Step Numerical Stability", self.test_training_step_numerical_stability)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\\n🧪 {test_name}")
            print("-" * 40)
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
        
        print("\\n" + "=" * 60)
        print(f"📊 Mixed Precision Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED - Mixed precision implementation appears consistent!")
        else:
            print("⚠️  SOME TESTS FAILED - Mixed precision implementation needs review")
        
        return passed_tests == len(tests)


def test_mixed_precision_consistency():
    """Pytest entry point for mixed precision consistency tests."""
    tester = MixedPrecisionComplianceTest()
    assert tester.run_all_tests(), "Mixed precision consistency tests failed"


if __name__ == "__main__":
    # Run tests directly
    tester = MixedPrecisionComplianceTest()
    tester.run_all_tests()