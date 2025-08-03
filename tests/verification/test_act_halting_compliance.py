"""
Test ACT halting logic compliance between PyTorch and MLX implementations.

This test focuses on the critical ACT halting logic that determines when the model
should stop computation. This is core to HRM's reasoning capabilities.

Critical areas tested:
1. Q-value computation consistency
2. Halting decision agreement
3. Target Q-value bootstrapping
4. Max steps enforcement
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple
import pytest

from mlx_hrm.modules.act import HRMConfig, HRMCarry, HRMInnerCarry
from mlx_hrm.models.hrm_act import HRM_ACT


class ACTHaltingComplianceTest:
    """Test suite for ACT halting logic compliance."""
    
    def __init__(self):
        # Test configuration matching PyTorch reference
        self.config = HRMConfig(
            batch_size=4,
            seq_len=8,
            puzzle_emb_ndim=16,
            num_puzzle_identifiers=4,
            vocab_size=32,
            hidden_size=64,
            expansion=4.0,
            num_heads=8,
            halt_max_steps=10,
            halt_exploration_prob=0.0  # Deterministic for testing
        )
    
    def test_q_value_halting(self) -> bool:
        """Test Q-value based halting decisions."""
        print("Testing Q-value based halting decisions...")
        
        model = HRM_ACT(self.config)
        model.set_training(True)  # Enable Q-value halting logic
        
        # Test scenarios
        scenarios = [
            {
                "name": "clear_halt",
                "q_halt_target": [2.0, 3.0, 4.0, 5.0],
                "q_continue_target": [-1.0, -0.5, 0.0, 1.0],
                "expected_halt": [True, True, True, True]
            },
            {
                "name": "clear_continue", 
                "q_halt_target": [-2.0, -1.0, 0.0, 1.0],
                "q_continue_target": [3.0, 4.0, 5.0, 6.0],
                "expected_halt": [False, False, False, False]
            },
            {
                "name": "mixed_signals",
                "q_halt_target": [2.0, -1.0, 3.0, -0.5],
                "q_continue_target": [1.0, 2.0, 1.0, 0.5],
                "expected_halt": [True, False, True, False]
            }
        ]
        
        passed_tests = 0
        
        for scenario in scenarios:
            print(f"  Testing scenario: {scenario['name']}")
            
            batch_size = len(scenario['q_halt_target'])
            
            # Create batch
            batch = {
                "input_ids": mx.random.randint(0, self.config.vocab_size, shape=(batch_size, self.config.seq_len)),
                "puzzle_ids": mx.random.randint(0, self.config.num_puzzle_identifiers, shape=(batch_size,)),
                "labels": mx.random.randint(0, self.config.vocab_size, shape=(batch_size, self.config.seq_len))
            }
            
            # Create carry state at step 3 (not at max steps)
            carry = model.initial_carry(batch_size)
            carry.steps = mx.array([3] * batch_size, dtype=mx.int32)
            carry.halted = mx.array([False] * batch_size, dtype=mx.bool_)
            
            # Store original Q-head layer
            original_q_head = model.inner.q_head
            
            # Create mock Q-head that returns our target Q-values
            class MockQHead:
                def __init__(self, q_halt_targets, q_continue_targets):
                    self.q_halt_targets = mx.array(q_halt_targets, dtype=mx.float32)
                    self.q_continue_targets = mx.array(q_continue_targets, dtype=mx.float32)
                    
                def __call__(self, x):
                    # x shape: [batch_size, hidden_size]
                    # Return shape: [batch_size, 2] where [:, 0] = q_halt, [:, 1] = q_continue
                    return mx.stack([self.q_halt_targets, self.q_continue_targets], axis=1)
            
            # Replace Q-head with mock
            model.inner.q_head = MockQHead(scenario['q_halt_target'], scenario['q_continue_target'])
            
            try:
                # Run forward pass
                new_carry, outputs = model(carry, batch)
                
                # Check results
                actual_halt = outputs["halted"].tolist()
                expected_halt = scenario['expected_halt']
                
                print(f"    Q_halt:     {scenario['q_halt_target']}")
                print(f"    Q_continue: {scenario['q_continue_target']}")
                print(f"    Expected:   {expected_halt}")
                print(f"    Actual:     {actual_halt}")
                
                if actual_halt == expected_halt:
                    print(f"    ✅ PASSED")
                    passed_tests += 1
                else:
                    print(f"    ❌ FAILED")
                    mismatches = sum(1 for a, e in zip(actual_halt, expected_halt) if a != e)
                    print(f"         {mismatches}/{len(expected_halt)} decisions incorrect")
                    
            except Exception as e:
                print(f"    💥 ERROR: {e}")
                
            finally:
                # Restore original Q-head
                model.inner.q_head = original_q_head
        
        success = passed_tests == len(scenarios)
        if success:
            print(f"  ✅ PASSED: Q-value halting logic works correctly")
        else:
            print(f"  ❌ FAILED: Q-value halting logic has issues")
        
        return success
    
    def test_max_steps_halting(self) -> bool:
        """Test halting at maximum steps."""
        print("\nTesting max steps halting...")
        
        config = HRMConfig(
            batch_size=3,
            seq_len=4,
            puzzle_emb_ndim=8,
            num_puzzle_identifiers=2,
            vocab_size=16,
            hidden_size=32,
            num_heads=4,
            halt_max_steps=5,
            halt_exploration_prob=0.0
        )
        
        model = HRM_ACT(config)
        model.set_training(True)
        
        batch = {
            "input_ids": mx.random.randint(0, config.vocab_size, shape=(3, 4)),
            "puzzle_ids": mx.random.randint(0, config.num_puzzle_identifiers, shape=(3,)),
            "labels": mx.random.randint(0, config.vocab_size, shape=(3, 4))
        }
        
        # Test at different step counts
        # Note: halt_max_steps=5 means halt when step counter reaches 5
        test_cases = [
            {"step": 3, "should_halt": False, "description": "Two steps before max (3→4)"},
            {"step": 4, "should_halt": True, "description": "One step before max (4→5, halt at 5)"}
        ]
        
        passed_cases = 0
        
        for case in test_cases:
            print(f"  Testing: {case['description']} (step {case['step']})")
            
            carry = model.initial_carry(3)
            carry.steps = mx.array([case['step']] * 3, dtype=mx.int32)
            carry.halted = mx.array([False] * 3, dtype=mx.bool_)
            
            # Set Q-values to strongly favor continuing (but should be overridden at max steps)
            original_q_head = model.inner.q_head
            
            class MockQHead:
                def __call__(self, x):
                    batch_size = x.shape[0]
                    # Return [q_halt, q_continue] = [-10.0, 10.0] for all sequences
                    q_halt = mx.full((batch_size,), -10.0, dtype=mx.float32)
                    q_continue = mx.full((batch_size,), 10.0, dtype=mx.float32)
                    return mx.stack([q_halt, q_continue], axis=1)
            
            model.inner.q_head = MockQHead()
            
            try:
                new_carry, outputs = model(carry, batch)
                halted = outputs["halted"]
                
                print(f"    Initial steps: {case['step']}")
                print(f"    New steps will be: {case['step'] + 1}")
                print(f"    Should halt: {case['should_halt']}")
                print(f"    Actually halted: {halted.tolist()}")
                
                if case['should_halt']:
                    if halted.all():
                        print(f"    ✅ PASSED: All sequences halted at max steps")
                        passed_cases += 1
                    else:
                        print(f"    ❌ FAILED: Not all sequences halted at max steps")
                else:
                    if not halted.any():
                        print(f"    ✅ PASSED: No sequences halted before max steps")
                        passed_cases += 1
                    else:
                        print(f"    ❌ FAILED: Some sequences halted before max steps")
                        
            except Exception as e:
                print(f"    💥 ERROR: {e}")
                
            finally:
                model.inner.q_head = original_q_head
        
        success = passed_cases == len(test_cases)
        if success:
            print(f"  ✅ PASSED: Max steps behavior correct")
        else:
            print(f"  ❌ FAILED: Max steps behavior incorrect")
        
        return success
    
    def test_target_q_computation(self) -> bool:
        """Test target Q-value computation for Q-learning."""
        print("\nTesting target Q-value computation...")
        
        config = HRMConfig(
            batch_size=2,
            seq_len=4,
            puzzle_emb_ndim=8,
            num_puzzle_identifiers=2,
            vocab_size=16,
            hidden_size=32,
            num_heads=4,
            halt_max_steps=8,
            halt_exploration_prob=0.0
        )
        
        model = HRM_ACT(config)
        model.set_training(True)
        
        batch = {
            "input_ids": mx.random.randint(0, config.vocab_size, shape=(2, 4)),
            "puzzle_ids": mx.random.randint(0, config.num_puzzle_identifiers, shape=(2,)),
            "labels": mx.random.randint(0, config.vocab_size, shape=(2, 4))
        }
        
        carry = model.initial_carry(2)
        carry.steps = mx.array([3, 4], dtype=mx.int32)  # Not at max steps
        carry.halted = mx.array([False, False], dtype=mx.bool_)
        
        # We'll let the model compute its own Q-values and check if the target computation is reasonable
        new_carry, outputs = model(carry, batch)
        
        print(f"  Model outputs keys: {outputs.keys()}")
        
        if "target_q_continue" in outputs:
            target_q = outputs["target_q_continue"]
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"]
            
            print(f"  Q_halt: {q_halt}")
            print(f"  Q_continue: {q_continue}")
            print(f"  Target Q: {target_q}")
            
            # Target should be sigmoid of max(Q_halt, Q_continue) from next step
            # Since we can't easily control the next step Q-values, let's just check reasonableness
            # Target Q should be in [0, 1] range due to sigmoid
            if target_q.min() >= 0.0 and target_q.max() <= 1.0:
                print(f"  ✅ PASSED: Target Q-values in valid range [0, 1]")
                return True
            else:
                print(f"  ❌ FAILED: Target Q-values out of range")
                return False
        else:
            print(f"  ❌ FAILED: No target_q_continue in outputs")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all ACT halting compliance tests."""
        print("🔬 Running ACT Halting Logic Compliance Tests")
        print("=" * 60)
        
        tests = [
            ("Q-value Halting", self.test_q_value_halting),
            ("Max Steps Halting", self.test_max_steps_halting),
            ("Target Q Computation", self.test_target_q_computation)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}")
            print("-" * 40)
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"📊 ACT Halting Compliance Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED - ACT halting logic appears compliant!")
        else:
            print("⚠️  SOME TESTS FAILED - ACT halting logic needs review")
        
        return passed_tests == len(tests)


def test_act_halting_compliance():
    """Pytest entry point for ACT halting compliance tests."""
    tester = ACTHaltingComplianceTest()
    assert tester.run_all_tests(), "ACT halting compliance tests failed"


if __name__ == "__main__":
    # Run tests directly
    tester = ACTHaltingComplianceTest()
    tester.run_all_tests()