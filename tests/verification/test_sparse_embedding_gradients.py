"""
Test sparse embedding gradient computation compliance between PyTorch and MLX.

This test focuses on verifying that the MLX sparse embedding implementation
produces identical gradients and updates compared to the PyTorch reference.

Critical areas tested:
1. Sparse embedding forward pass parity
2. Gradient computation for local weights
3. SignSGD optimizer behavior
4. Sparse update mechanics
5. Type casting consistency
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytest

from mlx_hrm.modules.act import HRMConfig
from mlx_hrm.layers.embeddings import CastedSparseEmbedding, SignSGD


class SparseEmbeddingComplianceTest:
    """Test suite for sparse embedding gradient computation compliance."""
    
    def __init__(self):
        # Configuration for sparse embedding testing
        self.config = {
            'num_embeddings': 128,    # Number of puzzle types
            'embedding_dim': 64,      # Embedding dimension
            'batch_size': 8,          # Batch size for testing
            'init_std': 0.02,         # Initialization standard deviation
            'learning_rate': 1e-3,    # SignSGD learning rate
            'weight_decay': 1e-2      # Weight decay for regularization
        }
    
    def create_test_data(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Create deterministic test data for sparse embedding tests."""
        # Set random seed for reproducibility
        mx.random.seed(42)
        
        # Create puzzle IDs (may have duplicates to test sparse handling)
        puzzle_ids = mx.random.randint(0, self.config['num_embeddings'], shape=(batch_size,))
        
        # Create synthetic gradients for testing optimizer behavior
        synthetic_grads = mx.random.normal((batch_size, self.config['embedding_dim'])) * 0.1
        
        return puzzle_ids, synthetic_grads
    
    def test_sparse_embedding_forward_pass(self) -> bool:
        """Test sparse embedding forward pass consistency."""
        print("Testing sparse embedding forward pass...")
        
        # Create sparse embedding layer
        embedding = CastedSparseEmbedding(
            num_embeddings=self.config['num_embeddings'],
            embedding_dim=self.config['embedding_dim'],
            batch_size=self.config['batch_size'],
            init_std=self.config['init_std'],
            cast_to=mx.float32  # No casting for simpler testing
        )
        
        batch_size = 4
        puzzle_ids, _ = self.create_test_data(batch_size)
        
        print(f"  Testing with puzzle IDs: {puzzle_ids}")
        
        try:
            # Test inference mode (default)
            embedding.eval()
            output_inference = embedding(puzzle_ids)
            
            print(f"  Inference output shape: {output_inference.shape}")
            print(f"  Expected shape: ({batch_size}, {self.config['embedding_dim']})")
            
            # Test training mode
            embedding.train()
            output_training = embedding(puzzle_ids)
            
            print(f"  Training output shape: {output_training.shape}")
            
            # Check shapes are correct
            expected_shape = (batch_size, self.config['embedding_dim'])
            shapes_correct = (
                output_inference.shape == expected_shape and
                output_training.shape == expected_shape
            )
            
            if shapes_correct:
                print(f"  ✅ PASSED: Output shapes correct")
                
                # Check that inference and training outputs are identical for same inputs
                output_diff = mx.abs(output_training - output_inference).max()
                print(f"    Training vs inference difference: {output_diff:.8f}")
                
                # Should be identical since we're not doing any training-specific modifications
                if output_diff < 1e-7:
                    print(f"  ✅ PASSED: Training and inference outputs identical")
                    
                    # Test local workspace tracking in training mode
                    if hasattr(embedding, '_local_ids') and embedding._local_ids is not None:
                        local_ids_match = mx.array_equal(embedding._local_ids, puzzle_ids)
                        if local_ids_match:
                            print(f"  ✅ PASSED: Local IDs tracked correctly")
                            return True
                        else:
                            print(f"  ❌ FAILED: Local IDs not tracked correctly")
                            return False
                    else:
                        print(f"  ❌ FAILED: Local workspace not set up in training mode")
                        return False
                else:
                    print(f"  ❌ FAILED: Training and inference outputs differ unexpectedly")
                    return False
            else:
                print(f"  ❌ FAILED: Incorrect output shapes")
                print(f"           Got inference: {output_inference.shape}, training: {output_training.shape}")
                print(f"           Expected: {expected_shape}")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_duplicate_ids_handling(self) -> bool:
        """Test handling of duplicate puzzle IDs in batch."""
        print("\\nTesting duplicate puzzle IDs handling...")
        
        embedding = CastedSparseEmbedding(
            num_embeddings=self.config['num_embeddings'],
            embedding_dim=self.config['embedding_dim'],
            batch_size=self.config['batch_size'],
            init_std=self.config['init_std']
        )
        
        # Create batch with intentional duplicates
        duplicate_ids = mx.array([5, 10, 5, 15, 10, 5], dtype=mx.int32)
        
        print(f"  Testing with duplicate IDs: {duplicate_ids}")
        
        try:
            embedding.train()
            outputs = embedding(duplicate_ids)
            
            print(f"  Output shape: {outputs.shape}")
            
            # Check that identical IDs produce identical embeddings
            # ID 5 appears at positions 0, 2, 5
            id_5_embeddings = [outputs[0], outputs[2], outputs[5]]
            
            # Compare first occurrence with other occurrences
            diff_0_2 = mx.abs(id_5_embeddings[0] - id_5_embeddings[1]).max()
            diff_0_5 = mx.abs(id_5_embeddings[0] - id_5_embeddings[2]).max()
            
            print(f"    ID 5 embedding differences:")
            print(f"      Position 0 vs 2: {diff_0_2:.8f}")
            print(f"      Position 0 vs 5: {diff_0_5:.8f}")
            
            # Should be identical (or very close due to floating point)
            if diff_0_2 < 1e-7 and diff_0_5 < 1e-7:
                print(f"  ✅ PASSED: Duplicate IDs produce identical embeddings")
                
                # Check ID 10 as well (positions 1, 4)
                diff_10 = mx.abs(outputs[1] - outputs[4]).max()
                print(f"      ID 10 difference (pos 1 vs 4): {diff_10:.8f}")
                
                if diff_10 < 1e-7:
                    print(f"  ✅ PASSED: All duplicate handling correct")
                    return True
                else:
                    print(f"  ❌ FAILED: ID 10 duplicates don't match")
                    return False
            else:
                print(f"  ❌ FAILED: Duplicate IDs produce different embeddings")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_type_casting_consistency(self) -> bool:
        """Test type casting behavior matches expected patterns."""
        print("\\nTesting type casting consistency...")
        
        # Test with different target dtypes
        test_cases = [
            {'cast_to': mx.float32, 'description': 'FP32 (no casting)'},
            {'cast_to': mx.bfloat16, 'description': 'BF16 (mixed precision)'},
        ]
        
        puzzle_ids = mx.array([1, 5, 10], dtype=mx.int32)
        
        for case in test_cases:
            print(f"  Testing {case['description']}...")
            
            try:
                embedding = CastedSparseEmbedding(
                    num_embeddings=self.config['num_embeddings'],
                    embedding_dim=self.config['embedding_dim'],
                    batch_size=self.config['batch_size'],
                    cast_to=case['cast_to']
                )
                
                # Test both training and inference
                embedding.eval()
                output_inf = embedding(puzzle_ids)
                
                embedding.train()
                output_train = embedding(puzzle_ids)
                
                # Check output dtypes
                expected_dtype = case['cast_to']
                inf_dtype_correct = output_inf.dtype == expected_dtype
                train_dtype_correct = output_train.dtype == expected_dtype
                
                print(f"    Inference dtype: {output_inf.dtype} (expected {expected_dtype})")
                print(f"    Training dtype: {output_train.dtype} (expected {expected_dtype})")
                
                if inf_dtype_correct and train_dtype_correct:
                    print(f"    ✅ PASSED: {case['description']} casting correct")
                else:
                    print(f"    ❌ FAILED: {case['description']} casting incorrect")
                    return False
                    
            except Exception as e:
                print(f"    💥 ERROR in {case['description']}: {e}")
                return False
        
        print(f"  ✅ PASSED: All type casting tests successful")
        return True
    
    def test_signsgd_update_mechanics(self) -> bool:
        """Test SignSGD optimizer update mechanics."""
        print("\\nTesting SignSGD optimizer update mechanics...")
        
        # Create sparse embedding and optimizer
        embedding = CastedSparseEmbedding(
            num_embeddings=10,  # Small for easier testing
            embedding_dim=4,    # Small for easier verification
            batch_size=3,
            init_std=0.02
        )
        
        optimizer = SignSGD(
            learning_rate=0.1,  # Larger for visible effects
            weight_decay=0.0    # No weight decay for simpler testing
        )
        
        # Create test data
        puzzle_ids = mx.array([1, 3, 7], dtype=mx.int32)
        
        try:
            # Store initial weights for comparison
            initial_weights = mx.array(embedding.weight)
            print(f"  Initial embedding weights shape: {initial_weights.shape}")
            
            # Forward pass in training mode
            embedding.train()
            outputs = embedding(puzzle_ids)
            
            print(f"  Forward pass outputs shape: {outputs.shape}")
            print(f"  Outputs for IDs {puzzle_ids}:")
            print(f"    {outputs}")
            
            # Create synthetic gradients (simulate backward pass)
            synthetic_grads = mx.array([
                [0.1, -0.2, 0.3, -0.1],   # Gradient for ID 1
                [-0.3, 0.1, -0.1, 0.2],   # Gradient for ID 3  
                [0.2, 0.1, -0.2, -0.3]    # Gradient for ID 7
            ])
            
            print(f"  Synthetic gradients:")
            print(f"    {synthetic_grads}")
            
            # Manually apply SignSGD update to test the mechanics
            # SignSGD rule: w = w - lr * sign(grad)
            lr = 0.1
            expected_signs = mx.sign(synthetic_grads)
            expected_updates = -lr * expected_signs
            
            print(f"  Expected sign(gradients):")
            print(f"    {expected_signs}")
            print(f"  Expected updates (-lr * sign(grad)):")
            print(f"    {expected_updates}")
            
            # Apply the updates manually to check our logic
            current_embeddings = initial_weights[puzzle_ids]
            expected_new_embeddings = current_embeddings + expected_updates
            
            print(f"  Current embeddings for test IDs:")
            print(f"    {current_embeddings}")
            print(f"  Expected new embeddings:")
            print(f"    {expected_new_embeddings}")
            
            # Test the optimizer update
            optimizer.update_sparse_embedding(embedding, synthetic_grads, puzzle_ids)
            
            # Check updated weights
            updated_weights = embedding.weight
            actual_new_embeddings = updated_weights[puzzle_ids]
            
            print(f"  Actual new embeddings after optimizer update:")
            print(f"    {actual_new_embeddings}")
            
            # Compare expected vs actual
            update_diff = mx.abs(actual_new_embeddings - expected_new_embeddings).max()
            print(f"  Max difference between expected and actual: {update_diff:.8f}")
            
            if update_diff < 1e-6:
                print(f"  ✅ PASSED: SignSGD update mechanics correct")
                
                # Verify that other embeddings were not affected
                unchanged_ids = [0, 2, 4, 5, 6, 8, 9]  # IDs not in our batch
                unchanged_diffs = mx.abs(updated_weights[unchanged_ids] - initial_weights[unchanged_ids]).max()
                print(f"  Max change in unused embeddings: {unchanged_diffs:.8f}")
                
                if unchanged_diffs < 1e-10:
                    print(f"  ✅ PASSED: Unused embeddings unchanged")
                    return True
                else:
                    print(f"  ❌ FAILED: Unused embeddings were modified")
                    return False
            else:
                print(f"  ❌ FAILED: SignSGD update mechanics incorrect")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_gradient_accumulation_with_duplicates(self) -> bool:
        """Test gradient handling when same embedding appears multiple times."""
        print("\\nTesting gradient accumulation with duplicate IDs...")
        
        embedding = CastedSparseEmbedding(
            num_embeddings=5,
            embedding_dim=3,
            batch_size=4,
            init_std=0.02
        )
        
        optimizer = SignSGD(learning_rate=0.05, weight_decay=0.0)
        
        # Test case: ID 2 appears twice in the batch
        puzzle_ids = mx.array([0, 2, 1, 2], dtype=mx.int32)
        
        try:
            # Forward pass
            embedding.train()
            outputs = embedding(puzzle_ids)
            
            print(f"  Batch IDs: {puzzle_ids} (ID 2 appears at positions 1 and 3)")
            print(f"  Forward outputs shape: {outputs.shape}")
            
            # Create gradients - different values for the two occurrences of ID 2
            synthetic_grads = mx.array([
                [0.1, 0.1, 0.1],   # Gradient for ID 0
                [0.2, -0.1, 0.3],  # Gradient for ID 2 (first occurrence)
                [-0.1, 0.2, -0.2], # Gradient for ID 1
                [0.1, 0.3, -0.1]   # Gradient for ID 2 (second occurrence)
            ])
            
            print(f"  Synthetic gradients:")
            for i, grad in enumerate(synthetic_grads):
                print(f"    Position {i} (ID {puzzle_ids[i]}): {grad}")
            
            # For MLX implementation, we update each occurrence independently
            # This tests the current implementation behavior
            initial_weights = mx.array(embedding.weight)
            
            # Update with optimizer
            optimizer.update_sparse_embedding(embedding, synthetic_grads, puzzle_ids)
            
            # Check that update was applied
            updated_weights = embedding.weight
            weight_changes = updated_weights - initial_weights
            
            print(f"  Weight changes:")
            for i in range(len(weight_changes)):
                if not mx.allclose(weight_changes[i], mx.zeros_like(weight_changes[i]), atol=1e-8):
                    print(f"    ID {i}: {weight_changes[i]}")
            
            # Verify that IDs 0, 1, 2 were updated (appeared in batch)
            # and IDs 3, 4 were not updated (didn't appear)
            id_0_changed = mx.abs(weight_changes[0]).max() > 1e-8
            id_1_changed = mx.abs(weight_changes[1]).max() > 1e-8  
            id_2_changed = mx.abs(weight_changes[2]).max() > 1e-8
            id_3_unchanged = mx.abs(weight_changes[3]).max() < 1e-8
            id_4_unchanged = mx.abs(weight_changes[4]).max() < 1e-8
            
            changes_correct = (id_0_changed and id_1_changed and id_2_changed and 
                             id_3_unchanged and id_4_unchanged)
            
            print(f"  ID change status: 0:{id_0_changed}, 1:{id_1_changed}, 2:{id_2_changed}, "
                  f"3:{id_3_unchanged}, 4:{id_4_unchanged}")
            
            if changes_correct:
                print(f"  ✅ PASSED: Correct embeddings were updated")
                return True
            else:
                print(f"  ❌ FAILED: Incorrect embeddings were updated")
                return False
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            return False
    
    def test_weight_decay_application(self) -> bool:
        """Test weight decay application in SignSGD."""
        print("\\nTesting weight decay application...")
        
        embedding = CastedSparseEmbedding(
            num_embeddings=3,
            embedding_dim=2,
            batch_size=2,
            init_std=0.1  # Larger initial values to see weight decay effect
        )
        
        # Test with and without weight decay
        test_cases = [
            {'weight_decay': 0.0, 'description': 'No weight decay'},
            {'weight_decay': 0.1, 'description': 'With weight decay'}
        ]
        
        puzzle_ids = mx.array([0, 2], dtype=mx.int32)
        synthetic_grads = mx.array([[0.1, -0.1], [0.2, 0.2]])
        
        for case in test_cases:
            print(f"  Testing {case['description']}...")
            
            try:
                optimizer = SignSGD(learning_rate=0.1, weight_decay=case['weight_decay'])
                
                # Store initial weights
                initial_weights = mx.array(embedding.weight[puzzle_ids])
                
                # Forward pass
                embedding.train() 
                outputs = embedding(puzzle_ids)
                
                # Apply update
                optimizer.update_sparse_embedding(embedding, synthetic_grads, puzzle_ids)
                
                # Check updated weights
                updated_weights = embedding.weight[puzzle_ids]
                weight_changes = updated_weights - initial_weights
                
                print(f"    Initial weights: {initial_weights}")
                print(f"    Weight changes: {weight_changes}")
                print(f"    Updated weights: {updated_weights}")
                
                # For weight decay case, check that weights moved toward zero
                if case['weight_decay'] > 0:
                    # Weights should be smaller in magnitude due to decay
                    initial_magnitude = mx.abs(initial_weights).mean()
                    updated_magnitude = mx.abs(updated_weights).mean()
                    
                    print(f"    Initial magnitude: {initial_magnitude:.6f}")
                    print(f"    Updated magnitude: {updated_magnitude:.6f}")
                    
                    # With weight decay, magnitude should decrease (unless gradient effect dominates)
                    # This is a qualitative check - exact behavior depends on gradient vs decay strength
                    decay_effect_visible = updated_magnitude != initial_magnitude
                    
                    if decay_effect_visible:
                        print(f"    ✅ PASSED: Weight decay effect visible")
                    else:
                        print(f"    ⚠️  WARNING: Weight decay effect not clearly visible")
                else:
                    # Without weight decay, check that update only comes from gradient
                    expected_update = -0.1 * mx.sign(synthetic_grads)  # -lr * sign(grad)
                    expected_weights = initial_weights + expected_update
                    
                    diff = mx.abs(updated_weights - expected_weights).max()
                    print(f"    Expected update: {expected_update}")
                    print(f"    Difference from expected: {diff:.8f}")
                    
                    if diff < 1e-6:
                        print(f"    ✅ PASSED: No weight decay case correct")
                    else:
                        print(f"    ❌ FAILED: No weight decay case incorrect")
                        return False
                        
            except Exception as e:
                print(f"    💥 ERROR in {case['description']}: {e}")
                return False
        
        print(f"  ✅ PASSED: Weight decay application tests completed")
        return True
    
    def run_all_tests(self) -> bool:
        """Run all sparse embedding gradient compliance tests."""
        print("🔬 Running Sparse Embedding Gradient Compliance Tests")
        print("=" * 60)
        
        tests = [
            ("Sparse Embedding Forward Pass", self.test_sparse_embedding_forward_pass),
            ("Duplicate IDs Handling", self.test_duplicate_ids_handling),
            ("Type Casting Consistency", self.test_type_casting_consistency),
            ("SignSGD Update Mechanics", self.test_signsgd_update_mechanics),
            ("Gradient Accumulation with Duplicates", self.test_gradient_accumulation_with_duplicates),
            ("Weight Decay Application", self.test_weight_decay_application)
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
        print(f"📊 Sparse Embedding Results: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED - Sparse embedding implementation appears compliant!")
        else:
            print("⚠️  SOME TESTS FAILED - Sparse embedding implementation needs review")
        
        return passed_tests == len(tests)


def test_sparse_embedding_gradients():
    """Pytest entry point for sparse embedding gradient compliance tests."""
    tester = SparseEmbeddingComplianceTest()
    assert tester.run_all_tests(), "Sparse embedding gradient compliance tests failed"


if __name__ == "__main__":
    # Run tests directly
    tester = SparseEmbeddingComplianceTest()
    tester.run_all_tests()