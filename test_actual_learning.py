#!/usr/bin/env python3
"""Test that the model can actually learn with higher learning rates."""

import mlx.core as mx
import mlx.nn as nn
from src.mlx_hrm.models.factory import create_hrm
from src.mlx_hrm.training.trainer import HRMTrainer

def create_single_batch():
    """Create a single batch for consistent testing."""
    # Simple pattern: input is [1,2,3,4], output should be [2,3,4,5]
    return {
        'input_ids': mx.array([[1, 2, 3, 4, 0, 0, 0, 0]]),  # [1, 8]
        'labels': mx.array([[2, 3, 4, 5, 0, 0, 0, 0]]),     # [1, 8]
        'puzzle_ids': mx.array([0])  # [1]
    }

def test_learning_with_higher_lr():
    """Test learning with higher learning rates."""
    print("🎯 Testing Actual Learning with Higher Learning Rates")
    print("=" * 60)
    
    # Create minimal config
    config = {
        'vocab_size': 12,
        'hidden_size': 32,  # Even smaller for faster learning
        'num_heads': 2,
        'H_layers': 1,
        'L_layers': 1,
        'H_cycles': 1,
        'L_cycles': 1,
        'halt_max_steps': 2,
        'expansion': 2.0,
        'rope_theta': 1000.0,
        'rms_norm_eps': 1e-5,
        'pos_encodings': 'rope',
        'puzzle_emb_ndim': 8,
        'num_puzzle_identifiers': 1,
        'batch_size': 4,
        'seq_len': 8
    }
    
    model = create_hrm(config)
    
    # Create simple training batch
    train_batch = create_single_batch()
    
    # Create trainer with higher learning rates
    trainer = HRMTrainer(
        model=model,
        train_dataloader=iter([train_batch]),  # Dummy dataloader
        optimizer_config={
            'dense_lr': 1e-2,      # 100x higher than default
            'sparse_lr': 1e-2,     # 100x higher than default  
            'dense_weight_decay': 0.01,   # Lower weight decay
            'sparse_weight_decay': 0.01
        },
        use_mixed_precision=False,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0  # Clip gradients to prevent instability
    )
    
    # Test initial predictions
    print("\n1. Initial Model Performance:")
    test_batch = train_batch  # Use same batch for consistency
    
    carry = trainer.loss_model.initial_carry(1)
    _, loss, metrics, outputs = trainer.loss_model(carry, test_batch)
    logits = outputs['logits'][0]  # Remove batch dim
    predictions = mx.argmax(logits, axis=-1)
    
    print(f"   Input:      {[int(x) for x in test_batch['input_ids'][0][:4]]}")
    print(f"   Expected:   {[int(x) for x in test_batch['labels'][0][:4]]}")
    print(f"   Predicted:  {[int(x) for x in predictions[:4]]}")
    print(f"   Initial loss: {float(loss):.4f}")
    
    # Training loop
    print(f"\n2. Training for 200 steps...")
    losses = []
    
    for step in range(200):
        try:
            loss, grads, metrics = trainer._training_step(train_batch)
            losses.append(float(loss))
            
            # Manual parameter update (since we're not using train_epoch)
            aligned_grads = trainer._align_gradients_with_model_params(grads)
            aligned_grads = trainer._clip_gradients(aligned_grads)
            
            model_params = trainer.loss_model.model.parameters()
            flat_params = trainer._flatten_nested_dict(model_params)
            flat_grads = trainer._flatten_nested_dict(aligned_grads)
            
            updated_flat_params = trainer.optimizer.update(flat_params, flat_grads)
            updated_params = trainer._unflatten_to_nested_dict(updated_flat_params, model_params)
            trainer.loss_model.model.update(updated_params)
            
            if step % 50 == 0:
                print(f"   Step {step}: loss={float(loss):.4f}")
                
        except Exception as e:
            print(f"   Error at step {step}: {e}")
            break
    
    print(f"\n3. Final Model Performance:")
    carry = trainer.loss_model.initial_carry(1)
    _, final_loss, metrics, outputs = trainer.loss_model(carry, test_batch)
    logits = outputs['logits'][0]
    final_predictions = mx.argmax(logits, axis=-1)
    
    print(f"   Input:      {[int(x) for x in test_batch['input_ids'][0][:4]]}")
    print(f"   Expected:   {[int(x) for x in test_batch['labels'][0][:4]]}")
    print(f"   Predicted:  {[int(x) for x in final_predictions[:4]]}")
    print(f"   Final loss: {float(final_loss):.4f}")
    
    # Calculate accuracy
    correct = sum(1 for i in range(4) if int(final_predictions[i]) == int(test_batch['labels'][0][i]))
    accuracy = correct / 4
    print(f"   Accuracy: {accuracy:.1%} ({correct}/4 correct)")
    
    # Analysis
    initial_loss = losses[0]
    final_loss_val = losses[-1]
    improvement = (initial_loss - final_loss_val) / initial_loss
    
    print(f"\n4. Training Analysis:")
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss:   {final_loss_val:.4f}")
    print(f"   Improvement:  {improvement:.1%}")
    
    if improvement > 0.1:
        print("   ✅ Model is learning!")
    elif improvement > 0.01:
        print("   ⚠️  Model shows some learning")
    else:
        print("   ❌ Model is not learning effectively")
    
    if accuracy >= 0.75:
        print("   ✅ Good final accuracy!")
    elif accuracy >= 0.5:
        print("   ⚠️  Decent final accuracy")
    else:
        print("   ❌ Poor final accuracy")
    
    print("\n" + "=" * 60)
    return improvement > 0.1 and accuracy >= 0.5

if __name__ == "__main__":
    success = test_learning_with_higher_lr()
    if success:
        print("🎉 LEARNING TEST: SUCCESS!")
    else:
        print("❌ LEARNING TEST: FAILED!")