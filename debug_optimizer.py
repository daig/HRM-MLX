#!/usr/bin/env python3
"""Debug optimizer parameter updates."""

import mlx.core as mx
import mlx.nn as nn
from src.mlx_hrm.models.factory import create_hrm
from src.mlx_hrm.training.trainer import HRMTrainer

def test_parameter_updates():
    """Test if parameters actually change during training."""
    print("🔍 Debug: Testing Parameter Updates")
    print("=" * 50)
    
    # Create minimal config and model
    config = {
        'vocab_size': 12,
        'hidden_size': 64,  # Smaller for debugging
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
        'puzzle_emb_ndim': 16,
        'num_puzzle_identifiers': 2,
        'batch_size': 1,
        'seq_len': 8
    }
    
    model = create_hrm(config)
    
    # Create minimal training batch
    batch = {
        'input_ids': mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]),  # [1, 8]
        'puzzle_ids': mx.array([0]),  # [1]
        'labels': mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])  # [1, 8]
    }
    
    # Create trainer
    trainer = HRMTrainer(
        model=model,
        train_dataloader=iter([batch]),
        use_mixed_precision=False
    )
    
    # Capture initial parameters
    print("\n1. Capturing initial parameters...")
    initial_params = {}
    for name, param in trainer.loss_model.model.named_parameters():
        initial_params[name] = mx.array(param)
        print(f"   {name}: shape={param.shape}, mean={float(mx.mean(param)):.6f}")
    
    print(f"\nInitial parameter count: {len(initial_params)}")
    
    # Run single training step
    print("\n2. Running training step...")
    try:
        loss, grads, metrics = trainer._training_step(batch)
        print(f"   Loss: {float(loss):.6f}")
        print(f"   Gradients computed: {len(grads) if isinstance(grads, dict) else 'Not dict'}")
        
        # Check gradient structure
        if isinstance(grads, dict):
            print("   Gradient structure:")
            def print_structure(d, indent=0):
                for k, v in d.items():
                    spaces = "    " * indent
                    if isinstance(v, dict):
                        print(f"{spaces}{k}:")
                        print_structure(v, indent + 1)
                    elif isinstance(v, mx.array):
                        print(f"{spaces}{k}: {v.shape}, mean={float(mx.mean(v)):.6f}")
                    else:
                        print(f"{spaces}{k}: {type(v)}")
            print_structure(grads, 1)
            
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now simulate the parameter update process manually
    print("\n3. Testing parameter update process...")
    
    # Align gradients
    aligned_grads = trainer._align_gradients_with_model_params(grads)
    print(f"   Aligned gradients: {len(aligned_grads) if isinstance(aligned_grads, dict) else 'Not dict'}")
    
    # Get model parameters
    model_params = trainer.loss_model.model.parameters()
    print(f"   Model parameters: {len(model_params) if isinstance(model_params, dict) else 'Not dict'}")
    
    # Flatten both
    flat_params = trainer._flatten_nested_dict(model_params)
    flat_grads = trainer._flatten_nested_dict(aligned_grads)
    
    print(f"   Flat parameters: {len(flat_params)}")
    print(f"   Flat gradients: {len(flat_grads)}")
    
    # Check overlap
    param_keys = set(flat_params.keys())
    grad_keys = set(flat_grads.keys())
    overlap = param_keys & grad_keys
    print(f"   Key overlap: {len(overlap)} / {len(param_keys)} parameters have gradients")
    
    if len(overlap) == 0:
        print("   ❌ No parameter-gradient key overlap! This is the problem.")
        print("   Parameter keys:", list(param_keys)[:5])
        print("   Gradient keys:", list(grad_keys)[:5])
        return
    
    # Update parameters  
    print(f"\n4. Updating parameters with optimizer...")
    updated_flat_params = trainer.optimizer.update(flat_params, flat_grads)
    print(f"   Updated parameters: {len(updated_flat_params)}")
    
    # Check if any parameters actually changed
    changes = 0
    for key in overlap:
        old_val = flat_params[key]
        new_val = updated_flat_params[key]
        if not mx.allclose(old_val, new_val, atol=1e-8):
            changes += 1
            diff = float(mx.mean(mx.abs(new_val - old_val)))
            print(f"   ✅ {key}: changed by {diff:.8f}")
    
    if changes == 0:
        print("   ❌ No parameters changed! Optimizer not working.")
    else:
        print(f"   ✅ {changes} parameters changed")
    
    # Reconstruct and apply updates 
    print(f"\n5. Applying updates to model...")
    updated_params = trainer._unflatten_to_nested_dict(updated_flat_params, model_params)
    trainer.loss_model.model.update(updated_params)
    
    # Check final parameters
    print(f"\n6. Checking final parameters...")
    final_changes = 0
    for name, initial_param in initial_params.items():
        # Get current parameter value
        current_params = dict(trainer.loss_model.model.named_parameters())
        if name in current_params:
            current_param = current_params[name]
            if not mx.allclose(initial_param, current_param, atol=1e-8):
                final_changes += 1
                diff = float(mx.mean(mx.abs(current_param - initial_param)))
                print(f"   ✅ {name}: final change {diff:.8f}")
    
    if final_changes == 0:
        print("   ❌ No final parameter changes detected!")
    else:
        print(f"   ✅ {final_changes} parameters have final changes")
    
    print("\n" + "=" * 50)
    print(f"RESULT: {final_changes} parameters successfully updated")

if __name__ == "__main__":
    test_parameter_updates()