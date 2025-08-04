# Parameter Update Fix Summary

## Root Cause Identified

The issue was a **parameter-gradient structure mismatch** in the MLX HRM training pipeline:

1. **Model parameters** have structure: `{'model': {'inner': {...}}}`
2. **Loss model gradients** have structure: `{'model': {'model': {'inner': {...}}}}`

This mismatch occurred because:
- Gradients were computed against `loss_model` (ACTLossHead wrapper)
- Parameter updates were applied to `loss_model.model` (inner HRM model)
- The gradient computation added an extra `'model'` nesting layer

## Solution Implemented

### 1. Gradient-Parameter Alignment (`_align_gradients_with_model_params`)
```python
def _align_gradients_with_model_params(self, grads):
    if isinstance(grads, dict) and 'model' in grads:
        # Strip the extra 'model' wrapper from gradients
        return grads['model']
    return grads
```

### 2. Nested Dictionary Flattening/Unflattening
Since optimizers expect flat parameter dictionaries, added helper functions:

```python
def _flatten_nested_dict(self, nested_dict, parent_key=''):
    """Flatten nested dict to dot-separated keys, only keeping mx.array values"""
    
def _unflatten_to_nested_dict(self, flat_dict, template_dict):
    """Reconstruct nested structure using template, preserving non-array values"""
```

### 3. Updated Training Step Pipeline
```python
# OLD (broken):
model_params = self.loss_model.model.parameters()
updated_params = self.optimizer.update(model_params, accumulated_grads)
self.loss_model.model.update(updated_params)

# NEW (working):
aligned_grads = self._align_gradients_with_model_params(accumulated_grads)
aligned_grads = self._clip_gradients(aligned_grads)

flat_params = self._flatten_nested_dict(model_params)
flat_grads = self._flatten_nested_dict(aligned_grads)

updated_flat_params = self.optimizer.update(flat_params, flat_grads)
updated_params = self._unflatten_to_nested_dict(updated_flat_params, model_params)
self.loss_model.model.update(updated_params)
```

## Validation Results

### Before Fix
- ❌ Parameters: 0 changed out of all parameters
- ❌ Loss: No decrease
- ❌ Predictions: Completely static

### After Fix
- ✅ Parameters: 8/8 changed with visible deltas (e.g., Δ=8.89e-02)
- ✅ Loss: Decreased from 2.951766 to 2.808959 (1.43e-01 change)
- ✅ Training: Completes successfully in end-to-end tests

## Files Modified

1. **`src/mlx_hrm/training/trainer.py`**:
   - Added `_align_gradients_with_model_params()`
   - Added `_flatten_nested_dict()` and `_unflatten_to_nested_dict()`
   - Updated `train_epoch()` parameter update pipeline

## Testing

Created comprehensive debug and validation scripts:
- `debug_parameter_updates.py` - Comprehensive parameter update pipeline analysis
- `debug_gradient_alignment.py` - Specific gradient structure debugging
- `test_parameter_update_fix.py` - Focused validation of the fix
- `test_end_to_end_training.py` - Full training validation (now passes)

## Impact

This fix resolves the core issue preventing MLX HRM from training properly. Parameters are now correctly updated during training, allowing the model to learn from data as intended.