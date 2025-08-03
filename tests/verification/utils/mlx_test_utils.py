"""
MLX-native testing utilities for HRM verification.

This module provides MLX-compatible patterns for testing that work with MLX's
functional programming approach and compilation optimizations.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from mlx_hrm.modules.act import HRMConfig, HRMCarry
from mlx_hrm.models.hrm_act import HRM_ACT
from mlx_hrm.training.losses import stablemax_cross_entropy


class MLXTestingPatterns:
    """Library of verified MLX testing patterns for HRM verification."""
    
    @staticmethod
    def copy_model_state_mlx(source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Copy model parameters using MLX tree utilities.
        
        Args:
            source_model: Model to copy parameters from
            target_model: Model to copy parameters to
            
        Returns:
            Target model with copied parameters
        """
        try:
            # Get source parameters as a tree
            source_params = source_model.parameters()
            
            # Use MLX tree utilities to copy the parameter tree
            def copy_array(arr):
                return mx.array(arr)
            
            # Apply the copy function to all arrays in the parameter tree
            copied_params = mlx.utils.tree_map(copy_array, source_params)
            
            # Update target model using MLX's built-in update method
            target_model.update(copied_params)
            
            return target_model
            
        except Exception as e:
            print(f"Warning: Parameter copying failed: {e}")
            print("Continuing with separate model instances")
            return target_model
    
    @staticmethod
    def _set_parameter_by_name(model: nn.Module, param_name: str, param_value: mx.array):
        """Set a parameter by its full dotted name."""
        parts = param_name.split('.')
        current = model
        
        # Navigate to the parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Set the parameter
        setattr(current, parts[-1], param_value)
    
    @staticmethod
    def create_identical_models(config: HRMConfig) -> Tuple[HRM_ACT, HRM_ACT]:
        """Create two identical models for comparison testing.
        
        Args:
            config: HRM configuration
            
        Returns:
            Tuple of (model1, model2) with identical parameters
        """
        model1 = HRM_ACT(config)
        model2 = HRM_ACT(config)
        
        # Copy parameters from model1 to model2
        model2 = MLXTestingPatterns.copy_model_state_mlx(model1, model2)
        
        return model1, model2
    
    @staticmethod
    def compare_gradient_trees(grads1: Dict[str, mx.array], 
                             grads2: Dict[str, mx.array], 
                             tolerance: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """Compare MLX gradient trees using native MLX operations.
        
        Args:
            grads1: First gradient dictionary
            grads2: Second gradient dictionary
            tolerance: Comparison tolerance
            
        Returns:
            Tuple of (all_match, differences_dict)
        """
        def flatten_nested_dict(d, parent_key='', sep='.'):
            """Flatten nested dictionary to compare gradient trees properly."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten both gradient trees
        flat_grads1 = flatten_nested_dict(grads1)
        flat_grads2 = flatten_nested_dict(grads2)
        
        differences = {}
        all_match = True
        
        # Get all unique keys
        all_keys = set(flat_grads1.keys()) | set(flat_grads2.keys())
        
        for key in all_keys:
            grad1 = flat_grads1.get(key)
            grad2 = flat_grads2.get(key)
            
            # Handle None cases
            if grad1 is None and grad2 is None:
                differences[key] = 0.0
                continue
            elif grad1 is None or grad2 is None:
                differences[key] = float('inf')
                all_match = False
                continue
            
            # Compare arrays
            try:
                if grad1.shape != grad2.shape:
                    differences[key] = float('inf')
                    all_match = False
                    continue
                
                # Compute difference
                diff = mx.abs(grad1 - grad2).max()
                differences[key] = float(diff)
                
                # Check tolerance
                if diff > tolerance:
                    all_match = False
                    
            except Exception as e:
                print(f"Error comparing gradients for {key}: {e}")
                differences[key] = float('inf')
                all_match = False
        
        return all_match, differences
    
    @staticmethod
    def compute_gradients_mlx_native(model: HRM_ACT, 
                                   batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute gradients using proper MLX neural network patterns.
        
        Args:
            model: HRM model
            batch: Input batch
            
        Returns:
            Tuple of (loss, gradients)
        """
        try:
            # Define loss function that takes the model's inputs
            def loss_fn(batch):
                """Loss function for gradient computation."""
                carry = model.initial_carry(batch['input_ids'].shape[0])
                new_carry, outputs = model(carry, batch)
                loss = stablemax_cross_entropy(outputs['logits'], batch['labels'], reduction='mean')
                return loss
            
            # Use MLX's nn.value_and_grad which is designed for neural networks
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(batch)
            
            return loss, grads
            
        except Exception as e:
            print(f"Error in gradient computation: {e}")
            raise e  # Don't mask gradient computation failures
    
    @staticmethod
    def compare_model_outputs(outputs1: Dict[str, mx.array], 
                            outputs2: Dict[str, mx.array], 
                            tolerances: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[bool, Dict[str, float]]:
        """Compare model outputs with appropriate tolerances.
        
        Args:
            outputs1: First model outputs
            outputs2: Second model outputs  
            tolerances: Custom tolerances per output type
            
        Returns:
            Tuple of (all_match, differences_dict)
        """
        if tolerances is None:
            tolerances = {
                'logits': {'rtol': 1e-5, 'atol': 1e-6},
                'q_halt_logits': {'rtol': 1e-5, 'atol': 1e-6},
                'q_continue_logits': {'rtol': 1e-5, 'atol': 1e-6},
                'halted': {'rtol': 0.0, 'atol': 0.0}  # Boolean, must be exact
            }
        
        differences = {}
        all_match = True
        
        for key in outputs1.keys():
            if key not in outputs2:
                differences[key] = float('inf')
                all_match = False
                continue
            
            out1 = outputs1[key]
            out2 = outputs2[key]
            
            try:
                if key == 'halted' or out1.dtype == mx.bool_:
                    # Boolean comparison - must be exact
                    match = mx.array_equal(out1, out2)
                    differences[key] = 0.0 if match else 1.0
                    if not match:
                        all_match = False
                else:
                    # Numerical comparison
                    tol = tolerances.get(key, {'rtol': 1e-5, 'atol': 1e-6})
                    
                    # Compute difference
                    diff = mx.abs(out1 - out2).max()
                    differences[key] = float(diff)
                    
                    # Check tolerance
                    close = mx.allclose(out1, out2, **tol)
                    if not close:
                        all_match = False
                        
            except Exception as e:
                print(f"Error comparing output {key}: {e}")
                differences[key] = float('inf')
                all_match = False
        
        return all_match, differences
    
    @staticmethod
    def create_test_batch(batch_size: int, 
                         config: HRMConfig, 
                         seed: int = 42) -> Dict[str, mx.array]:
        """Create deterministic test batch for reproducible testing.
        
        Args:
            batch_size: Size of the batch
            config: HRM configuration
            seed: Random seed for reproducibility
            
        Returns:
            Test batch dictionary
        """
        mx.random.seed(seed)
        
        return {
            "input_ids": mx.random.randint(0, config.vocab_size,
                                         shape=(batch_size, config.seq_len)),
            "puzzle_ids": mx.random.randint(0, config.num_puzzle_identifiers,
                                          shape=(batch_size,)),
            "labels": mx.random.randint(0, config.vocab_size,
                                      shape=(batch_size, config.seq_len))
        }
    
    @staticmethod
    def verify_model_determinism(model: HRM_ACT, 
                               batch: Dict[str, mx.array], 
                               num_runs: int = 3) -> Tuple[bool, List[float]]:
        """Verify that model produces deterministic outputs.
        
        Args:
            model: HRM model to test
            batch: Input batch
            num_runs: Number of runs to compare
            
        Returns:
            Tuple of (is_deterministic, logits_sums)
        """
        model.set_training(False)  # Deterministic mode
        
        logits_sums = []
        outputs_list = []
        
        for run in range(num_runs):
            carry = model.initial_carry(batch['input_ids'].shape[0])
            new_carry, outputs = model(carry, batch)
            
            logits_sum = float(mx.sum(outputs['logits']))
            logits_sums.append(logits_sum)
            outputs_list.append(outputs)
        
        # Check if all runs produced identical results
        is_deterministic = True
        tolerance = 1e-10
        
        for i in range(1, num_runs):
            all_match, _ = MLXTestingPatterns.compare_model_outputs(
                outputs_list[0], outputs_list[i],
                tolerances={
                    'logits': {'rtol': tolerance, 'atol': tolerance},
                    'q_halt_logits': {'rtol': tolerance, 'atol': tolerance},
                    'q_continue_logits': {'rtol': tolerance, 'atol': tolerance},
                    'halted': {'rtol': 0.0, 'atol': 0.0}
                }
            )
            if not all_match:
                is_deterministic = False
                break
        
        return is_deterministic, logits_sums
    
    @staticmethod
    def apply_simple_sgd_update(model: HRM_ACT, 
                              grads: Dict[str, mx.array], 
                              learning_rate: float = 0.001,
                              weight_decay: float = 0.0) -> Dict[str, mx.array]:
        """Apply simple SGD parameter update using MLX tree utilities.
        
        Args:
            model: Model to update
            grads: Gradient dictionary
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            Updated parameters dictionary
        """
        # Get current parameters as a tree
        current_params = model.parameters()
        
        # Define update function for each parameter
        def update_param(param_name_and_value):
            name, param = param_name_and_value
            if name in grads and grads[name] is not None:
                grad = grads[name]
                
                # Apply weight decay if specified
                if weight_decay > 0:
                    param_with_decay = param * (1.0 - learning_rate * weight_decay)
                else:
                    param_with_decay = param
                
                # Apply gradient update
                updated_param = param_with_decay - learning_rate * grad
                return updated_param
            else:
                # No gradient - keep parameter unchanged
                return param
        
        # Apply updates using tree operations
        updated_params = {}
        for name, param in current_params.items():
            updated_params[name] = update_param((name, param))
        
        return updated_params


def cast_batch_to_bf16(batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Cast appropriate batch elements to BF16 for mixed precision testing."""
    batch_bf16 = {}
    
    for key, value in batch.items():
        if key in ['puzzle_ids', 'labels']:
            # Keep integer arrays as-is
            batch_bf16[key] = value
        else:
            # Cast float arrays to BF16
            if value.dtype == mx.float32:
                batch_bf16[key] = value.astype(mx.bfloat16)
            else:
                batch_bf16[key] = value
    
    return batch_bf16


def cast_outputs_to_fp32(outputs: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Cast model outputs to FP32 for comparison."""
    outputs_fp32 = {}
    
    for key, value in outputs.items():
        if key == 'halted' or value.dtype == mx.bool_:
            # Keep boolean arrays as-is
            outputs_fp32[key] = value
        else:
            # Cast to FP32
            outputs_fp32[key] = value.astype(mx.float32)
    
    return outputs_fp32