"""PyTorch-MLX Bridge for Behavioral Compliance Testing.

This module provides utilities to run identical computations on both PyTorch
and MLX frameworks to validate behavioral equivalence with the reference HRM implementation.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add the HRM PyTorch implementation to path
HRM_PATH = Path(__file__).parent.parent.parent.parent / "HRM"
if HRM_PATH.exists():
    sys.path.insert(0, str(HRM_PATH))

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. PyTorch reference comparisons will be skipped.")

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. MLX comparisons will be skipped.")


class PyTorchMLXBridge:
    """Bridge class for comparing PyTorch and MLX computations."""
    
    def __init__(self):
        """Initialize the bridge with availability checks."""
        self.pytorch_available = PYTORCH_AVAILABLE
        self.mlx_available = MLX_AVAILABLE
        
        if not self.pytorch_available:
            print("Warning: PyTorch not available - reference comparisons disabled")
        if not self.mlx_available:
            print("Warning: MLX not available - MLX comparisons disabled")
    
    def requires_pytorch(self):
        """Decorator to skip tests when PyTorch is not available."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.pytorch_available:
                    print(f"Skipping {func.__name__}: PyTorch not available")
                    return None
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def requires_mlx(self):
        """Decorator to skip tests when MLX is not available."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.mlx_available:
                    print(f"Skipping {func.__name__}: MLX not available")
                    return None
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def numpy_to_torch(self, arr: np.ndarray) -> 'torch.Tensor':
        """Convert numpy array to PyTorch tensor."""
        if not self.pytorch_available:
            return None
        return torch.from_numpy(arr)
    
    def numpy_to_mlx(self, arr: np.ndarray) -> 'mx.array':
        """Convert numpy array to MLX array."""
        if not self.mlx_available:
            return None
        return mx.array(arr)
    
    def torch_to_numpy(self, tensor: 'torch.Tensor') -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        if not self.pytorch_available:
            return None
        return tensor.detach().cpu().numpy()
    
    def mlx_to_numpy(self, array: 'mx.array') -> np.ndarray:
        """Convert MLX array to numpy array."""
        if not self.mlx_available:
            return None
        return np.array(array)
    
    def compare_activations(
        self, 
        input_data: np.ndarray,
        pytorch_fn,
        mlx_fn,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        name: str = "activation"
    ) -> Dict[str, Any]:
        """Compare activation functions between PyTorch and MLX.
        
        Args:
            input_data: Input data as numpy array
            pytorch_fn: PyTorch activation function
            mlx_fn: MLX activation function  
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            name: Name of the activation for reporting
            
        Returns:
            Dictionary with comparison results
        """
        if not (self.pytorch_available and self.mlx_available):
            return {"skipped": True, "reason": "PyTorch or MLX not available"}
        
        # Convert input to both frameworks
        torch_input = self.numpy_to_torch(input_data)
        mlx_input = self.numpy_to_mlx(input_data)
        
        # Run computations
        torch_output = pytorch_fn(torch_input)
        mlx_output = mlx_fn(mlx_input)
        
        # Convert back to numpy for comparison
        torch_result = self.torch_to_numpy(torch_output)
        mlx_result = self.mlx_to_numpy(mlx_output)
        
        # Compare results
        are_close = np.allclose(torch_result, mlx_result, rtol=rtol, atol=atol)
        max_abs_diff = np.max(np.abs(torch_result - mlx_result))
        max_rel_diff = np.max(np.abs((torch_result - mlx_result) / (np.abs(torch_result) + 1e-10)))
        
        return {
            "name": name,
            "shapes_match": torch_result.shape == mlx_result.shape,
            "values_close": are_close,
            "max_abs_diff": float(max_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "torch_shape": torch_result.shape,
            "mlx_shape": mlx_result.shape,
            "input_shape": input_data.shape,
            "rtol": rtol,
            "atol": atol
        }
    
    def compare_silu_implementations(
        self,
        test_values: np.ndarray,
        rtol: float = 1e-5,  # Standard ML tolerance for cross-framework comparison
        atol: float = 5e-7   # Account for implementation differences (~1e-6 range)
    ) -> Dict[str, Any]:
        """Compare SiLU implementations between PyTorch and MLX.
        
        Args:
            test_values: Test input values as numpy array
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Comparison results dictionary
        """
        if not (self.pytorch_available and self.mlx_available):
            return {"skipped": True, "reason": "PyTorch or MLX not available"}
        
        # Define activation functions
        def pytorch_silu(x):
            return F.silu(x)
        
        def mlx_manual_silu(x):
            return x * mx.sigmoid(x)
        
        def mlx_native_silu(x):
            return nn.silu(x)
        
        # Compare PyTorch vs MLX manual
        manual_comparison = self.compare_activations(
            test_values, pytorch_silu, mlx_manual_silu, 
            rtol=rtol, atol=atol, name="PyTorch vs MLX Manual SiLU"
        )
        
        # Compare PyTorch vs MLX native
        native_comparison = self.compare_activations(
            test_values, pytorch_silu, mlx_native_silu,
            rtol=rtol, atol=atol, name="PyTorch vs MLX Native SiLU"
        )
        
        # Compare MLX manual vs native (should be identical)
        # Need to handle this differently since both are MLX functions
        mlx_input = self.numpy_to_mlx(test_values)
        mlx_manual_result = self.mlx_to_numpy(mlx_manual_silu(mlx_input))
        mlx_native_result = self.mlx_to_numpy(mlx_native_silu(mlx_input))
        
        are_close = np.allclose(mlx_manual_result, mlx_native_result, rtol=1e-15, atol=1e-15)
        max_abs_diff = np.max(np.abs(mlx_manual_result - mlx_native_result))
        max_rel_diff = np.max(np.abs((mlx_manual_result - mlx_native_result) / (np.abs(mlx_manual_result) + 1e-10)))
        
        mlx_internal_comparison = {
            "name": "MLX Manual vs Native SiLU",
            "shapes_match": mlx_manual_result.shape == mlx_native_result.shape,
            "values_close": are_close,
            "max_abs_diff": float(max_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "torch_shape": mlx_manual_result.shape,
            "mlx_shape": mlx_native_result.shape,
            "input_shape": test_values.shape,
            "rtol": 1e-15,
            "atol": 1e-15
        }
        
        return {
            "pytorch_vs_mlx_manual": manual_comparison,
            "pytorch_vs_mlx_native": native_comparison, 
            "mlx_manual_vs_native": mlx_internal_comparison,
            "all_equivalent": (
                manual_comparison.get("values_close", False) and
                native_comparison.get("values_close", False) and
                mlx_internal_comparison.get("values_close", False)
            )
        }
    
    def run_comprehensive_silu_test(self) -> Dict[str, Any]:
        """Run comprehensive SiLU behavioral compliance test.
        
        Returns:
            Complete test results
        """
        print("Running Comprehensive SiLU Behavioral Compliance Test")
        print("=" * 60)
        
        if not (self.pytorch_available and self.mlx_available):
            return {"skipped": True, "reason": "PyTorch or MLX not available"}
        
        # Test cases with different value ranges and shapes
        test_cases = [
            {
                "name": "Edge Values",
                "data": np.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0], dtype=np.float32),
                "expected_behavior": "Handle edge cases correctly"
            },
            {
                "name": "Random Small",
                "data": np.random.normal(0, 1, (10, 8)).astype(np.float32),
                "expected_behavior": "Standard random values"
            },
            {
                "name": "Random Large", 
                "data": np.random.normal(0, 1, (100, 256)).astype(np.float32),
                "expected_behavior": "Large tensor processing"
            },
            {
                "name": "Extreme Values",
                "data": np.array([1e-6, -1e-6, 1e6, -1e6], dtype=np.float32),
                "expected_behavior": "Numerical stability at extremes"
            },
            {
                "name": "Multidimensional",
                "data": np.random.normal(0, 1, (2, 4, 8, 16)).astype(np.float32),
                "expected_behavior": "Handle complex tensor shapes"
            }
        ]
        
        results = {}
        all_passed = True
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            print(f"Shape: {test_case['data'].shape}")
            print(f"Expected: {test_case['expected_behavior']}")
            
            comparison = self.compare_silu_implementations(test_case['data'])
            results[test_case['name'].lower().replace(" ", "_")] = comparison
            
            if not comparison.get("skipped", False):
                pytorch_mlx_manual_ok = comparison["pytorch_vs_mlx_manual"]["values_close"]
                pytorch_mlx_native_ok = comparison["pytorch_vs_mlx_native"]["values_close"]
                mlx_internal_ok = comparison["mlx_manual_vs_native"]["values_close"]
                
                print(f"  PyTorch vs MLX Manual: {'✅' if pytorch_mlx_manual_ok else '❌'}")
                print(f"  PyTorch vs MLX Native: {'✅' if pytorch_mlx_native_ok else '❌'}")
                print(f"  MLX Internal Consistency: {'✅' if mlx_internal_ok else '❌'}")
                
                if not (pytorch_mlx_manual_ok and pytorch_mlx_native_ok and mlx_internal_ok):
                    all_passed = False
                    print(f"  Max abs diff (PyTorch vs MLX Manual): {comparison['pytorch_vs_mlx_manual']['max_abs_diff']:.2e}")
                    print(f"  Max abs diff (PyTorch vs MLX Native): {comparison['pytorch_vs_mlx_native']['max_abs_diff']:.2e}")
            else:
                print(f"  Skipped: {comparison['reason']}")
        
        print(f"\n{'='*60}")
        print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        results["summary"] = {
            "all_passed": all_passed,
            "pytorch_available": self.pytorch_available,
            "mlx_available": self.mlx_available,
            "test_count": len(test_cases)
        }
        
        return results


# Global bridge instance
bridge = PyTorchMLXBridge()


def run_silu_pytorch_compliance_test():
    """Convenience function to run SiLU PyTorch compliance test."""
    return bridge.run_comprehensive_silu_test()


if __name__ == "__main__":
    # Run the comprehensive test when called directly
    results = run_silu_pytorch_compliance_test()
    
    if results.get("summary", {}).get("all_passed", False):
        print("\n🎉 SiLU implementations are behaviorally equivalent to PyTorch reference!")
    elif results.get("skipped", False):
        print(f"\n⚠️  Test skipped: {results['reason']}")
    else:
        print("\n⚠️  SiLU behavioral differences detected - review results above")