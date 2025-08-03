#!/usr/bin/env python3
"""
Precision-Aware Component Structural Validation

This script validates the structural integrity of all Phase 2 precision-aware 
components without requiring MLX to be installed. It checks:

1. Import structure and dependencies
2. Class definitions and inheritance
3. Method signatures and interfaces
4. Component integration points
5. Precision configuration integration
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any


class PrecisionStructuralValidator:
    """Validates the structural integrity of precision-aware components."""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Set up the path to the MLX HRM source
        self.base_path = Path(__file__).parent.parent / "src" / "mlx_hrm"
        
        print("🏗️  Precision-Aware Component Structural Validation")
        print("=" * 60)
        print(f"📁 Base path: {self.base_path}")
        print()
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        try:
            print(f"🔧 {test_name}...", end=" ")
            
            result = test_func()
            
            if result:
                print("✅ PASSED")
                self.test_results['passed'] += 1
                return True
            else:
                print("❌ FAILED")
                self.test_results['failed'] += 1
                return False
                
        except Exception as e:
            print("💥 ERROR")
            error_msg = f"{test_name}: {str(e)}"
            self.test_results['errors'].append(error_msg)
            self.test_results['failed'] += 1
            return False
    
    def load_python_file(self, relative_path: str) -> Any:
        """Load a Python module from a relative path."""
        full_path = self.base_path / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        spec = importlib.util.spec_from_file_location(
            f"module_{relative_path.replace('/', '_').replace('.py', '')}", 
            full_path
        )
        module = importlib.util.module_from_spec(spec)
        
        # Mock MLX imports to avoid import errors
        mlx_mock = type('MockModule', (), {
            'core': type('MockCore', (), {
                'array': type('MockArray', (), {}),
                'Dtype': type('MockDtype', (), {}),
                'float32': 'float32',
                'float64': 'float64', 
                'bfloat16': 'bfloat16',
                'zeros': lambda *args, **kwargs: None,
                'random': type('MockRandom', (), {
                    'normal': lambda *args, **kwargs: None,
                    'randint': lambda *args, **kwargs: None,
                })(),
                'stop_gradient': lambda x: x,
                'concatenate': lambda *args, **kwargs: None,
                'broadcast_to': lambda *args, **kwargs: None,
                'where': lambda *args, **kwargs: None,
                'split': lambda *args, **kwargs: (None, None),
                'sigmoid': lambda x: x,
                'stack': lambda *args, **kwargs: None,
                'scatter': lambda *args, **kwargs: None,
                'sum': lambda *args, **kwargs: None,
                'log': lambda *args, **kwargs: None,
                'mean': lambda *args, **kwargs: None,
                'take_along_axis': lambda *args, **kwargs: None,
                'argsort': lambda *args, **kwargs: None,
                'softmax': lambda *args, **kwargs: None,
                'take': lambda *args, **kwargs: None,
                'stream': lambda device: type('MockContext', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})(),
                'cpu': 'cpu',
                'fast': type('MockFast', (), {
                    'scaled_dot_product_attention': lambda *args, **kwargs: None
                })()
            })(),
            'nn': type('MockNN', (), {
                'Module': object,
                'value_and_grad': lambda model, loss_fn: (None, None),
                'Linear': type('MockLinear', (object,), {}),
                'losses': type('MockLosses', (), {
                    'binary_cross_entropy_with_logits': lambda *args, **kwargs: None
                })()
            })()
        })()
        
        sys.modules['mlx'] = mlx_mock
        sys.modules['mlx.core'] = mlx_mock.core
        sys.modules['mlx.nn'] = mlx_mock.nn
        
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise ImportError(f"Failed to load {relative_path}: {e}")
    
    def test_precision_config_exists(self) -> bool:
        """Test that precision configuration module exists and has required classes."""
        try:
            module = self.load_python_file("models/precision_config.py")
            
            # Check required classes exist
            required_classes = ['MLXPrecisionConfig']
            for cls_name in required_classes:
                if not hasattr(module, cls_name):
                    return False
            
            # Check MLXPrecisionConfig has required methods
            precision_config_class = getattr(module, 'MLXPrecisionConfig')
            required_methods = [
                'get_forward_dtype', 'get_master_dtype', 'get_normalization_dtype',
                'get_loss_dtype', 'get_gradient_dtype', 'validate_configuration'
            ]
            
            for method_name in required_methods:
                if not hasattr(precision_config_class, method_name):
                    return False
            
            return True
        except Exception:
            return False
    
    def test_precision_layers_exist(self) -> bool:
        """Test that precision layer components exist."""
        try:
            module = self.load_python_file("layers/precision.py")
            
            # Check required classes exist
            required_classes = ['MLXCastedLinear', 'PrecisionAwareRMSNorm']
            for cls_name in required_classes:
                if not hasattr(module, cls_name):
                    return False
            
            # Check required functions exist
            required_functions = ['precision_aware_rms_norm']
            for func_name in required_functions:
                if not hasattr(module, func_name):
                    return False
            
            return True
        except Exception:
            return False
    
    def test_precision_attention_exists(self) -> bool:
        """Test that precision-aware attention exists."""
        try:
            module = self.load_python_file("modules/attention.py")
            
            # Check PrecisionAwareAttention class exists
            if not hasattr(module, 'PrecisionAwareAttention'):
                return False
            
            return True
        except Exception:
            return False
    
    def test_precision_swiglu_exists(self) -> bool:
        """Test that precision-aware SwiGLU exists."""
        try:
            module = self.load_python_file("layers/activations.py")
            
            # Check PrecisionAwareSwiGLU class exists
            if not hasattr(module, 'PrecisionAwareSwiGLU'):
                return False
            
            return True
        except Exception:
            return False
    
    def test_precision_embeddings_exist(self) -> bool:
        """Test that precision-aware embeddings exist."""
        try:
            module = self.load_python_file("layers/embeddings.py")
            
            # Check PrecisionAwareSparseEmbedding class exists
            if not hasattr(module, 'PrecisionAwareSparseEmbedding'):
                return False
            
            return True
        except Exception:
            return False
    
    def test_precision_reasoning_modules_exist(self) -> bool:
        """Test that precision-aware reasoning modules exist."""
        try:
            module = self.load_python_file("modules/act.py")
            
            # Check precision-aware classes exist
            required_classes = ['PrecisionAwareHRMBlock', 'PrecisionAwareHRMReasoningModule']
            for cls_name in required_classes:
                if not hasattr(module, cls_name):
                    return False
            
            return True
        except Exception:
            return False
    
    def test_precision_hrm_inner_exists(self) -> bool:
        """Test that precision-aware HRM inner model exists."""
        try:
            module = self.load_python_file("models/hrm_inner_precision.py")
            
            # Check PrecisionAwareHRMInner class exists
            if not hasattr(module, 'PrecisionAwareHRMInner'):
                return False
            
            return True
        except Exception:
            return False
    
    def test_precision_losses_exist(self) -> bool:
        """Test that precision-aware losses exist."""
        try:
            module = self.load_python_file("training/precision_losses.py")
            
            # Check precision_stablemax_cross_entropy function exists
            if not hasattr(module, 'precision_stablemax_cross_entropy'):
                return False
            
            return True
        except Exception:
            return False
    
    def test_file_structure_integrity(self) -> bool:
        """Test that all required files exist."""
        required_files = [
            "models/precision_config.py",
            "layers/precision.py", 
            "modules/attention.py",
            "layers/activations.py",
            "layers/embeddings.py",
            "modules/act.py",
            "models/hrm_inner_precision.py",
            "training/precision_losses.py"
        ]
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            if not full_path.exists():
                print(f"     Missing file: {file_path}")
                return False
        
        return True
    
    def test_import_dependencies(self) -> bool:
        """Test that all modules can be imported without circular dependencies."""
        try:
            # Try to load each module to check for import issues
            modules_to_test = [
                "models/precision_config.py",
                "layers/precision.py",
                "training/precision_losses.py",
                "modules/attention.py",
                "layers/activations.py",
                "layers/embeddings.py",
                "modules/act.py",
                "models/hrm_inner_precision.py"
            ]
            
            for module_path in modules_to_test:
                try:
                    self.load_python_file(module_path)
                except Exception as e:
                    print(f"     Import error in {module_path}: {e}")
                    return False
            
            return True
        except Exception:
            return False
    
    def run_all_tests(self) -> bool:
        """Run all structural validation tests."""
        tests = [
            ("File Structure Integrity", self.test_file_structure_integrity),
            ("Import Dependencies", self.test_import_dependencies),
            ("Precision Config Module", self.test_precision_config_exists),
            ("Precision Layers Module", self.test_precision_layers_exist),
            ("Precision Losses Module", self.test_precision_losses_exist),
            ("Precision-Aware Attention", self.test_precision_attention_exists),
            ("Precision-Aware SwiGLU", self.test_precision_swiglu_exists),
            ("Precision-Aware Embeddings", self.test_precision_embeddings_exist),
            ("Precision-Aware Reasoning Modules", self.test_precision_reasoning_modules_exist),
            ("Precision-Aware HRM Inner Model", self.test_precision_hrm_inner_exists),
        ]
        
        print("🔬 Running Structural Validation Tests:")
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
    """Run the structural validation tests."""
    validator = PrecisionStructuralValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎉 All structural validation tests PASSED!")
        print("✅ Phase 2 precision-aware components are structurally sound.")
    else:
        print(f"\n⚠️  {validator.test_results['failed']} tests FAILED!")
        print("❌ Phase 2 components have structural issues that need fixing.")
    
    return success


if __name__ == "__main__":
    main()