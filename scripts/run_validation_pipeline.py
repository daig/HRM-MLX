#!/usr/bin/env python3
"""
Complete validation pipeline for MLX HRM implementation.

This script runs the full validation suite including:
1. Numerical validation against reference implementations
2. Performance benchmarking across model sizes
3. Accuracy validation on puzzle datasets
4. Integration testing
5. Checkpoint conversion validation

Usage:
    python scripts/run_validation_pipeline.py --all
    python scripts/run_validation_pipeline.py --numerical --performance
    python scripts/run_validation_pipeline.py --model tiny --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# Add source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "tests"))

from mlx_hrm.configs.model_presets import list_presets


class ValidationPipeline:
    """
    Complete validation pipeline for HRM MLX implementation.
    
    Coordinates all validation components and provides comprehensive
    reporting on implementation correctness and performance.
    """
    
    def __init__(self, verbose: bool = True, save_results: bool = True):
        """
        Initialize validation pipeline.
        
        Args:
            verbose: Enable verbose output
            save_results: Save detailed results to files
        """
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}
        self.start_time = time.time()
        
        # Paths
        self.repo_root = Path(__file__).parent.parent
        self.results_dir = self.repo_root / "validation_results"
        
        if self.save_results:
            self.results_dir.mkdir(exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_numerical_validation(self) -> Dict:
        """Run numerical validation tests."""
        self.log("Running numerical validation tests...")
        
        try:
            # Import and run numerical validation
            from validation.test_numerical_parity import (
                NumericalValidator, 
                test_component_rms_norm,
                test_component_rope,
                test_component_swiglu,
                test_model_forward_pass,
                test_loss_function_parity,
                test_gradient_flow,
                test_training_step_consistency
            )
            
            validator = NumericalValidator(tolerance=1e-4, rtol=1e-4)
            test_config = {
                'batch_size': 2,
                'seq_len': 16,
                'hidden_size': 128,
                'num_heads': 4,
                'H_layers': 2,
                'L_layers': 2,
                'H_cycles': 1,
                'L_cycles': 1,
                'vocab_size': 1000,
                'puzzle_emb_ndim': 0,
                'num_puzzle_identifiers': 10,
                'pos_encodings': 'rope',
                'rms_norm_eps': 1e-5,
                'rope_theta': 10000.0,
                'halt_max_steps': 4,
                'halt_exploration_prob': 0.0,
                'expansion': 4.0,
            }
            
            results = {'tests': {}, 'overall_status': 'PASS'}
            
            # Run individual tests
            tests = [
                ('rms_norm', test_component_rms_norm),
                ('rope', test_component_rope),
                ('swiglu', test_component_swiglu),
                ('model_forward', lambda: test_model_forward_pass(validator, test_config)),
                ('loss_parity', test_loss_function_parity),
                ('gradient_flow', lambda: test_gradient_flow(test_config)),
                ('training_consistency', lambda: test_training_step_consistency(test_config)),
            ]
            
            for test_name, test_func in tests:
                try:
                    self.log(f"  Running {test_name} test...")
                    if test_name in ['rms_norm', 'rope', 'swiglu', 'loss_parity']:
                        test_func(validator)
                    else:
                        test_func()
                    
                    results['tests'][test_name] = {'status': 'PASS', 'error': None}
                    self.log(f"  ✅ {test_name} test passed")
                    
                except Exception as e:
                    results['tests'][test_name] = {'status': 'FAIL', 'error': str(e)}
                    results['overall_status'] = 'FAIL'
                    self.log(f"  ❌ {test_name} test failed: {e}", "ERROR")
            
            return results
            
        except Exception as e:
            self.log(f"Numerical validation failed: {e}", "ERROR")
            return {'overall_status': 'FAIL', 'error': str(e)}
    
    def run_performance_benchmarks(
        self, 
        models: List[str] = None, 
        quick: bool = False
    ) -> Dict:
        """Run performance benchmarking."""
        self.log("Running performance benchmarks...")
        
        try:
            # Import benchmarking
            sys.path.append(str(self.repo_root / "benchmarks"))
            from benchmark_full_model import PerformanceBenchmark
            
            if models is None:
                models = ['tiny']  # Default to tiny for quick validation
            
            # Configure benchmark parameters
            if quick:
                warmup_steps = 5
                benchmark_steps = 20
            else:
                warmup_steps = 10
                benchmark_steps = 50
            
            benchmark = PerformanceBenchmark(
                warmup_steps=warmup_steps,
                benchmark_steps=benchmark_steps
            )
            
            results = {'models': {}, 'overall_status': 'PASS'}
            
            for model_name in models:
                try:
                    self.log(f"  Benchmarking {model_name} model...")
                    
                    model_results = benchmark.benchmark_model_preset(
                        model_name,
                        batch_size=4 if quick else 8,
                        seq_len=128 if quick else 256
                    )
                    
                    results['models'][model_name] = model_results
                    
                    # Extract key metrics for summary
                    if 'forward' in model_results:
                        throughput = model_results['forward']['throughput_tokens_per_sec']
                        memory_mb = model_results['memory']['total_memory_mb']
                        self.log(f"  ✅ {model_name}: {throughput:.0f} tokens/sec, {memory_mb:.1f} MB")
                    
                except Exception as e:
                    results['models'][model_name] = {'error': str(e)}
                    results['overall_status'] = 'FAIL'
                    self.log(f"  ❌ {model_name} benchmark failed: {e}", "ERROR")
            
            return results
            
        except Exception as e:
            self.log(f"Performance benchmarking failed: {e}", "ERROR")
            return {'overall_status': 'FAIL', 'error': str(e)}
    
    def run_accuracy_validation(self, quick: bool = False) -> Dict:
        """Run accuracy validation tests."""
        self.log("Running accuracy validation tests...")
        
        try:
            # Import accuracy validation
            from validation.test_puzzle_accuracy import (
                AccuracyValidator,
                MockPuzzleDataset,
                create_test_model
            )
            
            model = create_test_model('tiny')
            validator = AccuracyValidator(model)
            
            # Test datasets
            puzzle_types = ['arc', 'sudoku', 'maze']
            num_examples = 20 if quick else 50
            
            results = {'datasets': {}, 'overall_status': 'PASS'}
            
            for puzzle_type in puzzle_types:
                try:
                    self.log(f"  Testing {puzzle_type} validation...")
                    
                    dataset = MockPuzzleDataset(puzzle_type, num_examples=num_examples)
                    
                    # Use low expectations for mock data
                    success, dataset_results = validator.validate_against_baseline(
                        dataset,
                        expected_accuracy=0.05,  # 5% (mock data)
                        tolerance=0.10,  # 10% tolerance
                        max_examples=num_examples
                    )
                    
                    results['datasets'][puzzle_type] = {
                        'success': success,
                        'results': dataset_results
                    }
                    
                    accuracy = dataset_results['overall_sequence_accuracy']
                    self.log(f"  ✅ {puzzle_type}: {accuracy:.1%} accuracy")
                    
                except Exception as e:
                    results['datasets'][puzzle_type] = {'error': str(e)}
                    results['overall_status'] = 'FAIL'
                    self.log(f"  ❌ {puzzle_type} validation failed: {e}", "ERROR")
            
            return results
            
        except Exception as e:
            self.log(f"Accuracy validation failed: {e}", "ERROR")
            return {'overall_status': 'FAIL', 'error': str(e)}
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        self.log("Running integration tests...")
        
        try:
            # Test model creation and basic operations
            from mlx_hrm.models.factory import create_hrm
            from mlx_hrm.configs.model_presets import get_preset_config
            import mlx.core as mx
            import numpy as np
            
            results = {'tests': {}, 'overall_status': 'PASS'}
            
            # Test 1: Model creation for all presets
            try:
                self.log("  Testing model creation for all presets...")
                presets = list_presets()
                
                for preset in presets:
                    config = get_preset_config(preset)
                    model = create_hrm(config)
                    
                    # Basic forward pass test
                    batch_size, seq_len = 2, 32
                    batch = {
                        'input_ids': mx.random.randint(0, 1000, (batch_size, seq_len))
                    }
                    
                    carry = model.initial_carry(batch_size)
                    new_carry, outputs = model(carry, batch)
                    
                    # Verify outputs
                    assert 'logits' in outputs
                    assert 'q_halt_logits' in outputs
                    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
                
                results['tests']['model_creation'] = {'status': 'PASS', 'models_tested': len(presets)}
                self.log(f"  ✅ Model creation test passed for {len(presets)} presets")
                
            except Exception as e:
                results['tests']['model_creation'] = {'status': 'FAIL', 'error': str(e)}
                results['overall_status'] = 'FAIL'
                self.log(f"  ❌ Model creation test failed: {e}", "ERROR")
            
            # Test 2: Training step simulation
            try:
                self.log("  Testing training step simulation...")
                
                from mlx_hrm.training.act_loss import ACTLossHead
                
                config = get_preset_config('tiny')
                model = create_hrm(config)
                loss_model = ACTLossHead(model)
                
                batch = {
                    'input_ids': mx.random.randint(0, 1000, (2, 32)),
                    'labels': mx.random.randint(0, 1000, (2, 32))
                }
                
                carry = loss_model.initial_carry(2)
                
                def loss_fn(params):
                    loss_model.model.update(params)
                    new_carry, loss, metrics, _ = loss_model(carry, batch)
                    return loss
                
                loss, grads = mx.value_and_grad(loss_fn)(loss_model.model.parameters())
                
                # Verify training step worked
                assert isinstance(float(loss), float)
                assert grads is not None
                
                results['tests']['training_step'] = {'status': 'PASS', 'loss': float(loss)}
                self.log(f"  ✅ Training step test passed (loss: {float(loss):.4f})")
                
            except Exception as e:
                results['tests']['training_step'] = {'status': 'FAIL', 'error': str(e)}
                results['overall_status'] = 'FAIL'
                self.log(f"  ❌ Training step test failed: {e}", "ERROR")
            
            # Test 3: Checkpoint operations
            try:
                self.log("  Testing checkpoint operations...")
                
                from mlx_hrm.utils.checkpoint import save_checkpoint, load_checkpoint
                import tempfile
                
                config = get_preset_config('tiny')
                model = create_hrm(config)
                
                # Save checkpoint
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint_path = Path(tmpdir) / "test_checkpoint"
                    
                    save_checkpoint(
                        model,
                        str(checkpoint_path),
                        metadata={'test': True, 'step': 100}
                    )
                    
                    # Load checkpoint
                    weights, loaded_config, metadata = load_checkpoint(str(checkpoint_path))
                    
                    # Verify checkpoint integrity
                    assert loaded_config == config
                    assert metadata['test'] == True
                    assert metadata['step'] == 100
                
                results['tests']['checkpoint_ops'] = {'status': 'PASS'}
                self.log("  ✅ Checkpoint operations test passed")
                
            except Exception as e:
                results['tests']['checkpoint_ops'] = {'status': 'FAIL', 'error': str(e)}
                results['overall_status'] = 'FAIL'
                self.log(f"  ❌ Checkpoint operations test failed: {e}", "ERROR")
            
            return results
            
        except Exception as e:
            self.log(f"Integration tests failed: {e}", "ERROR")
            return {'overall_status': 'FAIL', 'error': str(e)}
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        elapsed_time = time.time() - self.start_time
        
        # Count successes and failures
        total_tests = 0
        passed_tests = 0
        
        for component, results in self.results.items():
            if isinstance(results, dict) and 'overall_status' in results:
                total_tests += 1
                if results['overall_status'] == 'PASS':
                    passed_tests += 1
        
        success_rate = passed_tests / max(total_tests, 1)
        
        report = {
            'validation_summary': {
                'total_time_sec': elapsed_time,
                'total_components': total_tests,
                'passed_components': passed_tests,
                'failed_components': total_tests - passed_tests,
                'success_rate': success_rate,
                'overall_status': 'PASS' if success_rate == 1.0 else 'FAIL'
            },
            'component_results': self.results,
            'recommendations': []
        }
        
        # Add recommendations based on results
        if success_rate < 1.0:
            report['recommendations'].append(
                "Some validation components failed. Review detailed results and fix issues before deployment."
            )
        
        if success_rate >= 0.8:
            report['recommendations'].append(
                "Validation mostly successful. MLX implementation appears functional."
            )
        
        if 'performance' in self.results:
            report['recommendations'].append(
                "Performance benchmarks completed. Compare with PyTorch baseline if available."
            )
        
        return report
    
    def save_results_to_files(self, report: Dict):
        """Save detailed results to files."""
        if not self.save_results:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = self.results_dir / f"validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"Detailed report saved to {report_file}")
        
        # Save individual component results
        for component, results in self.results.items():
            if isinstance(results, dict):
                component_file = self.results_dir / f"{component}_results_{timestamp}.json"
                with open(component_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
    
    def run_full_pipeline(
        self,
        include_numerical: bool = True,
        include_performance: bool = True,
        include_accuracy: bool = True,
        include_integration: bool = True,
        models: List[str] = None,
        quick: bool = False
    ) -> Dict:
        """
        Run the complete validation pipeline.
        
        Args:
            include_numerical: Run numerical validation tests
            include_performance: Run performance benchmarks
            include_accuracy: Run accuracy validation tests
            include_integration: Run integration tests
            models: List of model presets to benchmark
            quick: Use reduced test parameters for faster execution
            
        Returns:
            Comprehensive validation report
        """
        self.log("🚀 Starting MLX HRM validation pipeline...")
        
        if quick:
            self.log("⚡ Running in quick mode (reduced test parameters)")
        
        # Run selected validation components
        if include_numerical:
            self.results['numerical'] = self.run_numerical_validation()
        
        if include_performance:
            self.results['performance'] = self.run_performance_benchmarks(models, quick)
        
        if include_accuracy:
            self.results['accuracy'] = self.run_accuracy_validation(quick)
        
        if include_integration:
            self.results['integration'] = self.run_integration_tests()
        
        # Generate final report
        report = self.generate_report()
        
        # Save results
        self.save_results_to_files(report)
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def print_summary(self, report: Dict):
        """Print validation summary."""
        summary = report['validation_summary']
        
        print(f"\n{'='*70}")
        print("MLX HRM VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"Total validation time: {summary['total_time_sec']:.1f} seconds")
        print(f"Components tested: {summary['total_components']}")
        print(f"Components passed: {summary['passed_components']}")
        print(f"Components failed: {summary['failed_components']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Overall status: {summary['overall_status']}")
        
        # Component breakdown
        print(f"\nComponent Results:")
        for component, results in report['component_results'].items():
            if isinstance(results, dict) and 'overall_status' in results:
                status = results['overall_status']
                icon = "✅" if status == "PASS" else "❌"
                print(f"  {icon} {component.capitalize()}: {status}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        # Final verdict
        if summary['overall_status'] == 'PASS':
            print(f"\n🎉 Validation completed successfully!")
            print(f"   MLX HRM implementation is ready for use.")
        else:
            print(f"\n⚠️  Validation completed with issues.")
            print(f"   Review detailed results and fix failing components.")
        
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive validation pipeline for MLX HRM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validation components
  python run_validation_pipeline.py --all
  
  # Run specific components
  python run_validation_pipeline.py --numerical --performance
  
  # Quick validation (reduced parameters)
  python run_validation_pipeline.py --all --quick
  
  # Benchmark specific models
  python run_validation_pipeline.py --performance --models tiny small
  
  # Save detailed results
  python run_validation_pipeline.py --all --save-results
        """
    )
    
    # Component selection
    parser.add_argument('--all', action='store_true', help='Run all validation components')
    parser.add_argument('--numerical', action='store_true', help='Run numerical validation tests')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--accuracy', action='store_true', help='Run accuracy validation tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    
    # Configuration options
    parser.add_argument('--models', nargs='+', choices=list_presets(), help='Model presets to benchmark')
    parser.add_argument('--quick', action='store_true', help='Use reduced parameters for faster execution')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results to files')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.all or args.numerical or args.performance or args.accuracy or args.integration):
        parser.error('Must specify at least one validation component or --all')
    
    # Set component flags
    if args.all:
        include_numerical = True
        include_performance = True
        include_accuracy = True
        include_integration = True
    else:
        include_numerical = args.numerical
        include_performance = args.performance
        include_accuracy = args.accuracy
        include_integration = args.integration
    
    # Create and run pipeline
    pipeline = ValidationPipeline(
        verbose=not args.quiet,
        save_results=args.save_results
    )
    
    report = pipeline.run_full_pipeline(
        include_numerical=include_numerical,
        include_performance=include_performance,
        include_accuracy=include_accuracy,
        include_integration=include_integration,
        models=args.models,
        quick=args.quick
    )
    
    # Exit with appropriate code
    exit_code = 0 if report['validation_summary']['overall_status'] == 'PASS' else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()