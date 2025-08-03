"""
Checkpoint Management and Conversion Guide for HRM MLX.

This example demonstrates how to:
1. Save and load MLX checkpoints
2. Convert between PyTorch and MLX formats
3. Inspect checkpoint contents
4. Merge and modify checkpoints
5. Handle version compatibility
"""

import mlx.core as mx
from pathlib import Path
import pickle
import json
import time
from typing import Dict, Any, Optional, Union

# HRM imports
from mlx_hrm import create_hrm, create_hrm_from_checkpoint
from mlx_hrm.utils.checkpoint import save_checkpoint, load_checkpoint
from mlx_hrm.modules.act import HRMConfig


def basic_checkpoint_operations():
    """Demonstrate basic checkpoint save/load operations."""
    print("=" * 60)
    print("Basic Checkpoint Operations")
    print("=" * 60)
    
    # 1. Create and save a model
    print("\n1. Creating and saving model...")
    model = create_hrm('tiny')
    
    # Add some training metadata
    metadata = {
        'model_type': 'hrm_tiny',
        'training_step': 1000,
        'loss': 0.234,
        'accuracy': 0.789,
        'dataset': 'ARC-1',
        'training_time': '2.5 hours',
        'timestamp': time.time()
    }
    
    checkpoint_path = 'demo_checkpoint.pkl'
    save_checkpoint(model, checkpoint_path, metadata=metadata)
    print(f"Model saved to: {checkpoint_path}")
    
    # 2. Load the checkpoint
    print("\n2. Loading checkpoint...")
    loaded_model = create_hrm_from_checkpoint(checkpoint_path)
    print("Model loaded successfully!")
    
    # 3. Verify models produce same outputs
    print("\n3. Verifying model consistency...")
    test_input = {'input_ids': mx.random.randint(0, 1000, shape=(1, 16))}
    
    _, outputs1 = model(test_input)
    _, outputs2 = loaded_model(test_input)
    
    outputs_match = mx.allclose(outputs1['logits'], outputs2['logits'])
    print(f"Outputs match: {outputs_match}")
    
    # 4. Inspect checkpoint metadata
    print("\n4. Checkpoint metadata:")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    if 'metadata' in checkpoint_data:
        for key, value in checkpoint_data['metadata'].items():
            print(f"  {key}: {value}")
    
    # Cleanup
    Path(checkpoint_path).unlink()


def pytorch_mlx_conversion():
    """Demonstrate conversion between PyTorch and MLX formats."""
    print("\n" + "=" * 60)
    print("PyTorch ↔ MLX Conversion")
    print("=" * 60)
    
    # Note: This requires the conversion script
    print("\n1. Converting PyTorch checkpoint to MLX...")
    
    pytorch_checkpoint_path = "../HRM/checkpoints/model.pt"
    mlx_checkpoint_path = "converted_model.pkl"
    
    if Path(pytorch_checkpoint_path).exists():
        print(f"PyTorch checkpoint found at: {pytorch_checkpoint_path}")
        
        # Use the conversion script
        try:
            from scripts.convert_checkpoint import convert_pytorch_to_mlx
            
            convert_pytorch_to_mlx(
                pytorch_path=pytorch_checkpoint_path,
                mlx_path=mlx_checkpoint_path,
                verify=True  # Verify the conversion
            )
            
            print(f"Conversion successful: {mlx_checkpoint_path}")
            
            # Load converted model
            converted_model = create_hrm_from_checkpoint(mlx_checkpoint_path)
            print(f"Converted model loaded with {converted_model.num_parameters:,} parameters")
            
            # Cleanup
            Path(mlx_checkpoint_path).unlink()
            
        except ImportError:
            print("Conversion script not available - this is a demonstration")
            print("In practice, you would run:")
            print(f"  python scripts/convert_checkpoint.py --pytorch {pytorch_checkpoint_path} --mlx {mlx_checkpoint_path}")
    
    else:
        print(f"PyTorch checkpoint not found at: {pytorch_checkpoint_path}")
        print("This is expected in the demo - create a PyTorch checkpoint first")
    
    # 2. Converting MLX to PyTorch (reverse direction)
    print("\n2. Converting MLX checkpoint to PyTorch...")
    
    # Create an MLX checkpoint first
    model = create_hrm('tiny')
    temp_mlx_path = "temp_mlx_model.pkl"
    save_checkpoint(model, temp_mlx_path)
    
    try:
        from scripts.convert_checkpoint import convert_mlx_to_pytorch
        
        temp_pytorch_path = "temp_pytorch_model.pt"
        convert_mlx_to_pytorch(
            mlx_path=temp_mlx_path,
            pytorch_path=temp_pytorch_path
        )
        
        print(f"MLX → PyTorch conversion successful: {temp_pytorch_path}")
        
        # Cleanup
        Path(temp_mlx_path).unlink()
        Path(temp_pytorch_path).unlink()
        
    except ImportError:
        print("Conversion script not available - this is a demonstration")
        Path(temp_mlx_path).unlink()


def checkpoint_inspection():
    """Demonstrate checkpoint inspection and analysis."""
    print("\n" + "=" * 60)
    print("Checkpoint Inspection")
    print("=" * 60)
    
    # Create a checkpoint with detailed metadata
    model = create_hrm('small')
    
    detailed_metadata = {
        'model_config': {
            'preset': 'small',
            'hidden_size': model.config.hidden_size,
            'num_heads': model.config.num_heads,
            'H_layers': model.config.H_layers,
            'L_layers': model.config.L_layers,
            'vocab_size': model.config.vocab_size
        },
        'training_info': {
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'batch_size': 32,
            'training_steps': 5000,
            'dataset': 'Mixed puzzles',
            'loss_type': 'stablemax'
        },
        'performance': {
            'final_loss': 0.185,
            'validation_accuracy': 0.823,
            'best_accuracy': 0.847,
            'training_time_hours': 4.2
        },
        'version_info': {
            'mlx_hrm_version': '0.1.0',
            'mlx_version': '0.15.0',
            'python_version': '3.11.5'
        }
    }
    
    checkpoint_path = 'detailed_checkpoint.pkl'
    save_checkpoint(model, checkpoint_path, metadata=detailed_metadata)
    
    print(f"\n1. Checkpoint saved: {checkpoint_path}")
    
    # Inspect checkpoint structure
    print("\n2. Checkpoint structure:")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    print(f"  Keys: {list(checkpoint_data.keys())}")
    print(f"  Model parameters: {len(checkpoint_data['model'])} tensors")
    
    # Analyze parameter sizes
    print("\n3. Parameter analysis:")
    total_params = 0
    largest_param = 0
    param_stats = {}
    
    for name, param in checkpoint_data['model'].items():
        param_count = param.size
        total_params += param_count
        largest_param = max(largest_param, param_count)
        
        layer_type = name.split('.')[0] if '.' in name else 'other'
        if layer_type not in param_stats:
            param_stats[layer_type] = {'count': 0, 'params': 0}
        param_stats[layer_type]['count'] += 1
        param_stats[layer_type]['params'] += param_count
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Largest parameter tensor: {largest_param:,}")
    print(f"  Parameter breakdown:")
    
    for layer_type, stats in param_stats.items():
        pct = 100 * stats['params'] / total_params
        print(f"    {layer_type}: {stats['params']:,} params ({pct:.1f}%) in {stats['count']} tensors")
    
    # Display metadata in structured format
    print("\n4. Training metadata:")
    if 'metadata' in checkpoint_data:
        def print_nested_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_nested_dict(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")
        
        print_nested_dict(checkpoint_data['metadata'])
    
    # Cleanup
    Path(checkpoint_path).unlink()


def checkpoint_modification():
    """Demonstrate checkpoint modification and merging."""
    print("\n" + "=" * 60)
    print("Checkpoint Modification")
    print("=" * 60)
    
    # 1. Create base checkpoint
    base_model = create_hrm('tiny')
    base_checkpoint = 'base_model.pkl'
    
    base_metadata = {
        'model_name': 'base_model',
        'training_stage': 'pretrained',
        'accuracy': 0.75
    }
    
    save_checkpoint(base_model, base_checkpoint, metadata=base_metadata)
    print(f"1. Base checkpoint saved: {base_checkpoint}")
    
    # 2. Create fine-tuned checkpoint
    finetuned_model = create_hrm('tiny')
    
    # Simulate fine-tuning by modifying some parameters
    print("\n2. Simulating fine-tuning...")
    for name, param in finetuned_model.parameters().items():
        if 'embedding' in name.lower():
            # Add some noise to simulate fine-tuning updates
            noise = mx.random.normal(param.shape) * 0.001
            finetuned_model.parameters()[name] = param + noise
    
    finetuned_checkpoint = 'finetuned_model.pkl'
    finetuned_metadata = {
        'model_name': 'finetuned_model',
        'training_stage': 'finetuned',
        'base_model': 'base_model',
        'accuracy': 0.82,
        'finetune_dataset': 'ARC-specific'
    }
    
    save_checkpoint(finetuned_model, finetuned_checkpoint, metadata=finetuned_metadata)
    print(f"Fine-tuned checkpoint saved: {finetuned_checkpoint}")
    
    # 3. Compare checkpoints
    print("\n3. Comparing checkpoints...")
    
    with open(base_checkpoint, 'rb') as f:
        base_data = pickle.load(f)
    
    with open(finetuned_checkpoint, 'rb') as f:
        finetuned_data = pickle.load(f)
    
    # Compare parameter differences
    param_differences = {}
    for name in base_data['model'].keys():
        base_param = base_data['model'][name]
        finetuned_param = finetuned_data['model'][name]
        
        diff = mx.abs(finetuned_param - base_param)
        mean_diff = mx.mean(diff).item()
        max_diff = mx.max(diff).item()
        
        param_differences[name] = {
            'mean_diff': mean_diff,
            'max_diff': max_diff
        }
    
    print("Parameter differences (fine-tuned vs. base):")
    for name, diffs in param_differences.items():
        if diffs['mean_diff'] > 1e-6:  # Only show parameters that changed
            print(f"  {name}: mean={diffs['mean_diff']:.6f}, max={diffs['max_diff']:.6f}")
    
    # 4. Create ensemble checkpoint
    print("\n4. Creating ensemble checkpoint...")
    
    ensemble_weights = {'base': 0.3, 'finetuned': 0.7}
    ensemble_model = create_hrm('tiny')
    
    # Average parameters according to weights
    ensemble_params = {}
    for name in base_data['model'].keys():
        base_param = base_data['model'][name]
        finetuned_param = finetuned_data['model'][name]
        
        ensemble_param = (
            ensemble_weights['base'] * base_param + 
            ensemble_weights['finetuned'] * finetuned_param
        )
        ensemble_params[name] = ensemble_param
    
    # Update ensemble model parameters
    ensemble_model.update(ensemble_params)
    
    ensemble_checkpoint = 'ensemble_model.pkl'
    ensemble_metadata = {
        'model_name': 'ensemble_model',
        'training_stage': 'ensemble',
        'base_models': ['base_model', 'finetuned_model'],
        'ensemble_weights': ensemble_weights,
        'expected_accuracy': 0.79  # Weighted average
    }
    
    save_checkpoint(ensemble_model, ensemble_checkpoint, metadata=ensemble_metadata)
    print(f"Ensemble checkpoint saved: {ensemble_checkpoint}")
    
    # 5. Verify ensemble model
    print("\n5. Verifying ensemble model...")
    ensemble_loaded = create_hrm_from_checkpoint(ensemble_checkpoint)
    
    test_input = {'input_ids': mx.random.randint(0, 1000, shape=(1, 16))}
    _, ensemble_output = ensemble_loaded(test_input)
    
    print(f"Ensemble model verified - output shape: {ensemble_output['logits'].shape}")
    
    # Cleanup
    for checkpoint in [base_checkpoint, finetuned_checkpoint, ensemble_checkpoint]:
        Path(checkpoint).unlink()


def version_compatibility():
    """Demonstrate handling version compatibility."""
    print("\n" + "=" * 60)
    print("Version Compatibility")
    print("=" * 60)
    
    # 1. Create checkpoint with version info
    model = create_hrm('tiny')
    
    version_metadata = {
        'format_version': '1.0',
        'mlx_hrm_version': '0.1.0',
        'mlx_version': mx.__version__ if hasattr(mx, '__version__') else 'unknown',
        'checkpoint_schema': {
            'model': 'MLX state dict',
            'config': 'HRMConfig object', 
            'metadata': 'Dict with training info'
        },
        'compatibility': {
            'min_mlx_hrm_version': '0.1.0',
            'min_mlx_version': '0.15.0'
        }
    }
    
    versioned_checkpoint = 'versioned_model.pkl'
    save_checkpoint(model, versioned_checkpoint, metadata=version_metadata)
    print(f"1. Versioned checkpoint created: {versioned_checkpoint}")
    
    # 2. Version checking function
    def check_compatibility(checkpoint_path):
        """Check if checkpoint is compatible with current version."""
        
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        
        # Check format version
        format_version = metadata.get('format_version', '0.0')
        if format_version != '1.0':
            print(f"  WARNING: Checkpoint format {format_version} may not be compatible")
        
        # Check MLX HRM version
        checkpoint_hrm_version = metadata.get('mlx_hrm_version', 'unknown')
        current_hrm_version = '0.1.0'  # Would read from package
        
        print(f"  Checkpoint HRM version: {checkpoint_hrm_version}")
        print(f"  Current HRM version: {current_hrm_version}")
        
        # Check compatibility requirements
        compatibility = metadata.get('compatibility', {})
        min_hrm_version = compatibility.get('min_mlx_hrm_version', '0.0.0')
        
        if current_hrm_version < min_hrm_version:
            print(f"  ERROR: Requires HRM version {min_hrm_version} or higher")
            return False
        
        print("  ✓ Checkpoint is compatible")
        return True
    
    print("\n2. Checking compatibility:")
    is_compatible = check_compatibility(versioned_checkpoint)
    
    # 3. Migration function (example)
    def migrate_checkpoint_if_needed(checkpoint_path, output_path=None):
        """Migrate old checkpoint format to new format."""
        
        if output_path is None:
            output_path = checkpoint_path.replace('.pkl', '_migrated.pkl')
        
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        format_version = metadata.get('format_version', '0.0')
        
        if format_version == '1.0':
            print("  No migration needed")
            return checkpoint_path
        
        print(f"  Migrating from format {format_version} to 1.0...")
        
        # Example migration logic
        if format_version == '0.9':
            # Migrate 0.9 → 1.0
            if 'training_info' not in metadata:
                metadata['training_info'] = {
                    'optimizer': 'AdamW',
                    'learning_rate': metadata.get('lr', 1e-4)
                }
            
            metadata['format_version'] = '1.0'
            data['metadata'] = metadata
        
        # Save migrated checkpoint
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  Migrated checkpoint saved: {output_path}")
        return output_path
    
    print("\n3. Migration check:")
    migrate_checkpoint_if_needed(versioned_checkpoint)
    
    # Cleanup
    Path(versioned_checkpoint).unlink()
    migrated_path = versioned_checkpoint.replace('.pkl', '_migrated.pkl')
    if Path(migrated_path).exists():
        Path(migrated_path).unlink()


def checkpoint_utilities():
    """Demonstrate useful checkpoint utilities."""
    print("\n" + "=" * 60)
    print("Checkpoint Utilities")
    print("=" * 60)
    
    # 1. Checkpoint info utility
    def get_checkpoint_info(checkpoint_path):
        """Get comprehensive information about a checkpoint."""
        
        if not Path(checkpoint_path).exists():
            return {"error": "Checkpoint file not found"}
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            return {"error": f"Failed to load checkpoint: {e}"}
        
        info = {
            'file_size_mb': Path(checkpoint_path).stat().st_size / 1024**2,
            'keys': list(data.keys()),
            'has_model': 'model' in data,
            'has_config': 'config' in data,
            'has_metadata': 'metadata' in data
        }
        
        if 'model' in data:
            model_params = data['model']
            info['parameter_count'] = sum(param.size for param in model_params.values())
            info['parameter_tensors'] = len(model_params)
        
        if 'metadata' in data:
            metadata = data['metadata']
            info['metadata_keys'] = list(metadata.keys())
            
            # Extract common metadata fields
            for key in ['model_name', 'training_step', 'accuracy', 'loss']:
                if key in metadata:
                    info[key] = metadata[key]
        
        return info
    
    # 2. Checkpoint comparison utility
    def compare_checkpoints(checkpoint1_path, checkpoint2_path):
        """Compare two checkpoints."""
        
        info1 = get_checkpoint_info(checkpoint1_path)
        info2 = get_checkpoint_info(checkpoint2_path)
        
        comparison = {
            'file_sizes': {
                'checkpoint1_mb': info1.get('file_size_mb', 0),
                'checkpoint2_mb': info2.get('file_size_mb', 0)
            },
            'parameter_counts': {
                'checkpoint1': info1.get('parameter_count', 0),
                'checkpoint2': info2.get('parameter_count', 0)
            }
        }
        
        # Parameter differences (if both have models)
        if info1.get('has_model') and info2.get('has_model'):
            try:
                with open(checkpoint1_path, 'rb') as f:
                    data1 = pickle.load(f)
                with open(checkpoint2_path, 'rb') as f:
                    data2 = pickle.load(f)
                
                model1 = data1['model']
                model2 = data2['model']
                
                if set(model1.keys()) == set(model2.keys()):
                    total_diff = 0
                    max_diff = 0
                    
                    for name in model1.keys():
                        diff = mx.abs(model1[name] - model2[name])
                        param_mean_diff = mx.mean(diff).item()
                        param_max_diff = mx.max(diff).item()
                        
                        total_diff += param_mean_diff
                        max_diff = max(max_diff, param_max_diff)
                    
                    comparison['parameter_differences'] = {
                        'total_mean_diff': total_diff,
                        'max_diff': max_diff,
                        'parameters_match': max_diff < 1e-8
                    }
                else:
                    comparison['parameter_differences'] = {
                        'error': 'Different parameter structures'
                    }
            except Exception as e:
                comparison['parameter_differences'] = {
                    'error': f"Failed to compare parameters: {e}"
                }
        
        return comparison
    
    # Create test checkpoints
    model1 = create_hrm('tiny')
    model2 = create_hrm('tiny')
    
    checkpoint1 = 'test_checkpoint1.pkl'
    checkpoint2 = 'test_checkpoint2.pkl'
    
    save_checkpoint(model1, checkpoint1, metadata={'name': 'model1', 'accuracy': 0.75})
    save_checkpoint(model2, checkpoint2, metadata={'name': 'model2', 'accuracy': 0.80})
    
    print("\n1. Checkpoint information:")
    info1 = get_checkpoint_info(checkpoint1)
    for key, value in info1.items():
        print(f"  {key}: {value}")
    
    print("\n2. Checkpoint comparison:")
    comparison = compare_checkpoints(checkpoint1, checkpoint2)
    for category, data in comparison.items():
        print(f"  {category}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {data}")
    
    # Cleanup
    Path(checkpoint1).unlink()
    Path(checkpoint2).unlink()


def main():
    """Run all checkpoint management demonstrations."""
    print("HRM MLX Checkpoint Management Guide")
    print("This guide demonstrates comprehensive checkpoint operations")
    
    try:
        # Run all demonstrations
        basic_checkpoint_operations()
        pytorch_mlx_conversion()
        checkpoint_inspection()
        checkpoint_modification()
        version_compatibility()
        checkpoint_utilities()
        
        print("\n" + "=" * 60)
        print("Checkpoint Management Guide Complete!")
        print("\nKey takeaways:")
        print("1. Always include metadata when saving checkpoints")
        print("2. Verify checkpoint compatibility before loading")
        print("3. Use version information for future-proofing")
        print("4. Checkpoint inspection helps debug issues")
        print("5. Ensemble models can combine multiple checkpoints")
        print("6. Regular checkpoint comparison catches training issues")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("\nThis is a comprehensive demonstration showing various")
        print("checkpoint management techniques for HRM MLX models.")


if __name__ == "__main__":
    main()