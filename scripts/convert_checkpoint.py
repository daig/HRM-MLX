#!/usr/bin/env python3
"""
Checkpoint conversion between PyTorch and MLX formats.

This script provides bidirectional conversion between PyTorch HRM checkpoints
and MLX HRM checkpoints, with proper parameter name mapping and format handling.

Usage:
    # Convert PyTorch to MLX
    python scripts/convert_checkpoint.py pytorch_checkpoint.pt mlx_checkpoint.pkl --config config.json

    # Convert MLX to PyTorch
    python scripts/convert_checkpoint.py mlx_checkpoint.pkl pytorch_checkpoint.pt

    # Verify conversion
    python scripts/convert_checkpoint.py pytorch_checkpoint.pt mlx_checkpoint.pkl --verify
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import mlx.core as mx

# Import MLX HRM components
sys.path.append(str(Path(__file__).parent.parent / "src"))
from mlx_hrm.modules.act import HRMConfig
from mlx_hrm.models.factory import create_hrm
from mlx_hrm.utils.checkpoint import save_checkpoint, load_checkpoint


def build_parameter_mapping(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build parameter name mapping between PyTorch and MLX formats.
    
    PyTorch HRM uses:
    - embed_tokens.weight
    - puzzle_emb.embeddings
    - H_level.layers.{i}.self_attn.qkv_proj.weight
    - lm_head.weight
    - q_head.weight/bias
    
    MLX HRM uses:
    - dense_tok_emb.weight
    - sparse_tok_emb.embeddings
    - H_module.blocks.{i}.self_attn.qkv_proj.weight
    - lm_head.weight
    - q_head.weight/bias
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Mapping from PyTorch names to MLX names
    """
    mapping = {}
    
    # Token embeddings
    mapping['embed_tokens.weight'] = 'dense_tok_emb.weight'
    
    # Sparse embeddings
    mapping['puzzle_emb.embeddings'] = 'sparse_tok_emb.embeddings'
    
    # Initial states
    mapping['H_init'] = 'z_init_H'
    mapping['L_init'] = 'z_init_L'
    
    # Output heads
    mapping['lm_head.weight'] = 'lm_head.weight'
    mapping['q_head.weight'] = 'q_head.weight'
    mapping['q_head.bias'] = 'q_head.bias'
    
    # RoPE (if cached)
    mapping['rotary_emb.cos_cached'] = 'rope.cos_cached'
    mapping['rotary_emb.sin_cached'] = 'rope.sin_cached'
    
    # Hierarchical modules
    for module in ['H', 'L']:
        pt_module = f"{module}_level"
        mlx_module = f"{module}_module"
        
        # Get number of layers for this module
        n_layers = config.get(f'{module}_layers', 4)
        
        for i in range(n_layers):
            # Attention layers
            mapping[f'{pt_module}.layers.{i}.self_attn.qkv_proj.weight'] = \
                f'{mlx_module}.blocks.{i}.self_attn.qkv_proj.weight'
            mapping[f'{pt_module}.layers.{i}.self_attn.o_proj.weight'] = \
                f'{mlx_module}.blocks.{i}.self_attn.o_proj.weight'
            
            # MLP layers
            mapping[f'{pt_module}.layers.{i}.mlp.gate_up_proj.weight'] = \
                f'{mlx_module}.blocks.{i}.mlp.gate_up_proj.weight'
            mapping[f'{pt_module}.layers.{i}.mlp.down_proj.weight'] = \
                f'{mlx_module}.blocks.{i}.mlp.down_proj.weight'
    
    return mapping


def pytorch_to_mlx(
    pytorch_path: str,
    output_path: str,
    config_path: Optional[str] = None
) -> None:
    """
    Convert PyTorch checkpoint to MLX format.
    
    Args:
        pytorch_path: Path to PyTorch checkpoint (.pt or .pth)
        output_path: Output path for MLX checkpoint (.pkl)
        config_path: Optional path to model config JSON file
    """
    print(f"Converting PyTorch checkpoint: {pytorch_path}")
    print(f"Output MLX checkpoint: {output_path}")
    
    # Load PyTorch checkpoint
    try:
        import torch
        checkpoint = torch.load(pytorch_path, map_location='cpu')
        print(f"✓ Loaded PyTorch checkpoint")
    except ImportError:
        print("✗ PyTorch not available for loading checkpoint")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading PyTorch checkpoint: {e}")
        sys.exit(1)
    
    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print("✓ Found config in checkpoint")
    elif config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        print("✗ No configuration found. Please provide --config path or ensure config is in checkpoint")
        sys.exit(1)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume entire checkpoint is state dict
        state_dict = checkpoint
    
    print(f"✓ Found {len(state_dict)} parameters in PyTorch checkpoint")
    
    # Build parameter mapping
    mapping = build_parameter_mapping(config_dict)
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Convert parameters
    mlx_weights = {}
    converted_count = 0
    missing_mappings = []
    
    for pt_name, pt_tensor in state_dict.items():
        if pt_name in mapping:
            mlx_name = mapping[pt_name]
            
            # Convert tensor to MLX format
            if hasattr(pt_tensor, 'detach'):
                numpy_array = pt_tensor.detach().cpu().numpy()
            else:
                numpy_array = np.array(pt_tensor)
            
            mlx_array = mx.array(numpy_array)
            
            # Handle nested structure for MLX weights
            parts = mlx_name.split('.')
            current = mlx_weights
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = mlx_array
            converted_count += 1
            
            print(f"  ✓ {pt_name} -> {mlx_name} {list(numpy_array.shape)}")
        else:
            missing_mappings.append(pt_name)
    
    # Report conversion results
    print(f"\n✓ Converted {converted_count}/{len(state_dict)} parameters")
    
    if missing_mappings:
        print(f"⚠ Warning: {len(missing_mappings)} parameters without mappings:")
        for name in missing_mappings[:10]:  # Show first 10
            print(f"    {name}")
        if len(missing_mappings) > 10:
            print(f"    ... and {len(missing_mappings) - 10} more")
    
    # Create MLX model to verify structure
    try:
        mlx_config = HRMConfig(**config_dict)
        model = create_hrm(mlx_config)
        print("✓ Created MLX model for verification")
        
        # Verify parameter counts match
        model_params = model.parameters()
        def count_params(d):
            count = 0
            for v in d.values():
                if isinstance(v, dict):
                    count += count_params(v)
                else:
                    count += v.size
            return count
        
        mlx_param_count = count_params(mlx_weights)
        model_param_count = sum(p.size for p in model_params.values() if hasattr(p, 'size'))
        
        print(f"  Converted params: {mlx_param_count:,}")
        print(f"  Model expects: {model_param_count:,}")
        
        if mlx_param_count != model_param_count:
            print(f"⚠ Warning: Parameter count mismatch")
        else:
            print("✓ Parameter counts match")
            
    except Exception as e:
        print(f"⚠ Warning: Could not verify model structure: {e}")
    
    # Save MLX checkpoint
    metadata = {
        'source_format': 'pytorch',
        'source_checkpoint': str(pytorch_path),
        'conversion_timestamp': datetime.now().isoformat(),
        'pytorch_param_count': len(state_dict),
        'mlx_param_count': converted_count,
        'config': config_dict
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in MLX format (compatible with existing checkpoint utilities)
    checkpoint_data = {
        'weights': mlx_weights,
        'config': mlx_config,
        'metadata': metadata
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"✓ Saved MLX checkpoint to {output_path}")


def mlx_to_pytorch(
    mlx_path: str,
    output_path: str
) -> None:
    """
    Convert MLX checkpoint to PyTorch format.
    
    Args:
        mlx_path: Path to MLX checkpoint (.pkl)
        output_path: Output path for PyTorch checkpoint (.pt)
    """
    print(f"Converting MLX checkpoint: {mlx_path}")
    print(f"Output PyTorch checkpoint: {output_path}")
    
    # Load MLX checkpoint
    try:
        with open(mlx_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if isinstance(checkpoint_data, dict) and 'weights' in checkpoint_data:
            weights = checkpoint_data['weights']
            config = checkpoint_data.get('config')
            metadata = checkpoint_data.get('metadata', {})
        else:
            # Assume it's weights dict directly
            weights = checkpoint_data
            config = None
            metadata = {}
        
        print("✓ Loaded MLX checkpoint")
    except Exception as e:
        print(f"✗ Error loading MLX checkpoint: {e}")
        sys.exit(1)
    
    # Get config
    if config is None:
        print("✗ No configuration found in MLX checkpoint")
        sys.exit(1)
    
    config_dict = config._asdict() if hasattr(config, '_asdict') else vars(config)
    
    # Build reverse mapping
    mapping = build_parameter_mapping(config_dict)
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Flatten MLX weights
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items
    
    flat_weights = dict(flatten_dict(weights))
    print(f"✓ Found {len(flat_weights)} parameters in MLX checkpoint")
    
    # Convert to PyTorch format
    pytorch_state_dict = {}
    converted_count = 0
    missing_mappings = []
    
    for mlx_name, mlx_array in flat_weights.items():
        if mlx_name in reverse_mapping:
            pt_name = reverse_mapping[mlx_name]
            
            # Convert to numpy then PyTorch format
            numpy_array = np.array(mlx_array)
            
            # Save as numpy for PyTorch loading (avoids PyTorch dependency)
            pytorch_state_dict[pt_name] = numpy_array
            converted_count += 1
            
            print(f"  ✓ {mlx_name} -> {pt_name} {list(numpy_array.shape)}")
        else:
            missing_mappings.append(mlx_name)
    
    # Report conversion results
    print(f"\n✓ Converted {converted_count}/{len(flat_weights)} parameters")
    
    if missing_mappings:
        print(f"⚠ Warning: {len(missing_mappings)} parameters without reverse mappings:")
        for name in missing_mappings[:10]:
            print(f"    {name}")
        if len(missing_mappings) > 10:
            print(f"    ... and {len(missing_mappings) - 10} more")
    
    # Create PyTorch-compatible checkpoint
    checkpoint = {
        'model_state_dict': pytorch_state_dict,
        'config': config_dict,
        'metadata': {
            **metadata,
            'source_format': 'mlx',
            'source_checkpoint': str(mlx_path),
            'conversion_timestamp': datetime.now().isoformat(),
            'mlx_param_count': len(flat_weights),
            'pytorch_param_count': converted_count,
        }
    }
    
    # Save checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for Python-only loading
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✓ Saved PyTorch-compatible checkpoint to {output_path}")
    print("  Note: Checkpoint saved as pickle. Convert to .pt format using PyTorch if needed:")
    print(f"    import torch; import pickle")
    print(f"    with open('{output_path}', 'rb') as f: data = pickle.load(f)")
    print(f"    torch.save(data, '{output_path.with_suffix('.pt')}')")


def verify_conversion(
    original_path: str,
    converted_path: str,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that conversion preserved parameter values.
    
    Args:
        original_path: Original checkpoint path
        converted_path: Converted checkpoint path
        tolerance: Maximum allowed difference
        
    Returns:
        True if all parameters match within tolerance
    """
    print(f"\nVerifying conversion between {original_path} and {converted_path}")
    print(f"Tolerance: {tolerance}")
    
    original_path = Path(original_path)
    converted_path = Path(converted_path)
    
    # Load both checkpoints
    try:
        if original_path.suffix in ['.pt', '.pth']:
            # PyTorch -> MLX conversion
            import torch
            original = torch.load(original_path, map_location='cpu')
            
            # Extract state dict
            if 'model_state_dict' in original:
                original_state = original['model_state_dict']
            else:
                original_state = original
            
            # Load converted MLX checkpoint
            with open(converted_path, 'rb') as f:
                converted_data = pickle.load(f)
            
            if isinstance(converted_data, dict) and 'weights' in converted_data:
                converted_weights = converted_data['weights']
                config_dict = converted_data['config']._asdict()
            else:
                converted_weights = converted_data
                config_dict = {}
            
            # Flatten MLX weights
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key))
                    else:
                        items.append((new_key, v))
                return items
            
            converted_state = dict(flatten_dict(converted_weights))
            
        else:
            # MLX -> PyTorch conversion
            with open(original_path, 'rb') as f:
                original_data = pickle.load(f)
            
            if isinstance(original_data, dict) and 'weights' in original_data:
                original_weights = original_data['weights']
                config_dict = original_data['config']._asdict()
            else:
                original_weights = original_data
                config_dict = {}
            
            # Flatten original weights
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key))
                    else:
                        items.append((new_key, v))
                return items
            
            original_state = dict(flatten_dict(original_weights))
            
            # Load converted PyTorch checkpoint
            with open(converted_path, 'rb') as f:
                converted_data = pickle.load(f)
            
            if 'model_state_dict' in converted_data:
                converted_state = converted_data['model_state_dict']
            else:
                converted_state = converted_data
                
    except Exception as e:
        print(f"✗ Error loading checkpoints for verification: {e}")
        return False
    
    # Build mapping for comparison
    mapping = build_parameter_mapping(config_dict)
    
    # Compare parameters
    total_params = 0
    matching_params = 0
    max_diff = 0.0
    mismatched_params = []
    
    for original_name, original_tensor in original_state.items():
        if original_name in mapping:
            converted_name = mapping[original_name]
            
            if converted_name in converted_state:
                converted_tensor = converted_state[converted_name]
                
                # Convert both to numpy for comparison
                if hasattr(original_tensor, 'detach'):
                    original_np = original_tensor.detach().cpu().numpy()
                else:
                    original_np = np.array(original_tensor)
                
                if hasattr(converted_tensor, '__array__'):
                    converted_np = np.array(converted_tensor)
                else:
                    converted_np = converted_tensor
                
                # Compute difference
                diff = np.abs(original_np - converted_np)
                max_param_diff = np.max(diff)
                max_diff = max(max_diff, max_param_diff)
                
                total_params += 1
                
                if max_param_diff < tolerance:
                    matching_params += 1
                    status = "✓"
                else:
                    mismatched_params.append((original_name, max_param_diff))
                    status = "✗"
                
                print(f"  {status} {original_name}: max_diff = {max_param_diff:.2e}")
            else:
                print(f"  ✗ {original_name}: missing in converted checkpoint")
    
    # Summary
    print(f"\nVerification Results:")
    print(f"  Total parameters compared: {total_params}")
    print(f"  Matching parameters: {matching_params}")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")
    
    success = matching_params == total_params and max_diff < tolerance
    
    if success:
        print("✓ Verification PASSED")
    else:
        print("✗ Verification FAILED")
        if mismatched_params:
            print(f"  Parameters exceeding tolerance:")
            for name, diff in mismatched_params[:5]:
                print(f"    {name}: {diff:.2e}")
            if len(mismatched_params) > 5:
                print(f"    ... and {len(mismatched_params) - 5} more")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Convert checkpoints between PyTorch and MLX formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PyTorch to MLX
  python scripts/convert_checkpoint.py model.pt model.pkl --config config.json
  
  # Convert MLX to PyTorch
  python scripts/convert_checkpoint.py model.pkl model.pt
  
  # Convert with verification
  python scripts/convert_checkpoint.py model.pt model.pkl --verify --tolerance 1e-6
        """
    )
    
    parser.add_argument('input', help='Input checkpoint path')
    parser.add_argument('output', help='Output checkpoint path')
    parser.add_argument(
        '--config', 
        help='Model config path (required for PyTorch->MLX if not in checkpoint)'
    )
    parser.add_argument(
        '--verify', 
        action='store_true', 
        help='Verify conversion by comparing parameter values'
    )
    parser.add_argument(
        '--tolerance', 
        type=float, 
        default=1e-5, 
        help='Verification tolerance (default: 1e-5)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"✗ Input checkpoint not found: {input_path}")
        sys.exit(1)
    
    # Determine conversion direction based on file extensions
    if input_path.suffix in ['.pt', '.pth']:
        # PyTorch to MLX
        if output_path.suffix != '.pkl':
            print("⚠ Warning: MLX output should typically have .pkl extension")
        
        pytorch_to_mlx(str(input_path), str(output_path), args.config)
        
    elif input_path.suffix in ['.pkl']:
        # MLX to PyTorch
        if output_path.suffix not in ['.pt', '.pth', '.pkl']:
            print("⚠ Warning: PyTorch output should typically have .pt or .pth extension")
        
        mlx_to_pytorch(str(input_path), str(output_path))
        
    else:
        print(f"✗ Unsupported input format: {input_path.suffix}")
        print("Supported formats: .pt, .pth (PyTorch), .pkl (MLX)")
        sys.exit(1)
    
    # Verify conversion if requested
    if args.verify:
        if verify_conversion(str(input_path), str(output_path), args.tolerance):
            print("\n🎉 Conversion completed successfully!")
        else:
            print("\n❌ Conversion verification failed!")
            sys.exit(1)
    else:
        print("\n✓ Conversion completed!")
        print("  Use --verify to validate parameter preservation")


if __name__ == '__main__':
    main()