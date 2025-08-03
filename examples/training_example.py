"""
Complete training example for HRM in MLX.

This example demonstrates how to set up and run training for the HRM model
using the MLX framework. It includes data preparation, model creation,
training loop, validation, and checkpointing.
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import json
import time
from typing import Dict, List, Optional

# HRM imports
from mlx_hrm import create_hrm
from mlx_hrm.training.trainer import HRMTrainer
from mlx_hrm.training.act_loss import ACTLossHead
from mlx_hrm.data.dataset import EnhancedPuzzleDataset, SmartDataLoader
from mlx_hrm.utils.checkpoint import save_checkpoint, load_checkpoint


def create_mock_dataset(
    num_samples: int = 1000,
    vocab_size: int = 1000,
    seq_len: int = 32,
    num_puzzle_types: int = 3
) -> List[Dict]:
    """
    Create a mock puzzle dataset for training demonstration.
    
    In practice, you would load real puzzle data from ARC, Sudoku, etc.
    
    Args:
        num_samples: Number of training samples
        vocab_size: Vocabulary size
        seq_len: Sequence length
        num_puzzle_types: Number of different puzzle types
        
    Returns:
        List of training examples
    """
    print("Creating mock dataset...")
    
    dataset = []
    for i in range(num_samples):
        # Assign puzzle type (cyclical for balance)
        puzzle_type = i % num_puzzle_types
        
        # Create random input sequence
        input_ids = mx.random.randint(1, vocab_size - 1, shape=(seq_len,))
        
        # Create labels (shifted input for next token prediction)
        labels = mx.concatenate([input_ids[1:], mx.array([0])])  # 0 = padding/end token
        
        example = {
            'input_ids': input_ids,
            'labels': labels,
            'puzzle_type': puzzle_type,
            'sample_id': i
        }
        dataset.append(example)
    
    print(f"Created {len(dataset)} training examples")
    return dataset


def setup_training_config() -> Dict:
    """
    Set up training configuration parameters.
    
    Returns:
        Training configuration dictionary
    """
    return {
        # Model configuration
        'model_preset': 'tiny',  # Start with tiny model for demo
        
        # Training parameters
        'batch_size': 8,
        'gradient_accumulation_steps': 4,  # Effective batch size = 8 * 4 = 32
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'max_training_steps': 1000,
        'warmup_steps': 100,
        
        # Validation and checkpointing
        'eval_every_n_steps': 100,
        'checkpoint_every_n_steps': 200,
        'log_every_n_steps': 10,
        
        # Mixed precision training
        'use_mixed_precision': True,
        'mixed_precision_dtype': 'float16',
        'mixed_precision_components': ['attention', 'feedforward'],
        
        # Early stopping
        'early_stopping_patience': 5,
        'max_grad_norm': 1.0,
        
        # Loss configuration
        'loss_type': 'stablemax',  # Use HRM's novel stablemax loss
        
        # Data configuration
        'num_samples': 1000,
        'vocab_size': 1000,
        'seq_len': 32,
        'num_puzzle_types': 3,
        'val_split': 0.2,
        
        # Output directory
        'output_dir': './training_output'
    }


def create_data_loaders(config: Dict) -> tuple:
    """
    Create training and validation data loaders.
    
    Args:
        config: Training configuration
        
    Returns:
        (train_loader, val_loader) tuple
    """
    print("Creating data loaders...")
    
    # Create mock dataset
    full_dataset = create_mock_dataset(
        num_samples=config['num_samples'],
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_puzzle_types=config['num_puzzle_types']
    )
    
    # Split into train/validation
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_data = full_dataset[:train_size]
    val_data = full_dataset[train_size:]
    
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    
    # Create datasets with smart batching
    train_dataset = EnhancedPuzzleDataset(
        data=train_data,
        mode='train',  # Shuffled training mode
        puzzle_type_key='puzzle_type'
    )
    
    val_dataset = EnhancedPuzzleDataset(
        data=val_data,
        mode='test',  # Sequential validation mode
        puzzle_type_key='puzzle_type'
    )
    
    # Create data loaders with smart batching
    train_loader = SmartDataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_loader = SmartDataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader


def create_model_and_loss(config: Dict) -> tuple:
    """
    Create HRM model and loss function.
    
    Args:
        config: Training configuration
        
    Returns:
        (model, loss_head) tuple
    """
    print(f"Creating {config['model_preset']} HRM model...")
    
    # Create base model
    model = create_hrm(config['model_preset'])
    print(f"Model created with {model.num_parameters:,} parameters")
    
    # Create ACT loss head wrapper
    loss_head = ACTLossHead(
        model=model,
        loss_type=config['loss_type'],
        vocab_size=config['vocab_size']
    )
    
    print(f"Using {config['loss_type']} loss function")
    return model, loss_head


def setup_output_directory(config: Dict) -> Path:
    """
    Set up output directory for checkpoints and logs.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to output directory
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        # Convert any non-serializable values
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        json.dump(serializable_config, f, indent=2)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Configuration saved to: {config_path}")
    
    return output_dir


def run_training_example():
    """
    Run the complete training example.
    """
    print("=" * 60)
    print("HRM MLX Training Example")
    print("=" * 60)
    
    # 1. Setup configuration
    config = setup_training_config()
    print("\n1. Configuration loaded")
    
    # 2. Setup output directory
    output_dir = setup_output_directory(config)
    print("\n2. Output directory created")
    
    # 3. Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    print("\n3. Data loaders created")
    
    # 4. Create model and loss
    model, loss_head = create_model_and_loss(config)
    print("\n4. Model and loss function created")
    
    # 5. Setup trainer
    print("\n5. Setting up trainer...")
    trainer = HRMTrainer(
        model=loss_head,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer_config={
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'warmup_steps': config['warmup_steps']
        },
        loss_type=config['loss_type'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=config['max_grad_norm'],
        checkpoint_dir=str(output_dir),
        checkpoint_every_n_steps=config['checkpoint_every_n_steps'],
        eval_every_n_steps=config['eval_every_n_steps'],
        early_stopping_patience=config['early_stopping_patience'],
        use_mixed_precision=config['use_mixed_precision'],
        mixed_precision_dtype=config['mixed_precision_dtype'],
        mixed_precision_components=config['mixed_precision_components'],
        log_every_n_steps=config['log_every_n_steps'],
        max_training_steps=config['max_training_steps'],
        warmup_steps=config['warmup_steps']
    )
    
    # 6. Run training
    print("\n6. Starting training...")
    print("-" * 40)
    
    start_time = time.time()
    try:
        trainer.train()
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
        
    finally:
        training_time = time.time() - start_time
        print(f"Total training time: {training_time:.2f} seconds")
    
    # 7. Final evaluation
    print("\n7. Running final evaluation...")
    try:
        final_metrics = trainer.evaluate()
        print("Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Training example complete!")
    print(f"Check {output_dir} for checkpoints and logs")
    print("=" * 60)


def demo_inference_with_trained_model(checkpoint_path: str):
    """
    Demonstrate inference with a trained model.
    
    Args:
        checkpoint_path: Path to saved checkpoint
    """
    print(f"\nDemonstrating inference with checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    model = create_hrm('tiny')  # Should match training config
    model = load_checkpoint(model, checkpoint_path)
    
    # Example inference
    input_sequence = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    print(f"Input sequence: {input_sequence.tolist()}")
    
    # Single sequence inference
    logits = model.forward_single(input_sequence)
    next_token_probs = mx.softmax(logits[-1])
    predicted_token = mx.argmax(next_token_probs)
    
    print(f"Predicted next token: {predicted_token.item()}")
    print(f"Confidence: {next_token_probs[predicted_token].item():.4f}")
    
    # Generation example
    generated = model.generate(
        prompt=input_sequence[:4],  # Use first 4 tokens as prompt
        max_length=12,
        temperature=1.0
    )
    
    print(f"Generated sequence: {generated.tolist()}")


def main():
    """
    Main function to run the training example.
    """
    try:
        # Run training
        run_training_example()
        
        # Demonstrate inference (if checkpoint exists)
        checkpoint_path = "./training_output/checkpoint_final.pkl"
        if Path(checkpoint_path).exists():
            demo_inference_with_trained_model(checkpoint_path)
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("\nThis is a demonstration - in production you would:")
        print("1. Use real puzzle datasets (ARC, Sudoku, etc.)")
        print("2. Tune hyperparameters for your specific task")
        print("3. Use proper validation metrics")
        print("4. Implement early stopping based on task performance")
        print("5. Use distributed training for larger models")
        

if __name__ == "__main__":
    main()