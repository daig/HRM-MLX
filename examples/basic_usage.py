"""
Basic usage examples for HRM in MLX.

This script demonstrates how to use the HRM model for various tasks
including model creation, inference, generation, and checkpoint management.
"""

import mlx.core as mx
from mlx_hrm import create_hrm, list_available_presets, get_model_info
from mlx_hrm.utils import save_checkpoint, load_checkpoint


def example_model_creation():
    """Demonstrate different ways to create an HRM model."""
    print("=== Model Creation Examples ===\n")
    
    # Method 1: Using presets
    print("1. Creating model with preset:")
    model = create_hrm('small')  # 27M parameter model from paper
    print(f"Created model: {model}")
    print(f"Number of parameters: {model.num_parameters:,}\n")
    
    # Method 2: Custom configuration dict
    print("2. Creating model with custom config dict:")
    config = {
        'hidden_size': 384,
        'num_heads': 6,
        'H_layers': 3,
        'L_layers': 3,
        'vocab_size': 1000,
        'batch_size': 32,
        'seq_len': 256
    }
    custom_model = create_hrm(config)
    print(f"Created custom model with {custom_model.num_parameters:,} parameters\n")
    
    # Method 3: List available presets
    print("3. Available presets:")
    presets = list_available_presets()
    for preset in presets:
        info = get_model_info(preset)
        print(f"  - {preset}: {info['estimated_params_millions']}M parameters")


def example_forward_pass():
    """Demonstrate forward pass through the model."""
    print("\n=== Forward Pass Example ===\n")
    
    # Create model
    model = create_hrm('tiny')
    
    # Create input batch
    batch_size = 4
    seq_len = 32
    vocab_size = model.config.vocab_size
    
    batch = {
        'input_ids': mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
    }
    
    # Forward pass
    carry, outputs = model(batch)
    
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Q-halt logits shape: {outputs['q_halt_logits'].shape}")
    print(f"Halted sequences: {outputs['halted']}")
    print(f"ACT steps used: {carry.act_step}")


def example_single_sequence():
    """Demonstrate single sequence inference."""
    print("\n=== Single Sequence Inference ===\n")
    
    model = create_hrm('tiny')
    
    # Single sequence of tokens
    tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Get predictions
    logits = model.forward_single(tokens)
    
    print(f"Input tokens: {tokens}")
    print(f"Output shape: {logits.shape}")
    print(f"Next token predictions shape: {logits[-1].shape}")
    
    # Get most likely next token
    next_token = mx.argmax(logits[-1])
    print(f"Most likely next token: {next_token.item()}")


def example_generation():
    """Demonstrate text generation."""
    print("\n=== Text Generation Example ===\n")
    
    model = create_hrm('tiny')
    
    # Starting prompt
    prompt = mx.array([1, 2, 3, 4, 5])
    
    # Generate with different settings
    print("1. Basic generation:")
    generated = model.generate(prompt, max_length=20)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Length: {generated.shape[0]}")
    
    print("\n2. Generation with low temperature (more focused):")
    generated_focused = model.generate(prompt, max_length=20, temperature=0.5)
    print(f"Generated: {generated_focused}")
    
    print("\n3. Generation with high temperature (more random):")
    generated_random = model.generate(prompt, max_length=20, temperature=2.0)
    print(f"Generated: {generated_random}")
    
    print("\n4. Generation with top-k filtering:")
    generated_topk = model.generate(prompt, max_length=20, top_k=10)
    print(f"Generated: {generated_topk}")


def example_checkpointing():
    """Demonstrate checkpoint save/load."""
    print("\n=== Checkpoint Management Example ===\n")
    
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model
        model = create_hrm('tiny')
        checkpoint_path = Path(tmpdir) / 'my_checkpoint'
        
        # Save checkpoint with metadata
        print("1. Saving checkpoint...")
        save_checkpoint(
            model,
            str(checkpoint_path),
            metadata={
                'training_step': 1000,
                'loss': 0.234,
                'dataset': 'ARC-1'
            }
        )
        print(f"Saved to: {checkpoint_path}")
        
        # Load checkpoint
        print("\n2. Loading checkpoint...")
        from mlx_hrm import create_hrm_from_checkpoint
        loaded_model = create_hrm_from_checkpoint(str(checkpoint_path))
        print("Model loaded successfully!")
        
        # Verify they produce same outputs
        test_input = {'input_ids': mx.random.randint(0, 1000, shape=(1, 16))}
        outputs1 = model(test_input)[1]
        outputs2 = loaded_model(test_input)[1]
        
        print("\n3. Verification:")
        print(f"Outputs match: {mx.allclose(outputs1['logits'], outputs2['logits'])}")


def example_puzzle_solving():
    """Demonstrate using HRM for puzzle solving tasks."""
    print("\n=== Puzzle Solving Example ===\n")
    
    # Create model (in practice, you'd load a trained checkpoint)
    model = create_hrm('small')  # 27M model as in paper
    
    # Example: ARC-like puzzle input
    # In real usage, you'd encode the puzzle into token IDs
    print("Simulating ARC puzzle solving...")
    
    # Mock puzzle encoding (would come from actual tokenizer)
    puzzle_tokens = mx.array([
        10, 11, 12,  # Grid description tokens
        20, 21, 22,  # Pattern tokens
        30,          # Question token
    ])
    
    # Get model predictions
    logits = model.forward_single(puzzle_tokens)
    
    # The model uses ACT to reason through the puzzle
    # Multiple forward passes might be needed for complex puzzles
    carry = model.initial_carry(1)
    batch = {'input_ids': puzzle_tokens.reshape(1, -1)}
    
    for step in range(3):  # Simulate multiple reasoning steps
        carry, outputs = model(batch, carry)
        print(f"Step {step + 1}: ACT used {carry.act_step[0]} computation steps")
        if carry.halted[0]:
            print("Model halted - found solution!")
            break
    
    # Get final prediction
    prediction = mx.argmax(outputs['logits'][0, -1])
    print(f"Predicted answer token: {prediction.item()}")


def example_training_setup():
    """Demonstrate setting up model for training."""
    print("\n=== Training Setup Example ===\n")
    
    from mlx_hrm import create_model_for_training
    import mlx.optimizers as optim
    
    # Create model and optimizer
    model, optimizer = create_model_for_training(
        'tiny',
        learning_rate=1e-4,
        weight_decay=0.1
    )
    
    print(f"Model created with {model.num_parameters:,} parameters")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.learning_rate.item()}")
    
    # Example training step (simplified)
    print("\nSimulated training step:")
    
    # Mock batch
    batch = {
        'input_ids': mx.random.randint(0, 1000, shape=(4, 32)),
        'labels': mx.random.randint(0, 1000, shape=(4, 32))
    }
    
    # Forward pass
    carry, outputs = model(batch)
    
    # In real training, you'd compute loss here and backpropagate
    print(f"Forward pass complete - logits shape: {outputs['logits'].shape}")
    print(f"Ready for loss computation and backpropagation")


def main():
    """Run all examples."""
    print("HRM MLX Usage Examples")
    print("=" * 50)
    
    example_model_creation()
    example_forward_pass()
    example_single_sequence()
    example_generation()
    example_checkpointing()
    example_puzzle_solving()
    example_training_setup()
    
    print("\n" + "=" * 50)
    print("Examples complete! See individual functions for specific use cases.")


if __name__ == "__main__":
    main()