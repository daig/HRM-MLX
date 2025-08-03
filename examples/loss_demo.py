"""Demonstration of HRM loss functions."""

import mlx.core as mx
from mlx_hrm.models import create_hrm
from mlx_hrm.training import ACTLossHead, MetricsTracker


def main():
    """Demonstrate loss computation with HRM model."""
    print("HRM Loss Functions Demo")
    print("=" * 50)
    
    # Create a small model
    print("\n1. Creating HRM model (tiny configuration)...")
    model = create_hrm('tiny')
    print(f"   Model created successfully")
    
    # Wrap with ACT loss head
    print("\n2. Creating ACT loss head with stablemax loss...")
    loss_head = ACTLossHead(model, loss_type='stablemax')
    
    # Create sample batch
    batch_size = 2
    seq_len = 16
    vocab_size = model.config.vocab_size
    
    print(f"\n3. Creating sample batch:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Vocabulary size: {vocab_size}")
    
    batch = {
        'input_ids': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        'labels': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        'puzzle_id': mx.zeros((batch_size,), dtype=mx.int32)
    }
    
    # Forward pass
    print("\n4. Running forward pass...")
    # Use the model's initial_carry method
    carry = model.initial_carry(batch_size)
    new_carry, total_loss, metrics, outputs = loss_head(carry, batch)
    
    print(f"\n5. Results:")
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Language modeling loss: {metrics['lm_loss'].item():.4f}")
    print(f"   Q-halt loss: {metrics['q_halt_loss'].item():.4f}")
    if 'q_continue_loss' in metrics:
        print(f"   Q-continue loss: {metrics['q_continue_loss'].item():.4f}")
    
    print(f"\n6. Metrics:")
    print(f"   Valid sequences: {int(metrics['count'].item())}")
    if metrics['count'].item() > 0:
        print(f"   Token accuracy: {metrics['accuracy'].item() / metrics['count'].item():.2%}")
        print(f"   Sequence accuracy: {metrics['exact_accuracy'].item() / metrics['count'].item():.2%}")
        print(f"   Avg computation steps: {metrics['steps'].item() / metrics['count'].item():.1f}")
    
    # Compare with softmax
    print("\n7. Comparing with softmax loss...")
    loss_head_softmax = ACTLossHead(model, loss_type='softmax')
    carry_soft = loss_head_softmax.initial_carry(batch_size)
    _, total_loss_soft, _, _ = loss_head_softmax(carry_soft, batch)
    
    print(f"   Stablemax loss: {total_loss.item():.4f}")
    print(f"   Softmax loss: {total_loss_soft.item():.4f}")
    print(f"   Ratio: {total_loss.item() / total_loss_soft.item():.2f}x")
    
    # Demonstrate metrics tracking
    print("\n8. Demonstrating metrics tracking over multiple batches...")
    tracker = MetricsTracker()
    
    for i in range(5):
        batch = {
            'input_ids': mx.random.randint(0, vocab_size, (4, seq_len)),
            'labels': mx.random.randint(0, vocab_size, (4, seq_len)),
            'puzzle_id': mx.array([i % 3] * 4)  # Cycle through 3 puzzle types
        }
        
        carry = loss_head.initial_carry(4)
        _, loss, metrics, _ = loss_head(carry, batch)
        tracker.update(metrics)
    
    current_metrics = tracker.get_current()
    print(f"\n   Averaged metrics over 5 batches:")
    print(f"   - Average LM loss: {current_metrics['lm_loss']:.4f}")
    print(f"   - Total sequences: {int(current_metrics['count'])}")
    if current_metrics['count'] > 0:
        print(f"   - Average accuracy: {current_metrics['accuracy'] / current_metrics['count']:.2%}")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()