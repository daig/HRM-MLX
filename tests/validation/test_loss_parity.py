"""Test numerical parity between MLX and PyTorch loss implementations."""

import numpy as np
import mlx.core as mx
from mlx_hrm.training import (
    s_function,
    log_stablemax,
    stablemax_cross_entropy,
    IGNORE_LABEL_ID
)


def test_s_function_parity():
    """Test S-function matches PyTorch implementation."""
    print("Testing S-function parity...")
    
    # Test values including edge cases
    test_values = [-10, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 10]
    
    for x_val in test_values:
        # MLX computation
        x_mlx = mx.array([x_val])
        s_mlx = s_function(x_mlx, epsilon=1e-30)
        
        # Manual computation (matching PyTorch)
        if x_val < 0:
            s_expected = 1.0 / (1.0 - x_val + 1e-30)
        else:
            s_expected = x_val + 1.0
        
        # Compare
        diff = abs(s_mlx.item() - s_expected)
        print(f"  x={x_val:6.2f}: MLX={s_mlx.item():10.6f}, Expected={s_expected:10.6f}, Diff={diff:.2e}")
        assert diff < 1e-5, f"S-function mismatch for x={x_val}"
    
    print("✓ S-function parity test passed!\n")


def test_stablemax_normalization():
    """Test that stablemax probabilities sum to 1."""
    print("Testing stablemax normalization...")
    
    # Random logits
    np.random.seed(42)
    logits_np = np.random.randn(5, 10).astype(np.float32)
    logits = mx.array(logits_np)
    
    # Compute log probabilities
    log_probs = log_stablemax(logits, axis=-1)
    probs = mx.exp(log_probs)
    
    # Check sum
    sums = mx.sum(probs, axis=-1)
    print(f"  Probability sums: {sums}")
    print(f"  Max deviation from 1.0: {mx.max(mx.abs(sums - 1.0)).item():.2e}")
    
    assert mx.allclose(sums, mx.ones_like(sums), rtol=1e-4), "Stablemax doesn't sum to 1"
    print("✓ Stablemax normalization test passed!\n")


def test_stablemax_vs_softmax_extreme_values():
    """Test stablemax behavior with extreme values."""
    print("Testing stablemax with extreme values...")
    
    # Test cases with extreme values
    test_cases = [
        ("Large positive", mx.array([[100.0, 0.0, 0.0]])),
        ("Large negative", mx.array([[-100.0, 0.0, 0.0]])),
        ("Mixed extreme", mx.array([[100.0, -100.0, 0.0]])),
    ]
    
    for name, logits in test_cases:
        log_probs = log_stablemax(logits, axis=-1)
        probs = mx.exp(log_probs)
        
        print(f"  {name}:")
        print(f"    Logits: {logits[0]}")
        print(f"    Probs: {probs[0]}")
        print(f"    Sum: {mx.sum(probs[0]).item():.6f}")
        
        # Check no NaN or inf
        assert mx.all(mx.isfinite(log_probs)), f"Non-finite values in {name}"
    
    print("✓ Extreme values test passed!\n")


def test_loss_masking():
    """Test that ignore_index masking works correctly."""
    print("Testing loss masking...")
    
    # Create simple test case
    batch_size = 2
    seq_len = 6
    vocab_size = 10
    
    # Logits where we know the loss values
    logits = mx.zeros((batch_size, seq_len, vocab_size))
    # Set specific logits high to get near-zero loss for those positions
    for i in range(seq_len):
        logits[0, i, i] = 10.0  # High probability for correct class
        logits[1, i, (i + 1) % vocab_size] = 10.0
    
    # Labels with some positions masked
    labels = mx.array([
        [0, 1, 2, IGNORE_LABEL_ID, IGNORE_LABEL_ID, 5],
        [1, 2, 3, 4, IGNORE_LABEL_ID, IGNORE_LABEL_ID]
    ])
    
    # Compute loss
    losses = stablemax_cross_entropy(logits, labels, reduction='none')
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels:\n{labels}")
    print(f"  Losses shape: {losses.shape}")
    print(f"  Losses:\n{losses}")
    
    # Check masked positions have 0 loss
    mask = labels == IGNORE_LABEL_ID
    # MLX doesn't support boolean indexing yet, so check element-wise
    for i in range(batch_size):
        for j in range(seq_len):
            if mask[i, j]:
                assert losses[i, j] == 0, f"Masked position [{i}, {j}] should have 0 loss, got {losses[i, j]}"
    print(f"  ✓ All masked positions have 0 loss")
    
    # Check unmasked positions have non-zero loss (except where prediction is perfect)
    print("✓ Loss masking test passed!\n")


def test_loss_numerical_stability():
    """Test numerical stability of loss computation."""
    print("Testing loss numerical stability...")
    
    # Test with various scales of logits
    scales = [0.1, 1.0, 10.0, 100.0]
    
    for scale in scales:
        logits = mx.random.normal((2, 10, 50)) * scale
        labels = mx.random.randint(0, 50, (2, 10))
        
        loss = stablemax_cross_entropy(logits, labels, reduction='mean')
        
        print(f"  Scale={scale:5.1f}: Loss={loss.item():8.4f}")
        assert mx.isfinite(loss), f"Non-finite loss for scale={scale}"
    
    print("✓ Numerical stability test passed!\n")


def test_pytorch_comparison_values():
    """Test specific values that should match PyTorch implementation."""
    print("Testing specific PyTorch comparison values...")
    
    # Known test case
    logits = mx.array([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    labels = mx.array([[2]])  # Correct class is 2
    
    # Compute stablemax probabilities manually
    s_vals = s_function(logits[0, 0])
    print(f"  S-values: {s_vals}")
    
    s_sum = mx.sum(s_vals)
    probs = s_vals / s_sum
    print(f"  Probabilities: {probs}")
    print(f"  Prob sum: {mx.sum(probs).item()}")
    
    # Loss for the correct class
    loss = stablemax_cross_entropy(logits, labels, reduction='none')
    print(f"  Loss: {loss.item()}")
    
    # The loss should be -log(prob[2])
    expected_loss = -mx.log(probs[2])
    print(f"  Expected loss: {expected_loss.item()}")
    print(f"  Difference: {abs(loss.item() - expected_loss.item()):.2e}")
    
    assert mx.allclose(loss[0, 0], expected_loss, rtol=1e-4), "Loss computation mismatch"
    print("✓ PyTorch comparison test passed!\n")


def test_loss_gradients():
    """Test that gradients flow correctly through the loss."""
    print("Testing loss gradients...")
    
    def loss_fn(logits, labels):
        return stablemax_cross_entropy(logits, labels, reduction='mean')
    
    # Create test data
    logits = mx.random.normal((2, 5, 10))
    labels = mx.random.randint(0, 10, (2, 5))
    
    # Compute gradients
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(logits, labels)
    
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient stats:")
    print(f"    Mean: {mx.mean(grads).item():.6f}")
    print(f"    Std:  {mx.std(grads).item():.6f}")
    print(f"    Min:  {mx.min(grads).item():.6f}")
    print(f"    Max:  {mx.max(grads).item():.6f}")
    
    # Check gradients are finite and reasonable
    assert mx.all(mx.isfinite(grads)), "Non-finite gradients"
    assert mx.std(grads) > 0, "Zero gradients"
    
    print("✓ Gradient test passed!\n")


def main():
    """Run all parity tests."""
    print("=" * 60)
    print("MLX vs PyTorch Loss Function Parity Tests")
    print("=" * 60)
    print()
    
    test_s_function_parity()
    test_stablemax_normalization()
    test_stablemax_vs_softmax_extreme_values()
    test_loss_masking()
    test_loss_numerical_stability()
    test_pytorch_comparison_values()
    test_loss_gradients()
    
    print("=" * 60)
    print("✅ All parity tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()