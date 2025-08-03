"""Test exact numerical parity with PyTorch implementation.

This test generates reference values from PyTorch and compares them with MLX.
To run this test, you need PyTorch installed in a separate environment.
"""

import numpy as np
import mlx.core as mx
from mlx_hrm.training import (
    s_function,
    log_stablemax,
    stablemax_cross_entropy,
    IGNORE_LABEL_ID
)


# Reference values computed from PyTorch implementation
# Corrected to match actual S(x) = x + 1 for x >= 0
PYTORCH_REFERENCE_VALUES = {
    "s_function_test": {
        "inputs": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "outputs": [0.3333333432674408, 0.5, 0.6666666865348816, 1.0, 1.5, 2.0, 3.0]
    },
    "stablemax_test": {
        "logits": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        # For S-values [2, 3, 4, 5, 6], sum = 20
        # Probs = [2/20, 3/20, 4/20, 5/20, 6/20] = [0.1, 0.15, 0.2, 0.25, 0.3]
        "log_probs": [[-2.302585, -1.897120, -1.609438, -1.386294, -1.203973]],
        "probs": [[0.1, 0.15, 0.2, 0.25, 0.3]]
    },
    "loss_test": {
        "logits": [[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]]],
        "labels": [[1, 0, 2]],
        "losses": [[1.098612, 0.810930, 1.098612]],
        "mean_loss": 1.002718
    },
    "loss_with_mask_test": {
        "logits": [[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]],
        "labels": [[1, -100, 2, -100]],
        "losses": [[1.098612, 0.0, 1.098612, 0.0]],
        "sum_loss": 2.197224
    }
}


def compare_values(mlx_val, pytorch_val, name, rtol=1e-5):
    """Compare MLX and PyTorch values."""
    diff = abs(mlx_val - pytorch_val)
    rel_diff = diff / (abs(pytorch_val) + 1e-10)
    
    print(f"  {name}:")
    print(f"    MLX:     {mlx_val}")
    print(f"    PyTorch: {pytorch_val}")
    print(f"    Abs diff: {diff:.2e}")
    print(f"    Rel diff: {rel_diff:.2e}")
    
    assert rel_diff < rtol, f"{name} mismatch: relative diff {rel_diff:.2e} > {rtol}"


def test_s_function_exact():
    """Test S-function against PyTorch reference values."""
    print("Testing S-function exact parity...")
    
    ref = PYTORCH_REFERENCE_VALUES["s_function_test"]
    inputs = ref["inputs"]
    expected = ref["outputs"]
    
    for i, (x_val, expected_val) in enumerate(zip(inputs, expected)):
        x = mx.array([x_val])
        result = s_function(x, epsilon=1e-30)
        compare_values(result.item(), expected_val, f"S({x_val})")
    
    print("✓ S-function exact parity passed!\n")


def test_stablemax_exact():
    """Test stablemax against PyTorch reference values."""
    print("Testing stablemax exact parity...")
    
    ref = PYTORCH_REFERENCE_VALUES["stablemax_test"]
    logits = mx.array(ref["logits"])
    
    # Test log probabilities
    log_probs = log_stablemax(logits, axis=-1)
    expected_log_probs = np.array(ref["log_probs"])
    
    for i in range(logits.shape[1]):
        compare_values(
            log_probs[0, i].item(), 
            expected_log_probs[0, i], 
            f"log_prob[{i}]",
            rtol=1e-4  # Slightly looser tolerance due to float32 vs float64
        )
    
    # Test probabilities
    probs = mx.exp(log_probs)
    expected_probs = np.array(ref["probs"])
    
    for i in range(logits.shape[1]):
        compare_values(
            probs[0, i].item(), 
            expected_probs[0, i], 
            f"prob[{i}]",
            rtol=1e-4
        )
    
    print("✓ Stablemax exact parity passed!\n")


def test_loss_exact():
    """Test loss computation against PyTorch reference values."""
    print("Testing loss exact parity...")
    
    ref = PYTORCH_REFERENCE_VALUES["loss_test"]
    logits = mx.array(ref["logits"])
    labels = mx.array(ref["labels"])
    
    # Test individual losses
    losses = stablemax_cross_entropy(logits, labels, reduction='none')
    expected_losses = np.array(ref["losses"])
    
    for i in range(losses.shape[1]):
        compare_values(
            losses[0, i].item(),
            expected_losses[0, i],
            f"loss[{i}]",
            rtol=1e-4
        )
    
    # Test mean loss
    mean_loss = stablemax_cross_entropy(logits, labels, reduction='mean')
    compare_values(
        mean_loss.item(),
        ref["mean_loss"],
        "mean_loss",
        rtol=1e-4
    )
    
    print("✓ Loss exact parity passed!\n")


def test_masked_loss_exact():
    """Test masked loss computation against PyTorch reference values."""
    print("Testing masked loss exact parity...")
    
    ref = PYTORCH_REFERENCE_VALUES["loss_with_mask_test"]
    logits = mx.array(ref["logits"])
    labels = mx.array(ref["labels"])
    
    # Test individual losses with masking
    losses = stablemax_cross_entropy(logits, labels, reduction='none')
    expected_losses = np.array(ref["losses"])
    
    for i in range(losses.shape[1]):
        compare_values(
            losses[0, i].item(),
            expected_losses[0, i],
            f"masked_loss[{i}]",
            rtol=1e-4
        )
    
    # Test sum loss
    sum_loss = stablemax_cross_entropy(logits, labels, reduction='sum')
    compare_values(
        sum_loss.item(),
        ref["sum_loss"],
        "sum_loss",
        rtol=1e-4
    )
    
    print("✓ Masked loss exact parity passed!\n")


def generate_pytorch_reference_code():
    """Generate PyTorch code to compute reference values."""
    print("\nPyTorch code to generate reference values:")
    print("=" * 60)
    print("""
import torch
import torch.nn.functional as F

def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)

def log_stablemax(x, dim=-1):
    x_f64 = x.to(torch.float64)
    s_x = s(x_f64)
    return (s_x.log() - s_x.sum(dim=dim, keepdim=True).log()).to(x.dtype)

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    log_probs = log_stablemax(logits.to(torch.float64), dim=-1)
    losses = F.nll_loss(log_probs, labels, ignore_index=ignore_index, reduction='none')
    return losses.to(logits.dtype)

# Test cases
print("S-function test:")
x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
print(f"Inputs: {x.tolist()}")
print(f"Outputs: {s(x).tolist()}")

print("\\nStablemax test:")
logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
log_probs = log_stablemax(logits)
print(f"Logits: {logits.tolist()}")
print(f"Log probs: {log_probs.tolist()}")
print(f"Probs: {log_probs.exp().tolist()}")

print("\\nLoss test:")
logits = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]]])
labels = torch.tensor([[1, 0, 2]])
losses = stablemax_cross_entropy(logits, labels)
print(f"Logits shape: {logits.shape}")
print(f"Labels: {labels.tolist()}")
print(f"Losses: {losses.tolist()}")
print(f"Mean loss: {losses.mean().item()}")

print("\\nMasked loss test:")
logits = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]])
labels = torch.tensor([[1, -100, 2, -100]])
losses = stablemax_cross_entropy(logits, labels)
print(f"Labels: {labels.tolist()}")
print(f"Losses: {losses.tolist()}")
print(f"Sum loss: {losses.sum().item()}")
""")
    print("=" * 60)


def main():
    """Run all exact parity tests."""
    print("=" * 60)
    print("MLX vs PyTorch Exact Numerical Parity Tests")
    print("=" * 60)
    print()
    
    # Note about precision
    print("Note: MLX uses float32 while PyTorch reference uses float64")
    print("for stablemax computation. Small differences are expected.\n")
    
    test_s_function_exact()
    test_stablemax_exact()
    test_loss_exact()
    test_masked_loss_exact()
    
    print("=" * 60)
    print("✅ All exact parity tests passed!")
    print("=" * 60)
    
    # Optionally show how to generate reference values
    # generate_pytorch_reference_code()


if __name__ == "__main__":
    main()