# MLX HRM Implementation Notes

## Truncated Normal Initialization

We've implemented an exact port of the PyTorch HRM truncated normal initialization, which itself is based on JAX's implementation. Key characteristics:

### Algorithm Details

1. **Input**: `std` (desired standard deviation), `lower/upper` (truncation bounds in units of std)

2. **Process**:
   - Compute CDF values at bounds using `erf` function
   - Generate uniform samples in the CDF range
   - Apply inverse error function (`erfinv`) to convert to normal distribution
   - Apply variance correction by computing `comp_std = std / sqrt(variance_factor)`
   - Scale samples by `sqrt(2) * comp_std`
   - Clip to `[lower * comp_std, upper * comp_std]`

3. **Important Behavior**:
   - The clipping bounds are `lower/upper * comp_std`, NOT `lower/upper * std`
   - This causes the actual standard deviation to be larger than requested
   - For default [-2, 2] bounds, the std is inflated by approximately 29%
   - This matches the PyTorch HRM implementation exactly

### Key Differences from Standard Implementations

- PyTorch's `nn.init.trunc_normal_` does NOT properly handle variance correction
- Our implementation matches JAX's behavior (via the PyTorch HRM port)
- The confusing variable names in the original (`pdf_u` at lower bound, `pdf_l` at upper bound) are preserved for exact matching

### Testing Approach

Tests verify the actual behavior of the implementation rather than theoretical properties:
- Mean should be close to target (within sampling error)
- Std will be inflated due to the clipping behavior
- Bounds will be at `mean ± lower/upper * comp_std`

This ensures our MLX implementation produces identical behavior to the original PyTorch HRM code.