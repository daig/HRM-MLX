# Test Coverage for MLX Truncated Normal Implementation

## Core Functionality Tests

### `truncated_normal` function
- ✅ **Shape and dtype handling**: Tests both tuple and int shapes, different dtypes
- ✅ **Statistical properties**: Verifies mean and inflated std match PyTorch behavior
- ✅ **Deterministic initialization**: Same key produces same results
- ✅ **Custom bounds**: Tests with different truncation bounds
- ✅ **Zero std case**: Returns zeros when std=0
- ✅ **Asymmetric bounds**: Tests with different lower/upper bounds
- ✅ **Edge cases**: Very small std, large std with tight bounds
- ✅ **Variance correction**: Verifies the implementation matches PyTorch HRM
- ✅ **Clipping behavior**: Verifies values are clipped at expected bounds

### `init_truncated_normal` factory
- ✅ **Function creation**: Returns callable initializer
- ✅ **Determinism**: Initializer with key produces consistent results
- ✅ **Dtype support**: Works with different dtypes

### `LinearTruncNormal` layer
- ✅ **Layer creation**: Correct weight/bias shapes
- ✅ **Forward pass**: Produces correct output shape
- ✅ **Initialization std**: Default uses 1/sqrt(in_features), custom std works
- ✅ **Bias initialization**: Bias initialized to zeros
- ✅ **No bias option**: bias=False works correctly
- ✅ **Key handling**: Tests deterministic initialization
- ✅ **Weight diversity**: Verifies proper randomness in weights

### `EmbeddingTruncNormal` layer
- ✅ **Layer creation**: Correct weight shape and attributes
- ✅ **Embedding lookup**: Correct output shape for batched indices
- ✅ **Initialization std**: Default uses 1/sqrt(embedding_dim)

### Internal Functions
- ✅ **`_erfinv` approximation**: Tests inverse error function accuracy
- ✅ **Clipping verification**: Tests that clipping occurs at correct bounds

## Key Behavioral Tests

1. **Variance Inflation**: Tests verify that the PyTorch implementation's quirk of inflating std by ~29% for default bounds is preserved

2. **Exact Bounds**: Tests verify clipping occurs at `mean ± lower/upper * comp_std`, not at `mean ± lower/upper * std`

3. **Mean Preservation**: Tests verify mean is preserved (or shifted appropriately for asymmetric bounds)

4. **Determinism**: Multiple tests ensure reproducibility with fixed keys

## Edge Cases Covered

- Zero standard deviation
- Very small standard deviation (1e-6)
- Large std with tight bounds (tests extreme comp_std)
- Asymmetric bounds (e.g., [-1, 3])
- Different data types (float32, float16)
- Single dimension shapes
- Non-zero mean

## Implementation Fidelity

All tests are designed to verify that our MLX implementation exactly matches the PyTorch HRM implementation, including its quirks like the std inflation due to how clipping is handled. This ensures drop-in compatibility when porting models.