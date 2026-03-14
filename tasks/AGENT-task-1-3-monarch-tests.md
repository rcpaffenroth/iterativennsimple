# Task 1.3 — Test Plan for `MonarchLinear`

## File Location

`tests/test_MonarchLinear.py`

## Test Style

- Function-based tests using pytest (per project conventions).
- No class wrappers for tests.
- Use fixtures or helper functions for common setup.

---

## Helper Setup

```python
import torch
from iterativennsimple.MonarchLinear import MonarchLinear


def make_simple_monarch(in_f=16, out_f=16, num_blocks=4, bias=True, seed=42):
    """Create a small MonarchLinear for testing."""
    return MonarchLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=num_blocks, bias=bias, seed=seed
    )
```

---

## Test Cases

### 1. `test_construction_uniform`
- Create a `MonarchLinear` with `from_uniform_blocks`.
- Assert `in_features`, `out_features`, `num_blocks` are correct.
- Assert `perm_in` and `perm_out` have correct shapes and contain valid permutations (`sorted == arange`).
- Assert `blocks` has the correct number of elements, each with the right shape.

### 2. `test_construction_nonuniform`
- Create a `MonarchLinear` with `from_block_config` and non-uniform block sizes (e.g., `[4, 6, 6]` and `[5, 5, 6]`).
- Assert shapes are consistent.

### 3. `test_forward_shape`
- Forward a batch `(batch_size, in_features)` through the layer.
- Assert output shape is `(batch_size, out_features)`.

### 4. `test_forward_matches_dense`
- Construct a `MonarchLinear`.
- Compute dense weight via `to_dense()`.
- Compare `layer(x)` with `x @ dense.T + bias` for a random input batch.
- Use `torch.allclose` with reasonable tolerances.

### 5. `test_forward_no_bias`
- Same as above but with `bias=False`.

### 6. `test_gradient_flow`
- Forward through the layer, compute a scalar loss, call `.backward()`.
- Assert that each block in `blocks` has a non-None `.grad`.
- Assert that `bias.grad` is non-None (when bias is enabled).

### 7. `test_gradient_matches_dense`
- Compare gradients from `MonarchLinear` forward+backward with those from an equivalent dense `nn.Linear` whose weight is set to `to_dense()`.
- This validates correct backpropagation through the permute → block matmul → inverse-permute chain.

### 8. `test_optimizer_step`
- Create a `MonarchLinear`, run one SGD step.
- Verify that block values change after the step.
- Verify that `perm_in` and `perm_out` do **not** change (they are buffers).

### 9. `test_save_load`
- Create a `MonarchLinear`, forward to get output.
- Save `state_dict` to a temp file, create a new instance, `load_state_dict`.
- Forward with the same input and check outputs match exactly.
- Clean up temp file.

### 10. `test_to_device`
- Create on CPU, move to `cuda` (skip if no GPU), forward, move back.
- Assert all tensors (blocks, permutations, bias) are on the correct device.

### 11. `test_number_of_trainable_parameters`
- For known block sizes, check that `number_of_trainable_parameters()` returns the expected value.
- For example, 4 blocks of $(4 \times 4)$ with bias of length 16 → $4 \times 16 + 16 = 80$.

### 12. `test_sparsity_target`
- Create via `from_sparsity_target(256, 256, target_sparsity=0.75)`.
- Compute actual sparsity from `to_dense()`.
- Assert it is close to 0.75.

### 13. `test_from_block_config_validation`
- Assert that mismatched `sum(block_in_features) != in_features` raises `ValueError`.
- Assert that mismatched lengths of `block_in_features` and `block_out_features` raises `ValueError`.

### 14. `test_deterministic_seed`
- Create two instances with the same `seed`.
- Assert their `perm_in` and `perm_out` are identical.
- Create a third with a different seed and assert permutations differ.

### 15. `test_rectangular`
- Create a `MonarchLinear` where `in_features != out_features` (e.g., 32 → 16 with 4 blocks of $8 \times 4$).
- Forward and compare against `to_dense()`.

---

## Running Tests

```bash
uv run pytest tests/test_MonarchLinear.py -v
```

All tests should pass without GPU. GPU-specific tests (`test_to_device`) should be skipped with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")`.
