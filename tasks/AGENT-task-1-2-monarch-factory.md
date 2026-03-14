# Task 1.2 — Factory Functions for `MonarchLinear`

## Overview

Factory functions provide convenient ways to construct `MonarchLinear` instances, analogous to `MaskedLinear.from_description` and `SparseLinear.from_singleBlock`.

All factory functions are `@staticmethod` methods on the `MonarchLinear` class.

---

## Factory 1: `from_block_config` (general-purpose)

This is the most flexible constructor, analogous to `MaskedLinear.from_description`.

```python
@staticmethod
def from_block_config(
    in_features: int,
    out_features: int,
    block_in_features: list[int],
    block_out_features: list[int],
    initialization_type: str = "kaiming",
    bias: bool = True,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> "MonarchLinear":
    """
    Create a MonarchLinear with explicit block sizes.

    Args:
        in_features:  Total input dimension (must equal sum(block_in_features)).
        out_features: Total output dimension (must equal sum(block_out_features)).
        block_in_features:  List of column sizes for each diagonal block.
        block_out_features: List of row sizes for each diagonal block.
        initialization_type: How to initialize block weights.
            "kaiming" — Kaiming uniform (default, matches nn.Linear).
            "G" or "G=mu,sigma" — Gaussian.
            "U" or "U=lo,hi" — Uniform.
            "C=val" — Constant.
            "zeros" — All zeros.
        bias: Whether to include a bias vector.
        seed: Optional RNG seed for reproducible permutations.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A MonarchLinear instance.
    """
```

### Logic

1. Validate `sum(block_in_features) == in_features` and `sum(block_out_features) == out_features`.
2. Generate random permutations `perm_in` and `perm_out` (using `torch.randperm`, optionally seeded via a `torch.Generator`).
3. Construct the `MonarchLinear` with default Kaiming init.
4. If `initialization_type` is not `"kaiming"`, re-initialize each block using the pattern borrowed from `MaskedLinear._getInitializer`.

---

## Factory 2: `from_uniform_blocks` (convenience)

The most common case: all blocks have the same size.

```python
@staticmethod
def from_uniform_blocks(
    in_features: int,
    out_features: int,
    num_blocks: int,
    initialization_type: str = "kaiming",
    bias: bool = True,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> "MonarchLinear":
    """
    Create a MonarchLinear where all diagonal blocks have the same size.

    Args:
        in_features:  Total input dimension. Must be divisible by num_blocks.
        out_features: Total output dimension. Must be divisible by num_blocks.
        num_blocks:   Number of blocks along the diagonal.
        ... (remaining args same as from_block_config)

    Returns:
        A MonarchLinear instance.

    Raises:
        ValueError: If in_features or out_features is not divisible by num_blocks.
    """
```

### Logic

1. Compute `block_in = in_features // num_blocks`, `block_out = out_features // num_blocks`.
2. Validate divisibility.
3. Delegate to `from_block_config` with uniform lists.

---

## Factory 3: `from_sparsity_target` (sparsity-driven)

Automatically computes the number of uniform blocks to achieve approximately a given sparsity level.

```python
@staticmethod
def from_sparsity_target(
    in_features: int,
    out_features: int,
    target_sparsity: float,
    initialization_type: str = "kaiming",
    bias: bool = True,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> "MonarchLinear":
    """
    Create a MonarchLinear with approximately the given sparsity.

    The number of uniform blocks is chosen so that:
        nnz / (in_features * out_features) ≈ (1 - target_sparsity)

    For uniform blocks of size (out_features/k) × (in_features/k):
        nnz = k * (out_features/k) * (in_features/k)
            = out_features * in_features / k
        density = 1/k
        sparsity = 1 - 1/k
        k = 1 / (1 - target_sparsity)

    Args:
        target_sparsity: Desired fraction of zeros (0.0 = dense, 1.0 = all zeros).
            Must be in [0, 1). Rounded to nearest feasible k.
        ... (remaining args same as from_uniform_blocks)

    Returns:
        A MonarchLinear instance.
    """
```

### Logic

1. Compute ideal $k = 1 / (1 - \text{target\_sparsity})$.
2. Round $k$ to the nearest integer that divides both `in_features` and `out_features`. If no exact divisor exists, pick the closest feasible value and log the actual achieved sparsity.
3. Delegate to `from_uniform_blocks`.

---

## Initialization Helpers

Re-use the initialization pattern from `MaskedLinear._getInitializer` for consistency. The supported `initialization_type` values are:

| Value | Description |
|-------|-------------|
| `"kaiming"` | Kaiming uniform (default for `nn.Linear`). This is the default. |
| `"zeros"` | All zeros |
| `"G"` | Gaussian $\mu=0, \sigma=1$ |
| `"G=mu,sigma"` | Gaussian with specified $\mu, \sigma$ |
| `"U"` | Uniform $[-1, 1]$ |
| `"U=lo,hi"` | Uniform $[\text{lo}, \text{hi}]$ |
| `"C=val"` | Constant value |

### Implementation Note

Rather than duplicating the initializer logic, implement a private `_initialize_block(block, initialization_type)` method that handles the dispatch. This keeps the factory functions clean.

---

## Permutation Generation

```python
@staticmethod
def _generate_permutation(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate a random permutation of [0, n) as a LongTensor."""
    return torch.randperm(n, generator=generator)
```

When `seed` is provided in a factory function, create a `torch.Generator` seeded with that value to ensure reproducibility of permutations across runs.

---

## Example Usage

```python
from iterativennsimple.MonarchLinear import MonarchLinear

# 1. Uniform blocks — 4 blocks, each 64×64
layer = MonarchLinear.from_uniform_blocks(256, 256, num_blocks=4)

# 2. Custom block sizes
layer = MonarchLinear.from_block_config(
    in_features=256, out_features=128,
    block_in_features=[64, 64, 64, 64],
    block_out_features=[32, 32, 32, 32],
)

# 3. Target 75% sparsity → k=4 blocks
layer = MonarchLinear.from_sparsity_target(256, 256, target_sparsity=0.75)
```
