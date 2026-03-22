# Monarch Matrix Sparse Linear Layer — Implementation Plan Overview

## Summary

Implement `MonarchLinear`, a new `torch.nn.Module` in `iterativennsimple/` that represents a sparse linear layer using the Monarch matrix format: $S = P_1 M P_2$, where $M$ is block-diagonal and $P_1, P_2$ are random permutations stored as index arrays (not matrices).

## Motivation

- `SparseLinear` relies on unstructured COO sparsity, which is slow on GPUs.
- The previous optimized path used `torch-sparse`/`torch-scatter`, which are now deprecated and cause installation issues.
- Monarch matrices offer *structured* sparsity that maps naturally to dense batched operations (`torch.bmm`), giving GPU-friendly performance with no extra dependencies.

## Mathematical Background

A Monarch matrix $S \in \mathbb{R}^{m \times n}$ is defined as:

$$S = P_1 \, M \, P_2$$

where:
- $M \in \mathbb{R}^{m \times n}$ is block-diagonal with $k$ dense blocks of sizes $(m_i \times n_i)$, with $\sum m_i = m$ and $\sum n_i = n$.
- $P_1$ is a permutation on outputs (rows), stored as an index array `perm_out` of length $m$.
- $P_2$ is a permutation on inputs (columns), stored as an index array `perm_in` of length $n$.

The forward pass computes $y = x S^T + b$ as:

1. Permute input columns: $\tilde{x} = x[:, \text{perm\_in}]$
2. Block-diagonal multiply: split $\tilde{x}$ into chunks along dim=1 matching block column sizes, multiply each chunk by the corresponding block's transpose, concatenate results.
3. Permute output columns: $y[:, \text{perm\_out}] = \tilde{y}$ (i.e., apply inverse permutation).
4. Add bias.

Step 2 can be implemented efficiently with `torch.bmm` when all blocks are the same size, or with a loop over blocks when sizes differ.

## File Plan

| File | Purpose |
|------|---------|
| `AGENT-task-1-0-monarch-overview.md` | This file — high-level overview |
| `AGENT-task-1-1-monarch-module.md` | Detailed design of `MonarchLinear` class |
| `AGENT-task-1-2-monarch-factory.md` | Factory functions and initialization |
| `AGENT-task-1-3-monarch-tests.md` | Test plan |
| `AGENT-task-1-4-monarch-integration.md` | Integration with `Sequential2D` and `__init__.py` |
| `AGENT-task-1-5-monarch-performance.md` | GPU performance validation notes |

## Implementation Order

1. **`MonarchLinear` module** — core class with `__init__`, `forward`, `extra_repr`, `number_of_trainable_parameters`.
2. **Factory functions** — `from_block_config`, `from_uniform_blocks`, convenience constructors.
3. **Unit tests** — correctness against dense multiplication, gradient flow, save/load, device transfer.
4. **Integration** — register in `Sequential2D.from_config`, export from package.
5. **Performance notebook** — compare wall-clock time vs `MaskedLinear` and `SparseLinear` on GPU.

## Dependencies

None beyond what is already in `pyproject.toml`. The implementation uses only `torch` (batched matmul, indexing).
