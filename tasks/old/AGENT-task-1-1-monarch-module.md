# Task 1.1 — `MonarchLinear` Module Design

## File Location

`iterativennsimple/MonarchLinear.py`

## Class Signature

```python
class MonarchLinear(torch.nn.Module):
    """Sparse linear layer using Monarch matrix format: S = P1 @ M @ P2.

    M is block-diagonal, P1 and P2 are stored as index arrays.
    Forward computes y = x S^T + b efficiently via permute → block matmul → inverse-permute.
    """
```

## Constructor

```python
def __init__(
    self,
    in_features: int,
    out_features: int,
    block_in_features: list[int],
    block_out_features: list[int],
    perm_in: torch.Tensor,   # LongTensor of shape (in_features,)
    perm_out: torch.Tensor,  # LongTensor of shape (out_features,)
    bias: bool = True,
    device=None,
    dtype=None,
) -> None:
```

### Stored Attributes

| Attribute | Type | Trainable | Description |
|-----------|------|-----------|-------------|
| `in_features` | `int` | — | Total input dimension |
| `out_features` | `int` | — | Total output dimension |
| `num_blocks` | `int` | — | Number of diagonal blocks |
| `block_in_features` | `list[int]` | — | Column size of each block |
| `block_out_features` | `list[int]` | — | Row size of each block |
| `blocks` | `nn.ParameterList` | Yes | Each element is a `Parameter` of shape `(block_out_features[i], block_in_features[i])` |
| `perm_in` | `Tensor` (buffer) | No | Index array for input permutation, shape `(in_features,)` |
| `perm_out` | `Tensor` (buffer) | No | Index array for output permutation, shape `(out_features,)` |
| `inv_perm_out` | `Tensor` (buffer) | No | Inverse of `perm_out`, precomputed for efficiency |
| `bias` | `Parameter` or `None` | Yes | Bias vector of shape `(out_features,)` |

### Notes on Buffers vs Parameters

- `perm_in` and `perm_out` (and `inv_perm_out`) are registered as **buffers** (`self.register_buffer`), so they:
  - Move with the module on `.to(device)` / `.cuda()`.
  - Are saved/loaded via `state_dict`.
  - Do **not** receive gradients.

### Constructor Validation

- `sum(block_in_features) == in_features`
- `sum(block_out_features) == out_features`
- `len(block_in_features) == len(block_out_features)` (same number of blocks)
- `perm_in.shape == (in_features,)` and contains a valid permutation
- `perm_out.shape == (out_features,)` and contains a valid permutation

## Forward Pass

```python
def forward(self, input: torch.Tensor) -> torch.Tensor:
    # input shape: (batch, in_features)

    # 1. Permute input columns
    x = input[:, self.perm_in]  # (batch, in_features)

    # 2. Block-diagonal matmul
    #    Split x along dim=1 into chunks matching block_in_features
    #    For each block i, compute: y_i = x_i @ blocks[i].T
    #    Concatenate results along dim=1
    parts = torch.split(x, self.block_in_features, dim=1)
    out_parts = []
    for i, part in enumerate(parts):
        out_parts.append(part @ self.blocks[i].T)
    y = torch.cat(out_parts, dim=1)  # (batch, out_features)

    # 3. Inverse-permute output columns
    #    We want: result[:, perm_out[j]] = y[:, j]
    #    Equivalently: result[:, :] = y[:, inv_perm_out]
    result = y[:, self.inv_perm_out]

    # 4. Add bias
    if self.bias is not None:
        result = result + self.bias

    return result
```

### Performance Optimization (uniform block sizes)

When all blocks have the same size (the common case), the loop in step 2 can be replaced with a single `torch.bmm`:

```python
# Reshape x into (num_blocks, batch, block_in) → bmm with blocks stacked as (num_blocks, block_out, block_in)
x_3d = x.reshape(batch, self.num_blocks, block_in).permute(1, 0, 2)  # (num_blocks, batch, block_in)
W_3d = torch.stack(list(self.blocks))  # (num_blocks, block_out, block_in)
y_3d = torch.bmm(x_3d, W_3d.transpose(1, 2))  # (num_blocks, batch, block_out)
y = y_3d.permute(1, 0, 2).reshape(batch, out_features)
```

**Decision:** Start with the general loop implementation (handles non-uniform block sizes). Add the `torch.bmm` fast path behind a flag or auto-detect when all blocks are the same size. This keeps the code simple and correct first, then fast.

## Other Methods

### `extra_repr`

```python
def extra_repr(self) -> str:
    return (
        f"in_features={self.in_features}, "
        f"out_features={self.out_features}, "
        f"num_blocks={self.num_blocks}, "
        f"bias={self.bias is not None}"
    )
```

### `number_of_trainable_parameters`

```python
def number_of_trainable_parameters(self) -> int:
    total = sum(b.numel() for b in self.blocks)
    if self.bias is not None:
        total += self.out_features
    return total
```

This method is required by `Sequential2D.number_of_trainable_parameters()`.

### `to_dense`

A debugging/testing utility that constructs the full dense matrix $S = P_1 M P_2$:

```python
def to_dense(self) -> torch.Tensor:
    """Return the full dense weight matrix (out_features, in_features)."""
    # Build block-diagonal M
    M = torch.block_diag(*[b for b in self.blocks])
    # Apply permutations: S = P1 @ M @ P2
    # P2 permutes columns: M_permuted_cols = M[:, inv_perm_in]
    # P1 permutes rows: S = M_permuted[inv_perm_out, :]
    # But we stored perm_in as the forward permutation on input indices
    # so S[perm_out[i], perm_in[j]] = M[i, j]
    # equivalently S = P1 @ M @ P2 where P1 maps rows and P2 maps cols.
    S = torch.zeros(self.out_features, self.in_features,
                    device=M.device, dtype=M.dtype)
    S[self.perm_out] = M  # permute rows
    S = S[:, torch.argsort(self.perm_in)]  # permute columns
    return S
```

### `reset_parameters`

Initialize blocks with Kaiming uniform (matching `torch.nn.Linear` defaults) and bias with uniform $[-1/\sqrt{\text{fan\_in}}, 1/\sqrt{\text{fan\_in}}]$:

```python
def reset_parameters(self) -> None:
    for block in self.blocks:
        torch.nn.init.kaiming_uniform_(block, a=math.sqrt(5))
    if self.bias is not None:
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)
```

## Design Decisions

1. **`nn.ParameterList` for blocks** rather than a single 3D tensor — handles non-uniform block sizes naturally.
2. **Buffers for permutations** — ensures proper device handling and serialization without gradients.
3. **Precomputed `inv_perm_out`** — avoids recomputing `torch.argsort` on every forward pass.
4. **No explicit permutation matrices** — only index arrays, as specified in the task.
5. **Compatible interface** — has `in_features`, `out_features`, `bias`, `forward`, `number_of_trainable_parameters`, matching the patterns used by `Sequential2D` and `Sequential1D`.
