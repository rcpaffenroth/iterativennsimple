# Task 1.4 — Integration with Existing Codebase

## 1. Package Export

Add the import to `iterativennsimple/__init__.py`:

```python
from iterativennsimple.MonarchLinear import MonarchLinear
```

This keeps the module accessible at the package level, consistent with how `MaskedLinear` and `SparseLinear` are referenced throughout the codebase (though `__init__.py` is currently empty, the import patterns in tests and notebooks use the full module path).

---

## 2. `Sequential2D.from_config` Integration

Add a new block type to `Sequential2D.from_config` (in [Sequential2D.py](../iterativennsimple/Sequential2D.py)):

```python
elif block_type == 'MonarchLinear.from_uniform_blocks':
    block = MonarchLinear.from_uniform_blocks(
        in_features=in_features_list[i],
        out_features=out_features_list[j],
        num_blocks=cfg['block_kwargs'][i][j]['num_blocks'],
        initialization_type=cfg['block_kwargs'][i][j].get('initialization_type', 'kaiming'),
        bias=cfg['block_kwargs'][i][j].get('bias', True),
        seed=cfg['block_kwargs'][i][j].get('seed', None),
    )
elif block_type == 'MonarchLinear.from_block_config':
    block = MonarchLinear.from_block_config(
        in_features=in_features_list[i],
        out_features=out_features_list[j],
        block_in_features=cfg['block_kwargs'][i][j]['block_in_features'],
        block_out_features=cfg['block_kwargs'][i][j]['block_out_features'],
        initialization_type=cfg['block_kwargs'][i][j].get('initialization_type', 'kaiming'),
        bias=cfg['block_kwargs'][i][j].get('bias', True),
        seed=cfg['block_kwargs'][i][j].get('seed', None),
    )
elif block_type == 'MonarchLinear.from_sparsity_target':
    block = MonarchLinear.from_sparsity_target(
        in_features=in_features_list[i],
        out_features=out_features_list[j],
        target_sparsity=cfg['block_kwargs'][i][j]['target_sparsity'],
        initialization_type=cfg['block_kwargs'][i][j].get('initialization_type', 'kaiming'),
        bias=cfg['block_kwargs'][i][j].get('bias', True),
        seed=cfg['block_kwargs'][i][j].get('seed', None),
    )
```

Also add the import at the top of `Sequential2D.py`:

```python
from iterativennsimple.MonarchLinear import MonarchLinear
```

### Compatibility Notes

- `MonarchLinear` exposes `in_features` and `out_features` as attributes, so `Sequential2D.__init__` validation works unchanged.
- `MonarchLinear` implements `number_of_trainable_parameters()`, so `Sequential2D.number_of_trainable_parameters()` picks it up automatically.
- `MonarchLinear` supports `forward(input)` matching the expected signature.

---

## 3. YAML Configuration Example

For use with `Sequential2D.from_config`:

```yaml
in_features_list: [784, 64, 64, 10]
out_features_list: [784, 64, 64, 10]
block_types:
  - ['Identity', null, null, null]
  - ['MonarchLinear.from_uniform_blocks', null, null, null]
  - [null, 'MonarchLinear.from_uniform_blocks', null, null]
  - [null, null, 'MonarchLinear.from_uniform_blocks', null]
block_kwargs:
  - [null, null, null, null]
  - [{num_blocks: 4, bias: true, initialization_type: 'kaiming', activation: 'ReLU', activation_before: false}, null, null, null]
  - [null, {num_blocks: 4, bias: true, initialization_type: 'kaiming', activation: 'ReLU', activation_before: false}, null, null]
  - [null, null, {num_blocks: 4, bias: true, initialization_type: 'kaiming'}, null]
```

---

## 4. Files Modified (Summary)

| File | Change |
|------|--------|
| `iterativennsimple/MonarchLinear.py` | New file (Task 1.1 + 1.2) |
| `iterativennsimple/__init__.py` | Add import |
| `iterativennsimple/Sequential2D.py` | Add import + `from_config` block types |
| `tests/test_MonarchLinear.py` | New file (Task 1.3) |

No changes to `pyproject.toml` are needed — no new dependencies.
