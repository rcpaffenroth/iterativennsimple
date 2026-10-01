# Example configurations for `MaskedLinear_simplified`

Each file is a block matrix written as an edge list: named slots, and one entry per
non-zero block. Everything unlisted is zero and frozen. The format, the vocabulary
of `values` and `pattern`, and the reasoning behind both are in
`iterativennsimple/masked_linear_simplified_config.py`.

Print any of them as a block matrix, next to the shape and trainable-parameter
count of the layer it builds:

    uv run python -m iterativennsimple.masked_linear_simplified_config \
        examples/masked_linear_simplified_configs/mlp.yaml

which for that file draws

    z0(4)  .      .      .      .
    z1(8)  K/all  .      .      .
    z2(8)  .      K/all  .      .
    z3(3)  .      .      K/all  .

— the subdiagonal, visible as a subdiagonal. That is what the two formats are for:
you write the list and check the matrix.

| | |
| --- | --- |
| `recurrent_map.yaml` | one step of a recurrence, and the only rectangular example: three input slots, two output slots. A rectangular `diagonal` injects the input into the first coordinates of the hidden slot |
| `mlp.yaml` | a feed-forward net as one square matrix, the weights on the block subdiagonal. What `from_MLP` builds |
| `identity_plus_update.yaml` | a recurrence starting at `I` and learning its departure from it — the case `A = W_0 + U .* Omega` exists for. Also records what the format *cannot* say |
| `random_features.yaml` | a frozen random projection with a trained readout. The baseline a learned representation has to beat |
| `sparse_recurrence.yaml` | `bernoulli` density as a sweep axis, with the warning that density is also a capacity dial |
| `fixed_fan_in.yaml` | `per_row` instead of `bernoulli`: every output sees at most *k* inputs, and why "at most" |

Random patterns are drawn at build time and are not seeded by the config; seed the
process with `torch.manual_seed` if you need the same mask twice. Every file here is
built by `tests/test_masked_linear_simplified_config.py`, so a broken example fails
the suite.
