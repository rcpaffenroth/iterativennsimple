# LRA benchmark runs

Each subdirectory is one experiment: a `config.yaml` in, and `results.md`,
`curves.png`, `results.json` out, written back beside the config so the two stay
paired.

```bash
uv run python examples/lra_benchmark.py examples/lra_runs/image_smoke
```

Data comes from the `generatedata` library pinned in `pyproject.toml`, fetched from
its standard HTTP location. Nothing is cached to disk; each run re-downloads (a
second or two) and shares one copy across all models in that run.

## The runs

| directory | dataset | seq_len | cost | what it is for |
| --- | --- | ---: | --- | --- |
| `image_smoke` | `lra_image` | 64 | ~3 min | Patchified CIFAR-10. The one to run while iterating. |
| `pathfinder_smoke` | `lra_pathfinder` | 64 | ~3 min | Binary, so chance is 0.5 and "learned nothing" is unmistakable. |
| `listops_smoke` | `lra_listops` | 2048 | ~25 min | Exercises the embedding path. Expensive; see below. |
| `image_full` | `lra_image` | 1024 | ~1 hour | The real LRA definition. Run deliberately. |

## Read the wall-clock before believing anything

`torch.nn.RNN`/`LSTM`/`GRU` call fused cuDNN kernels that cover an entire sequence
in one launch. `Sequential2DRNN` runs a Python loop with a few kernel launches per
timestep, because the block map exists to be *inspected*. Measured at
`seq_len=1024`, batch 64, hidden 128 on an RTX 4090:

| | ms / batch |
| --- | ---: |
| `nn.RNN` (cuDNN) | 2 |
| `Sequential2DRNN`, K=1 | 232 |

**That gap is implementation, not architecture — and it is a small-hidden
artifact.** Measured at $L = 1024$, $B = 64$, $d_x = 1$:

| $d_h$ | our loop | cuDNN | gap |
| ---: | ---: | ---: | ---: |
| 128 | 162 ms | 1.2 ms | 94× |
| 512 | 164 ms | 21.7 ms | 7.5× |
| 1024 | 158 ms | 28.5 ms | 5.5× |
| 2048 | 160 ms | 77.7 ms | 2.1× |

Our column is **flat**, because the loop is launch-bound rather than compute-bound.
cuDNN's whole advantage is amortising launch overhead, so it evaporates once the
per-step $d_h^2$ matmul is real work. **The overhead is worst exactly where it
matters least**: small-hidden runs are cheap in absolute terms even at 94×, and the
large-hidden runs this project is aimed at go at roughly cuDNN speed.

Hoisting $W_{xh}x_t$ out of the loop was measured and **declined** — worth ~30%, and
only in combination with pre-unbinding; done naively it is a 25% *pessimisation*.
See §5.5b of `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` for the proposition and the
numbers.

## Making an expensive dataset cheap

Two knobs, and they are not equivalent:

- **`step_size`** patchifies: `seq_len = x_y_index // step_size`, trading sequence
  length for input width. Cheap and effective — `step_size: 16` is ~16× faster.
  **Only valid for continuous inputs.** With an embedding the script asserts
  `step_size == 1`, since an embedding consumes one integer per timestep.
- **`max_points`** subsamples rows. Keeps the task exactly intact and costs only
  statistical power. This is the lever for the token tasks.

Truncating the *sequence* is deliberately not offered. It would be cheapest of all
and on ListOps it would silently change the labels, since the answer depends on the
whole expression.

## Config reference

```yaml
dataset:
  name: lra_image        # lra_image | lra_listops | lra_pathfinder | lra_text | lra_pathx
  step_size: 16          # features per timestep; must be 1 when embedding is set
  max_points: null       # optional row subsample
  train_frac: 0.8
  val_frac: 0.1          # remainder is test
  split_seed: 0          # every model in a run sees the identical split
  embedding: null        # or {vocab_size: 17, dim: 32} for token tasks

training:
  epochs: 8
  batch_size: 64
  lr: 0.003
  grad_clip: 1.0         # recurrent models on long sequences need this
  seed: 0                # same init draw for every model
  device: cuda

models:
  - name: <label for the table>
    type: rnn | lstm | gru | sequential2d
    hidden_size: 128
    nonlinearity: tanh   # rnn and sequential2d only; tanh | relu
    # sequential2d only:
    K: 1                 # internal iterations per token
    W_xh: linear         # linear | monarch | masked
    W_hh: linear
    num_blocks: 4        # monarch only
    orthogonal_hh: true  # orthogonal init of W_hh...
    gain: 1.2            # ...with this gain
```

Every model is the same three pieces — optional embedding, recurrent core, linear
head on the final hidden state — with only the core differing, so a difference in
the table is attributable to the recurrence rather than to plumbing.

## Two deterministic observations from the first run

Both are properties of the implementation rather than results about the method, and
both are seed-independent measurements rather than accuracies — which is why they
can be stated from a single run:

- **`nn.RNN` has 19,978 parameters; `Sequential2DRNN` K=1 has 19,850.** The
  difference is exactly 128 = `hidden_size`, which is the second bias vector.
  That is §8.2's "one bias per slot, not two" showing up in a parameter count.
- **The Monarch `W_hh` row is the *slowest*, at a quarter of the parameters.**
  Block-diagonal structure means several small matmuls plus gathers, and at hidden
  128 a dense matmul is already trivially cheap, so the overhead dominates. This is
  the same conclusion `notebooks/advanced/sparse_scripts/README.md` reaches for CSR:
  structured sparsity pays off at scale, not at these sizes.
