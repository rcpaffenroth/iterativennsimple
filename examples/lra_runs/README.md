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

Every config is at `step_size: 1` — the LRA definition — except `image_monarch`,
which carries a warning about it. Cost comes from `max_points` and the epoch budget.

| directory | dataset | seq_len | cost | what it is for |
| --- | --- | ---: | --- | --- |
| `image_smoke` | `lra_image` | 1024 | ~2 min | 1000 rows, 3 epochs. "Does the pipeline run", never accuracy. The one to run while iterating. |
| `pathfinder_smoke` | `lra_pathfinder` | 1024 | ~1 hour | Binary, so chance is 0.5 and "learned nothing" is unmistakable. **Not yet run.** |
| `listops_smoke` | `lra_listops` | 2048 | ~2 hours | Exercises the embedding path. Expensive; see below. **Not yet run.** |
| `image_full` | `lra_image` | 1024 | ~1 hour | The real LRA definition at $d_h$ = 128. Undertrained at 20 epochs — see its banner. |
| `image_wide` | `lra_image` | 1024 | ~1 hour | Width sweep at fixed lr. **Superseded and its conclusion withdrawn** — see its banner. |
| `image_wide_lr` | `lra_image` | 1024 | ~1.8 hours | Width crossed with lr. Holds the best result so far (GRU h=512, 0.480 test). |
| `image_monarch` | `lra_image` | 256 | ~1.7 hours | Monarch block count at $d_h$ = 2048. **`step_size: 4`, so a different task** — see its banner. |
| `epoch_probe` | `lra_image` | 1024 | ~1 hour | Where the loss plateaus at 100 epochs. Not a model comparison; it sets the epoch budget every other run is multiplied by. |

Wall-clock figures are from an RTX 4090 and include per-epoch validation.
`overnight.log` records `START`/`END` timestamps for the last two, from a runner
that is not in this repo — so those two runs cannot be reproduced end-to-end from
what is committed, only re-launched by hand.

## Read the wall-clock before believing anything

`torch.nn.RNN`/`LSTM`/`GRU` call fused cuDNN kernels that cover an entire sequence
in one launch. `Sequential2DRNN` runs a Python loop with a few kernel launches per
timestep, because the block map exists to be *inspected*.

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

**Those absolute numbers are not training steps, and what they are was not written
down.** `image_wide/` measured 30.1 s/epoch at $d_h$ = 128, which over 125 train
batches is ~240 ms per training step — and `image_wide/config.yaml` quotes 234 /
231 / 225 ms across the three widths, also calling them "ms/batch". So there are two
different quantities under one label. The *ratios* against cuDNN are probably
unaffected if both columns above were measured the same way, but that is an
assumption. Re-measure, and record what is being timed, before quoting an absolute.

Hoisting $W_{xh}x_t$ out of the loop was measured and **declined** — worth ~30%, and
only in combination with pre-unbinding; done naively it is a 25% *pessimisation*.
See §5.5b of `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` for the proposition and the
numbers.

## Making an expensive dataset cheap

Two knobs, and they are not equivalent:

- **`max_points`** subsamples rows. Keeps the task exactly intact and costs only
  statistical power. **This is the cost lever.**
- **`step_size`** patchifies: it sets `seq_len = x_y_index // step_size` *and*
  `input_size = step_size`, so it shortens the sequence and widens the input. That
  is a **different task**, not a cheap preview of the same one, and two runs at
  different values may never be placed on the same axis. It is also only valid for
  continuous inputs — with an embedding the script asserts `step_size == 1`, since
  an embedding consumes one integer per timestep.

Using `step_size` as a cost dial is the most expensive mistake made on this project
so far; see §2.8 of `PRINCIPLES.md` and §4.1 of `RESEARCH_LOG.md`.

Truncating the *sequence* is deliberately not offered. It would be cheapest of all
and on ListOps it would silently change the labels, since the answer depends on the
whole expression.

## Config reference

```yaml
notes: |                 # optional; emitted verbatim above the results table
  ## READ THIS BEFORE USING THESE NUMBERS
  ...whatever caveat this run carries...

dataset:
  name: lra_image        # lra_image | lra_listops | lra_pathfinder | lra_text | lra_pathx
  step_size: 1           # features per timestep; changes the TASK, not the cost
  max_points: null       # row subsample -- this is the cost lever
  train_frac: 0.8
  val_frac: 0.1          # remainder is test
  split_seed: 0          # every model in a run sees the identical split
  embedding: null        # or {vocab_size: 17, dim: 32} for token tasks

training:
  epochs: 30             # models on these tasks learn late; 15 is far too few
  batch_size: 64
  lr: 0.001
  grad_clip: 1.0         # recurrent models on long sequences need this
  seed: 0                # same init draw and batch order for every model
  seeds: [0, 1, 2]       # optional; replaces `seed`, running every model once each
  device: cuda

models:
  - name: <label for the table>
    type: rnn | lstm | gru | sequential2d
    hidden_size: 128
    lr: 0.0003           # optional per-model override of training.lr
    nonlinearity: tanh   # rnn and sequential2d only; tanh | relu
    # sequential2d only:
    K: 1                 # internal iterations per token
    W_xh: linear         # linear | monarch
    W_hh: linear
    num_blocks: 4        # monarch only
    orthogonal_hh: true  # orthogonal init of W_hh...
    gain: 1.2            # ...with this gain
```

`notes:` exists because `results.md` is regenerated from scratch on every run. A
caveat written into `results.md` by hand is deleted the next time the config runs;
one written here survives, and `yaml.safe_dump` also carries it into the config
block at the foot of the report. Comments in the config are *not* carried — they are
for whoever edits the config, `notes:` is for whoever reads the results.

`seeds:` varies the initialisation draw and the batch order while leaving
`split_seed` — and therefore the data — untouched, which is what a replication has
to hold fixed. Each model then appears once per seed, with ` seed=N` appended to its
row name and a `seed` field in `results.json`. **The report does not average them.**
Grouping rows and computing a spread is analysis, and analysis frozen into a
generator goes stale silently (§1.3 of `PRINCIPLES.md`); the rows are the
measurement. Note that a spread over 3 seeds is *not* the evaluation sampling error
— they are different quantities and both matter.

`orthogonal_hh` / `gain` are listed because the code path exists, **not** because
they work: the hypothesis has failed to transfer at $L = 1024$ twice, at gain 1.2
(`image_full/`) and gain 1.0 (`image_wide/`). See §3.1 of `RESEARCH_LOG.md`.

Every model is the same three pieces — optional embedding, recurrent core, linear
head on the final hidden state — with only the core differing, so a difference in
the table is attributable to the recurrence rather than to plumbing.

## Two deterministic observations

Both are properties of the implementation rather than results about the method, and
both are seed-independent measurements rather than accuracies — which is why they
can be stated from a single run. Numbers are from `image_full/` and `image_smoke/`,
at $d_h$ = 128, `step_size: 1`:

- **`nn.RNN` has 18,058 parameters; `Sequential2DRNN` K=1 has 17,930.** The
  difference is exactly 128 = `hidden_size`, which is the second bias vector.
  That is §8.2's "one bias per slot, not two" showing up in a parameter count.
- **The Monarch `W_hh` row is the *slowest*, at a fifth of the parameters**
  (3,594 against 17,930; 11.4 s/epoch against 2.9). Block-diagonal structure means
  several small matmuls plus gathers, and at hidden 128 a dense matmul is already
  trivially cheap, so the overhead dominates. This is the same conclusion
  `notebooks/advanced/sparse_scripts/README.md` reaches for CSR: structured sparsity
  pays off at scale, not at these sizes. `image_monarch/` measures the same effect
  at $d_h$ = 2048, where cost is roughly linear in block count while
  $|W_{hh}| = 2 d_h^2/\text{nb}^2$ falls quadratically.
