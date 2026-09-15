# Research log — `Sequential2DRNN`

What has actually been done, what it showed, and what is still open. `PRINCIPLES.md`
covers *how* to work here; this file covers *what happened*.

Reverse-chronological within each section. Every quantitative claim is labelled
**[det]** for a deterministic measurement (timing, parameter count, exact
equivalence, divergence event — one measurement suffices) or **[stat]** for anything
resting on accuracy or loss on a finite sample. All **[stat]** results below are
**one seed**, so they carry no error bars beyond evaluation-set sampling noise.

---

## 1. What was built

### `iterativennsimple/Sequential2DRNN.py`

A `Sequential2D` block map driven as a discrete dynamical system:

$$z_{t+1} = (A \circ b \circ M)^{K} \circ \mathrm{Inject}_{t+1} \circ z_t$$

State is a list of slots; $M$ is the block map, $b$ a per-slot bias, $A$ a per-slot
activation, `Inject` overwrites the input slot with the next token, and $K$ is the
number of internal iterations per token. Based on Hershey, Paffenroth, Pathak &
Tavener, [arXiv:2404.00880](https://arxiv.org/abs/2404.00880).

Design rationale for every decision is in `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md`,
which is the authoritative document; the section numbers below refer to it.

Implemented: `from_rnn` (weight-copy from `torch.nn.RNN`), `from_3x3` (the general
three-slot map with six free blocks), the observation map `g` (§5.5), `PackedSequence`
support, arbitrary and nested block types, and $K > 1$.

Deliberately not implemented, each with reasons recorded: multi-layer stacking (§8.5),
`bidirectional`, `dropout`, the lifted formulation of eq. (15) (§8.1), a learnable
$M_{xx}$ (§8.6), and input lifting (§5.5b — a proven proposition, measured at 30%,
declined).

### `examples/lra_benchmark.py` + `examples/lra_runs/`

Directory-in, files-out: `config.yaml` → `results.md`, `curves.png`, `results.json`.
Every model is embedding → recurrent core → linear head on `h_n`, with only the core
differing, so table differences are attributable to the recurrence.

### Tutorials

- `notebooks/7-rcp-RNN-as-Sequential2D.ipynb` — builds the map by hand, checks it
  against `nn.RNN`, raises $K$. Symlinked into `tests/`, so it is also the test.
- `notebooks/advanced/12-claude-fixed-points-and-bistability.ipynb` — the
  dynamical-systems view at `hidden_size = 1`: cobwebs, bistability, an exactly
  located saddle-node fold. **Not reviewed by RCP in detail.**
- `examples/rnn_internal_iterations.py` — the $K$ experiment on copy-with-delay.
  Carries a warning header: its "fix" did not transfer (see §3.3).

133 tests pass, including both notebooks under nbmake.

---

## 2. Established results

> **Every `lra_image` result below sits on a test split in which 13.6% of rows are
> exact duplicates of training rows (§3.5).** The leakage is measured; its effect on
> these numbers is not. Do not quote an `lra_image` accuracy outside this file without
> reading §3.5 first.


### 2.1 `Sequential2DRNN` with $K=1$ **is** `torch.nn.RNN` **[det]**

`tests/test_Sequential2DRNN.py::test_matches_torch_rnn_across_options` asserts
`allclose` at 1e-6 on both `output` and `h_n`, across `tanh`/`relu`, both
`batch_first` settings, and several sequence lengths. This is the evidence.

Corroborating but *not* evidence: LRA accuracies agree (0.153 vs 0.149, $z = 0.2$ —
proves nothing at one seed), and parameter counts differ by exactly 128 =
`hidden_size`, which is PyTorch's second bias vector folded into one per §8.2.

### 2.2 Cost is flat in hidden width; the "~100×" gap is a small-hidden artefact **[det]**

Python-loop recurrence versus fused cuDNN, seq 1024, batch 64, $d_x = 1$:

| $d_h$ | ours | cuDNN | gap |
| ---: | ---: | ---: | ---: |
| 128 | 162 ms | 1.2 ms | 94× |
| 512 | 164 ms | 21.7 ms | 7.5× |
| 1024 | 158 ms | 28.5 ms | 5.5× |
| 2048 | 160 ms | 77.7 ms | 2.1× |

Our column is flat because the loop is launch-bound; cuDNN's whole advantage is
amortising launches, so it evaporates once the per-step $d_h^2$ matmul is real work.
Confirmed under real training: 29.9 / 29.4 / 29.1 s per epoch at $d_h$ = 128 / 512 /
2048, against cuDNN GRU's 5.2 → 20.5.

**Free to run is not free to train.** Over the same sweep `nn.GRU h=2048` diverged
to NaN, and the wide models needed much smaller learning rates. Flat wall-clock says
nothing about optimisation difficulty.

### 2.3 CORRECTED: Monarch's cost is *flat* in `nb`, not linear in it **[det]**

**What this section said until now:** "cost is roughly linear in `nb`", from
per-call timings at $d_h = 2048$, batch 64 — dense 0.024 ms, nb=2 0.124, nb=4
0.195, nb=16 0.771 — and the conclusion that "the FLOP-saving role of block count
needs $d_h$ in the tens of thousands".

**Why it was wrong.** Those timings measured two different models. `MonarchLinear.__init__`
called `_factorization(num_blocks)` unconditionally for square uniform blocks, so
whenever `nb` was a perfect power (4, 8, 16, but *not* 2) the layer silently stored
$n$ shared factors and rebuilt all $n^k$ blocks as products **on every forward
call**. The rising cost was those $nb(k-1)$ extra matmuls per call, not the block
structure. `MonarchLinear` now takes `use_factorization` (default `True`,
preserving the old behaviour); `examples/lra_benchmark.py` fixes it `False`.

Re-measured at the same $d_h = 2048$, batch 64, `use_views=False`, both paths:

| `nb` | factored (ms) | **independent (ms)** | $\|W_{hh}\|$ independent |
| ---: | ---: | ---: | ---: |
| dense | 0.022 | — | 4,194,304 |
| 2 | 0.113 | 0.112 | 2,097,152 |
| 4 | 0.175 | **0.113** | 1,048,576 |
| 8 | 0.352 | **0.118** | 524,288 |
| 16 | 0.669 | **0.125** | 262,144 |

The factored column reproduces the old numbers to ~15%, so this is the same
measurement, not a contradicting one. The independent column rises **12% across an
8× change in `nb`**.

**What is true.** Monarch at *any* `nb` costs about 5× a dense matmul at
$d_h = 2048$ (0.112 against 0.022) — the launch-bound floor of a `bmm` plus gathers
against one cuBLAS call — and that overhead is **flat in `nb`**. So block count is
close to free in wall-clock, and the old conclusion that sparsity cannot pay below
$d_h$ in the tens of thousands does not follow from this measurement.

**Not re-measured:** the training-time series 8.4 → 18.2 → 24.7 → 47.6 → 92.1
s/epoch as $W_{hh}$ fell 4.19 M → 0.033 M. That was also the factored path, and it
should flatten similarly — but that is a prediction, not a measurement.

Parameter counts differ between the two paths and this has bitten before:
$|W_{hh}| = d_h^2/\text{nb}$ with independent blocks, against $n\,d_h^2/\text{nb}^2$
when factored. (An earlier revision of this line also said 0.06 M and 70×, having
used nb=16's *total* parameter count where the $W_{hh}$ count was meant.)

Unaffected by any of the above: `MonarchLinear.forward(use_views=False)` is
1.8–2.2× faster than the default and is reached via the `MonarchNoViews` wrapper,
because `Sequential2D` calls `block.forward(x)` with no keyword arguments.

Also **[det]**, and the reason §3.3 below is now withdrawn: under factorization each
block is a product of $k$ Kaiming factors, so the layer's gain collapses. At
$d_h = 512$, comparing `to_dense()` against a dense `torch.nn.Linear`:

| `nb` | mode | $\sigma_{\max}$ | $\|W\|_F/\sqrt{d}$ |
| ---: | --- | ---: | ---: |
| dense | — | 1.148 | 0.578 |
| 2 | independent | 1.129 | 0.577 |
| 4 | factored | 0.871 | 0.335 |
| 8 | factored | 0.570 | 0.196 |
| 16 | factored | 0.363 | 0.105 |

At `nb` $\ge 4$ the factored $W_{hh}$ has $\sigma_{\max} < 1$: a strict contraction
in *every* direction at initialisation, so nothing survives a single step, let
alone 256. With `use_factorization=False` the gain sits at the dense $1/\sqrt3$ for
every `nb` from 2 to 64, so `nb` becomes a sparsity knob alone.

### 2.4 `torch.nn.GRU` at $d_h = 2048$ is unstable on LRA image **[det]**

NaN at epoch 7 with lr 1e-3, epoch 9 with 3e-4. At 1e-4 it is stable but reaches
only 0.170 in 30 epochs. No rate in that range both trains stably and progresses.
Not our code.

Note `clip_grad_norm_` cannot rescue a run once one NaN exists: the total norm
becomes non-finite and the rescale poisons every parameter. The harness stops at the
first non-finite loss and excludes non-finite epochs from "best".

### 2.5 Models on LRA image learn late **[stat, but a large effect]**

`nn.GRU h=512`, lr 1e-3, seq 1024: **0.159** val at epoch 15, **0.326** at 20,
**0.447** at 30 — and still climbing, with train loss falling 2.2174 → 1.3585. Any
comparison made before ~epoch 16 compares models that have not started learning.

This invalidated an earlier 15-epoch budget and makes `image_full/` (20 epochs)
undertrained; its config now says so.

### 2.6 Reproducibility **[det]**

Runs sharing `split_seed` reproduce each other to four decimal places, for both the
cuDNN path and our Python loop.

### 2.7 Best result so far **[stat]**

`nn.GRU h=512`, lr 1e-3, seq 1024, 30 epochs: **0.480 test**. Ours, dense
$d_h = 2048$ at lr 3e-5: 0.171 test. So a gated baseline needing no lr tuning is
roughly 2.5× ahead. Both truncated, so both understate.

---

## 3. Negative and retracted results

### 3.1 Orthogonal initialisation of $W_{hh}$ — no effect at seq 1024 **[stat]**

Tried twice and null both times: gain 1.2 gave 0.133 against 0.153 for default init
(`image_full/`), and gain 1.0 gave 0.131 against 0.132 (`image_wide/`).

The motivating argument — that gain > 1 offsets the contraction $\tanh' < 1$
introduces at every internal iteration — comes from
`examples/rnn_internal_iterations.py`, measured at seq 20, $K \le 4$, **and at
initialisation**. It has not transferred. Treat it as an untested hypothesis at long
sequence length. **Do not propose another gain value without first measuring the
memory horizon on a trained model at seq 1024.**

Two mechanisms were offered for why it failed and both were withdrawn — one
contradicted by the next data point, one by the following run.

### 3.2 RETRACTED: "width helps once the learning rate is scaled" **[stat]**

Reported as a headline finding. It came from taking the best result per width over
**unequal** learning-rate grids: one rate at $d_h = 128$, two at 512, three at 2048.
A maximum over unequal sample counts favours whoever got more samples, and the grids
do not overlap at $d_h = 128$ at all, so no matched-lr comparison including it exists.

Every matched comparison says width **hurts**: at lr 3e-4, h=512 0.1490 vs h=2048
0.1220; at lr 1e-4, 0.1940 vs 0.1450; and in `image_wide/` at fixed lr 1e-3,
0.153 → 0.132 → 0.111 across 128 → 512 → 2048.

**The width question is open.** Settling it needs the same lr grid at every width, at
fixed epochs.

### 3.3 WITHDRAWN: Monarch sparsity as a regulariser **[stat]**

**The one claim this sweep was said to support — "heavy sparsity (nb ≥ 8) is
clearly worse than dense or nb=2" — is withdrawn, not merely unresolved.** Its
`nb` = 4, 8, 16 rows ran on the factored path (§2.3) and its dense and `nb=2` rows
did not, so `nb` moved three things at once: the sparsity of $S$, whether the
blocks were independent or weight-tied products, and the initialisation gain of
$W_{hh}$. The third alone accounts for the result: at $d_h=2048$ those rows had
$\sigma_{\max}$ = 0.87 / 0.60 / 0.42, all below 1, so their hidden state was a
strict contraction in every direction at initialisation while dense and `nb=2`
(1.15, 1.15) were not. A model that cannot carry information across one step will
lose across 256 of them for reasons that have nothing to do with sparsity.

The $z$ values below are unaffected as arithmetic; what is withdrawn is the
attribution of the gap to sparsity. The sweep must be redone with
`use_factorization=False`, which is §5 item 3.

The original entry follows, retained because the resolution analysis in it is still
the right analysis:

$d_h = 2048$ fixed, `step_size: 4`, all rows at lr 1e-4 (val): dense 0.243, nb=2
0.239, nb=4 0.205, nb=8 0.161, nb=16 0.151, and the $W_{hh}$-matched control dense
$h{=}724$ 0.239.

With 1000 evaluation rows near $p = 0.2$, differences below ~0.04 are not resolvable:

| comparison | val $z$ | test $z$ | |
| --- | ---: | ---: | --- |
| dense vs nb=2 | 0.21 | 1.31 | tied |
| nb=2 vs nb=4 | 1.83 | 1.87 | unresolved |
| nb=8 vs nb=16 | 0.62 | 0.66 | tied |
| dense $h{=}724$ vs nb=4 | 1.83 | 1.22 | **unresolved** |
| dense vs nb=8 | 4.59 | 6.70 | real |
| dense vs nb=16 | 5.21 | 7.36 | real |

**Previously stated as supported:** heavy sparsity (nb ≥ 8) is clearly worse than
dense or nb=2 — **now withdrawn**, per the banner above: those are exactly the rows
whose initialisation gain collapsed. The parameter-matched control — previously
called "decisive" — settles nothing at $z = 1.2$ on test either.

Four further limits: one seed; the setting was weakly powered as a regularisation
test (dense's train/val gap was only 0.04); it ran at `step_size: 4`, so it is a
different task from every seq-1024 run (§4.1); and the parameter-matched control was
matched against *factored* counts, so `dense h=724` (524,176) pairs with what is now
`nb=8` (524,288), not `nb=4` (1,048,576).

### 3.4 $K > 1$ on copy-with-delay **[stat]**

On a pure memory task at seq 20, $K > 1$ did worse — $K=4$ at chance. The
memory-horizon measurement explaining it (§3.1) is **[det]**, computed by autograd at
initialisation; the accuracies are single-seed. The proposed fix did not transfer.
$K > 1$ has **not** been tested on a task requiring per-token computation rather than
memory, which is where it would have somewhere to put the effort.

### 3.5 Train/test leakage in `lra_image` from sampling with replacement **[det]**

`generatedata`'s LRA generators draw rows **with replacement** —
`rng.integers(0, len(dataset), size=num_points)` — so a dataset of `num_points` rows
drawn from a table of $N$ contains only $N(1-e^{-n/N})$ distinct rows in expectation.
Predicted 9,063 distinct for `lra_image`; measured 9,072. Full survey:

| dataset | rows | distinct | duplicate rows |
| --- | ---: | ---: | ---: |
| `lra_toy_bw` | 1000 | 1000 | 0 (0.0%) |
| `lra_image_mnist` | 1000 | 991 | 9 (0.9%) |
| **`lra_image`** | 10000 | 9072 | **928 (9.3%)** |
| `lra_pathfinder` | 10000 | 10000 | 0 (0.0%) |
| `lra_listops` | 10000 | 9665 | 335 (3.4%) |
| `lra_text` | 10000 | 9980 | 20 (0.2%) |
| `lra_pathx` | 2000 | 2000 | 0 (0.0%) |

The 0% tasks are *generated* per row rather than resampled from a table, so they have
nothing to collide in.

Replicating the harness split (`train_frac: 0.8`, `val_frac: 0.1`, `split_seed: 0`)
and counting test rows whose exact feature vector appears in train:

**`lra_image`: 136 of 1000 test rows (13.6%) and 142 of 1000 validation rows (14.2%)
are exact duplicates of training rows.** That applies to every `lra_image` result in
this log — §2.2, §2.4, §2.5, §2.7, §3.1, §3.2, §3.3.

**What this does not establish.** Nothing here measures how much any reported accuracy
was inflated. The inflation is bounded above by $0.136\,(1-a)$ — about 0.07 at the
$a = 0.480$ headline of §2.7 — but that bound assumes leaked rows are memorised
perfectly, which is unmeasured and probably false. The honest position: **the leakage
is measured, its effect is not.** Measuring it needs a rerun evaluating on a
deduplicated test split, since the harness does not persist weights.

`lra_image_mnist` (0.9%) and `lra_toy_bw` (0.0%) are clean enough to use as they ship.

**Must be resolved before any publication of `lra_image` numbers.** Two routes, not
yet chosen: fix the sampler to `rng.choice(..., replace=False)` and regenerate, or
keep the data and deduplicate at split time so no test row has a twin in train.

---

## 4. Process errors made, and what they cost

Recorded because the same mistakes are cheap to repeat. See `PRINCIPLES.md` for the
generalised rules.

### 4.1 `step_size` used as a cost dial when it changes the task

`step_size` sets both `seq_len = x_y_index // step_size` **and**
`input_size = step_size`. Three values were used on `lra_image` — 16, 4, 1 — across
five directories, with nothing marking them incomparable, and the cheap ones were
described as previews of the expensive ones. `pathfinder_smoke` was the only
pathfinder config and ran at 16, so the project had **no LRA pathfinder result at
all**, only a different task under its name.

`max_points` existed for exactly this purpose — it drops rows, costing only
statistical power — and was not used. **Fixed:** every config is now at
`step_size: 1` with `max_points` for cost, except `image_monarch/`, which carries a
warning at the top of both its config and its results.

### 4.2 Two configs could not answer their own question

`pathfinder_smoke` and `listops_smoke` had `orthogonal_hh: true` on their $K > 1$
rows and **not** on $K = 1$, so $K$ and initialisation moved together and neither
could be attributed. **Fixed:** orthogonal init removed from both, so all rows share
an initialisation.

### 4.3 Confounded comparisons reported as findings

Beyond §3.2: `image_wide/` held lr fixed across widths, producing a step-size
artefact reported as a capacity result; and the epoch budget moved 20 → 30 in the
same change as the learning rate, with the difference attributed to the learning
rate alone.

### 4.4 A statistical claim contradicted by its own caveat

The five Monarch numbers were called "monotonically decreasing" in one paragraph and
0.243-vs-0.239 called unresolvable four paragraphs later, in the same message.

### 4.5 Retracted claims left standing in comments and configs

The orthogonal-init hypothesis was asserted as fact in a config comment and silently
baked into two others, and remained after failing twice. A stale "~115×" timing
framing survived in a config header after being corrected in the module docstring.

### 4.6 Machinery proposed instead of care

Two automated checks were added to compensate for the above — one flagging
non-converged runs, and a proposed minimum-detectable-difference gate. RCP rejected
both: such code adds complexity that itself needs checking and can introduce bugs.
`was_truncated` was removed (77 lines, including two editorialising paragraphs in the
report generator). The distinction that survived: correctness fixes and compute
savings belong in the code; judgement aids do not.

### 4.7 A library default silently changed the model class

`num_blocks` was treated as one knob — sparsity — across `image_monarch/`, two
sections of this log and three of `TODO_Sequential2DRNN.md`. It was three knobs,
because `MonarchLinear.__init__` inferred the factored representation from whether
`num_blocks` happened to be a perfect power. Nothing in a config, a row name or a
results table showed which model had been built, and the affected values (4, 8, 16)
were exactly the ones a sweep naturally picks.

This is §2.8's lesson one level below the harness: a parameter that changes the
task is not a cost dial, and a parameter that changes the *model class* is not a
sparsity dial. The check that would have caught it is cheap and is now the habit —
**build the object and measure it before sweeping it**: parameter count, spectral
norm at initialisation, and per-call cost, tabulated against the knob. Three of
those numbers moved together and none of them was the one being swept.

`use_factorization` now makes the choice explicit; the harness fixes it `False` and
asserts the outcome. The default is unchanged, so existing work is untouched.

---

## 5. Open questions, in the order they seem worth doing

1. **$W_{hh} \supseteq I$ — an identity on the hidden diagonal.** GRU learns on LRA
   image and a vanilla recurrence barely does at matched width and lr. The plausible
   reason is that gates supply near-identity paths through 1024 steps, which a fixed
   contraction has none of. $M_{xx} = I$ already provides exactly such a path for the
   *input* channel (§8.6), and §8.6 notes that the same on the hidden diagonal is a
   separate and unexplored choice. Cheap, directly motivated, and the non-gated way
   to get what gating provides — which is the paper's thesis.
2. **Redo the width sweep properly** — the same lr grid at every width, fixed epochs
   (§3.2).
3. **Redo the Monarch sweep with `use_factorization=False`**, at a `step_size` where
   our model actually learns, and with more than one seed (§2.3, §3.3, §4.1). The
   `difficulty_ladder/` screen exists to locate that `step_size`.
4. **Seed replication.** Nothing in this log has error bars. Three seeds on any
   **[stat]** claim would change what can be said.
5. **A compute-bound rather than memory-bound task**, to test $K > 1$ where it has
   somewhere to put the effort (§3.4).
6. **Measure the trained spectrum** of $J_{T_x}$, rather than at initialisation —
   this is the measurement §3.1 is blocked on.
7. Deferred features and longer-range ideas are in `tasks/TODO_Sequential2DRNN.md`.

---

## 6. The dynamical-systems experiments

A separate line of work from `Sequential2DRNN`, sharing its datasets. The prediction
$y_k$ *is* the state of a discrete dynamical system, built one experiment at a time. Design
record, including every rejected reading: `tasks/OVERVIEW_DYNAMICAL_SYSTEM.md`.

### 6.1 Experiment 1 — $y_{k+1} = Ax_k + b$, solved exactly **[det given the split]**

`examples/dynamical_system_exp1.py` → `examples/dynamical_system_runs/exp1/`.

No optimiser and no epoch budget: all four losses are closed form at this experiment
(design record §3.2), so each number is the exact minimiser given its split. 5 split
seeds of 0.8/0.1/0.1 on the shipped 1000 rows; ± is the standard deviation across
those splits. **They share data, so they are not independent replicates** — the ± is
narrower than a true replication would give. Test split is 100 rows.

**Both controls pass.**

- `lra_toy_bw`: **1.000 ± 0.000 at every patch size from $p=1$ to $p=64$, all four
  losses.** One pixel separates a dark image from a bright one, so this is the
  positive control — the machinery works at $p=1$.
- `lra_image_mnist` at $p=1$: **0.106 ± 0.030 against chance 0.100.** $y_N$ sees the
  last pixel only, which in raster order is background. Negative control.

MNIST, terminal loss, across the patch homotopy:

| $p$ | 1 | 4 | 16 | 28 | 56 | 112 | 196 | 392 | 784 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test acc | 0.106 | 0.106 | 0.108 | 0.106 | 0.172 | 0.264 | 0.454 | 0.684 | **0.844** |

At $p=784$, $N=1$: that is exactly ridge regression on the flattened image, and **all
four losses return 0.844 ± 0.025 identically** — which they must, since with one step
there is no trajectory to weight. That identity is a useful implementation check.

**A pre-registered prediction, falsified.** Before running, the prediction was chance
through $p\approx112$ with the rise at $p\approx280$–$392$, on the argument that MNIST's
20×20 bounding box leaves rows 24–27 empty. Measured: the rise begins at $p=56$ and is
clear by $p=112$. Rows 24–27 carry **2.3% of total ink** (row 24 mean 0.056, row 25
0.021), enough to separate descending digits from non-descending ones. Direction right,
location wrong.

**Where the losses differ.** At intermediate $p$ the terminal loss is well ahead of
sum and weighted sum — 0.454 vs 0.238 / 0.288 at $p=196$ — because sum forces a single
$A$ to serve every patch position, while terminal fits the last position only.
`convergence+terminal` tracks `terminal` closely throughout (0.410 vs 0.454 there),
which is expected: it is terminal plus a Tikhonov term, and $\alpha$ is tuned.

### 6.2 `achieved(k)` at Experiment 1 — the null for the sign of life

The sign of life is whether `achieved(k)`, the accuracy of $\arg\max y_k$, **rises with
$k$** (design record §6). Experiment 1 cannot accumulate — $y_{k+1}$ depends on $x_k$ alone —
so its curve is the null, and it is not flat but a **hump**: on MNIST at $p=28$ under
sum loss it climbs from chance to ~0.30 around $k \approx 13$–$18$ and falls back to
chance by $k=28$, tracking the row-mass profile of the digit. Experiment 2's signal is a
*monotone rise* against that shape, not merely "above chance".

### 6.3 Experiment 2 — $y_{k+1} = Ax_k + By_k + b$ **[stat, 3 split seeds]**

`examples/dynamical_system_exp2.py` → `examples/dynamical_system_runs/exp2/`.
`lra_image_mnist`, full-batch Adam, 2000 steps at lr 3e-3, best-on-validation weights.
Test split is 100 rows, so $\mathrm{se} \approx 0.049$ near $p = 0.6$ and a single
matched pair needs to differ by ~0.14 to be resolved. The ± quoted is the spread over
3 splits of the same 1000 rows, so it is narrower than a true replication.

**The two implementations are bit-identical [det].** The direct recurrence and the
`Sequential2DRNN` block map on slots $[x,y]$ agree to **0.00e+00** — not merely
`allclose`. `check_two_implementations_agree` asserts it.

**Sign of life: yes, and large.** Terminal loss, cold, against Experiment 1 at the same
patch size:

| $p$ | $N$ | Experiment 1 | Experiment 2 cold | full-image LS |
| ---: | ---: | ---: | ---: | ---: |
| 28 | 28 | 0.106 | **0.593 ± 0.005** | 0.844 |
| 112 | 7 | 0.264 | **0.673 ± 0.012** | 0.844 |
| 196 | 4 | 0.454 | **0.720 ± 0.022** | 0.844 |

> **Re-measured at ten seeds** (see §6.4): the $p{=}28$ cold row is **0.565 / 0.239 /
> 0.253 / 0.513** on test for terminal / sum / weighted sum / convergence+terminal. The
> three-seed values above were optimistic by up to 0.064. The headline is unaffected;
> every other row in the table is still at three seeds and is good to ±0.03 typical,
> ±0.06 worst case (§6.5).

At $p=28$ that is chance → 0.565 at ten seeds, about 10 se; not in doubt. Reading 28 rows one at a
time through a **10-dimensional** state recovers 0.59 of the 0.844 available to a
model that sees the whole image at once.

**The losses produce qualitatively different trajectories, and this is the main
finding.** `achieved(k)` separates into two shapes:

- **sum and weighted sum** rise roughly monotonically — at $p=112$, from 0.08 at
  $k=1$ to 0.43 at $k=6$. These are genuinely running guesses.
- **terminal and convergence+terminal** sit near chance for the whole trajectory and
  then jump at the last step — at $p=28$, flat around 0.10-0.14 until $k \approx 25$,
  then 0.59 at $k=28$. Under these losses $y_k$ is **not a guess at all** until the
  end; it is an internal state that happens to have 10 coordinates.

So terminal wins on terminal accuracy and loses entirely on being a *running*
prediction. Which of those is wanted is a question about the goal, not about the data.

**CORRECTED: what $T+V$ actually does.** It was predicted here, from
$\max_k\lVert y_k-t\rVert \le T+V$, that convergence+terminal would be the *harsher*
trainer and would force uniformly-good trajectories, tracking sum loss. **It tracks
terminal loss instead.** The inequality is true; the behavioural inference drawn from
it was wrong. $T+V$ is minimised by a *short* path, and the shortest path from
$y_0 = 0$ to $t$ has length $\lVert t\rVert$ and passes nowhere near $t$ until it
arrives — so the bound's floor is $\lVert t \rVert \approx 1$, the same size as the
error it bounds, and it constrains early accuracy hardly at all. **$T+V$ penalises
path length, not earliness.**

**$\rho(B)$ concentrates at 1 when the sequence is long [det].** At $p=28$, $N=28$,
all eight runs land in $\rho(B) \in [0.97, 1.02]$ regardless of loss or arm. At
$N = 7$ and $N = 4$ it spreads (0.55 to 1.51). A linear recurrence given 28 steps
**learns a marginally-stable, near-identity recurrence on its own**, which is the
non-gated version of what §5 item 1 hypothesises gating supplies — and here every
eigenvalue is inspectable, which at $d_h = 2048$ it was not.

**Cold beats warm in 10 of 12 matched cells.** Freezing Experiment 1's $A$ and training only
$B$ — the boosting arm — was worse: 0.593 vs 0.487 at $p=28$ terminal, 0.673 vs 0.510
at $p=112$, 0.720 vs 0.657 at $p=196$. Individually these are 1-3 se on 100 test rows
and not decisive; the *consistency* across 3 patch sizes and 4 losses is the evidence,
and the cells are not independent. Read as: **an $A$ fitted to classify from one patch
alone is the wrong $A$ once $B$ can carry information**, so continuation-by-freezing
starts in a basin it then cannot leave. This is what Experiment 6 was going to ask, obtained
at Experiment 2.

> **Weakened by §6.5:** 12 cells at three seeds, where the measured movement to ten seeds is
> mean 0.027 / max 0.064 and paired differences are no more stable than absolute numbers.
> The gaps here are 0.06-0.16 — the same size as the noise. Not re-run at ten seeds.

**What the instrumented re-run added** (identical numbers to the first run, so the
instrumentation changed nothing; figures in `tasks/OVERVIEW_DYNAMICAL_SYSTEM.md` §7):

- **The optimisation budget is not binding.** Objectives and validation accuracy
  plateau by step 500-750 of 2000 — except terminal/cold, still rising slowly at 2000,
  so its 0.565 is a *lower* bound on what that configuration reaches.
- **The spectrum is a near-integrator plus a fast bulk [det].** At $N = 28$ the mean
  sorted $\lvert\lambda\rvert$ over 24 runs is 0.999, 0.970, 0.884, then a gap to
  0.674, 0.450, ... , 0.079. Roughly three slow modes carry memory and seven decay.
  At $N = 7$ and $N = 4$ eigenvalues sit outside the unit circle — **sequence length,
  not the loss, is what pins the spectrum**, since $\lvert\lambda\rvert^N$ only
  matters when $N$ is large.
- **CORRECTED mechanism.** The state-trajectory figure suggests that sum loss costs
  accuracy by holding $\lVert y_k \rVert$ small — terminal averages 2.317 over $k<N$
  against sum's 0.383. That explanation does **not** survive: convergence+terminal
  averages 0.444, essentially sum's scale, and scores 0.527 against sum's 0.303. The
  expensive constraint is **alignment with the one-hot target at every step**, not
  amplitude.

**Not established.** Whether cold-vs-warm survives more seeds or more data; Experiment 2 at
patch sizes other than 28, 112, 196; anything about Experiments 3-5.

### 6.4 Experiments 3, 4 and 5 — the $(\sigma, M)$ square **[stat, 3 split seeds]**

`examples/dynamical_system_exp3to5.py` → `dynamical_system_runs/exp3to5/`. One model with
two knobs covers four experiments: activation (identity or $\tanh$) and $M$, the number of
inner steps with $x_k$ held. Experiment 2 is the (identity, $M{=}1$) corner.

**All of the gain is in Experiment 2.** Experiments 3, 4 and 5 add nothing measurable.

**Experiment 3 ($\sigma$) is a null, and an earlier claim here is retracted.** Test
accuracy suggested $\tanh$ cost terminal loss 0.103 at $p{=}28$, and that was reported as
real. It is not: the training objectives agree within 3% (0.626 vs 0.610) and validation
accuracies within 0.013 (0.623 vs 0.613). A 0.103 gap on a 100-row test split is ~1.5 se.
**Retracted.**

What survives is **[det]**: $\tanh$ reaches the same training objective and validation
accuracy with **half the state amplitude**, mean $\lVert y_k\rVert$ 2.317 → 1.214. Losses
already inside $\tanh$'s linear region are unchanged to three decimals (0.383 → 0.383).
Saturation is 0.027 for terminal and exactly 0.000 for sum and weighted sum, so nothing
presses the rails — the large-amplitude regime the linear model builds is **available
rather than necessary**.

**Experiment 3's null and Experiment 5's gain are the same fact.** At $M{=}1$ the map is
applied once per patch, so the state never iterates toward $\sigma$'s fixed point — and
that fixed point is where $\sigma$'s value lies. Adding $\sigma$ alone buys nothing; adding
$\sigma$ *and* holding the input buys up to $+0.076$. A nonlinearity you never iterate is a
nonlinearity you have not used.

**Experiment 3b — genuine multiplicative mixing — is the largest effect in this work
[stat, n=10].** $y_{k+1} = \sigma(Ax_k + By_k + b) + (Cx_k)\odot(Dy_k)$: the only map here
in which $x$ and $y$ multiply. $p{=}28$, $M{=}1$, validation accuracy:

| loss | Exp 2 (linear) | Exp 3 ($\sigma$) | **Exp 3b (gated)** | 3b − 3 | $\rho(B)$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum | 0.241 | 0.257 | **0.417** | **+0.160** (5.5 se) | 1.02 → 1.11 |
| weighted sum | 0.243 | 0.267 | **0.454** | **+0.187** (8.2 se) | 1.00 → 1.11 |
| terminal | 0.583 | 0.577 | 0.541 | −0.036 (−1.4 se) | 1.07 → **0.91** |
| convergence+terminal | 0.575 | 0.600 | 0.589 | −0.011 (−0.3 se) | 1.10 → 1.15 |

Test agrees with validation throughout. **This settles the limit recorded in §11.6 of the
design record before any of it ran:** Experiment 3's null did not mean joint mixing fails,
it meant we had tested a *separable* pre-activation and called it mixing. Genuine
multiplication is worth +0.16 to +0.19 for the losses that ask for a running answer, at
$M{=}1$, with no inner iterations.

**The loss split appears a third time**, in a third map and at far greater magnitude. The
`achieved(k)` classification of §6.3 has now predicted the sign of three separate
interventions: $\sigma$, holding the input, and gating.

**A mechanism change visible only in the spectrum.** Terminal loss gains nothing in accuracy
(−1.4 se) while $\rho(B)$ falls 1.07 → 0.91, off the unit circle. §6.3 found Experiment 2
pinning $\rho\approx0.999$ because a near-identity path is the only way to carry memory 28
steps; the gate supplies an input-dependent path instead, relieving $B$ of the job. **That is
§5 item 1's gating hypothesis observed directly** — as a change of mechanism at unchanged
performance, which accuracy alone would not show.

**Registered predictions P7 and P9 (design record §10.1):** P7 — gating helps terminal most
— **refuted**; terminal is the only loss it does not help. P9 — $\rho(B)$ moves off 1 without
costing accuracy — **confirmed, for terminal specifically**, which is exactly where P7 failed.

**Combining gating with holding the input: gains compete, harms compound [stat, n=10].**
Validation change from the $\sigma$, $M{=}1$ baseline at $p{=}28$:

| loss | base | + gating | + holding | **+ both** | if additive |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum | 0.257 | +0.160 | +0.055 | **+0.167** | +0.215 |
| weighted sum | 0.267 | +0.187 | +0.074 | **+0.224** | +0.261 |
| terminal | 0.577 | −0.036 | −0.047 | **−0.083** | −0.083 |
| convergence+terminal | 0.600 | −0.011 | −0.057 | **−0.031** | −0.068 |

The gains are **sub-additive** (78% and 86% of the sum of parts); terminal's harm is
**exactly additive** to three decimals. When the two interventions help they are partly
substitutes competing for the same resource, so the trajectory losses hit a ceiling set by
something other than computation per patch; when they hurt, the costs simply add.

**Practical guidance the study ends on: if you add one thing, add the gate** — $+0.187$
against holding's $+0.074$, with no inner iterations, no extra sequential depth, and no
$\mu$ to tune.

**Experiment 4 confirms the containment derivation empirically.** §6's algebra says linear
$M{>}1$ is a strict subset of Experiment 2, so it cannot win. **Zero violations in 8
comparisons**, on test and validation, margins 0.08-0.18.

**Experiment 5 shows $\sigma$ does not rescue inner iterations.** Across Experiments 4 and
5, **0 of 16 rows beat their own $M{=}1$ value on validation.** One beats it on test (tanh
terminal, $M{=}2$: 0.527 vs 0.490) and that row reads 0.563 vs 0.613 on validation. The
difference-in-differences between tanh and identity is +0.107 on test but +0.020 on
validation — "$\sigma$ reduces the damage" is unresolved; "it helps" is refuted.

**Four equivalence controls, all bit-identical [det]**, each verified to fire under a
$10^{-6}$ perturbation: direct vs `Sequential2DRNN` block map; Experiment 4 $M{=}1$ vs
Experiment 2 (different script, different model class); Experiment 5 $M{=}1$ vs Experiment
3; and all four losses coinciding at $N{=}1$.

**ATTRIBUTION RESOLVED, and it inverts the result above.** Every $\mu = 1$ row moved two
things at once (§2.3): the model class shrank to a strict subset, *and* an inner-convergence
penalty switched on that was identically zero at $M{=}1$. The $\mu = 0$ controls hold the
penalty at nothing. **Median validation recovery is 88%** (mean 91%, range 17-183%), and
$\mu{=}0$ rows reach or beat their own $M{=}1$ value in **6 of 16** cells against **0 of 16**
with the penalty on.

So **"holding the input hurts" is an artefact of a penalty we put in the objective.** The
corrected claim: holding the input is roughly neutral; penalising it for not settling costs
0.12-0.15 accuracy.

**Mechanism [det] per run.** For sum and weighted sum the penalty crushes $\rho(B)$ from
~1.00 to **0.62-0.70**, and removing it restores ~1.05. Settling fast requires a strict
contraction, and a contraction destroys exactly the near-integrator memory path of §6.3.
**Convergence and memory are in direct tension.** Terminal and convergence+terminal keep
$\rho \approx 1$ in every condition, so a second, unmeasured mechanism operates there;
$\lVert A \rVert$ would separate them and is now recorded, though not for these runs.

**RESOLVED at ten seeds; two earlier readings here were RETRACTED on the way [stat, n=10].**
Change in validation accuracy from each arm's own $M{=}1$ row, at $\mu{=}0$, ten seeds in
every cell (se multiples in parentheses):

| loss | identity $M{=}2$ | identity $M{=}4$ | $\tanh$ $M{=}2$ | $\tanh$ $M{=}4$ |
| --- | ---: | ---: | ---: | ---: |
| sum | −0.022 (−0.9) | +0.016 (+0.7) | +0.055 (+1.9) | **+0.075 (+2.8)** |
| weighted sum | +0.014 (+0.6) | +0.037 (+1.7) | **+0.074 (+3.3)** | **+0.076 (+3.2)** |
| terminal | −0.004 (−0.2) | −0.055 (−2.2) | −0.047 (−1.7) | −0.117 (−3.5) |
| convergence+terminal | −0.067 (−2.1) | −0.127 (−2.7) | −0.057 (−2.4) | −0.162 (−5.9) |

**The loss type sets the sign** — 7 of 8 cells, under both activations; the exception is
$-0.9$ se. Losses whose `achieved(k)` rises and holds (§6.3) are non-negative; losses that
sit at chance and spike at the last step are negative. **The nonlinearity sets the
magnitude**: $\tanh$ gains reach $+0.075$ at 2.8-3.3 se, identity only $+0.037$ at $\le 1.7$ se.

**Why, from the containment derivation.** With linear $f$ the held-input fixed point
$y^\ast=(I-B)^{-1}Ax$ is a linear map of $x$, reachable in one step — the same fact as
linear Experiment 4 being a strict subset of Experiment 2 — so iterating sharpens what a
one-step model already represents. With $\sigma$, $y^\ast=\sigma(Ax+By^\ast+b)$ is not
reachable by any one-step linear model, so settling toward it is new computation.

**Conjunctive claim:** holding $x_k$ pays when the map is nonlinear **and** the loss asks
for a running answer. **Under terminal loss with a linear map — the two natural defaults,
and what `examples/lra_benchmark.py` uses — it is a straight loss**, $-0.127$ at $M{=}4$.

**Two retractions on the way, both the same error.** Read from Experiment 5 alone it looked
loss-driven and activation-independent ($\sigma$ and the loss family are confounded there);
read from Experiment 4 at $M{=}2$ alone it looked like "nothing gains without $\sigma$" (an
over-correction — $M{=}4$ shows weak positives). Both were two-variable tables read as
though one variable were fixed.

**Limits.** One dataset, one patch size, $M \le 4$; ten seeds vary the *split* of the same
1000 rows. Three cells clear 3 se; the rest is sign agreement across 16 cells.

**And the process point.** Had the Experiment 4/5 result been reported without this control,
the log would have recorded a confident negative that closed off a region of the space — the
§2.2 failure exactly. The control existed only because the $M$ axis visibly moved two things
at once.

### 6.5 Method: test accuracy is not readable at this sample size

Three times in one session a delta of 0.03-0.10 on the 100-row test split was contradicted
by validation accuracy. **Below ~0.14, a test-accuracy delta is not reportable on its own**
— it needs validation agreement or an independently measured mechanism.

**CORRECTED statement of why.** The first version of this section said validation "was right
each time", implying it is the more reliable measure. It is not. Re-running Experiment 5's
$M{=}1$ configuration at **10 seeds** instead of 3 moved the estimates by a mean of 0.018
and a maximum of 0.036 — and validation moved just as much as test (max 0.036 vs 0.030).
For terminal loss the two moved in **opposite directions**, test $+0.030$ and validation
$-0.036$.

So validation did not win by being less noisy. **When two independently noisy measures of
the same models disagree, the disagreement is itself the signal that neither has resolved
anything.** Agreement between them is weak evidence; disagreement is strong evidence of
nothing-to-see. That is the usable rule.

**Measured noise floor [det], from 8 configurations re-run at 10 seeds:** a three-seed
estimate here moves by a **mean of 0.027 and a maximum of 0.064** when taken to ten seeds.
So **any effect below about 0.06 measured at three seeds depends on which seeds were
drawn**, and two comparisons in this work flipped sign between 3 and 10.

**And a tempting correction that does NOT hold.** It is natural to argue that *differences*
between two losses should be more stable than absolute accuracies, because both are
measured on the same splits and the common split effect cancels. Measured over 12 loss
pairs: the difference moves by a mean of **0.029**, max 0.056 — **no better than the
marginals** (0.027 / 0.064). Different losses respond differently to the same split, so
there is no common effect to cancel. Paired comparison buys nothing here; only seeds do.

Two qualifications found while applying that rule. The **training objective is not
comparable across $M$**, because the inner-convergence term is identically zero at $M{=}1$
and non-negative beyond it. And an instrument that should have existed from the start:
none of the $\mu{=}1$ runs measure whether $y$ actually *converges* while $x_k$ is held —
the quantity the whole design is named after went into the objective and was never read
back off held-out data. `inner_gap` and `first_to_last_inner_gap` were added afterwards
and cover the $\mu{=}0$ rows only; backfilling costs ~50 minutes and is deferred.

---

## 7. Where things are

| | |
| --- | --- |
| `PRINCIPLES.md` | how to work here; read before writing code or reporting a result |
| `RESEARCH_LOG.md` | this file |
| `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` | design record, authoritative on every decision |
| `tasks/TODO_Sequential2DRNN.md` | deferred work and the experiment queue |
| `tasks/README_RCP.md` | RCP's review checklist with his responses |
| `README_Sequential2DRNN.md` | user-facing entry point for the module |
| `examples/lra_runs/README.md` | harness, config schema, cost model |
| `tasks/OVERVIEW_DYNAMICAL_SYSTEM.md` | the design record: settled decisions **and rejected readings** |
| `examples/dynamical_system_exp1.py` | Experiment 1, closed form; writes `dynamical_system_runs/exp1/` |
| `examples/dynamical_system_exp2.py` | Experiment 2, written twice and checked by `allclose`; writes `.../exp2/` |
| `examples/dynamical_system_exp3to5.py` | Experiments 3-5 as the $(\sigma, M)$ square; resumes from `results.json` |
