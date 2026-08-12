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

## 6. Where things are

| | |
| --- | --- |
| `PRINCIPLES.md` | how to work here; read before writing code or reporting a result |
| `RESEARCH_LOG.md` | this file |
| `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` | design record, authoritative on every decision |
| `tasks/TODO_Sequential2DRNN.md` | deferred work and the experiment queue |
| `tasks/README_RCP.md` | RCP's review checklist with his responses |
| `README_Sequential2DRNN.md` | user-facing entry point for the module |
| `examples/lra_runs/README.md` | harness, config schema, cost model |
