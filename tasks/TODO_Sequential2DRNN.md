# TODO: `Sequential2DRNN` future work

Items RCP marked "out of scope for the moment, put on a TODO list" during the
review of `README_RCP.md` on 2026-07-28. Nothing here is on the critical path.

**The critical path is the training loop.** RCP wants that working before any of
the experiments below, and has tasks in mind to propose once it exists. Several
items are marked as blocked on that.

Design rationale for everything here is in `OVERVIEW_RNN_SEQUENTIAL_2D.md` (same directory); the
section numbers are the pointers.

---

## Features not implemented

- [x] **`PackedSequence` support.** ~~Not started.~~ **Done** —
      `Sequential2DRNN.forward_packed`. It did segregate cleanly: $M$, $b$, $A$,
      `Inject`, `internal_step` and `external_step` are all untouched, because
      packing only changes *how many rows of the batch are alive* at each step.
      Validated against `nn.RNN` for both `enforce_sorted` settings and against
      per-sequence runs for a three-slot state with feedback.
- [ ] **`bidirectional`.** Mechanically a second parameter set run over reversed
      time, outputs concatenated. Not needed now.
- [ ] **`dropout`.** Not needed now.
- [ ] **The rest of the $x$-row, $W_{yx}$ and $W_{hx}$** (§10.1). Inert at
      $K = 1$, live at $K > 1$, and in direct tension with the input-persistence
      invariant of §8.6 — they cannot both be switched on. Revisit only once
      $K > 1$ experiments are running.
- [ ] **Multi-hidden-slot states $z = [x, y, h_1, \ldots, h_k]$** (§4.1). The
      blocking question is semantic, not technical: do the $h_i$ share a
      timescale? The core is already $N$-slot generic, so nothing precludes this,
      but per-slot $K$ does not exist. This is where §8.5's wavefront discussion
      finally becomes non-vacuous.
- [ ] **Varying $K$** — per slot, per token, adaptive. Note §10.3: if the internal
      iteration converges, adaptive $K$ falls out for free by stopping when
      $\|z^{(k)} - z^{(k-1)}\| < \epsilon$, which is Graves' adaptive computation
      time derived from the dynamics rather than bolted on.
- [ ] **The scalar leak $\lambda$** (§8.6). The sanctioned relaxation of
      $M_{xx} = I$: $\lambda^{k}$ decay of the held forcing, interpolating $I$ at
      $\lambda = 1$ and absent at $\lambda = 0$. Meaningless at $K = 1$. Carries a
      conceptual cost flagged in §10.3 — at $\lambda < 1$ the attractor landscape
      *drifts* during the internal iteration, so "the fixed point for this input"
      stops being well defined.

---

## Experiments — blocked on the training loop

RCP has task suggestions to make once a training loop exists; do not pick tasks
unilaterally.

- [ ] **Measure the trained spectrum** (highest value of these). Everything in
      `examples/rnn_internal_iterations.py` is measured at *initialisation*, which
      makes "a random contraction has no memory" nearly a tautology. Train, then
      measure the spectral radius of $J_{T_x}$. If training pushes it past 1 on its
      own, the orthogonal initialisation in that script is a convenience rather
      than a fix, and that example's headline is much weaker than it reads.
      **RCP: "yes! This is exactly something I want to run... run this experiment
      early in the testing process."**
- [ ] **A compute-bound task rather than a memory-bound one.** Copy-with-delay is
      pure memory, which is precisely what iterating a contraction destroys, so
      $K > 1$ was never going to win there. Something requiring several steps of
      work on the *current* token is where $K > 1$ has somewhere to put the
      effort. This is the experiment that would actually test the hypothesis.
      **RCP: discuss task choice first.**
- [ ] **The convergence loss.** See §10.3 and the amendment recording RCP's
      point that training on varied $y$ should itself prevent the degenerate
      solution. The remaining question is one of optimisation dynamics, not of
      optima, and it is settled empirically rather than by argument: sweep the
      loss weight and watch whether task loss ever drops.
- [ ] **Factor the memory diagnostic.** The current measurement,
      $\partial h_t / \partial x_0$, is a *product* of the two sensitivities that
      matter and cannot distinguish a model that stopped *responding* from one
      that stopped *remembering*. Measure $\partial\Phi/\partial x$ and
      $\partial\Phi/\partial h$ separately (§10.3).
- [ ] **Sparse $W_{hh}$ at larger $K$ versus dense at $K = 1$, equal parameter
      count.** `MonarchLinear` and `MaskedLinear` drop in and are already tested.
      This is the sparsity question the paper is actually about, and the cheapest
      of these to run.

---

## Findings from the overnight LRA image runs (2026-07-29)

Four runs on `lra_image`, all sharing `split_seed: 0` so rows are comparable across
directories. Tables in `examples/lra_runs/*/results.md`.

### Answered

- [ ] **RETRACTED: "width helps once lr is scaled".** This was reported as settled
      and is not. The claim came from comparing, per width, the *best* result over
      the lr values tried -- but the grids were **unequal**: $d_h$ 128 was run at one
      lr (1e-3), 512 at two (3e-4, 1e-4), 2048 at three (3e-4, 1e-4, 3e-5). A maximum
      over unequal sample counts favours whichever row got more samples, and the
      grids do not even overlap at $d_h = 128$, so no matched-lr comparison including
      it exists.

      Every **matched-lr** comparison available says width *hurts*:

      | comparison | narrow | wide |
      | --- | ---: | ---: |
      | `image_wide_lr`, lr 3e-4 | h=512: 0.1490 | h=2048: 0.1220 |
      | `image_wide_lr`, lr 1e-4 | h=512: 0.1940 | h=2048: 0.1450 |
      | `image_wide`, lr 1e-3, 20 ep | h=128: 0.153 | h=512: 0.132, h=2048: 0.111 |

      **The width question is open.** Settling it needs the same lr grid at every
      width -- at minimum {1e-3, 3e-4, 1e-4, 3e-5} x {128, 512, 2048} at fixed
      epochs. Note also that `image_wide` ran 20 epochs and `image_wide_lr` ran 30,
      so the two runs differ in *two* variables and the earlier claim that lr alone
      explained the difference was unsupported.
- [x] **Epoch budget dominated the comparisons made before it was fixed.** `nn.GRU h=512` sat at 0.159
      val at epoch 15, 0.326 at 20, and 0.447 at 30 — still climbing. Any comparison
      made before epoch ~16 on this task compares models that have not started
      learning. Check the `epoch` column against the epoch budget: a best epoch at
      or near the last one means the number is a lower bound. (A `was_truncated()`
      helper used to flag this automatically and was removed — a reader sees it from
      the table unaided, so the detector was machinery standing in for noticing.)
- [ ] **Monarch as a regulariser: UNRESOLVED, not answered.** At $d_h = 2048$
      fixed, seq 256, one seed, all rows at lr 1e-4 (val): dense 0.243, nb=2 0.239,
      nb=4 0.205, nb=8 0.161, nb=16 0.151, and the parameter-matched control
      dense $h{=}724$ 0.239.

      With 1000 evaluation rows near $p = 0.2$, differences below about 0.04 are not
      resolvable. Taking $z = |\Delta| / \sqrt{\mathrm{SE}_1^2 + \mathrm{SE}_2^2}$:

      | comparison | val $z$ | test $z$ | |
      | --- | ---: | ---: | --- |
      | dense vs nb=2 | 0.21 | 1.31 | tied |
      | nb=2 vs nb=4 | 1.83 | 1.87 | unresolved |
      | nb=8 vs nb=16 | 0.62 | 0.66 | tied |
      | dense $h{=}724$ vs nb=4 | 1.83 | 1.22 | **unresolved** |
      | dense vs nb=8 | 4.59 | 6.70 | real |
      | dense vs nb=16 | 5.21 | 7.36 | real |

      So what the run supports: **heavy sparsity (nb $\ge$ 8) is clearly worse than
      dense or nb=2.** Nothing else. In particular the parameter-matched control —
      previously called "decisive" here — is $z = 1.2$ on test and settles nothing.
      An earlier version of this entry described the five numbers as "falling
      monotonically"; three of the four gaps are below the resolution limit.

      Two further limits: one seed, so seed-to-seed variance is entirely unmeasured
      and the $z$ values above are a floor on the true uncertainty; and the setting
      was weakly powered as a regularisation test — dense's train/val gap was only 0.04, and a regulariser needs a model
      that overfits substantially. A real test needs a regime with a large
      generalisation gap.

      **This sweep is internally clean but does not transfer to the LRA task.** All
      five block-count rows and the parameter-matched control ran at the same lr
      (1e-4), epochs (30) and `step_size` (4), so the *ranking by block count* is
      sound. But `step_size: 4` is not the LRA definition: it makes `seq_len` 256
      instead of 1024 **and** `input_size` 4 instead of 1, so the model sees four
      adjacent pixels per step. That is a 1D patch embedding -- a different task with
      a wider input, not a shorter version of the same one. The value 4 was chosen
      arbitrarily; the argument for shortening gave a direction, not a number.
      Re-run at `step_size: 1` before quoting this against anything at seq 1024.
- [x] **More Monarch blocks costs more time, not less.** 8.4 -> 18.2 -> 24.7 -> 47.6
      -> 92.1 s/epoch as $W_{hh}$ parameters fell 4.19 M -> 0.033 M: an 11x slowdown
      for a 128x parameter reduction. (An earlier revision said 4.23 M -> 0.06 M and
      70x, mixing the *total* parameter count into a $W_{hh}$ comparison.
      $|W_{hh}| = 2 d_h^2/\text{nb}^2$.) Each block is a small matmul plus a gather and we are
      launch-bound, so cost is roughly linear in `nb` while FLOPs fall as
      $1/\text{nb}^2$. The FLOP-saving role of block count needs $d_h$ in the tens of
      thousands. `MonarchLinear.forward(use_views=False)` is 1.8-2.2x faster and is
      used via the `MonarchNoViews` wrapper in `examples/lra_benchmark.py`, since
      `Sequential2D` cannot pass keyword arguments to blocks.
- [x] **Flat compute cost in width is about wall-clock only, not trainability.**
      Our s/epoch is flat (29.9 / 29.4 / 29.1 at $d_h$ 128 / 512 / 2048) while cuDNN
      GRU goes 5.2 -> 20.5. But `nn.GRU h=2048` diverged to NaN at both lr 1e-3
      (epoch 7) and 3e-4 (epoch 9), and at 1e-4 reached only 0.170. Width being free
      to *run* is not width being free to *train*, and §9.6 should not be read as
      implying otherwise.

### Where the comparison stands

**Only compare within a `step_size`.** `step_size` sets both `seq_len` and
`input_size`, so runs at different values are different tasks. An earlier version of
this table listed a `step_size: 4` row next to `step_size: 1` rows, which invited
reading 0.252 > 0.171 as progress. It is not: those are different problems and the
first also has a 4x wider input.

At `step_size: 1` (seq 1024), the real task:

| | best test |
| --- | ---: |
| `nn.GRU` h=512, lr 1e-3, 30 ep | **0.480** |
| ours, dense h=2048, lr 3e-5, 30 ep | 0.171 |
| ours, dense h=128, lr 1e-3, 30 ep | 0.180 |

At `step_size: 4` (seq 256), a different and easier task:

| | best test |
| --- | ---: |
| ours, dense h=2048, lr 1e-4, 30 ep | 0.252 |

Our module trains and is roughly 2.5x behind a gated baseline that needs no lr
tuning. Whether it *benefits from width* is unresolved -- see the retraction above.
Most of our rows are truncated and therefore understated.

### The next question, and it is now well posed

GRU learns on this task and a vanilla recurrence barely does, at matched width and
learning rate. The plausible reason is that gates supply near-identity paths through
1024 steps, which a fixed contraction has none of — and **the module already has the
machinery to supply one without hand-designed gates**: $M_{xx} = I$ does exactly this
for the input channel (§8.6). §8.6 explicitly notes that an identity on the *hidden*
diagonal, $W_{hh} \supseteq I$, is a separate and unexplored choice. That is a far
better experiment than another learning-rate sweep, and it is cheap.

Note two ideas that have now failed and should not be retried as-is: orthogonal
initialisation at gain 1.2 and at gain 1.0 (§10.3, and the `image_full` /
`image_wide` tables).

---

## Findings from the LRA image runs (2026-07-28)

Recorded so the negative results are not rediscovered. Full tables in
`examples/lra_runs/*/results.md`.

- [x] **Cost is flat in hidden width.** 30.1 / 29.6 / 29.3 s per epoch at
      $d_h$ = 128 / 512 / 2048 (seq 1024, batch 64) — 1.03x over a 16x width
      increase, against cuDNN GRU's 39x. The Python loop is launch-bound, so width
      is nearly free for us and expensive for cuDNN. This is the single most useful
      fact for planning experiments; see §9.6 of the overview.
- [x] **`Sequential2DRNN` K=1 is `nn.RNN`.** Proven by
      `tests/test_Sequential2DRNN.py::test_matches_torch_rnn_across_options`, which
      asserts `allclose` at 1e-6 on both `output` and `h_n` across `tanh`/`relu`,
      both `batch_first` settings, and several sequence lengths. That is the
      evidence. Corroborating but *not* evidence: their LRA accuracies agree
      (0.153 vs 0.149, $z = 0.2$, which at one seed proves nothing), and their
      parameter counts differ by exactly 128 = `hidden_size` = the folded second
      bias.
- [x] **Everything is reproducible to four decimals** across runs sharing a
      `split_seed`, both for cuDNN and for our loop.
- [ ] **Orthogonal init has now failed twice.** `gain=1.2` (image_full) and
      `gain=1.0` (image_wide) both did nothing at $L = 1024$: 0.133 and 0.131
      against 0.132 for default init. The memory-horizon argument in
      `examples/rnn_internal_iterations.py` was established at $L = 20$, $K \le 4$
      and **at initialisation**. Treat it as unsupported at long sequence length
      until measured on a *trained* model at $L = 1024$. Do not offer another gain
      value as a fix without that measurement.
- [ ] **The width question is still open, and the first attempt was botched.**
      `image_wide` held `lr = 0.001` across all widths; train loss rose above
      $\ln 10$ and worsened monotonically with width, and GRU h=2048 diverged. That
      is a step-size artefact, not a capacity result. `image_wide_lr/` crosses
      $d_h$ with `lr` and is written but not yet run.
- [ ] **More Monarch blocks costs more time, not less, at these sizes.** At
      $d_h = 2048$: per-call 0.024 ms dense, 0.124 (nb=2), 0.195 (nb=4), 0.771
      (nb=16) — roughly linear in `nb` while FLOPs fall as $1/\text{nb}^2$, because
      each block is a small matmul plus a gather and we are launch-bound. The
      FLOP-saving role of block count needs $d_h$ in the tens of thousands. The
      *regularisation* role is untested; `image_monarch/` is written but not run.
      `MonarchLinear.forward(use_views=False)` is 1.8-2.2x faster and is now used
      via a wrapper, since `Sequential2D` cannot pass keyword arguments to blocks.
- [ ] **`torch.nn.GRU` h=2048 diverges** at `lr=1e-3` on this task (NaN at epoch 7).
      Not our code. Note `clip_grad_norm_` cannot save a run once a single NaN
      exists — the total norm becomes non-finite and the rescale poisons every
      parameter. The harness now stops at the first non-finite loss and flags the
      row.

---

## Exploratory

- [ ] **Two hidden units.** `notebooks/advanced/12-claude-fixed-points-and-bistability.ipynb`
      ends by pointing at it: complex multipliers admit a Neimark–Sacker
      bifurcation, so the internal iteration could converge to an invariant circle
      rather than to a point. Whether that is a useful kind of memory or merely
      decorative is genuinely open.
- [ ] **Review notebook 12 in detail.** RCP likes it but has not had time to check
      it closely, and it is the file where the most errors were made and corrected
      during authoring. Not on the critical path.

---

## Housekeeping

- [ ] **Export `Sequential2DRNN` from `iterativennsimple/__init__.py`?** Currently
      that file exports only `MonarchLinear`, so the present import path matches
      how `Sequential2D` is used everywhere else. RCP: fine as is for now.
