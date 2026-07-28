# TODO: `Sequential2DRNN` future work

Items RCP marked "out of scope for the moment, put on a TODO list" during the
review of `README_RCP.md` on 2026-07-28. Nothing here is on the critical path.

**The critical path is the training loop.** RCP wants that working before any of
the experiments below, and has tasks in mind to propose once it exists. Several
items are marked as blocked on that.

Design rationale for everything here is in `OVERVIEW_RNN_SEQUENTIAL_2D.md`; the
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
