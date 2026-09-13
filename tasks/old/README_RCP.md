# RCP's list

Things waiting on **you** — decisions, reviews, and experiments. Written 2026-07-28,
at the end of the session that built `Sequential2DRNN`.

This is deliberately not a design document. The *why* lives in
`OVERVIEW_RNN_SEQUENTIAL_2D.md`; the *how* lives in `README_Sequential2DRNN.md`.
This file is only the list of things not yet settled, with pointers.

---

## 1. Nothing is committed yet

Everything below is uncommitted work on `develop`. Last commit is `e4fbaac`.

```
 M  OVERVIEW_RNN_SEQUENTIAL_2D.md          design doc, heavily extended
 M  README.md                              two new entries in the file/notebook lists
 ?? README_Sequential2DRNN.md              user-facing entry point
 ?? README_RCP.md                          this file
 ?? iterativennsimple/Sequential2DRNN.py   the module (302 lines)
 ?? notebooks/7-rcp-RNN-as-Sequential2D.ipynb
 ?? notebooks/8-rcp-fixed-points-and-bistability.ipynb
 ?? tests/test_Sequential2DRNN.py          14 unit tests
 ?? tests/test_7-...ipynb                  symlink -> notebooks/7-...
 ?? tests/test_8-...ipynb                  symlink -> notebooks/8-...
```

`uv run pytest` — **91 passed**.

- [ ] Review, then commit. Probably worth a `CHANGELOG.md` entry under a new
      version heading, since the existing changelog tracks at that granularity.
RCP - I made the commit, so create an appropriate CHANGELOG.md entry.
- [ ] Decide whether `Sequential2DRNN` should be exported from
      `iterativennsimple/__init__.py`. Right now that file exports only
      `MonarchLinear`, so the current import path (`from
      iterativennsimple.Sequential2DRNN import Sequential2DRNN`) matches how
      `Sequential2D` is used everywhere else. Consistent, but easy to miss.
RCP - That is fine and can left that way for the moment.

---

## 2. Decisions I made that you have not confirmed

Each is defensible and each is reversible cheaply *now* and expensively later.

- [ ] **The name `Sequential2DRNN`, in its own file.** Chosen to match
      `Sequential1D` / `Sequential2D`. §10.2 of the overview had "module name and
      file location" as an open question and I picked one to make progress.
RCP - I am fine with the name and the file location.

- [ ] **`from_rnn` builds a *two*-slot map $[x, h]$, not the three-slot $[x, y, h]$
      of §5.2.** `torch.nn.RNN` has no readout, so the $y$-slot would be
      vestigial. This is the one place the code deliberately diverges from the
      overview; flagged at the top of that document. `from_3x3` is where the
      three-slot layout lives.
RCP - I am confused.  How is the final answer $y$ created?  Isn't that normally a linear layer applied to the hidden state $h$?  I think it would be cleaner to this build a three-slot map for this reason.  Is there a reason not to do this?

- [ ] **`from_3x3` takes its six blocks as ordinary keyword arguments.**
      Given how the source-first/target-first confusion went, making them
      **keyword-only** would turn a reversed name into a `TypeError` instead of a
      silently misplaced block. I did not do it. I think it is worth doing —
      the naming convention is exactly the kind of thing that will be got wrong
      again, and this makes the failure loud. Your call.

RCP - Great idea!  I would definitely make them keyword-only arguments.  I would also add a check that the blocks are all 2D tensors and that they have compatible shapes.  That should help catch some tranpose errors.

- [ ] **Notebook 8 loads plotly.js from a CDN** (`pio.renderers.default =
      'notebook_connected'`). Without it the notebook is 5.3 MB, against 1.5 MB
      for your largest existing one. Cost: the plots need a network connection to
      view. Switch the one line to `'notebook'` if you would rather have it
      offline and large.

RCP - That is fine, but note I moved Notebook 8 to notebooks/advanced/12-claude... . I am not sure this is on the critical path.  I like it, but I don't have time to review it in detail right now.      

---

## 3. Pre-existing, unrelated, not touched

- [ ] **Four broken notebook symlinks in `tests/`.** The notebooks were renamed
      and the links were not updated, so these four are not being tested at all:

      test_2-rcp-seqential-2D-problems.ipynb  -> 2-rcp-seqential-simple-problems.ipynb
      test_3-rcp-iterated-2D-problems.ipynb   -> 3-rcp-iterated-simple-problems.ipynb
      test_4-rcp-pulled-apart.ipynb           -> 5-rcp-pulled-apart.ipynb
      test_5-rcp-MLP.ipynb                    -> 4-rcp-MLP.ipynb

      Note the last two also have their numbers crossed. Say the word and I will
      fix them; I left it alone as out of scope.

RCP - I fixed these!

- [ ] **`Sequential2D.forward_vector` drops `dtype`** (`Sequential2D.py:127`):
      `torch.zeros((...), device=X_in.device)` with no `dtype=`, so the
      accumulator silently promotes to fp32 under AMP or bf16. `Sequential2DRNN`
      sidesteps it by using `forward_list`, and `test_dtype_is_preserved` pins
      that. The underlying bug is still there for other callers.

RCP - This is a good catch.  Can you fix this throughout the code base?  I think it is a good idea to add a test that checks that the dtype is preserved for all forward methods. 

---

## 4. Open design questions

Recorded in §10 of the overview. Listed here so they are not lost.

- [ ] **`bidirectional`** — mechanically a second parameter set over reversed
      time. Not scoped.
RCP - not needed now
- [ ] **`PackedSequence`** — real `nn.RNN` accepts it; meaningfully more code.
      Do you need it?
RCP - This might be worthwhile, I want to be able to run experiments with PackedSequence.  Is there some way to segragate the code so that adding this does not mess up the rest of the code?
- [ ] **`dropout`** — mechanical, but confirm it is in scope at all.
RCP - not needed now
- [ ] **The rest of the $x$-row, $W_{yx}$ and $W_{hx}$** (§10.1). Inert at
      $K = 1$, live at $K > 1$, and in direct tension with input persistence —
      they cannot both be on. Revisit once $K > 1$ experiments are actually
      running.
RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.
- [ ] **Multi-hidden-slot states $z = [x, y, h_1, \ldots, h_k]$** (§4.1).
      Deferred on purpose; the blocking question is whether the $h_i$ share a
      timescale. The code is $N$-slot generic so nothing precludes it, but the
      per-slot-$K$ machinery does not exist.
RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.
- [ ] **Should $K$ vary** — per slot, per token, adaptive? Note §10.3: if the
      internal iteration converges, adaptive $K$ falls out for free by stopping
      when $\|z^{(k)} - z^{(k-1)}\| < \epsilon$.
RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.

---

## 5. Experiments worth running

Roughly in order of how much I expect them to teach.

- [ ] **Measure the trained spectrum.** Everything in
      `examples/rnn_internal_iterations.py` is measured *at initialisation*, which
      makes "a random contraction has no memory" nearly a tautology. Train, then
      measure the spectral radius of $J_{T_x}$. **If training pushes it past 1 on
      its own, the orthogonal initialisation in that script is a convenience
      rather than a fix**, and the headline result of that example is much
      weaker than it reads.
RCP - This is a nice idea, but let's wait until we have a working training loop.  I will have suggestions for some experiments to run once we have a working training loop.  I think we should focus on getting the training loop working first, and then we can run these experiments.

- [ ] **Factor the diagnostic.** The script measures $\partial h_t / \partial x_0$,
      which is a *product* — roughly
      $\|\partial\Phi/\partial x\| \cdot \|\partial\Phi/\partial h\|^{t}$ — so it
      cannot distinguish a model that stopped *responding* from one that stopped
      *remembering*. Measure the two separately. §10.3.

RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.

- [ ] **A compute-bound task rather than a memory-bound one.** Copy-with-delay is
      pure memory, which is exactly what iterating a contraction destroys, so
      $K > 1$ was never going to win there. Something needing several steps of
      work on the *current* token is where $K > 1$ has somewhere to put the
      effort. This is the experiment that would actually test the hypothesis.

RCP - Yes!  Wait until we talk about the training loop and then we can discuss what tasks to run.  I have some ideas for tasks that would be good to run.     

- [ ] **The convergence loss — carefully.** §10.3 argues that the obvious
      $\|z^{(K)} - z^{(K-1)}\|$ is *globally minimised by the memoryless
      solution*, so it points at the failure mode rather than merely permitting
      it. If you try it, monitor $\partial\Phi/\partial h$ alongside, and consider
      the spectral form instead (push $\rho(J_T)$ to just below 1, not to 0) —
      which is where the DEQ people landed for the same problem (Bai, Koltun &
      Kolter 2021).

RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.

- [ ] **Sparse $W_{hh}$ at larger $K$ versus dense at $K = 1$, equal parameter
      count.** `MonarchLinear` and `MaskedLinear` drop straight in and are tested.
      This is the sparsity question the paper is actually about, and it is the
      cheapest of these to run.

RCP - yes!  This is exactly something I want to run.  I think we should run this experiment early in the testing process.  Let's discuss when we discuss tasks to run.

- [ ] **Two hidden units.** Notebook 8 ends by pointing at it: complex multipliers
      admit a Neimark–Sacker bifurcation, so the internal iteration could converge
      to an invariant circle rather than a point. Whether that is a useful kind of
      memory or merely pretty is genuinely open. Your call whether it is worth a
      notebook 9.

RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.

- [ ] **The scalar leak $\lambda$** (§8.6). The sanctioned relaxation of
      $M_{xx} = I$: $\lambda^{k}$ decay of the forcing, interpolating $I$ at
      $\lambda = 1$ and absent at $\lambda = 0$. Meaningless at $K = 1$. Note the
      caveat from §10.3 — at $\lambda < 1$ the landscape *drifts* during the
      internal iteration, so "the fixed point for this input" stops being well
      defined, which is a real conceptual cost and not just a numerical one.

RCP - yes, this is out of scope for the moment, but can be put onto a TODO list for future work.
\
---

## 6. Things I would like a second opinion on

Places where I am least confident, in descending order.

1. **§10.3 as a whole.** It is the newest material and came together in the last
   few exchanges of the session, so it has had far less scrutiny than §8. The
   convergence-versus-contraction distinction, the claim about the naive loss,
   and the three-way diagnostic are all plausible and none is checked.

RCP - I think it is fine, but I will just point out that training should take care of this.  I.e., if different $y$ outputs are part of the training process, and we achieve low loss, then the model will have learned to converge to the *various* correct $y$ outputs. Do you agree?

2. **Notebook 8's framing.** It is the furthest into your territory and it is
   where I made and corrected the most errors — a root finder that double-counted
   exact roots, and $w_{hh} = 1$ mislabelled as unstable when it is
   non-hyperbolic. Both are fixed, but it is the file most likely to still contain
   something wrong.

RCP - As I mentioned, I moved Notebook 8 to notebooks/advanced/12-claude... . I am not sure this is on the critical path.  I like it, but I don't have time to review it in detail right now.      

3. **Whether the wavefront decision (§8.5) is right.** Giving up exact
   `nn.RNN` stacking to keep the construction a single block map seemed clearly
   correct at the time and you agreed, but it is the largest single thing this
   module does not do.

RCP - Your grammar is a little strange here.  Just to confirm, you are overwriting h as a function of K?  So when you say "giving up exact nn.RNN stacking", you mean that we are not stacking the RNNs in the usual way, but instead we are iterating and overwriting the hidden state K times?  I think this is fine, but I just want to make sure I understand what you mean.

---

## 7. Standing note

`OVERVIEW_RNN_SEQUENTIAL_2D.md` §3.3 records your instruction that transposes are
your known weak point and that orientation claims in your prose should be
re-derived rather than trusted. That already caught one real error this session:
the six-block list in `from_3x3` was written target-first while the matrix in
§5.2 was source-first, which would have silently excluded $W_{xh}$ — the
input-to-hidden matrix — from the exposed blocks. Kept in the overview as a
worked example of the detection method.
