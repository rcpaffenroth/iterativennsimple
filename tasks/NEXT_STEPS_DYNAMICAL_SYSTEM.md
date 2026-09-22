# Next steps — the dynamical-system experiments on other LRA datasets

*A note to whoever picks this up. Written at the end of the session that produced
Experiments 1-5 and 3b on MNIST, by the assistant that ran them, for the assistant that
runs the next round.*

**Read `tasks/OVERVIEW_DYNAMICAL_SYSTEM.md` first** — it is the single document for this
work: design decisions, results, figures, re-run instructions, and what was rejected.
`RESEARCH_LOG.md` §6 has the same results chronologically. `PRINCIPLES.md` governs how to
write code and report results here and is not optional. This file is only about **what to do
next and what to avoid**.

---

## 1. The hypothesis this round exists to test

**RCP's, and it is the reason for this round:** *the interesting behaviour — particularly
what extra iterations buy — may be masked by how simple the current datasets are.*

The evidence that he is right is already in our own numbers:

- MNIST is **nearly linearly separable**. Whole-image ridge regression gets **0.844** (§3).
  A model that can almost solve the task in one step has little reason to iterate.
- Holding the input for $M$ inner steps bought at most **+0.075**, and only for the losses
  that ask for a running answer (§7.1).
- Adding a multiplicative gate bought **+0.187** — two and a half times as much (§5.1).

Read together: on MNIST, **per-step expressiveness beat iteration decisively.** That is
exactly what you would expect if the task needs a better one-step map rather than a
computation unrolled over time. It is *not* evidence that iteration is useless; it is
evidence that MNIST cannot tell us.

**The prediction to test:** on a task that genuinely requires composition or tracing, the
$M$ effect should grow and may overtake gating. If it does not — on ListOps, which is
*definitionally* compositional — that is a real negative about iteration in this
architecture, and worth far more than another MNIST number.

---

## 2. Do this first. It is free and it decides everything else.

**Run Experiment 1's closed-form solve at $p =$ the full feature count, on every LRA
dataset.** That is one `lstsq` per dataset — seconds — and it gives the **linear ceiling**:
what a single linear map of the whole input achieves.

That number is the screening instrument for this entire round:

- **Linear ceiling near chance** → the task cannot be done in one linear step → iteration
  has somewhere to go. **These are the datasets worth running.**
- **Linear ceiling high** (MNIST's 0.844) → a one-step map nearly suffices → you will
  measure what we already measured.

`examples/dynamical_system_exp1.py` already does this; point `PATCH_SIZES` at
`{name: [n_features]}` and read the $N{=}1$ row. Do this before committing compute to
anything below. It is the cheapest decisive measurement available and it directly serves
`PRINCIPLES.md` §2.12 — *a dataset is an object to be measured, not a given.*

---

## 3. The datasets, in the order they are worth doing

All feature counts are powers of two or $784$, so **$N{=}32$ is reachable on every one** —
which matters, because holding sequence length fixed across datasets keeps $N$ from
confounding the comparison (`PRINCIPLES.md` §2.3).

| dataset | rows | features | classes | $p$ for $N{=}32$ | blocker |
| --- | ---: | ---: | ---: | ---: | --- |
| `lra_image` (CIFAR-10 grey) | 10000 | 1024 | 10 | 32 | **13.6% train/test leakage** |
| `lra_listops` | 10000 | 2048 | 10 | 64 | needs an embedding |
| `lra_pathfinder` | 10000 | 1024 | 2 | 32 | 2-dim state (§4) |
| `lra_text` | 10000 | 4096 | 2 | 128 | embedding **and** 2-dim state |
| `lra_pathx` | 2000 | 16384 | 2 | 512 | both, plus only 2000 rows |

### Tier 1 — `lra_image`, the drop-in

Ten classes, continuous values, 1024 features. **Runs with no code changes at all**: add it
to `DATASET`/`PATCH_SIZES` and go. Harder than MNIST (linear baselines on grey CIFAR are
usually ~0.25-0.30, but *measure it*, §2).

**Blocker, and do not skip it:** `lra_image` has **13.6% of its test set duplicated in its
training set** from sampling with replacement (`RESEARCH_LOG.md` §3.5). Either regenerate it
with `rng.choice(..., replace=False)` in `generatedata`, or deduplicate at split time so no
test row has a twin in train. Every `lra_image` number in `RESEARCH_LOG.md` sits on this and
the effect on accuracy has never been measured.

### Tier 2 — `lra_listops`, the one that actually tests the hypothesis

Ten classes, so the state stays 10-dimensional. **Definitionally compositional** — evaluating
nested operations is the thing iteration is supposed to be for. If holding the input pays
anywhere, it pays here; if it does not pay here, that is a substantive negative.

**Blocker:** the values are token IDs, not continuous measurements. Treating them as real
numbers is wrong — token 7 is not "seven times" token 1. It needs an embedding before the
recurrence, which is **new machinery and a new axis** (`examples/lra_benchmark.py` has the
pattern in `build_model`). Budget for that being the actual work.

### Tier 3 — the binary tasks, and why they need Experiment 6 first

`lra_pathfinder`, `lra_text`, `lra_pathx` are two-class. **In this architecture $y_k$ *is*
the prediction, so a two-class task gives you a two-dimensional state.** That is not a small
inconvenience — it is a memory bottleneck so severe it will mask everything you are trying to
measure, and it confounds dataset with state capacity.

**This is the concrete argument for building Experiment 6 ($h_k$) before touching them.**
Internal state decoupled from the class count is precisely what these tasks need, and until
now Experiment 6 had no forcing reason to exist. It has one now.

---

## 4. Keep these fixed. They were expensive to establish.

- **Ten seeds, not three.** Measured: a three-seed estimate here moves by a **mean of 0.027
  and a max of 0.064** when taken to ten, and two comparisons flipped sign. Paired
  differences are **no more stable** than absolute numbers (0.029 vs 0.027) — the plausible
  argument that matched splits cancel is false here and was tested. `RESEARCH_LOG.md` §6.5.
- **Check test *and* validation.** Three times in one session a test-split delta was
  contradicted by validation. The usable rule is not "trust validation" — validation moves
  just as much — it is that **disagreement between two noisy measures means nothing is
  resolved.** Agreement is weak evidence; disagreement is strong evidence of no effect.
- **All four losses.** The whole loss family is load-bearing. Under terminal loss alone —
  the default for sequence classification — Experiments 4 and 5 read as a clean negative, and
  the gating result would have been invisible. The two-way split of `achieved(k)` shapes (§8)
  has now **predicted the sign of three separate interventions**; it is the most useful idea
  in the work.
- **$\mu = 0$.** The inner convergence penalty costs 0.12-0.15 and crushes $\rho(B)$ from
  ~1.00 to 0.62-0.70, destroying the near-integrator the memory depends on (§6). Convergence
  and memory are in direct tension. Measure convergence; do not optimise it.
- **Report $\rho(B)$ every run.** It is [det], costs three lines, and caught a change of
  mechanism at unchanged accuracy that the accuracy column could not show (§5.1).

---

## 5. Traps, each of which cost real time here

1. **`step_size` must divide the feature count** — `load_data_as_sequence` raises otherwise.
   Use the $p$ column in §3.
2. **Do not move two variables and attribute to one.** This was the single most repeated
   error of the last session, committed *while writing warnings about it*: $\sigma$ confounded
   with the loss family; $M$ confounded with the penalty. Enumerate the axes of the specific
   table in front of you (`PRINCIPLES.md` §2.10), not from memory.
3. **The training objective is not comparable across $M$** — the inner-convergence term is
   identically zero at $M{=}1$ and non-negative beyond. Validation accuracy is the clean
   cross-$M$ measure.
4. **Do not rescale targets when you change the activation.** That moves two variables.
   Accuracy is `argmax`, which is scale-invariant, so saturation cannot change it — measure
   saturation instead (§11.7).
5. **A check that can pass on an empty set is worse than no check.** Bit three times in one
   session, including a control that compared zero rows and printed `0.00e+00`. Assert the
   match count *before* the tolerance. `PRINCIPLES.md` §1.10.
6. **Initialising a new term to all-zeros can kill its gradient.** For the gate
   $(Cx)\odot(Dy)$, zeroing both $C$ and $D$ makes $\partial/\partial C$ vanish identically —
   the term would be dead and the experiment would have returned a meaningless null. Zero
   one, randomise the other, and **verify the gradient is live at init**.
7. **`exp3to5.py` resumes** on `(experiment, p, M, mu, loss, seed)`. Edit `CONFIGURATIONS`
   and re-run; only new cells cost anything. This is how ten-seed and $\mu{=}0$ rows were
   added without repeating hours of work.

---

## 6. Predictions worth registering before you run anything

Register them, then score them honestly — including the ones that land for the wrong reason.
In the last round, **one prediction hit numerically and was still scored a miss**, because it
hit on the only measure that could not support it. That is the standard.

- **N1.** On a dataset whose **linear ceiling is near chance**, the $M$ effect exceeds
  MNIST's $+0.075$. *Mechanism:* iteration pays when a one-step map cannot do the job.
- **N2.** Gating still beats iteration on `lra_image` but the gap narrows from MNIST's
  2.5×. *Mechanism:* CIFAR is harder than MNIST but still perceptual, not compositional.
- **N3.** On `lra_listops`, iteration **overtakes** gating. *Mechanism:* nested evaluation is
  recursive; no amount of per-step expressiveness substitutes for unrolling it. **This is the
  hypothesis. If it fails, it is a real negative about iteration in this architecture, and
  the most valuable result available in this round.**
- **N4.** The `achieved(k)` two-way split (§8) reproduces on every dataset. *Mechanism:* it
  is a property of what the loss asks for, not of the data. Three interventions so far.

---

## 7. What is deliberately still open, and is RCP's to decide

- **Experiment 6 ($h_k$, and $g$).** Left unspecified on purpose until 1-5 ran. §3 above now
  gives it a forcing reason: binary tasks are unusable without it.
- **The warm-start protocol for Experiments 3+** (§11.8). Experiment 3 has the *same
  parameters* as Experiment 2, so there is no new block to freeze and the §11.1 protocol does
  not extend. Do not invent a replacement; ask.
- **Whether any of this transfers off MNIST.** That is this round.
