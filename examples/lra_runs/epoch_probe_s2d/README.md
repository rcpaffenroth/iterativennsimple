# INCOMPLETE RUN — killed, no `results.md`

This directory has a `config.yaml` and a log, and **no results**. The process was
killed at epoch 75/100 of the second model (175 of 200 epochs total, ~87%), so
`write_report` never ran: there is no `results.md`, `results.json` or `curves.png`.
Nothing here was produced by a completed run.

`run-killed-at-175of200-epochs.log` holds the per-epoch history that did complete —
all 100 epochs of `S2D h=2048 lr=3e-5`, and 75 of `S2D h=2048 lr=1e-5`. That is why
it was not simply re-run. **Parse the log; do not trust the summary below.**

Re-running this config as-is is a perfectly valid thing to do (~2 h) and would
produce the missing artefacts plus the last 25 epochs.

## Measurements

One seed, one width, `step_size: 1`, seq 1024, 8000 training rows, `grad_clip: 1.0`.

`S2D h=2048 lr=3e-5`, train loss by epoch (chance is $\ln 10 = 2.3026$):

| epochs | train loss | val acc |
| --- | --- | --- |
| 1 | 2.3105 | 0.103 |
| 38–51 | 2.09 – 2.14 | 0.21 – 0.23 |
| 52 | **2.2639** | 0.123 |
| 53–100 | 2.25 – 2.32 | 0.10 – 0.17 |

The change at epoch 52 happened within one epoch: 2.0933 → 2.2639.

`S2D h=2048 lr=1e-5`: train loss 2.3051 → 2.2279 over 75 epochs, monotonic, slope
−0.00026 per epoch over the last 20.

## What is *not* established

- **The cause of the epoch-52 step is not diagnosed.** No gradient norms were
  recorded. Calling it a "training instability" is a guess about one event in one
  run; an Adam moment blow-up, a batch-ordering effect, or something else are all
  still open.
- **Whether it reproduces is unknown.** $n = 1$, one seed.
- **Whether any of it generalises past $d_h$ = 2048 is unknown.** One width.
- The val accuracies at epochs 38–51 are corroborated by the train-loss drop, so
  they are probably not selection artefacts. But `best val acc` over a long budget
  *is* biased upward for any row that is not learning — roughly +0.03 at 100 epochs
  and $n$ = 1000, since it is a maximum over that many noisy estimates. Judge
  learning by the train-loss trajectory, which is immune to that selection.

## Why it was run

`epoch_probe/` (complete, in the neighbouring directory) measured the epoch budget
using `S2D h=128 lr=1e-3`. That row never left chance — train loss 2.281 → 2.257
over 100 epochs — so it could not show a plateau, and half the probe answered
nothing. This one used the setting that had trained furthest in `image_wide_lr/`,
h=2048 at lr=3e-5, which was the *lowest* rate in that grid, plus one step below it
to ask whether the grid should extend downward.

Relevant open question: `RESEARCH_LOG.md` §5 item 1, $W_{hh} \supseteq I$.
