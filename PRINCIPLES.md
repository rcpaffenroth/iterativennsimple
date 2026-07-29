# Principles for working on this project

**Read this before writing code or reporting a result.** It exists so that the
arguments behind it do not have to be had again. It was written after a session in
which the assistant repeatedly stated conclusions that did not survive light
pushback, and repeatedly proposed adding machinery to prevent that instead of
simply not doing it.

Two halves: how to write the code, and how to talk about results. The second half
is the one that was actually violated, and it is the more important of the two.

The errors were not subtle and not few. In one session: three claims retracted under
pushback, two configs that could not answer their own question, one parameter used as
a cost dial that silently changed the task, one baseline quoted while undertrained by
a measured factor, and a statistical claim contradicted by its own caveat in the same
message. The corrective is not more machinery. It is enumerating what changed before
saying what caused it, and not stating a conclusion you would abandon if questioned.

---

## Part 1 — Code

The project-wide style is in the `research-code` skill and applies here in full.
The short version: **mathematically transparent code for experiments and teaching,
not production software.** Clarity beats robustness, generality, and defensiveness.
When they conflict, clarity wins.

What that has meant in practice on this project:

### 1.1 Every line of machinery is a liability

Code added to compensate for a mistake is itself code that can be wrong, and it
needs checking too. That cost is real and it is paid by the person reading the code
later, not by whoever wrote it.

Concretely: an automated check that flags "this run had not converged" sounds
prudent and is not. It adds a function with a window parameter and a tolerance
parameter, two heuristics, and an output path — all to restate something a reader
sees immediately by looking at whether the best epoch equals the last epoch. It was
added during that session, and removed.

**Before adding a guard, a flag, a validator, or a summary statistic, ask whether
the thing it prevents is better prevented by not making the mistake.** If a human
reading the output would notice the problem unaided, do not build the detector.

### 1.2 What does earn its place

Not everything defensive is scaffolding. The distinction that has held up:

- **Correctness fixes belong in the code.** Excluding non-finite epochs from
  "best" is not a convenience — without it a NaN epoch can win on a fluke and the
  reported weights are poisoned. That is a bug, and bugs get fixed.
- **Compute savings belong in the code.** Stopping at the first NaN saves 20 epochs
  of guaranteed-garbage training. Cheap, obviously correct.
- **Judgement aids do not.** Anything whose purpose is to remind the author of
  something they should have noticed is a substitute for noticing.

### 1.3 Generated output reports, it does not interpret

Scripts write numbers, tables, and figures. They do not write paragraphs telling
the reader what the numbers mean. Interpretation is the reader's job and it changes
as understanding changes, whereas a paragraph baked into a generator is frozen and
goes stale silently. Two such paragraphs were written into `lra_benchmark.py` and
both were wrong within a day.

Stating a *fact* about the measurement — the split sizes, the resulting standard
error, that there is one seed — is reporting. Saying what to conclude is not.

### 1.4 Keep the mathematics visible

`Sequential2DRNN` is one equation:

    z_{t+1} = (A . b . M)^K . Inject_{t+1} . z_t

`internal_step` and `external_step` are each about four lines and map onto it
directly. That is deliberate and worth protecting. The one optimisation that was
considered and declined — hoisting the input matmul out of the loop, §5.5b of
`tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` — was declined partly because it would have
split the equation across a precompute and a loop for a measured 30%.

---

## Part 2 — Claims

This is the part that went wrong.

### 2.1 Rough and inconclusive is fine. Wrongly decisive is not.

Preliminary runs produce weak numbers. That is expected and nobody minds. What
causes damage is a confident statement that has to be withdrawn — because it wastes
the reader's time twice, and because it makes every *future* confident statement
worthless. Confidence is only useful as a signal if it is withheld when unearned.

**The test:** if light pushback would make you retract a claim, do not make the
claim in that form. Say what you measured and what it does not settle.

### 2.2 Deciding which parts of the algorithm space are dead is not your call

This is a research project exploring a large space of architectures. "Sparsity does
not help", "width does not help", "$K > 1$ is not useful" are claims that close off
regions of that space, and closing off a region on weak evidence is expensive — it
stops work that should have continued.

Hand over observations with their limits attached. Let the PI decide what is worth
another run. "Six numbers, one seed, three of five gaps below the resolution limit"
is a useful thing to say. "Sparsity does not regularise" is not, from that data.

### 2.3 Change one variable at a time, or say plainly that you did not

Three separate errors in one session, all the same shape: more than one variable
moved, and the outcome was attributed to one of them.

- `step_size` was 4 in one run and 1 in the others. It sets **both**
  `seq_len = x_y_index // step_size` **and** `input_size = step_size`, so those are
  different tasks with different input widths — not a long and a short version of
  one task. Results were compared across it anyway.
- A width sweep took the best result per width over **unequal** learning-rate grids
  (one lr at $d_h$=128, two at 512, three at 2048). A maximum over unequal sample
  counts favours whoever got more samples. This produced a "width helps" conclusion
  that the matched-lr comparisons contradict.
- Epoch budget moved from 20 to 30 in the same change as the learning rate, and the
  difference was attributed to the learning rate.

**Before comparing two rows, list every axis on which they differ.** If more than
one differs, either fix that or state the confound in the same sentence as the
result.

### 2.4 Know what your sample size can resolve, before quoting a ranking

With $n$ evaluation rows and accuracy near $p$, the standard error is
$\sqrt{p(1-p)/n}$, and the standard error on a *difference* between two runs is
$\sqrt{2}$ times that. At $n = 1000$ and $p \approx 0.2$, **differences below about
0.04 are not resolvable.**

And that is a floor, not the real uncertainty: it counts only sampling noise in the
evaluation set, not seed-to-seed variation in training, which is unmeasured whenever
there is one seed. With one seed there are no error bars at all.

A specific failure to avoid repeating: a five-point sequence 0.243 / 0.239 / 0.205 /
0.161 / 0.151 was described as "monotonically decreasing". Three of the four gaps
were below the resolution limit. The defensible statement was that the two ends
differ and the middle is unresolved.

### 2.5 Separate deterministic measurements from statistical ones

They deserve different confidence and it is worth saying which is which.

- **Deterministic:** wall-clock timings, parameter counts, memory, whether a run
  produced NaN, whether two implementations agree to `allclose` tolerance. One
  measurement is often enough. These have carried the weight of every claim on this
  project that survived scrutiny.
- **Statistical:** anything involving accuracy or loss on a finite sample. Needs a
  resolution check, and ideally replication.

When the deterministic version of a claim is available, lead with it. "K=1
reproduces `torch.nn.RNN`" is proven by a unit test asserting `allclose` to 1e-6 —
not by two accuracies agreeing to 0.004, which proves nothing and was cited anyway.

### 2.6 Put the uncertainty in the sentence

Not in a caveats section below, not in a footnote. In the sentence making the
claim. A caveat four paragraphs down does not undo a headline, and in that session
a claim was listed as trustworthy and contradicted by its own caveat in the same
message.

### 2.7 Record retractions where the claim lives

When a result is withdrawn, the correction goes in the file that made the claim —
not only in conversation, which is lost. `examples/lra_runs/*/results.md` carry
banners for exactly this reason: someone reading those tables later must not be
able to miss that the width comparison is confounded.

### 2.8 A parameter that changes the task is not a cost dial

The single most damaging error of that session. `step_size` in the LRA harness sets

    seq_len    = x_y_index // step_size
    input_size = step_size

so it shortens the sequence **and widens the input**. Two runs at different values
are two different tasks, not a cheap preview and an expensive real version. Yet
three different values were used on `lra_image` across five directories — 16, 4 and
1 — with nothing marking them as incomparable, and the cheap ones were described as
previews of the expensive ones.

The harness already had `max_points`, which drops rows and therefore costs only
statistical power while leaving the task identical. It was written for exactly this
purpose and then not used, because `step_size` was the more obvious lever.

**When you need a run to be cheaper, ask whether the lever changes the task.** Fewer
rows, fewer models, fewer epochs, smaller hidden size: same task. Different sequence
length, different input width, different vocabulary, truncated inputs: different
task. Only the first kind may be used to make a preview of something.

All configs are now at `step_size: 1` and use `max_points` for cost. The one
exception, `image_monarch/`, carries a warning at the top of its config and its
results.

### 2.9 Stale claims propagate into comments and never get revisited

A belief written into a comment, a config, or a model name is a copy that does not
update when the belief dies. In that session the orthogonal-initialisation
hypothesis was asserted as fact in a config comment, silently baked into two other
configs as `orthogonal_hh: true`, and left in place after it had failed twice.

**When retracting a claim, grep for it.** It will be in more places than you
remember: docstrings, config comments, row names, report generators, TODO entries.
§2.7 says record the retraction where the claim lives — plural.

### 2.10 List the axes before comparing two rows

Not a mental check, an actual enumeration. For this harness the axes are: dataset,
`step_size`, `max_points`, split fractions, `split_seed`, epochs, batch size,
`grad_clip`, seed, learning rate, hidden size, `K`, block type, `num_blocks`,
initialisation. Fifteen of them.

Errors of this kind found in one session: `step_size` differing (§2.8); unequal
learning-rate grids (§2.3); epochs moving with learning rate (§2.3); and two configs
in which `orthogonal_hh` was set on the `K > 1` rows but not on `K = 1`, so `K` and
initialisation could not be separated at all. That last one made those configs
unable to answer the question they were written to ask.

A cheap habit that would have caught most of them: after writing a config, print one
line per model listing every axis, and look at the columns.

---

## Part 3 — Where things are

| | |
| --- | --- |
| `tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md` | design record: every decision and why, including those deliberately not taken. Read before changing the module. |
| `tasks/TODO_Sequential2DRNN.md` | deferred work, the experiment queue, and the findings log with its retractions |
| `README_Sequential2DRNN.md` | user-facing entry point for the module |
| `examples/lra_runs/README.md` | the benchmark harness, config schema, and cost model |
| `PRINCIPLES.md` | this file |

Two standing notes that are easy to get wrong and expensive to get wrong:

- **`Sequential2D.blocks[i][j]` is indexed (input, output)** — the transpose of the
  block matrix as normally written. This does *not* extend to weight tensors:
  PyTorch stores `Linear.weight` as `(out_features, in_features)`, the mathematics
  convention, so weights copy across with no transpose.
- **RCP has said that transposes are his known weak point** and that orientation
  claims in his prose should be re-derived rather than trusted. He wants to be told
  when something is wrong. That instruction has already caught one real error.
