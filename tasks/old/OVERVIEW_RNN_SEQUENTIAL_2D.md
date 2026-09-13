# Overview: An RNN-compatible module built on `Sequential2D`

**Status:** implemented in `iterativennsimple/Sequential2DRNN.py`.
**Audience:** whoever (human or LLM) picks this up next. This document exists so the
ideas below do not have to be relitigated.

**Where things live:**

| | |
| --- | --- |
| `iterativennsimple/Sequential2DRNN.py` | the module |
| `README_Sequential2DRNN.md` | user-facing entry point, quickstart, the gotchas |
| `notebooks/7-rcp-RNN-as-Sequential2D.ipynb` | tutorial; symlinked into `tests/`, run by nbmake |
| `examples/rnn_internal_iterations.py` | extended experiment on $K > 1$ |
| `tests/test_Sequential2DRNN.py` | the invariants, mostly the silent ones |

| `TODO_Sequential2DRNN.md` | deferred work and the experiment queue |
| `README_RCP.md` | RCP's review checklist, annotated with his responses |
| `notebooks/advanced/12-claude-fixed-points-and-bistability.ipynb` | the dynamical-systems tutorial at `hidden_size = 1` |

This document remains the record of *why*; the README is the record of *how*.

Note one divergence from §5.2 below: `from_rnn` builds the **two**-slot map
$[x, h]$ of eq. (9), since `torch.nn.RNN` has no readout — its `output` *is* the
hidden sequence — so a $y$-slot would be vestigial *and* lagged. The readout you
would normally compose afterwards is available as the **observation map** of §5.5,
which is the right home for it. The three-slot layout is what `from_3x3` gives
you, and its $y$-slot exists for feedback rather than for readout.

---

## 1. Goal

Build a new PyTorch module in which a `Sequential2D` block map is driven as a discrete
dynamical system, such that:

1. There is a **setting of the parameters and structure that reproduces a single-layer
   `torch.nn.RNN` exactly** — copy weights from a trained `nn.RNN`, get bit-comparable
   outputs. This is a *proof of concept and a pytest*, **not** the target use case.
   (Multi-layer stacking is deliberately not reproduced; §8.5 explains why the divergence
   is the intended behaviour rather than a gap.)
2. The module exposes strictly more structure than `nn.RNN`: arbitrary block sparsity
   patterns, extra state slots, sparse/structured block types, and — most importantly —
   **a decoupling of the internal iteration count from the external sequence length**
   (§7). That decoupling is where the research value lies.

The theoretical basis is Hershey, Paffenroth, Pathak & Tavener, *"Rethinking the
Relationship between Recurrent and Non-Recurrent Neural Networks: A Study in Sparsity"*,
[arXiv:2404.00880](https://arxiv.org/abs/2404.00880). Equation numbers below refer to that
paper.

### 1.1 The central idea, stated up front

Everything else in this document is machinery in service of one equation:

$$z_{t+1} = \underbrace{(A \circ b \circ M)^{K}}_{\text{fast, autonomous}} \circ
\underbrace{\mathrm{Inject}_{t+1}}_{\text{slow, forcing}} \circ\; z_t$$

A block map $M$ (a `Sequential2D`), a per-slot bias $b$, and a per-slot activation $A$ are
composed into one **internal** step, and that internal step runs $K$ times per input token.
The input is overwritten into a dedicated slot by `Inject` once per token.

**$K$ is the whole point.** It decouples the network's own dynamics from the sequence's
clock: a fast internal system driven by a slow external forcing. $K = 1$ collapses to an
ordinary RNN, which is why `nn.RNN` compatibility falls out as a special case rather than
being designed for. $K > 1$ is the research surface — it is also the dial between the paper's
MLP and RNN readings of eq. (48), and $K \to \infty$ is a deep equilibrium model.

Full treatment in §7; the operators are defined in §5.

---

## 2. Existing code this builds on

| File | What it provides |
| --- | --- |
| `iterativennsimple/Sequential2D.py` | `Sequential2D` — the 2-D array of modules with the `+` combiner. This is the paper's block non-linear function, Definition 1 / eq. (3). |
| `iterativennsimple/Sequential2D.py:8` | `Identity` — an identity module carrying `in_features`/`out_features`, so an `I` block can literally appear in the block matrix. |
| `iterativennsimple/Sequential2D.py:98` | `Sequential2D.forward_list` — the list-valued forward. **Use this one** (see §9). |
| `iterativennsimple/MaskedLinear.py`, `SparseLinear.py`, `MonarchLinear.py` | Drop-in replacements for `torch.nn.Linear` as block types. |
| `iterativennsimple/Sequential1D.py` | `torch.nn.Sequential` + `in_features`/`out_features`. |

---

## 3. Notation and conventions

### 3.1 The paper's convention

Eq. (3): $f_{i,j} : \mathbb{R}^{|h_j|} \to \mathbb{R}^{|h_i|}$. **Row = output, column =
input** — ordinary matrix convention. Acts on column vectors, mini-batches are
$\mathbb{R}^{|x| \times n}$ (features × batch), eq. (5).

### 3.2 The code's convention — it is the transpose

`Sequential2D.py:79-80` requires `blocks[i][j].in_features == in_features_list[i]` and
`blocks[i][j].out_features == out_features_list[j]`. So in code, **`i` = input slot,
`j` = output slot**, and `forward` takes `(batch, features)`.

> **`blocks` is the transpose of the paper's $M$, and the batch axis is also transposed.**
> Writing the paper's equations directly into `blocks` puts every off-diagonal block in the
> wrong cell. This is the single most likely source of a silent bug.

### 3.3 Standing instruction on transposes

> **RCP (the author of this project) has decades of habit with right dot-products — the
> mathematics standard — while deep learning uses left dot-products. He states plainly that
> transposes are his known weak point and that he makes these errors routinely.**
>
> **Therefore: every orientation claim in RCP's prose, diagrams, and comments must be
> independently re-derived before being acted on, including ones in this very document.**
> Do not treat a stated orientation as authoritative. Check it against a concrete shape and
> say so when it disagrees. He wants to be told.

There is one genuinely reassuring fact worth internalising, because it removes a whole class
of confusion:

**PyTorch stores weights in the *math* convention, and transposes only in the operation.**
`torch.nn.Linear(in, out).weight` has shape `(out_features, in_features)` — row = output,
exactly the paper's convention — and `forward` computes `x @ W.T + b` to accommodate the
`(batch, features)` layout. So `nn.RNN`'s `weight_ih_l0` has shape
`(hidden_size, input_size)` and reads directly as $W_{ih}$ in eq. (8). **The weight-copy in
§6 therefore needs no transpose at all.**

The transposition in §3.2 is about *which cell of the block array a module goes in*, not
about the orientation of any weight tensor. Two distinct issues that are easy to conflate:

| issue | where it bites | fix |
| --- | --- | --- |
| `blocks[i][j]` is indexed (input, output) | choosing the cell | index carefully, or build in paper convention and transpose once |
| `(batch, features)` vs `(features, batch)` | reading the math onto tensors | handled inside `nn.Linear`; not your problem |
| weight tensor orientation | copying from `nn.RNN` | **no transpose needed** — both are `(out, in)` |

### 3.4 Naming — source-first

**$W_{ab}$ means "from slot $a$ to slot $b$". Source first. Decided.**

In the paper's matrix $M$ it sits at row $b$, column $a$; in code it sits at
`blocks[a][b]`. This matches §5.2's matrix, and it matches PyTorch, whose `W_ih` means
input→hidden.

> **Historical note, kept deliberately.** An earlier draft of the `from_3x3` block list was
> written target-first ($W_{yx}, W_{yy}, W_{yh}, W_{hx}, W_{hy}, W_{hh}$) while the matrix in
> §5.2 was source-first. The *set* of six blocks was correct — the $y$ and $h$ rows — but the
> labels were reversed, and under source-first that same list would have excluded $W_{xh}$,
> the input-to-hidden matrix, without which there is no RNN at all. This is a live example of
> §3.3: the error was in the naming, not the intent, and it was caught by checking whether
> the named set was coherent rather than by trusting the labels.

---

## 4. State layout

The state is a list of slots. The canonical three-slot layout is

$$z = [\,x,\; y,\; h\,]$$

- $x$ — input slot, size `input_size`. Overwritten by `Inject` each external step.
- $y$ — output/readout slot, size `output_size`. Optional.
- $h$ — hidden slot, size `hidden_size`.

Nothing in the design is limited to three slots; this is just the layout that makes the
`nn.RNN` correspondence readable.

### 4.1 Forward compatibility: $z = [x, y, h_1, \ldots, h_k]$

Multi-hidden-slot states are **an explicit future goal, deliberately not built now.** The
reason for deferring is not implementation difficulty but an unresolved semantic question:
*do all $h_i$ share a timescale?* Once $K$ can differ per slot — some slots updating every
internal iteration, others every $n$-th — the "two timescales" of §7 become a hierarchy, and
that needs its own design pass.

**Nothing in the current implementation may preclude it.** Concretely, the core must be
written $N$-slot generic from the start:

- The core module takes `in_features_list` / `out_features_list` of arbitrary length. Three
  slots is a *factory convenience* (§5.5), never a hard-coded assumption.
- $b$ and $A$ are lists indexed by slot, of arbitrary length — already true as specified.
- **Do not bake a scalar $K$ into the state-stepping loop in a way that resists becoming
  per-slot later.** Keep the "how many internal iterations, and which slots participate in
  each" decision in one place, even while the answer is a constant.
- Slot identity should be positional-or-named, not the literal triple `(x, y, h)` threaded
  through signatures.
- The `Inject` operator should take *which slot* to overwrite as data, not assume slot 0.

---

## 5. The four operators

One **external** step is the composition

$$z_{t+1} = A \circ b \circ M \circ \mathrm{Inject}_{t+1} \circ z_t$$

Applied right to left.

### 5.1 `Inject` — overwrite, not add

$$\mathrm{Inject}_{t+1}\big([x_t,\, y_t,\, h_t]\big) = [\,x_{t+1},\, y_t,\, h_t\,]$$

**Overwrite semantics.** The $x$-slot is clobbered with the next input; $y$ and $h$ carry
through untouched. This is the paper's forced (non-autonomous) reading, eq. (11), and it is
what makes sequence length a *runtime* property — essential for `nn.RNN` compatibility,
since `nn.RNN` accepts any $L$ per batch.

Injecting **before** $M$ (rather than after) is what makes the time indices line up with
PyTorch's; see §6.

### 5.2 `M` — the `Sequential2D` block map

For the RNN-compatible setting, in paper convention (row = output):

$$
M =
\begin{bmatrix}
I & \texttt{None} & \texttt{None} \\
\texttt{None} & \texttt{None} & W_{hy} \\
W_{xh} & \texttt{None} & W_{hh}
\end{bmatrix}
\circ
\begin{bmatrix} x_{t+1} \\ \text{(ignored)} \\ h_t \end{bmatrix}
=
\begin{bmatrix}
x_{t+1} \\
W_{hy}\,h_t \\
W_{xh}\,x_{t+1} + W_{hh}\,h_t
\end{bmatrix}
$$

Translated to code indices (slots $x{=}0$, $y{=}1$, $h{=}2$):

| paper cell | meaning | code cell |
| --- | --- | --- |
| $M_{x,x} = I$ | $x \to x$ | `blocks[0][0] = Identity(...)` |
| $M_{y,h} = W_{hy}$ | $h \to y$ | `blocks[2][1]` |
| $M_{h,x} = W_{xh}$ | $x \to h$ | `blocks[0][2]` |
| $M_{h,h} = W_{hh}$ | $h \to h$ | `blocks[2][2]` |

All other cells `None`.

$M$ is **fixed** — it does not depend on $t$. All external time dependence lives in
`Inject`. (This is a cleaner factorization than eq. (9), which hides the non-autonomy
inside $F_x$'s argument.)

### 5.3 `b` — per-slot bias

One bias vector per **output slot**, added after $M$:

$$b = [\,b_x,\; b_y,\; b_h\,]$$

**Blocks are constructed with `bias=False`.** Bias is a property of the slot, not of the
individual blocks. Rationale and consequences in §8.2.

### 5.4 `A` — per-slot activation

One `nn.Module` per slot, applied after $b$:

$$A = \mathrm{diag}(A_x,\, A_y,\, A_h)$$

For RNN compatibility $A = \mathrm{diag}(I,\, I,\, \tanh)$.

Two things this must get right:

- **Order is $A \circ b \circ M$, never $M \circ A$.** $A$ applies to the *sum* of all
  incoming blocks, which is what gives $\tanh(W_{xh}x + W_{hh}h + b)$. Applying it before
  $M$ would give $W_{xh}x + W_{hh}\tanh(h)$ — a different (pre-activation) model. It also
  means the stored state is post-activation, matching `nn.RNN`'s convention that $h_0$ is
  itself a valid hidden state.
- **`A` is a list of modules, not literally elementwise functions.** $\tanh$ is
  elementwise so nothing is lost, but the same mechanism then covers `softmax` or
  `LayerNorm` on a slot at no extra cost.

> **Do not use `Sequential2D.from_config`'s `block_kwargs['activation']` mechanism**
> (`Sequential2D.py:255-272`). It wraps each block individually in a `Sequential1D`, giving
> $\sigma(W_{xh}x) + \sigma(W_{hh}h)$, which is **not** $\sigma(W_{xh}x + W_{hh}h)$. That
> path cannot express an RNN. The per-slot `A` above is the fix, and it is the one genuinely
> new primitive this module needs.

### 5.5 The observation map $g$ — a readout is not a slot

The four operators above are the **state map**. The reported output is separate:

$$z_{t+1} = F(z_t, x_{t+1}) \qquad\text{(state map, the composition above)}$$
$$y_t = g(z_t) \qquad\text{(observation map)}$$

This is the classical state-space form — S4, MAMBA and ordinary linear control all
write it this way, and their $C$ matrix is an *observation*, not a state component.
Implemented as `Sequential2DRNN.observe`.

**Why this is a distinct mechanism from a $y$-slot**, rather than a stylistic
choice. There are two things one might mean by "the output", and they behave
differently:

| | lag | can feed back | is it in $M$? |
| --- | --- | --- | --- |
| $y$ as a **state slot**, via $W_{hy}$ | one internal iteration (§8.3) | yes, via $W_{yh}$ | yes |
| $y$ as an **observation**, via $g$ | none | no, by construction | no |

The lag is not incidental: the update is Jacobi, so a block writing into the
$y$-slot reads the $h$ from *before* the step. At $K = 1$ that is a full token of
delay, which is simply the wrong answer for an ordinary readout. As $K$ grows and
the internal iteration converges the lag shrinks, since
$h^{(K-1)} \to h^{(K)}$ — so it is worst exactly in the RNN-compatible regime.

Conversely, feedback *requires* the slot: the paper's above-diagonal $S$ (§6.4)
cannot be an observation, because an observation cannot influence the dynamics.

So: **use `readout` (the observation) for an ordinary readout; use the $y$-slot
when you want feedback.** Both can coexist — the slot may carry $W_{yh}$ feedback
while `readout` supplies the reported output.

One consequence worth stating because it looks like an inconsistency: **the
"bias belongs to the slot" rule of §8.2 does not apply to $g$.** A plain
`torch.nn.Linear` with its own bias is exactly right as a `readout`, because $g$
sits outside the block matrix and there is no slot sum for it to double-count
against.

`from_rnn(rnn, readout=None)` returns $h_t$, which is what makes it bit-comparable
with `nn.RNN` — that module has no readout at all, and its `output` *is* the
hidden sequence.

### 5.5b Input lifting — a proposition, and why it is not implemented

The input's contribution can be hoisted out of the internal loop entirely. This is
a *theorem about the existing map*, not a variant of it.

> **Proposition (input lifting).** Suppose the input-persistence invariant of §8.6
> holds — $M_{xx} = I$, $b_x = 0$, $A_x = \mathrm{id}$ — and no other block writes
> into the $x$-slot. Write $M_{\setminus x}$ for $M$ with the $x$ row and column
> deleted, and define the **forcing**
> $$u_t \;=\; \big(W_{xs}\,x_t\big)_{s \neq x}$$
> the contribution the input makes to each non-input slot. Then for every $K \ge 0$
> $$\Big[(A \circ b \circ M)^{K} \circ \mathrm{Inject}_{x_t}\, z\Big]_{\setminus x}
> \;=\; \big(A \circ (b + u_t) \circ M_{\setminus x}\big)^{K} z_{\setminus x}$$
>
> *Proof.* Induction on $k$; the base case is trivial. The three hypotheses give
> $x^{(k)} = x_t$ for every $k$, so the total arriving at a slot $s \neq x$ on
> iteration $k$ is $\sum_{s' \neq x} W_{s's} z^{(k)}_{s'} + W_{xs} x_t$, whose last
> term is the constant $(u_t)_s$. One step of each side therefore agrees. $\square$

**Corollary.** $W_{xs}$ need never be applied inside the internal loop; $u_t$ can be
precomputed for a whole sequence in one batched matmul.

Three things worth noting.

- **The hypothesis is exactly §8.6.** The invariant insisted on there for dynamical
  reasons — impulse response, gradient path length, well-posedness of
  $K \to \infty$ — is precisely what licenses this optimisation. It also survives the
  scalar leak of §8.6: $M_{xx} = \lambda I$ gives $x^{(k)} = \lambda^{k} x_t$ and
  forcing $\lambda^{k} u_t$, still precomputable. It does **not** survive a general
  learnable $W_{xx}$.
- **This is the right form, rather than widening the $x$-slot.** Precomputing
  $\hat{x}_t = W_{xh} x_t$ and setting $W_{xh} \to I_{d_h}$ also works, but breaks as
  soon as the input feeds two slots: one identity cannot serve two differently-sized
  targets, so $W_{xh}$ and $W_{xy}$ would have to become projections $[I\ 0]$ and
  $[0\ I]$ out of a concatenated slot. The forcing form takes any number of targets
  by stacking, and is the classical form for a forced discrete system.
- **What it costs.** $u_t$ is $\Theta(L B d_h)$ where the raw input was
  $\Theta(L B d_x)$, and $d_x \ll d_h$ is the intended regime (one pixel per step,
  large hidden). But BPTT already stores $\Theta(L B d_h)$, so this is a
  constant-factor increase rather than an asymptotic one — 512 MB at $d_h = 2048$,
  $L = 1024$, $B = 64$.

**Not implemented, deliberately.** Measured on an RTX 4090, $L = 1024$, $B = 64$,
$d_h = 128$:

| variant | ms / batch |
| --- | ---: |
| $W_{xh}$ applied inside the loop, slicing `x[:, t, :]` | 176 |
| $W_{xh}$ inside, input pre-unbound into a list | 162 |
| hoisted, slicing `xh[:, t, :]` | **220** |
| hoisted, forcing pre-unbound into a list | **117** |

The lifting is worth about **30%**, and *only* in combination with pre-unbinding.
Hoisting on its own is a 25% **pessimisation**, because it trades a tiny
$(B, d_x) \times (d_x, d_h)$ matmul for 1024 slices of a 33 MB tensor.

Thirty percent does not justify a second code path that must be kept provably
equivalent to the first — particularly since the gap it narrows shrinks as $d_h$
grows, and vanishes in the regime this project is aimed at (§9.6). Recorded because
the proposition is worth knowing, and because it is the first thing a reader will
propose.

### 5.6 Construction: the factory pattern

Follow the existing `Sequential2D.from_config` style (`Sequential2D.py:163`). The core module
stays $N$-slot generic (§4.1); the ergonomics live in classmethod factories.

**`from_rnn(rnn)`** — build the module by copying weights and biases out of an existing
`torch.nn.RNN`. Produces the §6 configuration; asserts `num_layers == 1`,
`bidirectional == False`; maps `bias_ih_l0 + bias_hh_l0 → b_h`. No transpose needed on the
weight tensors (§3.3). This *is* the equivalence pytest, promoted to a supported entry point.

**`from_3x3(...)`** — the general three-slot constructor. Same input/output contract as an
`nn.RNN` (takes the same data, returns the same shapes) but exposes the six free blocks of
the $y$ and $h$ rows. In source-first naming (§3.4) those are

$$W_{xy},\; W_{yy},\; W_{hy} \quad (\text{into } y) \qquad
W_{xh},\; W_{yh},\; W_{hh} \quad (\text{into } h)$$

**These will generally give different answers than a standard RNN, by design.** That is the
point of having it. The $x$-row is not exposed: $M_{xx} = I$ is fixed (§8.6) and the
remaining two cells are deferred (§10.1).

**Blocks may themselves be `Sequential2D`.** `Sequential2D` carries `in_features` and
`out_features`, so it already satisfies the block contract — nesting works today, no new code
required. So `from_3x3` should take **six `nn.Module` arguments** rather than a
configuration dictionary: any of `Linear`, `MaskedLinear`, `SparseLinear`, `MonarchLinear`,
`Sequential1D`, or a nested `Sequential2D` drops straight in, and `None` means "absent". This
keeps the factory signature small and the composition open-ended.

Two obligations that come with taking pre-built modules:

- **`bias=False` is the caller's responsibility.** Bias belongs to the slot, not the block
  (§8.2), and `torch.nn.Linear` defaults to `bias=True`. `from_3x3` must either assert that
  supplied blocks are bias-free or document loudly that supplying a biased block double-counts.
- **A nested `Sequential2D` block does not get the outer $A$.** The outer per-slot activation
  applies to the *slot sum* after all blocks have contributed; a nested block is inside one
  of those contributions. If a nested block needs its own internal nonlinearity it must carry
  it itself (via `Sequential1D`). Worth an explicit test — it is exactly the sort of thing
  that silently produces a linear model where a nonlinear one was intended.

Nesting must be **tested and exemplified**, not merely asserted to work.

---

## 6. `nn.RNN` compatibility

PyTorch's recurrence:

$$h_t = \tanh\big(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\big)$$

Ours, with the structure above:

$$h_{t+1} = A_h\big(W_{xh}\,x_{t+1} + W_{hh}\,h_t + b_h\big)$$

Injecting before $M$ makes the indices agree. The weight map is:

| `nn.RNN` | ours |
| --- | --- |
| `weight_ih_l0` | $W_{xh}$ |
| `weight_hh_l0` | $W_{hh}$ |
| `bias_ih_l0 + bias_hh_l0` | $b_h$ |
| `nonlinearity='tanh'` | $A_h = \tanh$ |

### Required conditions

1. $W_{yh} = \texttt{None}$ — no $y \to h$ feedback. A nonzero one is the paper's
   above-diagonal $S$ (§6.4) and breaks equivalence.
2. $b_x = 0$, $A_x = \mathrm{id}$ — required for input persistence (§8.6), though at $K = 1$
   the violation would be invisible here since `Inject` overwrites the slot regardless.
3. $K = 1$ internal iteration per injection (see §7).
4. Read `output` from the **$h$-slot**, not the $y$-slot (see §8.3).
5. `num_layers=1`. Multi-layer stacking is deliberately **not** reproduced — see §8.5.

### The pytest

Construct an `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh',
batch_first=True)`, build ours via `from_rnn` (§5.5), run both on the same random input and
`h_0`, and assert `allclose` on both `output` and `h_n`. Vary sequence length across cases to
confirm $L$ really is a runtime property.

Companion tests worth writing at the same time, since each guards a decision that is silent
when violated:

- **Input persistence** (§8.6): with $K > 1$, assert the $x$-slot is bit-identical across all
  $K$ internal iterations. Catches a stray $b_x$ or $A_x$.
- **`nonlinearity='relu'`** as well as `'tanh'` — confirms $A$ is genuinely per-slot
  pluggable rather than hard-coded.
- **Nested blocks** (§5.5): a `Sequential2D` used as a block inside `from_3x3`, asserting the
  outer $A$ does *not* reach inside it.
- **Structured block types** (§9.5): the same test with `MonarchLinear` or `SparseLinear`
  substituted for $W_{hh}$, asserting only that shapes and gradients flow — not equivalence.

---

## 7. The point: internal iterations and two timescales

This is the actual research goal; §6 is scaffolding.

Let $T = A \circ b \circ M$ be one **internal** step. The external step is $T$ applied $K$
times per injection:

$$z_{t+1} = \underbrace{(A \circ b \circ M)^{K}}_{\text{fast, autonomous}} \circ
\underbrace{\mathrm{Inject}_{t+1}}_{\text{slow, forcing}} \circ\; z_t$$

$K = 1$ is the degenerate RNN case. $K > 1$ is a **fast internal dynamical system driven by
a slow external forcing** — the network performs $K$ steps of its own dynamics per input
token. This is a genuinely different timescale, and it is where the interesting behaviour is
expected to live.

*(A condensed statement of this appears in §1.1. The full version is kept here because it
needs $A$, $b$, $M$ and `Inject` to already be defined; forward-referencing them would cost
more than the front-loading gains.)*

### 7.1 $M_{xx} = I$ is load-bearing once $K > 1$

Under $K = 1$ with overwrite injection, whatever $M$ writes to the $x$-slot is discarded on
the next `Inject`, so $M_{xx}$ is inert. Under $K > 1$ it is not:

- $M_{xx} = I$ — the $x$-slot holds $x_{t+1}$ constant across all $K$ internal iterations.
  Every internal step sees the same forcing: a genuinely forced fast system. This is
  eq. (50)'s $M_\infty$, infinite impulse response.
- $M_{xx} = \texttt{None}$ — $x$ is consumed on internal iteration 1; iterations $2\ldots K$
  run **unforced**, relaxing freely from the initial kick.

**Default: $M_{xx} = I$, fixed and non-learnable**, which also makes the $x$-slot an input
skip connection across the internal iterations. See §8.6 — and note the argument there is
from principles, not from measured performance.

### 7.2 What $K$ buys, theoretically

- **$K$ is the MLP↔RNN dial.** Eq. (48) argues MLPs and RNNs are the same object with
  different sparsity patterns and parameter tying. $K$ internal iterations of a strictly
  lower-triangular $M$ is *depth per token*; $K = 1$ with $M_{hh} \neq 0$ is *width in
  time*. $K$ turns the paper's equivalence from an architecture choice into a runtime knob.
- **$K \to \infty$ with contractive $M$ is a deep equilibrium model.** The fixed point
  $z^\star = A(b + M z^\star)$ under held injection is exactly the DEQ formulation
  (Bai, Kolter & Koltun, 2019), and is the same statement as the paper's fixed-point
  discussion in §4.3.2. Finite $K$ is the truncated unroll. Relevant because it means the
  implicit-differentiation literature applies if $O(1)$-memory backward is ever wanted.
- **BPTT depth is $K \cdot L$, not $L$.** Harmless at $K = 2$; document it loudly at
  $K = 20$.

---

## 8. Decisions taken, with rationale

### 8.1 Forced formulation, not the lifted one

The paper gives two constructions. We use the **forced** one, eq. (11): state is
$O(|x|+|h|)$, $x_t$ re-injected each step, $L$ discovered at runtime.

We do **not** use the **lifted** one, eq. (15) — $M_{RNN_3}$ of dimension
$|h| + T(|x|+|h|)$ iterated $T$ times. It is the more elegant object (it makes RNN and MLP
literally the same fixed map) but it is disqualified as a drop-in module because:

- $T$ is baked into the architecture, so variable sequence length is impossible.
- $T$ nonzero blocks × $T$ applications $= \Theta(T^2)$ block applications vs. $2T$.
  Eqs. (17)–(19) show why: on iteration 1, blocks $2\ldots T$ compute $f_\theta(x_j, 0)$ —
  work that is discarded.
- Its blocks $f_\theta(x_j, \cdot)$ depend on the **data**, not just $\theta$. The "fixed
  map" is fixed only per-sample, which is awkward for a module with registered parameters.

The lifted version remains worth building **separately**, for studying the continuum. It
should not wear the `nn.RNN` API.

### 8.2 Bias is per-slot, not per-block

$b_h = b_{ih} + b_{hh}$ when copying from `nn.RNN`.

- Forward outputs match `nn.RNN` **exactly**; the pytest just sums PyTorch's two bias
  vectors.
- The map `nn.RNN state_dict` → ours is **not injective**, so it does not round-trip back.
  That direction is not needed.
- The alternative (bias inside each block, bijective with PyTorch's two-bias layout) was
  rejected: it makes "which of the $N$ incoming blocks carries the bias" ambiguous the moment
  a slot has three or more sources — which is exactly the generality this module exists for.
- **Double-counting hazard:** `torch.nn.Linear` defaults to `bias=True`, and
  `Sequential2D.from_config` constructs it that way (`Sequential2D.py:189`). Blocks must be
  built with `bias=False` or every slot gets two biases.

### 8.3 The $y$-slot lags one internal step

`Sequential2D` is a **Jacobi** update — every block reads the *old* state and results are
summed into the *new* state. So

$$y^{(k+1)} = A_y\big(W_{hy}\,h^{(k)} + b_y\big)$$

reads $h^{(k)}$, not $h^{(k+1)}$. After $K$ internal iterations, $y = W_{hy}h^{(K-1)}$ while
$h = h^{(K)}$ — **lag is exactly one internal step, independent of $K$.**

Consequences:
- For `nn.RNN`'s `output` tensor, read the **$h$-slot** after each external step.
- **For an ordinary readout, use the observation map $g$ of §5.5, not a slot.** That is the
  real resolution of this trap: $g$ reads the current state, so it has no lag. The slot route
  exists for when feedback is wanted, and then the lag is the price.
- If you do read the $y$-slot, take it shifted by one and run one extra flush iteration at
  the end of the sequence.
- **Document this in the module docstring.** It is self-consistent but surprising, and it is
  a standing trap in every readout-slot configuration.

### 8.4 `None` means "not yet alive", not "the zero vector"

`forward_list` leaves a slot `None` when nothing wrote to it. Two readings were considered:

1. `None` $\equiv$ zero vector. Then $b \circ \texttt{None} = b$ and
   $A \circ b \circ \texttt{None} = A(b)$, which is generally neither `None` nor zero — so
   `None` must be materialised on every slot with a nonzero bias, destroying the sparsity
   exactly where it is wanted.
2. **`None` $\equiv$ absent / not-yet-alive. `b` and `A` skip `None` slots entirely; a slot
   stays `None` until some block writes to it.** ← **This is the chosen semantics.**

The two readings agree exactly when $A(b) = 0$ on the dead slots — which happens to hold in
RNN-compat mode ($b_y = 0$, $A_y = I$, $\tanh(0) = 0$). That coincidence would hide the
discrepancy until someone puts a `sigmoid` or `LayerNorm` on a slot, so it must be an
explicit, documented choice rather than an accident.

Exploit the resulting algebra throughout: `None` $\circ$ anything = `None`,
$f \circ \texttt{None} = \texttt{None}$, `None` contributes nothing to a sum. It is both the
efficiency mechanism and the notation that keeps the code looking like the mathematics.

### 8.5 Multi-slot depth is a wavefront, and that is intended

**Exact `nn.RNN` multi-layer stacking is explicitly out of scope.** The Jacobi update is the
intended semantics, not a limitation to be worked around.

> **This section is about $L$, not $K$ — they are different axes, and an earlier draft read
> as though they were the same.**
>
> - **$K$** (§7) is *how many times the internal map runs per token*. Each internal step does
>   overwrite the hidden slot, so after $K$ steps the hidden state has been replaced $K$
>   times. That is intended, and is what §7 is about.
> - **$L$** (this section) is *how many hidden slots there are* —
>   $h^{(1)}, \ldots, h^{(L)}$, i.e. a stacked or deep RNN. The wavefront is about how
>   information moves *between* those slots.
>
> The two are independent. The wavefront arises already at $K = 1$, and — first corollary
> below — **raising $K$ does not fix it.**
>
> **In the code as it stands there is exactly one hidden slot, so this section is currently
> vacuous.** It is a pre-commitment about what happens when the multi-slot states of §4.1 get
> built, written down now so the decision is not made silently later by whoever writes that
> code.

For context, PyTorch's stacked RNN is

$$h_t^{(l)} = \tanh\big(W_{ih}^{(l)} h_t^{(l-1)} + W_{hh}^{(l)} h_{t-1}^{(l)}\big),
\qquad h_t^{(0)} = x_t$$

Layer $l$ at time $t$ reads layer $l-1$ at time $\mathbf{t}$, **not** $t-1$. But
`Sequential2D` is a **Jacobi** update — every block reads the *old* state and results are
summed into the *new* state — so a single $(L{+}1)\times(L{+}1)$ block map applied once per
timestep gives

$$h_t^{(l)} = \tanh\big(W_{ih}^{(l)} h_{t-1}^{(l-1)} + W_{hh}^{(l)} h_{t-1}^{(l)}\big)$$

a **diagonal wavefront**: information takes $l$ timesteps to reach slot $l$. This is a
different model from a stacked `nn.RNN`, and it is the one we want. It keeps the whole
construction a single block map, which is the premise of the entire design.

Two corollaries to record:

- **$K > 1$ does not convert the wavefront into exact stacking.** It is tempting to assume
  extra internal iterations subsume depth; they do not. On internal iteration 2 the
  recurrent term is already corrupted, because $h^{(1)}$ has overwritten the $h_{t-1}^{(l)}$
  that $W_{hh}^{(l)}$ needs to read. Exact stacking would require holding $h_{t-1}$ in
  separate slots — state $[x;\, h_{t-1}^{(1..L)};\, h_t^{(1..L)}]$ with $K = L$ sweeps, which
  is exact because the $h_t$ sub-block is then strictly lower triangular. We are **not**
  doing this.
- **Do not "fix" this with Gauss-Seidel.** Updating slots in place, in order, would also be
  exact and cost no extra work, but it makes `forward` slot-order-dependent. At that point it
  is no longer a block map and the paper's entire framing stops applying.

Consequence for §6: the `nn.RNN` equivalence test is **single-layer only**
(`num_layers=1`), where Jacobi and Gauss-Seidel coincide and the question does not arise.

### 8.6 $M_{xx} = I$, fixed and non-learnable — input persistence

**Default:** $M_{xx} = I$ (via `Identity`, `Sequential2D.py:8` — no parameters, so
non-learnable for free), $A_x = \mathrm{id}$, $b_x = 0$. Together these make the $x$-slot an
exact identity wire across the $K$ internal iterations.

This is a **default arrived at from principles, not an empirically validated choice** — see
the closing subsection before acting on it as if it were settled fact.

**Why it must be $I$ and not `None`: this is FIR vs. IIR, and the paper already proves it.**
Eq. (39) shows that iterating $M$ past $T$ times makes the result independent of $h_0$ —
finite impulse response, the input is forgotten. Eq. (50)'s $M_\infty$ adds exactly this
identity in the top-left corner, and eq. (54) shows the input dependence survives — infinite
impulse response. §7.1 is that theorem applied to the *internal* timescale.

The consequence is structural, not merely desirable: with $M_{xx} = \texttt{None}$, as
$K \to \infty$ the fast subsystem converges to a fixed point of an **unforced** map — the
same fixed point regardless of input. The model degenerates to a constant. $M_{xx} = I$ is
precisely what makes the $K \to \infty$ limit well-posed, and is the same device DEQ calls
input injection ($z^\star = f(z^\star, x)$ is only meaningful because $x$ is still present at
the fixed point).

**It is a skip connection — specifically an *input skip*, not a *state residual*.** Unroll
the $K$ internal iterations as a $K$-layer network. With $M_{xx} = I$, $A_x = \mathrm{id}$,
$b_x = 0$, the $x$-slot carries $x$ unchanged to every layer, bypassing all intervening
$W_{hh}$ and nonlinearities. Topologically that is a skip connection, and it buys the usual
thing skips buy — a short gradient path:

$$\frac{\partial h^{(K)}}{\partial x}
= \sum_{k=1}^{K}\left[\prod_{j=k+1}^{K} A_h'\,W_{hh}\right] W_{xh}$$

The $k = K$ term is $A_h'\,W_{xh}$, depth 1. With $M_{xx} = \texttt{None}$, $x$ enters only
at iteration 1 and the sum collapses to the single depth-$K$ term
$\left[\prod_{j=2}^{K} A_h'\,W_{hh}\right]W_{xh}$, which vanishes or explodes in $K$.

The one distinction worth preserving: this is an **input skip** (input → every layer,
DenseNet-flavoured), not a **state residual** (ResNet's $h \leftarrow h + f(h)$). The
$h$-recursion's own Jacobian is still $W_{hh}\,\mathrm{diag}(A_h')$, unchanged. So it
shortens the gradient path to the **input** but does nothing for vanishing gradients along
the **hidden** path. Do not expect ResNet-style conditioning of the state dynamics from it —
that would need an identity on the $h$-slot, i.e. $W_{hh} \supseteq I$, which is a separate
and currently unexplored choice.

**Why non-learnable.** The tempting argument — "a learnable $W_{xx}$ would be redundant with
$W_{xh}$" — is **wrong**. $W_{xx}$ composes with itself $K$ times, giving
$x^{(k)} = W_{xx}^{k}\,x$: genuine learned dynamics on the forcing (decay, rotation) that
$W_{xh}$ cannot express. It is not redundant, it is *extra* — and it carries a stability
hazard, since $\|W_{xx}\| > 1$ makes the forcing grow geometrically over the $K$ internal
steps. Fixed $I$ is the neutral choice. If this is ever relaxed, the right generalization is
a **scalar leak** $\lambda \in [0,1]$ giving $\lambda^{k}$ decay — interpolating $I$ at
$\lambda = 1$ and `None` at $\lambda = 0$ — not a full learnable matrix. Meaningless at
$K = 1$.

> **Trap: input persistence is a property of the whole $x$-row, not of $M_{xx}$ alone.**
> The $x$-slot passes through the full internal step $A \circ b \circ M$:
> $$x^{(k+1)} = A_x\big(x^{(k)} + b_x\big)$$
> With $M_{xx} = I$ but $b_x \neq 0$, the input **drifts linearly in $K$**:
> $x^{(k)} = x + k\,b_x$. With $A_x = \tanh$ it saturates. The invariant requires **all
> three** of $M_{xx} = I$, $b_x = 0$, $A_x = \mathrm{id}$.
>
> **Assert this in code, do not merely document it.** At $K = 1$ the violation is invisible
> (the $x$-slot is overwritten before it matters); at $K = 10$ it is a silent bug.

#### This is a principled default, not an empirical claim

Everything above argues from *principles* — impulse response, well-posedness of the
$K \to \infty$ limit, gradient path length. **None of it is a theorem that this configuration
performs better.** It is a defensible place to stop, chosen so the semantics are clean and
the invariants are assertable, not because the alternatives were tried and lost.

Each of the three constraints might be worth relaxing, and any of them could plausibly win in
practice:

| relaxation | what it would buy | why it is deferred |
| --- | --- | --- |
| **learnable $W_{xx}$** | learned dynamics on the forcing ($W_{xx}^{k}x$): decay, rotation, input-dependent gating over internal time | not redundant with $W_{xh}$, but stability hazard at $\|W_{xx}\| > 1$; prefer the scalar leak $\lambda$ first |
| **$A_x \neq \mathrm{id}$** | squashing/normalising the persistent forcing, e.g. `LayerNorm` on the held input | breaks exact persistence; $\tanh$ in particular saturates it |
| **$b_x \neq 0$** | a learned offset on the forcing | causes linear drift $x + k\,b_x$ over internal iterations — almost certainly not wanted, but it is a *choice*, not an impossibility |

Treat the table as a list of experiments, not a list of mistakes. Anyone revisiting this
should feel free to turn them on deliberately — the assertion guarding the invariant should
therefore be *disableable by an explicit flag*, not a hard `assert` that has to be deleted to
run the experiment.

---

## 9. Implementation notes and traps

1. **Use `forward_list` (`Sequential2D.py:98`), not `forward_vector`.** The state stays a
   list `[x, y, h]`; no concatenation, no slice-assignment, and `None`-propagation is free.
   `forward_vector` allocates and slice-accumulates on every call, which for a recurrent
   module happens $K \cdot L$ times per forward pass.
2. **`forward_vector` drops `dtype`** — `Sequential2D.py:127` calls
   `torch.zeros((...), device=X_in.device)` with no `dtype=`, silently promoting the
   accumulator to fp32 under AMP/bf16. Pre-existing and out of scope, but another reason to
   stay on the list path.
3. **Use `Identity` (`Sequential2D.py:8`)** for $I$ cells so the block matrix in code reads
   like the block matrix in the paper. Note it returns its input unchanged, so
   `in_features` must equal `out_features` for it to be meaningful — it does not check this.
4. **Watch the transpose (§3.2).** Consider writing the block matrix in paper convention and
   transposing once, in one clearly-marked place, rather than transposing mentally at every
   construction site.
5. **Block types are pluggable.** Every cell can be `MaskedLinear`, `SparseLinear`, or
   `MonarchLinear` instead of `torch.nn.Linear`. An RNN with a Monarch $W_{hh}$ behind an
   unchanged outer API is the paper's §8 sparsity study with no new plumbing — this is close
   to free and should be supported from the start.

---

### 9.6 The Python loop is a small-model problem

`Sequential2DRNN` steps a Python loop with a handful of kernel launches per
timestep, where `torch.nn.RNN` calls a fused cuDNN kernel that runs the whole
recurrence on-device. The resulting gap is often quoted as ~100x, which is true and
badly misleading: it is a statement about *small hidden sizes*, not about the
method. Measured on an RTX 4090, $L = 1024$, $B = 64$, $d_x = 1$:

| $d_h$ | our loop | cuDNN | gap |
| ---: | ---: | ---: | ---: |
| 128 | 162 ms | 1.2 ms | 94x |
| 512 | 164 ms | 21.7 ms | 7.5x |
| 1024 | 158 ms | 28.5 ms | 5.5x |
| 2048 | 160 ms | 77.7 ms | 2.1x |

Note our column is **flat** — 158-164 ms regardless of $d_h$ — because the loop is
launch-bound, not compute-bound. cuDNN's entire advantage is amortising launch
overhead, so it evaporates once the per-step $d_h^2$ matmul is real work.

> **Flat cost is about wall-clock, not trainability.** Measured later on real
> training: our s/epoch is flat (29.9 / 29.4 / 29.1 at $d_h$ = 128 / 512 / 2048)
> while cuDNN GRU goes 5.2 -> 20.5. But over the same sweep `nn.GRU h=2048`
> diverged to NaN at two learning rates, and *our* wide models needed the learning
> rate scaled roughly as $1/d_h$ (1e-3 -> 1e-4 -> 3e-5) before width helped at all.
> Width being free to **run** is not width being free to **train**. Do not read this
> section as saying large $d_h$ is available for nothing.

The consequence for planning: **the overhead is worst exactly where it matters
least.** Small-hidden experiments are cheap in absolute terms even at 100x, and the
large-hidden experiments this project is aimed at — large $h$, large Monarch blocks —
run at roughly cuDNN speed. Paying the cost to keep the code readable is therefore
a much easier trade than the headline number suggests. See §5.5b for the one
optimisation considered and declined.

---

## 10. Open questions

Not yet decided; do not guess. Note that `num_layers > 1` is **not** on this list — it is a
settled design decision, §8.5.

### 10.1 The rest of the $x$-row: $W_{yx}$ and $W_{hx}$

$M_{xx}$ is settled (§8.6). The other two cells writing *into* the $x$-slot — $W_{yx}$
($y \to x$) and $W_{hx}$ ($h \to x$), source-first per §3.4 — are not exposed by `from_3x3`,
which covers only the $y$ and $h$ rows.

That is a $K=1$-flavoured choice: at $K = 1$ anything written into $x$ is discarded by the
next `Inject`, so excluding them costs nothing. At $K > 1$ they become live, and would let
the internal dynamics modify their own forcing between injections — feedback onto the input
channel. Note this would break the input-persistence invariant of §8.6 by construction, so
the two features are in tension and cannot simply both be switched on.

Deferred, not rejected. Revisit when $K > 1$ experiments are actually running.

### 10.2 Other unresolved items

- **`bidirectional`** — mechanically a second parameter set run over reversed time, outputs
  concatenated. Not yet scoped.
- ~~**`PackedSequence`**~~ — **done**, `forward_packed`. It turned out to segregate
  cleanly: packing changes only how many rows of the batch are alive per step, so the
  four operators of §5 are untouched and the whole of it is confined to one method.
- **`batch_first`, `dropout`** — mechanical, but confirm they are in scope.
- **Should $K$ be allowed to vary** (per layer, per timestep, adaptive)? Start constant.
  See §10.3 — adaptive $K$ falls out of convergence for free.
- **Does $K > 1$ ever win?** The one experiment run so far
  (`examples/rnn_internal_iterations.py`) says no on a pure *memory* task, and
  explains why: every internal iteration applies a contraction, so the memory
  horizon shrinks geometrically in $K$. Orthogonal initialisation of $W_{hh}$
  largely repairs it — $K = 4$ goes from chance to 0.89 accuracy — but $K = 1$
  still wins. **The comparison across $K$ is meaningless unless initialisation is
  controlled for**, which was not anticipated anywhere above. Untested: a
  compute-bound rather than memory-bound task, which is where $K > 1$ has
  somewhere to put the effort.

### 10.3 Convergence of the internal iteration — a feature, and the trap inside it

Not an open question so much as a research direction with one sharp hazard in it.

**Convergence of $T = A \circ b \circ M$ in $k$ is desirable, not a failure.** If the
internal iteration settles, you can stop early and save compute, the answer stops
depending on the exact value of $K$, and — the part easy to miss — the
$K \cdot L$ BPTT depth of §7.2 **goes away**: at a fixed point the implicit
function theorem gives the gradient without storing the internal trajectory,
$O(1)$ memory in $K$. Per-token early stopping is also exactly Graves' adaptive
computation time, derived from the dynamics rather than bolted on.

**The trap.** Define the fixed-point map

$$\Phi(x, h) \;=\; \lim_{K \to \infty} T^{K}\big(\mathrm{Inject}_x(h)\big),
\qquad\text{so}\qquad h_{t+1} = \Phi(x_{t+1}, h_t)$$

Convergence in $k$ and memory in $t$ are *different quantities*, and they are in
tension. If $T$ is a **global contraction** with rate $\rho < 1$, then
$\|\partial\Phi/\partial h\| \le \rho^{K} \to 0$: the fixed point stops depending
on $h$ at all and the model becomes **memoryless**, a feedforward network applied
per token. That is §4.3.1's finite-impulse-response result restated on the
internal timescale.

So the condition to aim for is sharp:

> $T$ must **converge without being a global contraction in $h$** — which means
> multiple fixed points, so that *which basin* is reached encodes the memory.

Memory then lives in the identity of an attractor rather than in a decaying
transient, and does not decay at all. This is an attractor/associative-memory
network, and is the Radhakrishnan et al. iterated-autoencoder line the paper
already cites as close in spirit.

#### The input conditions the landscape — do not read the above as $h$-only

The framing just given is incomplete in an important way. For a held input $x$,
the internal map $T_x$ has a fixed-point set $\mathcal{F}(x)$, and

> **$x$ determines what that set *is*; $h$ determines which element of it is
> reached.**

Both dependencies are required, and they do different jobs:

$$\frac{\partial \Phi}{\partial x} \neq 0 \quad \text{(the model responds to input at all)}
\qquad
\frac{\partial \Phi}{\partial h} \neq 0 \quad \text{(the model remembers)}$$

**The $x$-dependence is the more basic of the two.** A converged state that
ignores $x_t$ computes nothing, memory or no memory. Whereas
$\partial\Phi/\partial x \neq 0$ with $\partial\Phi/\partial h = 0$ is still a
functional model — a DEQ, a feedforward network per token; useful, merely not
recurrent. So the collapse warned about above is the *less* severe of the two
failure modes.

The right mental picture is therefore not "a fixed landscape whose basins are
indexed by $h$" but **$x_t$ reshapes the landscape that $h_t$ navigates**. The
input behaves as a *bifurcation parameter* rather than a perturbation: as $x$
varies, fixed points of $T_x$ are created and destroyed, changing how many basins
exist and where they sit.

> **Terminology, loosely used.** "Bifurcation parameter" here is descriptive, not
> a claim that any particular normal form has been identified. No continuation has
> been run and nothing has been classified as saddle-node, pitchfork or Hopf in
> the general high-dimensional case. The one place the term is used *literally*
> in this repository is
> `notebooks/8-rcp-fixed-points-and-bistability.ipynb`, which reduces to
> `hidden_size = 1`, where the fold is genuinely a saddle-node and can be
> exhibited.

Two consequences:

- **This is a third and stronger justification for $M_{xx} = I$ (§8.6).** Holding
  $x$ constant across the $K$ internal iterations is what makes $T_x$, and hence
  $\mathcal{F}(x)$, *well defined at all*. Without input persistence there is no
  input-conditioned map to have fixed points of, and the whole landscape picture
  evaporates. §8.6 argues from impulse response and gradient path length; this is
  the deeper reason.
- **Input-conditioned dynamics is the load-bearing ingredient in the closest
  state-of-the-art relatives.** Mamba's selectivity is exactly input-dependent
  state transition — the input conditioning the map rather than merely forcing
  it. The paper already cites Gu & Dao alongside S4 in §1, so this is not a
  speculative extension.

**Consequence for a convergence loss.** The obvious
$\mathcal{L}_{\text{conv}} = \|z^{(K)} - z^{(K-1)}\|$ is **globally minimized by
the degenerate solution** — one attractor, no $h$-dependence, zero memory. It
does not merely permit the failure mode, it points at it. The task loss opposes
this, so the pair may well train fine, but they are adversarial and the balance
must be monitored rather than assumed.

Two safer formulations:

- **Measure three things, not one.** Convergence $\|z^{(k)} - z^{(k-1)}\| \to 0$,
  input sensitivity $\|\partial\Phi/\partial x\| \not\to 0$, and memory
  $\|\partial\Phi/\partial h\| \not\to 0$. Together they say "settles, responds,
  and still remembers"; any one alone is misleading, and the three fail
  independently for different reasons.

  Note the diagnostic currently in `examples/rnn_internal_iterations.py` measures
  $\partial h_t/\partial x_0$, which is a *product* of these — roughly
  $\|\partial\Phi/\partial x\| \cdot \|\partial\Phi/\partial h\|^{t}$ — and so
  cannot distinguish a model that has stopped responding from one that has
  stopped remembering. It should be factored.
- **Penalize the spectrum, not the difference.** Push the spectral radius of
  $J_T$ toward slightly below 1 rather than toward 0 — convergence with a long
  memory, rather than convergence by collapse. This is what the DEQ literature
  settled on for the same problem (Bai, Koltun & Kolter, *Stabilizing Equilibrium
  Models by Jacobian Regularization*, 2021, penalizing $\|J\|_F$ with a
  Hutchinson estimator).

#### RCP's amendment: training already opposes the degenerate solution

Reviewing the above, RCP made the following point, and it is right:

> If different $y$ outputs are part of the training process, and we achieve low loss, then the
> model will have learned to converge to the *various* correct $y$ outputs.

Agreed, and it materially weakens the warning as originally written. The degenerate
solution — one attractor, no $h$-dependence — has *high task loss* on any task that needs
memory. So it is not a competing optimum of $\mathcal{L}_{\text{task}} +
\mathcal{L}_{\text{conv}}$; at low total loss the model necessarily distinguishes whatever
the task requires it to distinguish, which is exactly $\partial\Phi/\partial h \neq 0$ where
it matters. The warning above is therefore **not** an argument against adding a convergence
loss.

What survives is narrower, and is about **optimisation dynamics rather than optima**:

- $\nabla\mathcal{L}_{\text{conv}}$ is well conditioned near the degenerate solution, while
  $\nabla\mathcal{L}_{\text{task}}$ through a near-contraction is exponentially small in $K$.
  The memory-horizon table in `examples/rnn_internal_iterations.py` shows this reaching
  *numerically zero* at $K = 4$ — and a zero gradient means the task loss cannot inform those
  parameters at all.
- So the failure mode is not "training picks the degenerate optimum", it is "training never
  reaches a region where the task loss has anything to say". The premise *if we achieve low
  loss* is precisely what is at risk.

Which makes this an empirical question, not one to be settled by argument. **Sweep the
convergence-loss weight and watch whether the task loss ever drops**, monitoring
$\partial\Phi/\partial h$ alongside. If it trains, RCP is right and the concern was
overstated. Initialisation will matter more than the loss weight, for the reasons in §8.6 and
the orthogonal-init result in that example.

**Caveat on the existing measurement.** The memory-horizon numbers in
`examples/rnn_internal_iterations.py` are taken **at initialization**, where a
randomly-initialized contraction having no memory is nearly a tautology. What the
*trained* spectrum does is the real question and has not been looked at. Training
may push $\rho$ toward 1 on its own when the task demands memory, in which case
the orthogonal initialization is providing a better starting point rather than
repairing anything fundamental.

---

## 11. One-paragraph summary for a cold reader

A `Sequential2D` block map $M$ over a slot-partitioned state $[x, y, h]$, composed with a
per-slot bias $b$ and a per-slot activation $A$, iterated $K$ times per input token, with the
$x$-slot overwritten by `Inject` between token steps and held by a fixed $M_{xx} = I$ in
between. Setting $K = 1$, three slots, four nonzero cells,
$A = \mathrm{diag}(\mathrm{id}, \mathrm{id}, \tanh)$, and no $y \to h$ feedback reproduces a
single-layer `torch.nn.RNN` exactly — that is the regression test, and `from_rnn` makes it a
supported entry point rather than only a test. Everything else — $K > 1$, extra slots,
arbitrary sparsity patterns, structured block types, feedback cells — is the actual research
surface, and rests on reading the network as a fast internal dynamical system driven by a
slow external forcing.
