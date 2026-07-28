# `Sequential2DRNN`: an RNN as an iterated block map

A `torch.nn.RNN` is one point in a much larger family of models. This module makes
that concrete: it runs a `Sequential2D` block map as a discrete dynamical system,
reproduces `torch.nn.RNN` exactly at one setting of its parameters, and opens up
everything around that setting.

## The whole module is one equation

Writing $z$ for the state, split into slots,

$$z_{t+1} = \underbrace{(A \circ b \circ M)^{K}}_{\text{fast, internal}} \circ
\underbrace{\mathrm{Inject}_{t+1}}_{\text{slow, external}} \circ\; z_t$$

| symbol | what it is |
| --- | --- |
| $M$ | a `Sequential2D` — a matrix whose entries are (possibly nonlinear) maps |
| $b$ | a bias, one vector per slot |
| $A$ | an activation, one module per slot |
| $\mathrm{Inject}$ | overwrite the input slot with the next token |
| $K$ | how many times the internal map runs per token |

Read right to left: a token arrives and is written into the input slot; the
internal map runs $K$ times; the next token arrives.

**$K$ is the point.** It decouples the network's own dynamics from the sequence's
clock — a fast internal system driven by a slow external forcing. $K = 1$
collapses to an ordinary RNN, which is why `torch.nn.RNN` compatibility falls out
as a special case rather than being designed for. $K > 1$ is the research surface.

## Quickstart

```python
import torch
from iterativennsimple.Sequential2DRNN import Sequential2DRNN

# Exactly reproduce a torch.nn.RNN.
rnn = torch.nn.RNN(input_size=4, hidden_size=6, batch_first=True)
model = Sequential2DRNN.from_rnn(rnn)

x = torch.randn(3, 7, 4)                       # (batch, seq, features)
assert torch.allclose(model(x)[0], rnn(x)[0], atol=1e-6)

# Same weights, but the internal map now runs 3 times per token.
# This is no longer an RNN, and that is the idea.
thinking = Sequential2DRNN.from_rnn(rnn, K=3)
```

The general three-slot map, state $z = [x, y, h]$, exposes six free blocks:

```python
linear = lambda i, o: torch.nn.Linear(i, o, bias=False)   # bias lives on the slot

model = Sequential2DRNN.from_3x3(
    input_size=4, output_size=2, hidden_size=32,
    W_xh=linear(4, 32),        # x -> h    input
    W_hh=linear(32, 32),       # h -> h    recurrence
    W_hy=linear(32, 2),        # h -> y    readout
    #  W_xy, W_yy, W_yh also available
    K=3, batch_first=True,
)
```

Variable-length batches work as they do with `nn.RNN` — pass a `PackedSequence`
and get one back:

```python
from torch.nn.utils.rnn import pack_padded_sequence
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
output, h_n = model(packed)          # output is a PackedSequence; h_n is (1, batch, hidden)
```

Naming is **source-first**: `W_ab` maps slot `a` to slot `b`. Any module carrying
`in_features` and `out_features` works as a block — `MaskedLinear`,
`MonarchLinear`, `SparseLinear`, `Sequential1D`, or a nested `Sequential2D`.
`None` means the block is absent, and costs nothing to evaluate.

## Where to start reading

| | |
| --- | --- |
| **`notebooks/7-rcp-RNN-as-Sequential2D.ipynb`** | **Start here.** Builds the block map by hand, checks it against `torch.nn.RNN`, then turns $K$ up and watches what happens. Runs in seconds. |
| **`notebooks/8-rcp-fixed-points-and-bistability.ipynb`** | The dynamical-systems view, with `hidden_size = 1` so everything can be drawn: cobwebs, bistability, and an honest saddle-node fold. Where memory-as-attractor-identity comes from. Also runs in seconds. |
| `examples/rnn_internal_iterations.py` | Trains at several $K$ on a task that needs memory. A worked negative result: $K > 1$ collapses, the reason is measured, the fix is applied, the comparison is redone. Takes a couple of minutes. |
| `OVERVIEW_RNN_SEQUENTIAL_2D.md` | Every design decision and why, including the ones deliberately not taken. Read before changing anything. |
| `tests/test_Sequential2DRNN.py` | The invariants — mostly the ones that fail silently. |
| [arXiv:2404.00880](https://arxiv.org/abs/2404.00880) | Hershey, Paffenroth, Pathak & Tavener. The mathematics this implements. |

## Five things that will bite you

These are all deliberate, and each is explained at length in the overview.

**1. `blocks[i][j]` is indexed (input, output) — the transpose of the matrix on
the page.** In `Sequential2D`, `blocks[i][j]` takes slot `i` and produces slot
`j`, so the code array is the transpose of the block matrix as normally written.
The factories hide this; direct construction does not.

Reassuringly, this does *not* extend to weight tensors: PyTorch stores
`Linear.weight` as `(out_features, in_features)` — the mathematics convention —
and transposes inside `forward`. So `rnn.weight_ih_l0` copies across with **no
transpose at all**.

**2. The activation goes on the slot, not on the block.** $A$ is applied after
everything arriving at a slot has been summed, giving $\tanh(W_{xh}x + W_{hh}h)$
rather than $\tanh(W_{xh}x) + \tanh(W_{hh}h)$. Only the first is an RNN. This is
also why `Sequential2D.from_config`'s per-block `activation` option cannot express
one.

**3. Bias belongs to the slot, so blocks must be bias-free.** `torch.nn.Linear`
defaults to `bias=True`, which double-counts. `from_3x3` asserts against it.
PyTorch's two RNN biases become one: $b_h = b_{ih} + b_{hh}$, which is exact for
the forward pass but does not round-trip back to a `torch.nn.RNN` `state_dict`.

**4. A readout is an *observation*, not a slot — and the difference is a one-step
lag.** The state map and the reported output are separate objects:

$$z_{t+1} = F(z_t, x_{t+1}) \qquad\qquad y_t = g(z_t)$$

which is how S4, MAMBA and linear control are all written. Pass `readout=` and it
becomes $g$: reads the current state, no lag, cannot feed back. Carry the readout
in the $y$-*slot* instead and it goes through the block map, which is a **Jacobi**
update, so it lands one internal iteration behind — a full token of delay at
$K = 1$. Use the slot only when you want feedback (the paper's $S$ block); use
`readout` otherwise.

Note the bias rule in item 3 does *not* apply to `readout`: it sits outside the
block matrix, so an ordinary biased `nn.Linear` is correct there.

**5. `None` means "not yet alive", not "zero".** A slot nothing has written to
stays `None`, and $b$ and $A$ skip it. This is *not* the same as holding a zero
vector, since $A(0 + b)$ is generally neither. The two agree only when
$A(b) = 0$ — which happens to hold in RNN-compatible mode, so the difference
stays invisible until you put a `sigmoid` or `LayerNorm` on a slot.

## What is deliberately not here

Not gaps to be filled in without thinking — each was decided against, with
reasons in the overview.

- **Multi-layer stacking.** A single block map applied once per timestep gives a
  diagonal *wavefront*, where information takes $l$ steps to reach slot $l$. That
  differs from `torch.nn.RNN`'s stacking, and keeping the construction one block
  map is worth more than matching it. `from_rnn` asserts `num_layers == 1`.
- **`bidirectional`, `dropout`.** Not scoped yet.
- **The lifted formulation** (dimension $|h| + T(|x|+|h|)$, eq. 15 of the paper).
  Elegant — it makes RNN and MLP literally the same fixed map — but it bakes the
  sequence length into the architecture, costs $\Theta(T^2)$, and its blocks
  depend on the data rather than only on the parameters.
- **A learnable $M_{xx}$.** Fixed at $I$, which holds the injected token across
  all $K$ internal iterations. This is an input skip connection across the
  unrolled steps and is what makes the $K \to \infty$ limit well-posed. It is a
  principled default, *not* a measured one — the overview lists it as an
  experiment worth running.

## Status

Implemented and tested: the `torch.nn.RNN` equivalence, the general three-slot
map, the observation map, `PackedSequence`, arbitrary and nested block types, and
$K > 1$. Deferred work is in `TODO_Sequential2DRNN.md`; open design questions are
in §10 of the overview.
