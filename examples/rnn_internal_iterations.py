"""Internal iterations trade memory for computation -- a worked negative result.

NOTE: this "notebook" is actually a .py file that can either be run
interactively in a Jupyter environment or as a standard python script.  It is a
.py rather than a notebook because it trains seven models and takes a couple of
minutes, which is longer than a notebook that runs in CI should take.

The question.  In

    z_{t+1} = (A . b . M)^K . Inject_{t+1} . z_t

the number of internal iterations K is free.  K = 1 is an ordinary RNN.  K > 1
lets the network run its own dynamics several steps per input token -- a fast
internal timescale under a slow external one.  Does that extra computation help?

The short answer, on a task that needs memory: no, and at first it looks
catastrophic.  The interesting part is *why*, and that the reason is fixable.
The script walks the whole loop -- measure, diagnose, fix, re-measure -- because
that loop is more useful to copy than any particular number in it.

See OVERVIEW_RNN_SEQUENTIAL_2D.md for the mathematics.
"""

import time

import torch
import plotly.graph_objects as go

from iterativennsimple.Sequential2DRNN import Sequential2DRNN

torch.manual_seed(0)

# %%
# The task: emit the token seen `delay` steps ago.
#
# Tokens are one-hot vectors of dimension `n_symbols`.  Chosen because the *only*
# thing that solves it is memory, so any difference between models is
# attributable to the recurrent state rather than to per-token capacity.

n_symbols = 8
delay = 5
seq_len = 20
batch_size = 128
hidden_size = 24


def copy_with_delay(batch_size):
    """Returns x, y of shape (batch, seq_len, n_symbols), with y[t] = x[t-delay]."""
    symbols = torch.randint(0, n_symbols, (batch_size, seq_len))     # (batch, seq)
    x = torch.nn.functional.one_hot(symbols, n_symbols).float()      # (batch, seq, sym)
    y = torch.zeros_like(x)
    y[:, delay:, :] = x[:, :-delay, :]     # first `delay` targets are meaningless
    return x, y


x_check, y_check = copy_with_delay(2)
print(f'x {tuple(x_check.shape)}  y {tuple(y_check.shape)}')
print(f'target at t={delay} equals input at t=0: '
      f'{torch.equal(y_check[:, delay], x_check[:, 0])}')


# %%
# The model.
#
# State z = [x, y, h].  We use three of the six free blocks: the input drives the
# hidden state, the hidden state is recurrent, and the hidden state is read out.
# `orthogonal` is the knob the diagnosis below will turn out to need.

def build(hidden_size, K, orthogonal=False, gain=1.2, seed=0):
    torch.manual_seed(seed)
    linear = lambda i, o: torch.nn.Linear(i, o, bias=False)   # bias is on the slot

    W_hh = linear(hidden_size, hidden_size)
    if orthogonal:
        # Orthogonal W has all singular values 1, and gain > 1 pushes back
        # against the contraction that tanh' < 1 introduces at every iteration.
        torch.nn.init.orthogonal_(W_hh.weight, gain=gain)

    return Sequential2DRNN.from_3x3(
        input_size=n_symbols, output_size=n_symbols, hidden_size=hidden_size,
        W_xh=linear(n_symbols, hidden_size),   # x -> h
        W_hh=W_hh,                             # h -> h, the recurrence
        W_hy=linear(hidden_size, n_symbols),   # h -> y, the readout
        A_h=torch.nn.Tanh(),
        A_y=torch.nn.Identity(),               # logits; the loss applies softmax
        K=K, batch_first=True)


# %%
# Training.  Cross-entropy against the delayed symbol, scored only where the
# target is a real symbol.

def train(model, steps=400, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = copy_with_delay(batch_size)
        logits, _ = model(x)                                  # (batch, seq, sym)

        logits_scored = logits[:, delay:, :].reshape(-1, n_symbols)
        targets = y[:, delay:, :].argmax(dim=-1).reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits_scored, targets)

        optimizer.zero_grad()
        loss.backward()
        # An RNN on a memory task will blow up without this; the clip is doing
        # real work here, not defensive scaffolding.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def accuracy(model, batch_size=512):
    x, y = copy_with_delay(batch_size)
    with torch.no_grad():
        logits, _ = model(x)
    predicted = logits[:, delay:, :].argmax(dim=-1)
    actual = y[:, delay:, :].argmax(dim=-1)
    return (predicted == actual).float().mean().item()


# %%
# ---------------------------------------------------------------------------
# 1. The naive comparison
# ---------------------------------------------------------------------------
#
# Hidden width fixed, K varied.  Note K changes *no parameters at all* -- it only
# changes how many times the same map is applied.

configurations = [1, 2, 4]
naive = {}

for K in configurations:
    model = build(hidden_size, K)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start = time.time()
    losses = train(model)
    naive[K] = {'losses': losses, 'accuracy': accuracy(model),
                'time': time.time() - start, 'params': n_params}
    print(f'K={K}  params={n_params}  final loss={losses[-1]:.4f}  '
          f'accuracy={naive[K]["accuracy"]:.3f}  ({naive[K]["time"]:.1f}s)')

print(f'\nchance accuracy is 1/{n_symbols} = {1/n_symbols:.3f}')
print('K > 1 is not merely worse -- K=4 is at chance.  Why?')

# %%
# ---------------------------------------------------------------------------
# 2. The diagnosis: how far does information actually travel?
# ---------------------------------------------------------------------------
#
# Measure the memory horizon directly, at initialisation, before any training
# muddies it:  || d h_t / d x_0 ||.  If this reaches zero before t = delay, the
# task is not hard for the model -- it is unsolvable by it.

def memory_horizon(model, probe_times=(1, 3, 5, 10, 19)):
    x = torch.randn(1, seq_len, n_symbols, requires_grad=True)
    z = [None, None, torch.zeros(1, model.features_list[2])]
    norms = {}
    for t in range(seq_len):
        z = model.external_step(z, x[:, t, :])
        if t in probe_times:
            gradient = torch.autograd.grad(z[2].sum(), x, retain_graph=True)[0]
            norms[t] = gradient[0, 0, :].norm().item()   # sensitivity to token 0
    return norms


probe_times = (1, 3, 5, 10, 19)
print('|| d h_t / d x_0 ||   (sensitivity of the hidden state to the first token)')
print('  t =       ' + '  '.join(f'{t:>8d}' for t in probe_times))
for K in configurations:
    norms = memory_horizon(build(hidden_size, K))
    print(f'  K = {K}:  ' + '  '.join(f'{norms[t]:8.1e}' for t in probe_times))

# %%
# There it is.  Every internal iteration applies tanh, whose derivative is below
# one, so the map contracts.  Running it K times per token contracts the memory
# K times as fast, and the horizon shrinks *geometrically* in K.  At K = 4 the
# influence of token 0 is numerically zero well before the delay of 5 -- the
# model cannot represent the task, let alone learn it.
#
# This is not a fact about neural networks; it is a fact about iterating a
# contraction, and it is exactly the kind of question the dynamical-systems
# framing makes obvious to ask.

# %%
# ---------------------------------------------------------------------------
# 3. The fix: stop the recurrence from contracting
# ---------------------------------------------------------------------------
#
# Initialise W_hh orthogonally with gain > 1, so that the linear part expands
# just enough to offset tanh' < 1.

print('|| d h_t / d x_0 ||   with orthogonal W_hh (gain 1.2)')
print('  t =       ' + '  '.join(f'{t:>8d}' for t in probe_times))
for K in configurations:
    norms = memory_horizon(build(hidden_size, K, orthogonal=True))
    print(f'  K = {K}:  ' + '  '.join(f'{norms[t]:8.1e}' for t in probe_times))

print('\nK=4 now has a longer memory horizon than the *default* K=1 did.')
print('The original comparison was measuring initialisation, not K.')

# %%
# ---------------------------------------------------------------------------
# 4. The fair comparison
# ---------------------------------------------------------------------------

fair = {}
for K in configurations:
    model = build(hidden_size, K, orthogonal=True)
    start = time.time()
    losses = train(model)
    fair[K] = {'losses': losses, 'accuracy': accuracy(model),
               'time': time.time() - start}
    print(f'K={K} (orthogonal)  final loss={losses[-1]:.4f}  '
          f'accuracy={fair[K]["accuracy"]:.3f}  ({fair[K]["time"]:.1f}s)')

# %%
# A width-matched control.  K multiplies compute, so the honest question is not
# "does K > 1 beat K = 1" but "does K > 1 beat spending the same extra compute on
# a wider hidden state".

wide = build(hidden_size * 2, K=1, orthogonal=True)
wide_params = sum(p.numel() for p in wide.parameters() if p.requires_grad)
wide_losses = train(wide)
print(f'K=1 wide (hidden={hidden_size*2})  params={wide_params}  '
      f'final loss={wide_losses[-1]:.4f}  accuracy={accuracy(wide):.3f}')

# %%
# Training curves.  Smoothed, since minibatch noise is larger than the
# differences being compared.

def smooth(values, window=20):
    values = torch.tensor(values)
    kernel = torch.ones(window) / window
    return torch.nn.functional.conv1d(
        values.view(1, 1, -1), kernel.view(1, 1, -1)).flatten().tolist()


figure = go.Figure()
for K in configurations:
    figure.add_trace(go.Scatter(y=smooth(naive[K]['losses']),
                                name=f'K = {K}, default init',
                                line=dict(dash='dot')))
for K in configurations:
    figure.add_trace(go.Scatter(y=smooth(fair[K]['losses']),
                                name=f'K = {K}, orthogonal init'))
figure.add_trace(go.Scatter(y=smooth(wide_losses),
                            name=f'K = 1, hidden = {hidden_size*2}',
                            line=dict(dash='dash', color='black')))
figure.update_layout(
    title=f'Copy-with-delay ({delay} steps, {n_symbols} symbols): '
          f'internal iterations, initialisation, and width',
    xaxis_title='training step', yaxis_title='cross-entropy (smoothed)',
    template='plotly_white')
figure.show()

# %%
# What K costs.  It multiplies both the work per token and the
# backpropagation-through-time depth, which is K * seq_len rather than seq_len.

print(f'\n{"K":>3}  {"naive acc":>10}  {"fair acc":>9}  {"train time":>11}  {"BPTT depth":>11}')
for K in configurations:
    print(f'{K:>3}  {naive[K]["accuracy"]:>10.3f}  {fair[K]["accuracy"]:>9.3f}  '
          f'{fair[K]["time"]:>10.1f}s  {K * seq_len:>11}')

# %%
# What this does and does not show.
#
# It does show that internal iterations interact strongly with the spectrum of
# the recurrence, that the interaction is geometric in K, and that a comparison
# across K is meaningless until initialisation is controlled for.  That is a
# statement about iterated maps and it will hold wherever this construction is
# used.
#
# It should NOT be read as "the internal iteration converging is bad".  Convergence
# is the goal: it lets you stop early, makes the answer independent of the exact K,
# and removes the K*seq_len backpropagation depth entirely, since at a fixed point
# the gradient follows from the implicit function theorem.  What the numbers above
# measure is something narrower -- a *global* contraction, which converges to a
# fixed point that has forgotten h_t.  Writing
#
#     Phi(x, h) = lim_K T^K(Inject_x(h)),   so   h_{t+1} = Phi(x_{t+1}, h_t)
#
# there are two quantities that matter, and they fail independently:
#
#     d Phi / d x   the model responds to the current input at all
#     d Phi / d h   the model remembers
#
# A global contraction with rate rho sends the second to zero like rho^K --
# convergent and memoryless.  What you want instead is convergence *without*
# global contraction: several fixed points, so that which basin is reached carries
# the memory.  Memory then lives in the identity of an attractor rather than in a
# decaying transient, and does not decay at all.
#
# Crucially the fixed points depend on x as well as on h, and that dependence is
# the more basic of the two: x determines what the fixed-point set *is*, while h
# determines which element of it is reached.  A converged state that ignores x_t
# computes nothing.  So the input is better thought of as a bifurcation parameter
# reshaping the landscape than as a perturbation within a fixed one.
#
# Note the measurement above, d h_t / d x_0, is a *product* of these -- roughly
# ||d Phi/d x|| * ||d Phi/d h||^t -- so it cannot tell a model that stopped
# responding from one that stopped remembering.  Factoring it is the obvious next
# diagnostic.
#
# The practical warning that follows: a naive convergence loss ||z_K - z_(K-1)||
# is globally minimised by the degenerate solution -- one attractor, no
# h-dependence, no memory.  See Sec. 10.3 of the overview.
#
# Also note every memory-horizon number above is measured at *initialisation*.
# What the trained spectrum does is a different and more interesting question,
# and is not answered here.
#
# It does not show that K > 1 is useless.  Copy-with-delay is a pure *memory*
# task, and memory is precisely what iterating a contraction destroys.  A task
# needing per-token *computation* -- several reasoning steps on the current input
# rather than long retention -- is where K > 1 has somewhere to put the effort.
# This experiment is small, synthetic, and single-seed; treat it as a worked
# example of the method rather than as evidence.
#
# Things to try from here:
#
# * Run a compute-bound task instead of a memory-bound one, e.g. requiring
#   several steps of arithmetic on each token, and see whether the sign flips.
# * Sweep `gain`, or constrain W_hh to be exactly orthogonal during training, and
#   see how much of the K effect survives.
# * Turn on W_yy (y -> y) so the readout has its own memory, or W_yh (y -> h) so
#   the output feeds back -- the above-diagonal block S of the paper.
# * Replace W_hh with MonarchLinear or MaskedLinear and ask whether a sparse
#   recurrence at larger K beats a dense one at K = 1 for equal parameter count.
#   That is the sparsity question the paper is really about.
# * Watch the internal iterations converge: drive model.internal_step by hand
#   between tokens and track ||z_k - z_(k-1)||.  If it contracts, the model is
#   closer to a deep equilibrium model than to an RNN -- and the memory result
#   above says that contraction is exactly what costs you the memory.
