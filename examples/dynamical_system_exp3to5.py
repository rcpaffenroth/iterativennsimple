"""Experiments 3, 4 and 5 — the (activation, inner steps) square.

    y_{k,j+1} = sigma( A x_k + B y_{k,j} + b ),   x_k held for j = 1..M

Two knobs, four experiments, one model:

                    M = 1                       M > 1  (x_k held)
    identity        Experiment 2                Experiment 4   (a control, see below)
    tanh            Experiment 3                Experiment 5

So **M = 1 reproduces Experiments 2 and 3 exactly**, which makes every run of this file
carry its own control.  `check_contains_experiment_2` asserts the identity/M=1 corner is
bit-identical to `dynamical_system_exp2.LinearRecurrence`.

Experiment 4 is a control, not an experiment (design record Sec. 6).  Holding x for M inner
steps in the LINEAR case unrolls to

    y <- B^M y + (I - B^M)(I - B)^{-1} A x

which is Experiment 2 with B' = B^M constrained to have an M-th root -- a strict SUBSET.
Inner iterations cannot buy representational power there, so if Experiment 4 beats
Experiment 2 the implementation is wrong.  What is NOT vacuous is that the objective
differs: inner steps carry a convergence term, so Experiment 4 measures what the trajectory
penalty costs with representational power held fixed.

TWO DEPARTURES FROM THE DESIGN RECORD, both deliberate:

  * **Targets are NOT scaled into sigma's interior.**  The record called for t in {0, 0.9}
    to avoid tanh saturating against a target of 1.  But accuracy is argmax, which is
    scale-invariant, so saturation cannot change the reported number -- while moving the
    target alongside the activation would move two variables at once (PRINCIPLES Sec. 2.3).
    Targets stay one-hot everywhere.  The real cost of saturation is a vanishing gradient,
    so `saturation` below MEASURES it rather than patching it in advance.
  * **Cold start only.**  The frozen-warm arm does not extend here: Experiment 3 has exactly
    the same parameters as Experiment 2, so there is no new block to freeze around.  What to
    do instead is a design decision, not one to invent mid-run.

    uv run python examples/dynamical_system_exp3to5.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'examples')
import dynamical_system_exp1 as exp1          # data loading, splits, accuracy_per_step
import dynamical_system_exp2 as exp2          # Experiment 2, for the containment check

OUT = Path('examples/dynamical_system_runs/exp3to5')

DATASET = 'lra_image_mnist'
SEEDS = range(3)
# More seeds where three could not resolve the question.  Two places need it.  Experiment 5
# at mu = 0 is where holding the input turned out to HELP the trajectory losses and hurt the
# terminal ones; Experiment 4 at mu = 0 is the control for that result -- if the split is
# driven by what the LOSS asks for, as claimed, it must appear without sigma too.
RESOLVE = range(10)
STEPS, LR = 2000, 3e-3

# Experiment 3 sweeps patch size at M = 1; Experiments 4 and 5 sweep M at p = 28, where
# N = 28 leaves the most room for inner iterations to matter.
#
# mu is the weight on the inner convergence term, and it is a REAL axis rather than a
# constant, because raising M moves two things at once: the model class shrinks to a
# strict subset (B' = B^M must have an M-th root) AND a penalty switches on that was
# identically zero at M = 1.  The mu = 0 rows hold the second fixed so the first can be
# attributed -- PRINCIPLES Sec. 2.3.
# Seeds are per-configuration for the same reason mu is: the question each configuration
# answers needs a different amount of statistical power, and re-running the whole sweep at
# ten seeds would cost hours to sharpen three cells.
#                 experiment      activation    p    M   mu   seeds
CONFIGURATIONS = ([('Experiment 3', 'tanh',      p,   1,  1.0, SEEDS) for p in (28, 112, 196)]
                  + [('Experiment 4', 'identity', 28, 1,  1.0, RESOLVE)]
                  + [('Experiment 4', 'identity', 28, M,  1.0, SEEDS) for M in (2, 4)]
                  + [('Experiment 4', 'identity', 28, M,  0.0, RESOLVE) for M in (2, 4)]
                  + [('Experiment 5', 'tanh',     28, 1,  1.0, RESOLVE)]
                  + [('Experiment 5', 'tanh',     28, M,  1.0, SEEDS) for M in (2, 4)]
                  + [('Experiment 5', 'tanh',     28, M,  0.0, RESOLVE) for M in (2, 4)]
                  # Experiment 3b: genuinely non-separable mixing.  M=1 is the direct
                  # comparison with Experiment 3; M=2 is where Sec. 7.1 shows a
                  # nonlinearity starts to pay.
                  + [('Experiment 3b', 'tanh_gated', 28, 1, 1.0, RESOLVE)]
                  + [('Experiment 3b', 'tanh_gated', 28, 2, 0.0, RESOLVE)])

# The map's kind, not merely its activation: 'tanh_gated' adds a multiplicative term and is
# Experiment 3b.  Experiment 3's sigma has a SEPARABLE pre-activation -- x and y are added
# then squashed, never multiplied -- so its null could not distinguish "joint mixing does not
# help" from "we never tested joint mixing" (design record Sec. 11.6).  This tests it.
ACTIVATIONS = {'identity': torch.nn.Identity(), 'tanh': torch.nn.Tanh(),
               'tanh_gated': torch.nn.Tanh()}


# =============================================================================
# The map
# =============================================================================

class Recurrence(torch.nn.Module):
    """y_{k,j+1} = sigma(A x_k + B y_{k,j} + b)  [ + (C x_k) * (D y_{k,j}) if gated ],

    with x_k held for M inner steps.  The bracketed term is Experiment 3b: the only form
    here in which x and y actually multiply rather than being added and then squashed.
    """

    def __init__(self, patch_size, num_classes, kind, inner_steps):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(num_classes, patch_size))
        self.B = torch.nn.Parameter(torch.empty(num_classes, num_classes))
        self.b = torch.nn.Parameter(torch.zeros(num_classes))
        self.activation = ACTIVATIONS[kind]
        self.gated = kind.endswith('_gated')
        self.M = inner_steps
        # torch.nn.Linear's own initialisation, so `cold` is what a reader expects.
        torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        torch.nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        if self.gated:
            # C = 0 makes the gate contribute nothing at initialisation, so the model starts
            # out BEING Experiment 3 and contains it.  D must NOT also be zero: the gradient
            # of (Cx)*(Dy) with respect to C is x*(Dy), which vanishes identically at D = 0,
            # so zeroing both would leave the whole term dead with no way to grow.
            self.C = torch.nn.Parameter(torch.zeros(num_classes, patch_size))
            self.D = torch.nn.Parameter(torch.empty(num_classes, num_classes))
            torch.nn.init.kaiming_uniform_(self.D, a=5 ** 0.5)

    def forward(self, Xp):                              # Xp: (n, N, p)
        y = torch.zeros(len(Xp), len(self.b), dtype=Xp.dtype, device=Xp.device)
        reads = []
        for k in range(Xp.shape[1]):
            inner = []
            for _ in range(self.M):                     # x_k is held across this loop
                mixed = self.activation(Xp[:, k, :] @ self.A.T + y @ self.B.T + self.b)
                if self.gated:                          # the only place x and y multiply
                    mixed = mixed + (Xp[:, k, :] @ self.C.T) * (y @ self.D.T)
                y = mixed
                inner.append(y)
            reads.append(torch.stack(inner, dim=1))
        return torch.stack(reads, dim=1)                # (n, N, M, C)


def outer(Y):
    """The state each outer read ends at -- what the fit terms see.  (n,N,M,C) -> (n,N,C)"""
    return Y[:, :, -1, :]


def check_contains_experiment_2():
    """[det]: the identity / M=1 corner IS Experiment 2.  One measurement suffices."""
    torch.manual_seed(0)
    reference = exp2.LinearRecurrence(patch_size=28, num_classes=10)
    model = Recurrence(28, 10, 'identity', 1)
    with torch.no_grad():                               # same weights, not just same shape
        model.A.copy_(reference.A); model.B.copy_(reference.B); model.b.copy_(reference.b)
    Xp = torch.randn(7, 28, 28)
    gap = (outer(model(Xp)) - reference(Xp)).abs().max().item()
    assert gap == 0.0, gap
    print(f'identity/M=1 reproduces Experiment 2 to {gap:.2e}', flush=True)


# =============================================================================
# Losses.  The four fit terms act on the OUTER states; inner steps carry only the
# convergence term, so that M does not change the weight on the fit (design record Sec. 8.1).
# =============================================================================

def terminal(Y, T):
    return ((outer(Y)[:, -1] - T) ** 2).sum(-1).mean()


def sum_over_steps(Y, T):
    return ((outer(Y) - T[:, None]) ** 2).sum(-1).mean()


def weighted_sum_over_steps(Y, T):
    lam = torch.arange(1, Y.shape[1] + 1, dtype=Y.dtype, device=Y.device)
    lam = lam / lam.sum()
    return (((outer(Y) - T[:, None]) ** 2).sum(-1) * lam).sum(1).mean()


def convergence_and_terminal(Y, T, mu=1.0):
    Z = outer(Y)
    return terminal(Y, T) + mu * ((Z[:, 1:] - Z[:, :-1]) ** 2).sum(-1).mean()


LOSSES = {'terminal': terminal,
          'sum': sum_over_steps,
          'weighted sum': weighted_sum_over_steps,
          'convergence+terminal': convergence_and_terminal}


def inner_convergence(Y):
    """mean_{rows,k,j} ||y_{k,j+1} - y_{k,j}||^2 -- identically zero when M = 1.

    This is what makes "settle, then answer" a property we asked for rather than hoped for.
    It is never the whole objective: V alone is minimised by the identity map.
    """
    if Y.shape[2] == 1:
        return Y.new_zeros(())
    return ((Y[:, :, 1:] - Y[:, :, :-1]) ** 2).sum(-1).mean()


# =============================================================================
# Train
# =============================================================================

def train(model, loss, data, mu):
    """Full-batch Adam, keeping the best-on-validation weights."""
    Xp_train, T_train, Xp_val, labels_val = data
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best = {'val_acc': -1.0, 'state': None, 'step': -1}
    history = {'step': [], 'objective': [], 'val_acc': []}
    for step in range(STEPS):
        Y = model(Xp_train)
        objective = loss(Y, T_train) + mu * inner_convergence(Y)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if step % 50 == 0 or step == STEPS - 1:
            with torch.no_grad():
                val_acc = exp1.accuracy_per_step(outer(model(Xp_val)), labels_val)[-1].item()
            history['step'].append(step)
            history['objective'].append(objective.item())
            history['val_acc'].append(val_acc)
            if val_acc > best['val_acc']:
                best = {'val_acc': val_acc, 'step': step,
                        'state': {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    best['history'] = history
    return best


def inner_gap(Y):
    """mean ||y_{k,j+1} - y_{k,j}|| at TEST time -- does the state actually settle?

    Holding x_k for M steps is only worth doing if y goes somewhere while it is held, and
    the design's stated aim is that it *converges*.  The objective contains this quantity
    (squared, as a penalty); this measures it on held-out data, in the norm, so that
    "the state settled" is a reading rather than an assumption.  NaN when M = 1, where
    there is no inner transition to measure.
    """
    if Y.shape[2] == 1:
        return float('nan')
    return (Y[:, :, 1:] - Y[:, :, :-1]).norm(dim=3).mean().item()


def first_to_last_inner_gap(Y):
    """mean ||y_{k,M} - y_{k,1}|| -- how far the state moves across the whole hold.

    Read against `inner_gap`: if the per-step gap is small AND this is small, the state
    was already at rest and the inner steps did nothing.  If the per-step gap is small but
    this is large, it moved steadily rather than settling.
    """
    if Y.shape[2] == 1:
        return float('nan')
    return (Y[:, :, -1] - Y[:, :, 0]).norm(dim=2).mean().item()


def saturation(Y):
    """Fraction of state coordinates with |y| > 0.95 -- how close tanh is to its rails.

    Reported rather than designed around: one-hot targets are deliberately NOT rescaled
    (see the module docstring), so this is the quantity that would show the cost.
    """
    return (Y.abs() > 0.95).double().mean().item()


def main():
    check_contains_experiment_2()
    OUT.mkdir(parents=True, exist_ok=True)

    X, labels = exp1.load_flat(DATASET)
    n, n_features = X.shape
    num_classes = int(labels.max()) + 1
    T = torch.nn.functional.one_hot(labels, num_classes).double()

    # Resume: anything already in results.json is not recomputed.  Records written before
    # mu became an axis were all at mu = 1.0.
    results_path = OUT / 'results.json'
    records = json.loads(results_path.read_text()) if results_path.exists() else []
    for r in records:
        r.setdefault('mu', 1.0)
    already_done = {(r['experiment'], r['p'], r['M'], r['mu'], r['loss'], r['seed'])
                    for r in records}

    for experiment, activation, p, M, mu, seeds in CONFIGURATIONS:
        N = n_features // p
        Xp = X.reshape(n, N, p)

        for loss_name, loss in LOSSES.items():
            todo = [s for s in seeds
                    if (experiment, p, M, mu, loss_name, s) not in already_done]
            if not todo:
                print(f'  skipping {experiment} p={p} M={M} mu={mu:g} {loss_name} '
                      f'-- all {len(list(seeds))} seeds already in results.json', flush=True)
                continue
            for seed in todo:
                train_idx, val_idx, test_idx = exp1.split(n, seed)
                torch.manual_seed(seed)
                model = Recurrence(p, num_classes, activation, M).double()

                data = (Xp[train_idx], T[train_idx], Xp[val_idx], labels[val_idx])
                best = train(model, loss, data, mu)

                with torch.no_grad():
                    Y_test = model(Xp[test_idx])
                    achieved = exp1.accuracy_per_step(outer(Y_test), labels[test_idx])
                    eigenvalues = torch.linalg.eigvals(model.B)
                records.append({
                    'experiment': experiment, 'activation': activation, 'p': p, 'N': N,
                    'M': M, 'mu': mu, 'loss': loss_name, 'seed': seed,
                    'rho_B': eigenvalues.abs().max().item(),
                    'eigenvalues': [[z.real.item(), z.imag.item()] for z in eigenvalues],
                    'val_acc': best['val_acc'], 'test_acc': achieved[-1].item(),
                    'achieved': achieved.tolist(),
                    'saturation': saturation(Y_test),
                    'mean_state_norm': Y_test.norm(dim=3).mean().item(),
                    'inner_gap': inner_gap(Y_test),
                    'first_to_last_inner_gap': first_to_last_inner_gap(Y_test),
                    # The inner gap is ||(B - I) y + A x||, so a model can make it small
                    # two ways: contract (rho << 1) or keep B near I and starve A.  These
                    # two norms separate those routes; rho alone does not.
                    'norm_A': model.A.norm().item(),
                    'norm_B_minus_I': (model.B - torch.eye(num_classes,
                                       dtype=model.B.dtype)).norm().item(),
                    'history': best['history']})
            recent = [r for r in records
                      if (r['experiment'], r['p'], r['M'], r['mu'], r['loss'])
                      == (experiment, p, M, mu, loss_name)]
            print(f'  {experiment}  {activation:8s} p={p:4d} N={N:3d} M={M} mu={mu:g}  '
                  f'{loss_name:22s} n={len(recent):2d} '
                  f'test {np.mean([r["test_acc"] for r in recent]):.4f} '
                  f'+- {np.std([r["test_acc"] for r in recent]):.4f}   '
                  f'rho(B) {np.mean([r["rho_B"] for r in recent]):.3f}   '
                  f'sat {np.mean([r["saturation"] for r in recent]):.3f}', flush=True)
            # Written after every loss, not every configuration: a ten-seed
            # configuration is an hour on its own, and resume is per (config, loss, seed).
            (OUT / 'results.json').write_text(json.dumps(records, indent=1))

    (OUT / 'results.json').write_text(json.dumps(records, indent=1))
    print(f'\nwrote {OUT / "results.json"}  ({len(records)} records)')
    return records


if __name__ == '__main__':
    main()
