"""Experiment 2 of the dynamical-systems experiments:

    y_{k+1} = A x_k + B y_k + b

The first experiment with a recurrence, so the first experiment where `achieved(k)` can rise --
which is the sign of life (tasks/OVERVIEW_DYNAMICAL_SYSTEM.md Sec. 6).  Experiment 1 supplies
the null: its achieved(k) is a hump tracking where the ink is, returning to chance.

Two things this file is deliberately doing at once.

1. The recurrence is written TWICE -- once directly, once as a `Sequential2DRNN` block
   map -- and `check_two_implementations_agree` asserts they match to 1e-5.  The pair is
   there to be read side by side; the assert makes the comparison a test as well.

2. Every experiment is run COLD and WARM.  Warm means Experiment 1's exact closed-form solution for
   the same loss, same patch size and same split, with A and b FROZEN and B initialised
   at zero -- so the warm model starts life *being* Experiment 1, and anything it gains is
   attributable to B.  Cold trains everything from a random start.  The gap between them
   is what Experiment 6 was going to ask about, obtained here for free.

Unrolling with y_0 = 0 shows what this model class is:

    y_N = sum_k B^{N-1-k} A x_k = [B^{N-1}A  B^{N-2}A  ...  A] vec(image)

i.e. full least squares on the flattened image constrained to companion/Krylov
structure -- Cp + C^2 parameters standing in for a C x Np map.  B is 10x10, so rho(B)
is the whole memory story and is printed for every run.

    uv run python examples/dynamical_system_exp2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'examples')          # Experiment 1 holds the data loading and the exact solve
import dynamical_system_exp1 as exp1

from iterativennsimple.Sequential2D import Identity
from iterativennsimple.Sequential2DRNN import Sequential2DRNN

OUT = Path('examples/dynamical_system_runs/exp2')

DATASET = 'lra_image_mnist'
PATCH_SIZES = [28, 112, 196]    # Experiment 1 is at chance (0.106), 0.264 and 0.454 here
SEEDS = range(3)
STEPS, LR = 2000, 3e-3
TRAJECTORY_PATCH, TRAJECTORY_ROWS = 28, 6   # keep y_k itself only here          # full batch: the split and the init are the only randomness


# =============================================================================
# The recurrence, written directly
# =============================================================================

class LinearRecurrence(torch.nn.Module):
    """y_{k+1} = A x_k + B y_k + b, iterated over the patches of one image."""

    def __init__(self, patch_size, num_classes):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(num_classes, patch_size))
        self.B = torch.nn.Parameter(torch.empty(num_classes, num_classes))
        self.b = torch.nn.Parameter(torch.zeros(num_classes))
        # torch.nn.Linear's own initialisation, so `cold` matches what a reader expects.
        torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        torch.nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)

    def forward(self, Xp):                                  # Xp: (n, N, p)
        y = torch.zeros(len(Xp), len(self.b), dtype=Xp.dtype, device=Xp.device)
        trajectory = []
        for k in range(Xp.shape[1]):
            y = Xp[:, k, :] @ self.A.T + y @ self.B.T + self.b       # the governing equation
            trajectory.append(y)
        return torch.stack(trajectory, dim=1)               # (n, N, C)


# =============================================================================
# The same recurrence, as a Sequential2D block map
# =============================================================================

def as_sequential2d(model):
    """The same dynamics as z_{k+1} = (A . b . M) . Inject_{k+1} . z_k on slots [x, y]:

        M = [[ I   A ]
             [ 0   B ]]

    `Sequential2D.blocks[i][j]` is indexed (input, output) -- the transpose of the block
    matrix as normally written (PRINCIPLES Part 3) -- so A, which maps x to y, sits at
    blocks[0][1].  Weight tensors need no transpose: torch stores Linear.weight as
    (out, in), the mathematics convention.

    M_xx = I holds the injected patch across the step; the y activation is Identity
    because Experiment 2 is linear -- Experiment 3 is the same map with a sigma there.
    """
    num_classes, patch_size = model.A.shape

    W_xy = torch.nn.Linear(patch_size, num_classes, bias=False)
    W_yy = torch.nn.Linear(num_classes, num_classes, bias=False)
    W_xy.weight.data = model.A.detach().clone()
    W_yy.weight.data = model.B.detach().clone()

    return Sequential2DRNN(
        features_list=[patch_size, num_classes],
        blocks=[[Identity(in_features=patch_size, out_features=patch_size), W_xy],
                [None,                                                      W_yy]],
        bias=[None, model.b.detach().clone()],
        activation=[torch.nn.Identity(), torch.nn.Identity()],
        inject_slot=0, hidden_slot=1, output_slot=1, K=1, batch_first=True)


def check_two_implementations_agree():
    """[det]: the two spellings are the same map.  One measurement suffices."""
    torch.manual_seed(0)
    model = LinearRecurrence(patch_size=28, num_classes=10)
    Xp = torch.randn(7, 28, 28)
    direct = model(Xp)
    block_map, _ = as_sequential2d(model)(Xp)
    assert torch.allclose(direct, block_map, atol=1e-5), \
        (direct - block_map).abs().max().item()
    print(f'two implementations agree to '
          f'{(direct - block_map).abs().max().item():.2e}', flush=True)


# =============================================================================
# The four losses, on a whole trajectory
# =============================================================================
#
# Same four as Experiment 1 and written the same way -- means of squared norms -- so the
# relative weight mu = 1 below compares like with like, and so the experiments are comparable.

def terminal(Y, T):
    """mean_rows ||y_N - t||^2"""
    return ((Y[:, -1] - T) ** 2).sum(-1).mean()


def sum_over_steps(Y, T):
    """mean_{rows,k} ||y_k - t||^2"""
    return ((Y - T[:, None]) ** 2).sum(-1).mean()


def weighted_sum_over_steps(Y, T):
    """sum_k lambda_k mean_rows ||y_k - t||^2,  lambda_k proportional to k"""
    lam = torch.arange(1, Y.shape[1] + 1, dtype=Y.dtype, device=Y.device)
    lam = lam / lam.sum()                                   # (N,) convex weights
    return (((Y - T[:, None]) ** 2).sum(-1) * lam).sum(1).mean()


def convergence_and_terminal(Y, T, mu=1.0):
    """mean_rows ||y_N - t||^2  +  mu * mean_{rows,k} ||y_{k+1} - y_k||^2

    V alone is degenerate -- its minimum is the identity map, converge perfectly and
    score zero -- so it is only ever optimised paired with a fit term.
    """
    return terminal(Y, T) + mu * ((Y[:, 1:] - Y[:, :-1]) ** 2).sum(-1).mean()


LOSSES = {'terminal': terminal,
          'sum': sum_over_steps,
          'weighted sum': weighted_sum_over_steps,
          'convergence+terminal': convergence_and_terminal}


# =============================================================================
# Train
# =============================================================================

def train(model, loss, data):
    """Full-batch Adam, keeping the best-on-validation weights.

    Full batch because 800 rows fit trivially and it removes batch order as a source
    of randomness: the split and the initialisation are then the only two.
    """
    Xp_train, T_train, Xp_val, labels_val = data
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR)

    best = {'val_acc': -1.0, 'state': None, 'step': -1}
    history = {'step': [], 'objective': [], 'val_acc': []}
    for step in range(STEPS):
        objective = loss(model(Xp_train), T_train)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if step % 50 == 0 or step == STEPS - 1:
            with torch.no_grad():
                val_acc = exp1.accuracy_per_step(model(Xp_val), labels_val)[-1].item()
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


def main():
    check_two_implementations_agree()
    OUT.mkdir(parents=True, exist_ok=True)

    X, labels = exp1.load_flat(DATASET)
    n, n_features = X.shape
    num_classes = int(labels.max()) + 1
    T = torch.nn.functional.one_hot(labels, num_classes).double()
    records = []

    for p in PATCH_SIZES:
        N = n_features // p
        Xp = X.reshape(n, N, p)

        for loss_name, loss in LOSSES.items():
            for arm in ('cold', 'warm'):
                for seed in SEEDS:
                    train_idx, val_idx, test_idx = exp1.split(n, seed)
                    torch.manual_seed(seed)
                    model = LinearRecurrence(p, num_classes).double()

                    if arm == 'warm':
                        # Experiment 1's exact solution for THIS loss, patch size and split,
                        # frozen.  With B = 0 the model starts out *being* Experiment 1.
                        G, c = exp1.LOSSES[loss_name](Xp[train_idx], T[train_idx])
                        W = exp1.solve(G, c, alpha=_best_alpha(
                            G, c, Xp[val_idx], labels[val_idx]))
                        with torch.no_grad():
                            model.A.copy_(W[:-1].T)
                            model.b.copy_(W[-1])
                            model.B.zero_()
                        model.A.requires_grad_(False)
                        model.b.requires_grad_(False)

                    data = (Xp[train_idx], T[train_idx], Xp[val_idx], labels[val_idx])
                    best = train(model, loss, data)

                    with torch.no_grad():
                        Y_test = model(Xp[test_idx])
                        achieved = exp1.accuracy_per_step(Y_test, labels[test_idx])
                        eigenvalues = torch.linalg.eigvals(model.B)
                    record = {'p': p, 'N': N, 'loss': loss_name, 'arm': arm,
                              'seed': seed, 'rho_B': eigenvalues.abs().max().item(),
                              'eigenvalues': [[z.real.item(), z.imag.item()]
                                              for z in eigenvalues],
                              'val_acc': best['val_acc'],
                              'test_acc': achieved[-1].item(),
                              'achieved': achieved.tolist(),
                              'history': best['history']}
                    # The state path y_k itself, for the longest sequence and one split
                    # only -- enough to look at, small enough to keep in the json.
                    if p == TRAJECTORY_PATCH and seed == 0:
                        record['y_traj'] = Y_test[:TRAJECTORY_ROWS].tolist()
                        record['y_labels'] = labels[test_idx][:TRAJECTORY_ROWS].tolist()
                    records.append(record)
                recent = records[-len(list(SEEDS)):]
                print(f'  p={p:4d} N={N:4d}  {loss_name:22s} {arm:4s}  '
                      f'test {np.mean([r["test_acc"] for r in recent]):.4f} '
                      f'+- {np.std([r["test_acc"] for r in recent]):.4f}   '
                      f'rho(B) {np.mean([r["rho_B"] for r in recent]):.3f}', flush=True)

    (OUT / 'results.json').write_text(json.dumps(records, indent=1))
    print(f'\nwrote {OUT}/results.json  ({len(records)} records)')
    return records



def _best_alpha(G, c, Xp_val, labels_val):
    """Experiment 1's ridge choice, on the same criterion Experiment 1 used."""
    best_acc, best_alpha = -1.0, None
    for alpha in exp1.ALPHAS:
        W = exp1.solve(G, c, alpha)
        acc = exp1.accuracy_per_step(exp1.trajectory(Xp_val, W), labels_val)[-1].item()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    return best_alpha


if __name__ == '__main__':
    main()
