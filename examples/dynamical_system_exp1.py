"""Experiment 1 of the dynamical-systems experiments, solved exactly:

    y_{k+1} = A x_k + b

x_k is patch k of one image, y_k is the running guess in R^C, and y_k *is* the
prediction -- there is no readout head.  Design record: tasks/OVERVIEW_DYNAMICAL_SYSTEM.md.

Experiment 1 has no recurrence, so y_{k+1} depends on x_k alone and the model cannot
accumulate anything across patches.  That is the point: it is the null against which
Experiment 2's `achieved(k)` curve is read.  Two predictions worth checking against the
output, both made before running:

  * lra_image_mnist at p=1  ~ chance.  y_N sees the last pixel only, which in raster
    order is background.  POSITIVE control: lra_toy_bw at p=1 should NOT be at chance,
    because one pixel already separates a dark image from a bright one.
  * lra_image_mnist at p=784 is N=1, i.e. exactly ridge regression on the flattened
    image -- the classical anchor.  With 800 training rows against 7840 parameters it
    is underdetermined, so that end is ridge-dominated (OVERVIEW Sec. 5.4).

All four losses are closed form here (OVERVIEW Sec. 3.2), so this file contains no
optimiser and no training loop, and every number in it is exact given the split.

    uv run python examples/dynamical_system_exp1.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from generatedata.load_data import load_data_as_sequence

DATA_DIR = '../generatedata/data/processed'
OUT = Path('examples/dynamical_system_runs/exp1')

# Divisors only: load_data_as_sequence raises otherwise (load_data.py:297).
PATCH_SIZES = {'lra_toy_bw':      [1, 2, 4, 8, 16, 32, 64],
               'lra_image_mnist': [1, 4, 16, 28, 56, 112, 196, 392, 784]}

ALPHAS = torch.logspace(-8, 4, 13, dtype=torch.float64)   # ridge grid, tuned on val
SPLIT_SEEDS = range(5)      # closed form, so the split is the ONLY source of variance


# =============================================================================
# The four losses, each as one set of normal equations
# =============================================================================
#
# Every loss below is quadratic in W = [A^T ; b], so each reduces to a Gram matrix G
# and a cross-term c with  G W = c.  Returning that pair rather than a design matrix
# keeps each loss's own mathematics on its own page.  All are written as MEANS, so the
# relative weight mu = 1 in `convergence_and_terminal` compares like with like.

def design(x):
    """[x, 1] -- the ones column is what makes A and b a single matrix W."""
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)


def terminal(Xp, T):
    """L = mean_rows ||y_N - t||^2.   Only the last patch is ever looked at."""
    Phi = design(Xp[:, -1, :])                              # (n, p+1)
    return Phi.T @ Phi / len(Phi), Phi.T @ T / len(Phi)


def sum_over_steps(Xp, T):
    """L = mean_{rows,k} ||y_k - t||^2.   Every (patch, label) pair, pooled."""
    n, N, p = Xp.shape
    Phi = design(Xp.reshape(n * N, p))                      # (nN, p+1)
    T_rep = T.repeat_interleave(N, dim=0)                   # (nN, C)
    return Phi.T @ Phi / len(Phi), Phi.T @ T_rep / len(Phi)


def weighted_sum_over_steps(Xp, T):
    """L = sum_k lambda_k mean_rows ||y_k - t||^2   with lambda_k proportional to k.

    After k patches the model has seen kp of the Np pixels, so the weight tracks the
    fraction of the evidence that is actually available to it (OVERVIEW Sec. 2).
    """
    n, N, p = Xp.shape
    lam = torch.arange(1, N + 1, dtype=Xp.dtype)
    lam = lam / lam.sum()                                   # (N,) convex weights
    Phi = design(Xp.reshape(n * N, p))                      # (nN, p+1)
    T_rep = T.repeat_interleave(N, dim=0)                   # (nN, C)
    w = lam.repeat(n)[:, None]                              # (nN, 1), matches reshape order
    return (Phi * w).T @ Phi / n, (Phi * w).T @ T_rep / n


def convergence_and_terminal(Xp, T, mu=1.0):
    """L = mean_rows ||y_N - t||^2  +  mu * mean_{rows,k} ||y_{k+1} - y_k||^2.

    y_{k+1} - y_k = A(x_k - x_{k-1}), so b cancels and the convergence term is
    quadratic in A alone:

        mean_k ||A (x_k - x_{k-1})||^2  =  tr(A C_delta A^T)

    i.e. generalised Tikhonov whose metric is the patch-difference covariance.  In the
    linear case "convergence loss" IS smoothness regularisation, exactly -- which is
    the mechanism behind  max_k ||y_k - t||  <=  T + V  (OVERVIEW Sec. 3).
    """
    G, c = terminal(Xp, T)
    D = (Xp[:, 1:, :] - Xp[:, :-1, :]).reshape(-1, Xp.shape[2])     # (n(N-1), p)
    penalty = torch.zeros_like(G)
    if len(D):                       # N == 1 (whole image as one patch) has no differences
        penalty[:-1, :-1] = mu * (D.T @ D) / len(D)                 # bias column unpenalised
    return G + penalty, c


LOSSES = {'terminal': terminal,
          'sum': sum_over_steps,
          'weighted sum': weighted_sum_over_steps,
          'convergence+terminal': convergence_and_terminal}


# =============================================================================
# Solve, and evaluate the trajectory
# =============================================================================

def solve(G, c, alpha):
    """W = (G + alpha I)^{-1} c, with the bias row left unshrunk.

    Naive form is W = G^{-1} c.  alpha is not a nicety here: MNIST's border pixels are
    identically zero, so G is singular by construction and the naive form does not
    exist.  Everything is float64 because these Gram matrices are badly conditioned.
    """
    ridge = alpha * torch.eye(len(G), dtype=G.dtype)
    ridge[-1, -1] = 0.0
    return torch.linalg.solve(G + ridge, c)                 # (p+1, C)


def trajectory(Xp, W):
    """The whole trajectory y_1 .. y_N at once:  y_{k+1} = A x_k + b.

    Returns (n, N, C); entry [i, k] is row i's guess after reading k+1 patches.
    """
    A, b = W[:-1], W[-1]                                    # (p, C), (C,)
    return Xp @ A + b


def accuracy_per_step(Y, labels):
    """achieved(k) -- the sign of life.  (n, N, C) -> (N,)"""
    return (Y.argmax(dim=2) == labels[:, None]).double().mean(dim=0)


# =============================================================================
# Data
# =============================================================================

def load_flat(name):
    """(n, features) float64 and integer labels.  Loaded once; patched by reshaping."""
    X, Y = load_data_as_sequence(name, step_size=1, label_every_step=False,
                                 local=True, data_dir=DATA_DIR)
    X = torch.from_numpy(np.asarray(X, dtype=np.float64)).squeeze(-1)   # (n, features)
    labels = torch.from_numpy(np.asarray(Y)).argmax(dim=1)              # (n,)
    return X, labels


def split(n, seed):
    """The lra_benchmark convention: 0.8 / 0.1 / 0.1 under a fixed permutation."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


# =============================================================================
# Sweep
# =============================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    for name, patch_sizes in PATCH_SIZES.items():
        X, labels = load_flat(name)
        n, n_features = X.shape
        num_classes = int(labels.max()) + 1
        T = torch.nn.functional.one_hot(labels, num_classes).double()
        print(f'\n{name}: {n} rows, {n_features} features, {num_classes} classes '
              f'(chance {1 / num_classes:.3f})', flush=True)

        for p in patch_sizes:
            N = n_features // p
            Xp = X.reshape(n, N, p)                         # (n, N, p) raster order

            for loss_name, loss in LOSSES.items():
                for seed in SPLIT_SEEDS:
                    train, val, test = split(n, seed)
                    G, c = loss(Xp[train], T[train])

                    # alpha is chosen on TERMINAL validation accuracy for every loss
                    # alike, so the selection criterion is not an axis that moves with
                    # the loss (PRINCIPLES Sec. 2.3).
                    val_acc, alpha, W = -1.0, None, None
                    for candidate_alpha in ALPHAS:
                        candidate_W = solve(G, c, candidate_alpha)
                        Y_val = trajectory(Xp[val], candidate_W)
                        candidate_acc = accuracy_per_step(Y_val, labels[val])[-1].item()
                        if candidate_acc > val_acc:
                            val_acc, alpha, W = candidate_acc, candidate_alpha.item(), candidate_W

                    achieved = accuracy_per_step(trajectory(Xp[test], W), labels[test])
                    records.append({'dataset': name, 'p': p, 'N': N, 'loss': loss_name,
                                    'seed': seed, 'alpha': alpha, 'val_acc': val_acc,
                                    'test_acc': achieved[-1].item(),
                                    'achieved': achieved.tolist()})
                print(f'  p={p:4d} N={N:4d}  {loss_name:22s} '
                      f'test {np.mean([r["test_acc"] for r in records[-len(SPLIT_SEEDS):]]):.4f}'
                      f' +- {np.std([r["test_acc"] for r in records[-len(SPLIT_SEEDS):]]):.4f}',
                      flush=True)

    (OUT / 'results.json').write_text(json.dumps(records, indent=1))
    print(f'\nwrote {OUT / "results.json"}  ({len(records)} records). '
          f'Figures: examples/dynamical_system_figures.py')
    return records



if __name__ == '__main__':
    main()
