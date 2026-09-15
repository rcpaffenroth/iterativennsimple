"""Figures for tasks/OVERVIEW_DYNAMICAL_SYSTEM.md, built from the saved results.

Reads `dynamical_system_runs/exp{1,2}/results.json` and writes into `tasks/figures/`.
Separate from the two experiment scripts so the figures can be redrawn without
refitting anything.

    uv run python examples/dynamical_system_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path('examples/dynamical_system_runs')
FIGURES = Path('tasks/figures')
LOSSES = ['terminal', 'sum', 'weighted sum', 'convergence+terminal']
INNER_STEPS = [1, 2, 4]
COLOURS = dict(zip(LOSSES, ['C0', 'C1', 'C2', 'C3']))


def where(records, **match):
    return [r for r in records if all(r[k] == v for k, v in match.items())]


def mean_of(records, field, **match):
    return np.mean([r[field] for r in where(records, **match)], axis=0)


# =============================================================================

def patch_homotopy(exp1, exp2):
    """Accuracy against patch size: the homotopy from the degenerate floor to full LS.

    At p = 1 Experiment 1 sees one pixel; at p = 784 it is N = 1, i.e. exactly ridge
    regression on the flattened image.  Experiment 2 is drawn where it was run.
    """
    figure, axis = plt.subplots(figsize=(7.5, 5))
    patch_sizes = sorted({r['p'] for r in exp1 if r['dataset'] == 'lra_image_mnist'})

    for loss in LOSSES:
        acc = [mean_of(exp1, 'test_acc', dataset='lra_image_mnist', p=p, loss=loss)
               for p in patch_sizes]
        axis.plot(patch_sizes, acc, ':o', color=COLOURS[loss], alpha=0.55, markersize=4,
                  label=f'Experiment 1  {loss}')

    exp2_patches = sorted({r['p'] for r in exp2})
    for loss in LOSSES:
        acc = [mean_of(exp2, 'test_acc', p=p, loss=loss, arm='cold')
               for p in exp2_patches]
        axis.plot(exp2_patches, acc, '-s', color=COLOURS[loss], markersize=6,
                  label=f'Experiment 2  {loss}')

    anchor = mean_of(exp1, 'test_acc', dataset='lra_image_mnist', p=784, loss='terminal')
    axis.axhline(anchor, color='k', linestyle='-.', linewidth=1,
                 label=f'full-image ridge LS = {anchor:.3f}')
    axis.axhline(0.1, color='grey', linestyle=':', label='chance')
    axis.set_xscale('log', base=2)
    axis.set_xticks(patch_sizes)
    axis.set_xticklabels(patch_sizes)
    axis.set_xlabel('patch size p   (N = 784 / p patches per image)')
    axis.set_ylabel('test accuracy  (argmax y_N)')
    axis.set_title('The patch homotopy: Experiment 1 (dotted) vs Experiment 2 cold (solid)')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2, loc='upper left')
    figure.tight_layout()
    figure.savefig(FIGURES / 'patch_homotopy.png', dpi=130)
    plt.close(figure)


def optimisation_paths(exp2, p=28):
    """The training trajectory: what Adam is actually doing, cold against warm.

    Objectives are NOT comparable across losses -- each is a different functional -- so
    the left panel is on a log scale and read within a colour, not across them.  The
    right panel is validation accuracy, which is comparable.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for loss in LOSSES:
        for arm, style in (('cold', '--'), ('warm', '-')):
            runs = where(exp2, p=p, loss=loss, arm=arm)
            steps = runs[0]['history']['step']
            axes[0].plot(steps, np.mean([r['history']['objective'] for r in runs], axis=0),
                         style, color=COLOURS[loss], label=f'{loss} {arm}')
            axes[1].plot(steps, np.mean([r['history']['val_acc'] for r in runs], axis=0),
                         style, color=COLOURS[loss])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('training objective (log scale)')
    axes[1].set_ylabel('validation accuracy  (argmax y_N)')
    axes[1].axhline(0.1, color='grey', linestyle=':')
    for axis in axes:
        axis.set_xlabel('Adam step')
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(f'Optimisation paths, p={p}  (dashed = cold, solid = warm/frozen A)')
    figure.tight_layout()
    figure.savefig(FIGURES / 'optimisation_paths.png', dpi=130)
    plt.close(figure)


def state_trajectories(exp2, p=28):
    """The state path itself: all ten coordinates of y_k against k, for single images.

    This is the dynamical system as a dynamical system.  The true class coordinate is
    drawn heavy; a perfect run would drive it to 1 and the other nine to 0.  Terminal
    loss and sum loss are asked for different things and visibly do different things.
    """
    shown = ['terminal', 'sum']
    examples = 3
    figure, axes = plt.subplots(len(shown), examples,
                                figsize=(4.4 * examples, 3.6 * len(shown)), squeeze=False)

    for row, loss in enumerate(shown):
        run = where(exp2, p=p, loss=loss, arm='cold', seed=0)[0]
        Y = np.array(run['y_traj'])                     # (rows, N, C)
        labels = run['y_labels']
        for column in range(examples):
            axis = axes[row][column]
            for c in range(Y.shape[2]):
                true = (c == labels[column])
                axis.plot(range(1, Y.shape[1] + 1), Y[column, :, c],
                          linewidth=2.4 if true else 0.9,
                          color='C3' if true else 'grey',
                          alpha=1.0 if true else 0.5,
                          label=f'class {c} (true)' if true else None)
            axis.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
            axis.axhline(0.0, color='k', linestyle=':', linewidth=0.8)
            axis.set_title(f'{loss}  —  true class {labels[column]}', fontsize=10)
            axis.set_xlabel('k  (patches read)')
            axis.set_ylabel('$y_k$ coordinates')
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7, loc='upper left')
    figure.suptitle(f'State trajectories $y_k \\in \\mathbb{{R}}^{{10}}$, p={p}, cold '
                    f'(heavy red = true class coordinate)')
    figure.tight_layout()
    figure.savefig(FIGURES / 'state_trajectories.png', dpi=130)
    plt.close(figure)


def spectrum(exp2):
    """Every eigenvalue of the learned B, against the unit circle.

    B is 10x10, so unlike d_h = 2048 the whole spectrum can simply be looked at.
    Inside the circle the recurrence forgets; outside it grows.
    """
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    theta = np.linspace(0, 2 * np.pi, 400)

    for axis, p in zip(axes, sorted({r['p'] for r in exp2})):
        for loss in LOSSES:
            points = np.array([z for r in where(exp2, p=p, loss=loss)
                               for z in r['eigenvalues']])
            axis.scatter(points[:, 0], points[:, 1], s=18, alpha=0.65,
                         color=COLOURS[loss], label=loss)
        axis.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        axis.axhline(0, color='grey', linewidth=0.5)
        axis.axvline(0, color='grey', linewidth=0.5)
        N = where(exp2, p=p)[0]['N']
        axis.set_title(f'p={p},  N={N} steps')
        axis.set_xlabel('Re')
        axis.set_ylabel('Im')
        axis.set_aspect('equal')
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.suptitle('Spectrum of the learned recurrence B, all seeds and arms')
    figure.tight_layout()
    figure.savefig(FIGURES / 'spectrum.png', dpi=130)
    plt.close(figure)


def achieved_curves(exp1, exp2):
    """achieved(k) for both experiments -- the sign-of-life instrument, side by side."""
    patches = sorted({r['p'] for r in exp2})
    figure, axes = plt.subplots(1, len(patches), figsize=(5.2 * len(patches), 4.3))
    for axis, p in zip(axes, patches):
        for loss in LOSSES:
            null = mean_of(exp1, 'achieved', dataset='lra_image_mnist', p=p, loss=loss)
            axis.plot(range(1, len(null) + 1), null, ':', color=COLOURS[loss], alpha=0.5)
            curve = mean_of(exp2, 'achieved', p=p, loss=loss, arm='cold')
            axis.plot(range(1, len(curve) + 1), curve, '-', color=COLOURS[loss],
                      label=loss)
        axis.axhline(0.1, color='grey', linestyle=':', label='chance')
        axis.set_title(f'p={p}   (dotted = Experiment 1 null)')
        axis.set_xlabel('k  (patches read)')
        axis.set_ylabel('achieved(k)')
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.suptitle('achieved(k): Experiment 2 cold (solid) against the experiment 1 null (dotted)')
    figure.tight_layout()
    figure.savefig(FIGURES / 'achieved_k.png', dpi=130)
    plt.close(figure)


# =============================================================================
# Experiments 3-5: the (activation, inner steps) square
# =============================================================================

def activation_and_inner_steps(exp345):
    """Accuracy against M, identity beside tanh, at p = 28.

    The four corners of the square are Experiment 2 (identity, M=1), Experiment 3 (tanh,
    M=1), Experiment 4 (identity, M>1) and Experiment 5 (tanh, M>1).  Registered
    prediction P4 is that the identity curves never rise above their own M=1 point,
    because Experiment 4's model class is a strict subset of Experiment 2's; a rise there
    is a bug, not a result.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, activation in zip(axes, ('identity', 'tanh')):
        for loss in LOSSES:
            here = [r for r in exp345 if r['activation'] == activation
                    and r['p'] == 28 and r['loss'] == loss]
            # mu = 1 is the sweep; mu = 0 removes the inner convergence penalty, which is
            # what separates "holding the input hurts" from "penalising it hurts".
            for mu, style, alpha in ((1.0, '-o', 1.0), (0.0, '--s', 0.7)):
                runs = [r for r in here if r.get('mu', 1.0) == mu]
                Ms = sorted({r['M'] for r in runs})
                if not Ms:
                    continue
                if mu == 0.0:                       # anchor the dashed line at its own M=1
                    Ms = [1] + Ms
                mean, sd = [], []
                for M in Ms:
                    source = [r for r in here
                              if r['M'] == M and r.get('mu', 1.0) == (1.0 if M == 1 else mu)]
                    mean.append(np.mean([r['test_acc'] for r in source]))
                    sd.append(np.std([r['test_acc'] for r in source]))
                axis.errorbar(Ms, mean, yerr=sd, fmt=style, capsize=3, alpha=alpha,
                              color=COLOURS[loss],
                              label=f'{loss}' + ('' if mu == 1.0 else r'  ($\mu=0$)'))
                if mu == 1.0:
                    axis.axhline(mean[0], color=COLOURS[loss], linestyle=':', alpha=0.35)
        axis.set_xscale('log', base=2)
        axis.set_xticks(INNER_STEPS)
        axis.set_xticklabels(INNER_STEPS)
        axis.set_xlabel('M   (inner steps with $x_k$ held)')
        axis.set_title(f'{activation}   —   '
                       + ('Experiment 2 at M=1, Experiment 4 beyond'
                          if activation == 'identity'
                          else 'Experiment 3 at M=1, Experiment 5 beyond'))
        axis.grid(alpha=0.3)
    axes[0].set_ylabel('test accuracy  (argmax $y_N$)')
    axes[0].legend(fontsize=6, ncol=2)
    figure.suptitle('Holding the input: accuracy against M at p=28.  '
                    r'Solid = with the inner convergence penalty ($\mu=1$), '
                    r'dashed = without it ($\mu=0$)')
    figure.tight_layout()
    figure.savefig(FIGURES / 'activation_and_inner_steps.png', dpi=130)
    plt.close(figure)


def activation_vs_patch(exp2, exp345):
    """Experiment 3 (tanh) against Experiment 2 (identity) across patch size, M = 1.

    Tests P1 (tanh hurts terminal) and P2 (tanh leaves sum loss alone).
    """
    figure, axis = plt.subplots(figsize=(7.5, 5))
    patches = sorted({r['p'] for r in exp345 if r['M'] == 1 and r['activation'] == 'tanh'})
    for loss in LOSSES:
        linear = [np.mean([r['test_acc'] for r in exp2
                           if r['p'] == p and r['loss'] == loss and r['arm'] == 'cold'])
                  for p in patches]
        nonlinear = [np.mean([r['test_acc'] for r in exp345
                              if r['p'] == p and r['loss'] == loss
                              and r['activation'] == 'tanh' and r['M'] == 1])
                     for p in patches]
        axis.plot(patches, linear, '--s', color=COLOURS[loss], alpha=0.55,
                  label=f'Experiment 2  {loss}')
        axis.plot(patches, nonlinear, '-o', color=COLOURS[loss],
                  label=f'Experiment 3  {loss}')
    axis.axhline(0.1, color='grey', linestyle=':', label='chance')
    axis.set_xscale('log', base=2)
    axis.set_xticks(patches); axis.set_xticklabels(patches)
    axis.set_xlabel('patch size p')
    axis.set_ylabel('test accuracy  (argmax $y_N$)')
    axis.set_title('Does $\\sigma$ help? Experiment 3 (solid) vs Experiment 2 (dashed)')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES / 'activation_vs_patch.png', dpi=130)
    plt.close(figure)


def spectrum_and_saturation(exp2, exp345):
    """Left: does tanh let rho(B) escape 1?  Right: is tanh actually saturating?

    P3 says the spectrum escapes, because what pinned the identity runs to |lambda| ~ 1
    was |lambda|^N blowing up, and tanh bounds the state regardless.  P6 says terminal
    loss saturates heavily -- the quantity that decides whether one-hot targets were the
    wrong call (design record Sec. 8.7).
    """
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    theta = np.linspace(0, 2 * np.pi, 400)

    for loss in LOSSES:
        identity = np.array([z for r in exp2 if r['p'] == 28 and r['loss'] == loss
                             for z in r['eigenvalues']])
        tanh = np.array([z for r in exp345 if r['p'] == 28 and r['M'] == 1
                         and r['activation'] == 'tanh' and r['loss'] == loss
                         for z in r['eigenvalues']])
        if len(identity):
            axes[0].scatter(identity[:, 0], identity[:, 1], s=14, alpha=0.35,
                            color=COLOURS[loss], marker='x')
        if len(tanh):
            axes[0].scatter(tanh[:, 0], tanh[:, 1], s=20, alpha=0.75,
                            color=COLOURS[loss], label=loss)
    axes[0].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    axes[0].axhline(0, color='grey', linewidth=0.5); axes[0].axvline(0, color='grey', linewidth=0.5)
    axes[0].set_aspect('equal'); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel('Re'); axes[0].set_ylabel('Im')
    axes[0].set_title('Spectrum of B at p=28  (x = identity, o = tanh)')
    axes[0].legend(fontsize=7)

    width = 0.2
    for i, loss in enumerate(LOSSES):
        runs = [r for r in exp345 if r['activation'] == 'tanh' and r['p'] == 28
                and r['loss'] == loss]
        Ms = sorted({r['M'] for r in runs})
        if not Ms:
            continue
        values = [np.mean([r['saturation'] for r in runs if r['M'] == M]) for M in Ms]
        axes[1].bar(np.arange(len(Ms)) + i * width, values, width,
                    color=COLOURS[loss], label=loss)
        axes[1].set_xticks(np.arange(len(Ms)) + 1.5 * width)
        axes[1].set_xticklabels([f'M={M}' for M in Ms])
    axes[1].axhline(0.3, color='k', linestyle='--', linewidth=1,
                    label='P6 predicted > 0.3')
    axes[1].set_ylabel('fraction of coordinates with $|y| > 0.95$')
    axes[1].set_title('Saturation under tanh, p=28')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(FIGURES / 'spectrum_and_saturation.png', dpi=130)
    plt.close(figure)


def gated_mixing(exp345):
    """Experiment 3b against Experiments 2 and 3: what genuine multiplication buys.

    Left: accuracy.  The trajectory losses gain enormously; the terminal-type losses do not.
    Right: the spectral radius of the learned B.  Terminal loss gains no accuracy but its
    rho falls off the unit circle -- the gate takes over carrying memory, so B no longer has
    to be a near-integrator.  A change of mechanism that the left panel cannot show.
    """
    maps = [('Experiment 4', 'Exp 2: linear', 'C7'),
            ('Experiment 5', r'Exp 3: $\sigma$', 'C0'),
            ('Experiment 3b', 'Exp 3b: gated', 'C3')]
    present = [mp for mp in maps if any(r['experiment'] == mp[0] and r['M'] == 1 for r in exp345)]
    if len(present) < 3:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width, x = 0.26, np.arange(len(LOSSES))

    for i, (exp, label, colour) in enumerate(present):
        acc, rho = [], []
        for loss in LOSSES:
            runs = [r for r in exp345 if r['experiment'] == exp and r['M'] == 1
                    and r['loss'] == loss]
            acc.append(np.mean([r['val_acc'] for r in runs]))
            rho.append(np.mean([r['rho_B'] for r in runs]))
        axes[0].bar(x + i * width, acc, width, color=colour, label=label)
        # NOT a bar: for rho what matters is distance from 1, and a bar from zero makes
        # 0.91 and 1.11 look alike when they mean opposite things.
        axes[1].plot(x + width, rho, 'o', markersize=11, color=colour, label=label,
                     alpha=0.85)

    axes[0].axhline(0.1, color='grey', linestyle=':', label='chance')
    axes[0].set_ylabel('validation accuracy  (argmax $y_N$)')
    axes[0].set_title('What multiplicative mixing buys, by loss  ($M=1$, n=10)')
    axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1, label=r'$|\lambda|=1$')
    axes[1].set_ylabel(r'$\rho(B)$')
    axes[1].set_title('Spectral radius: does $B$ still have to carry the memory?')
    for axis in axes:
        axis.set_xticks(x + width)
        axis.set_xticklabels([l.replace(' ', '\n') for l in LOSSES], fontsize=8)
        axis.grid(alpha=0.3, axis='y')
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES / 'gated_mixing.png', dpi=130)
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    exp1 = json.loads((RUNS / 'exp1' / 'results.json').read_text())
    exp2 = json.loads((RUNS / 'exp2' / 'results.json').read_text())

    patch_homotopy(exp1, exp2)
    optimisation_paths(exp2)
    state_trajectories(exp2)
    spectrum(exp2)
    achieved_curves(exp1, exp2)

    # Experiments 3-5 land configuration by configuration, so this file may be absent or
    # partial while the sweep is still running.
    exp345_path = RUNS / 'exp3to5' / 'results.json'
    if exp345_path.exists():
        exp345 = json.loads(exp345_path.read_text())
        activation_and_inner_steps(exp345)
        activation_vs_patch(exp2, exp345)
        spectrum_and_saturation(exp2, exp345)
        gated_mixing(exp345)

    print(f'wrote {len(list(FIGURES.glob("*.png")))} figures to {FIGURES}')


if __name__ == '__main__':
    main()
