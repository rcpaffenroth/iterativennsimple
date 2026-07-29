"""Compare Sequential2DRNN against torch.nn.RNN / LSTM / GRU on Long Range Arena tasks.

Usage:

    uv run python examples/lra_benchmark.py examples/lra_runs/image_smoke

The directory must contain `config.yaml`.  Results are written back into the same
directory as `results.md` and `curves.png`, so a config and its output stay paired.

Every model is the same three pieces -- optional embedding, recurrent core,
linear head on the final hidden state -- with *only the core* differing.  That is
the point: any difference in the table is attributable to the recurrence and not
to the input or output plumbing.

A warning about wall-clock.  `torch.nn.RNN` and friends call fused cuDNN kernels
that process a whole sequence in one launch; Sequential2DRNN steps a Python loop
with a few launches per timestep, because the block map is meant to be
*inspectable* rather than fast.  The resulting gap is a *small-hidden* artefact and
shrinks fast, because our loop is launch-bound while cuDNN's advantage is
amortising launches.  Measured on an RTX 4090, seq_len=1024, batch 64:

    d_h     ours    cuDNN    gap
    128    162 ms   1.2 ms    94x
    512    164 ms  21.7 ms   7.5x
   2048    160 ms  77.7 ms   2.1x

`step_size` trades sequence length for input width and is the main cost lever, but
it is NOT a free speed dial: it sets both `seq_len = x_y_index // step_size` and
`input_size = step_size`, so two runs at different values are different tasks.
Never compare across `step_size` values.  Times are reported in the output table so
the cost is never invisible.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                      # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from generatedata.load_data import load_data_as_sequence

from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential2D import Identity
from iterativennsimple.Sequential2DRNN import Sequential2DRNN


# =============================================================================
# Data
# =============================================================================

def load_task(cfg):
    """Load one LRA dataset and split it into train / validation / test.

    `generatedata` ships each dataset as a single block of `num_points` rows with
    no split of its own, so we make one here.  The split is a fixed permutation
    under the configured seed, so every model in a run sees exactly the same data.

    `step_size` is how many raw features become one timestep:
    `seq_len = x_y_index // step_size`.  At step_size 1 each timestep is a single
    scalar -- the LRA definition -- and the sequence is as long as it gets.  Larger
    values patchify, trading sequence length for input width, which is the only
    practical lever on run time here.
    """
    name, step_size = cfg['name'], cfg['step_size']

    # An embedding consumes one integer per timestep, so patchifying and embedding
    # are mutually exclusive.  Fail here rather than inside Embedding's index check.
    assert not (cfg.get('embedding') and step_size != 1), (
        f'{name}: embedding needs one token per timestep, so step_size must be 1, '
        f'not {step_size}.  Patchifying only makes sense for continuous inputs.')

    X, Y = load_data_as_sequence(name, step_size=step_size, label_every_step=False)

    X = torch.from_numpy(np.asarray(X, dtype=np.float32))   # (N, seq_len, step_size)
    labels = torch.from_numpy(np.asarray(Y)).argmax(dim=1)  # (N,) class indices

    # Subsampling rows keeps the task intact and only costs statistical power,
    # which is the right way to make a smoke test out of an expensive dataset.
    # (Truncating the *sequence* would be cheaper still, but on ListOps it would
    # silently change the labels, so it is deliberately not offered.)
    if cfg.get('max_points'):
        keep = torch.randperm(
            len(X), generator=torch.Generator().manual_seed(0))[:cfg['max_points']]
        X, labels = X[keep], labels[keep]

    generator = torch.Generator().manual_seed(cfg.get('split_seed', 0))
    order = torch.randperm(len(X), generator=generator)
    n_train = int(cfg['train_frac'] * len(X))
    n_val = int(cfg['val_frac'] * len(X))

    splits = {'train': order[:n_train],
              'val': order[n_train:n_train + n_val],
              'test': order[n_train + n_val:]}
    return {split: (X[idx], labels[idx]) for split, idx in splits.items()}, X.shape[1:]


def batches(X, y, batch_size, shuffle, device, generator=None):
    """Iterate minibatches.  The whole dataset lives on the CPU and moves per batch.

    No DataLoader: the data is already a tensor in memory, so workers and
    collation would add machinery without buying anything.
    """
    order = (torch.randperm(len(X), generator=generator) if shuffle
             else torch.arange(len(X)))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx].to(device), y[idx].to(device)


# =============================================================================
# Models -- identical except for the core
# =============================================================================

class SequenceClassifier(torch.nn.Module):
    """embedding (optional) -> recurrent core -> linear head on the final state.

    Classification reads only `h_n`, the hidden state after the last token.  Every
    core returns it with a leading dimension of 1 (`num_layers * num_directions`
    for the torch modules, always 1 for ours), and LSTM returns `(h_n, c_n)`, so
    the two lines in `forward` below are all the reconciliation needed.
    """

    def __init__(self, core, hidden_size, num_classes, embedding=None):
        super().__init__()
        self.embedding = embedding
        self.core = core
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):                       # x: (batch, seq_len, step_size)
        if self.embedding is not None:
            # Token tasks arrive as integer-valued floats with step_size == 1.
            x = self.embedding(x.squeeze(-1).long())        # (batch, seq, embed)
        _, h_n = self.core(x)
        if isinstance(h_n, tuple):
            h_n = h_n[0]                        # LSTM hands back (h_n, c_n)
        return self.head(h_n[-1])               # (batch, num_classes)


class MonarchNoViews(torch.nn.Module):
    """MonarchLinear on its faster code path.

    `MonarchLinear.forward` takes `use_views`, and `False` is 1.8-2.2x faster here
    (at d_h=2048, batch 64: nb=16 costs 1.67 ms with views, 0.77 ms without).
    `Sequential2D` calls `block.forward(x)` with no keyword arguments, so the only
    way to reach it is to wrap.

    Carries `in_features` / `out_features` to satisfy the block contract, and
    `bias = None` because bias belongs to the slot (Sec. 8.2).  Deliberately has no
    `.weight`, so `_check_block`'s shape test skips it -- Monarch keeps factors, not
    a matrix.
    """

    def __init__(self, monarch):
        super().__init__()
        self.monarch = monarch
        self.in_features = monarch.in_features
        self.out_features = monarch.out_features
        self.bias = None

    def forward(self, x):
        return self.monarch(x, use_views=False)


def make_block(kind, in_features, out_features, spec):
    """One block of the Sequential2DRNN map.  Bias-free: bias belongs to the slot."""
    if kind == 'linear':
        return torch.nn.Linear(in_features, out_features, bias=False)
    if kind == 'monarch':
        return MonarchNoViews(MonarchLinear.from_uniform_blocks(
            in_features, out_features, num_blocks=spec.get('num_blocks', 4),
            bias=False, seed=spec.get('seed', 0)))
    if kind == 'masked':
        return MaskedLinear(in_features, out_features, bias=False)
    raise ValueError(f'unknown block type {kind!r}')


def build_model(spec, input_size, num_classes, seed):
    """Build one row of the comparison from its config entry."""
    torch.manual_seed(seed)                     # same init draw for every model

    embedding = None
    core_input = input_size
    if spec.get('embedding'):
        embedding = torch.nn.Embedding(spec['embedding']['vocab_size'],
                                       spec['embedding']['dim'])
        core_input = spec['embedding']['dim']

    hidden_size = spec['hidden_size']
    kind = spec['type']

    if kind in ('rnn', 'lstm', 'gru'):
        cls = {'rnn': torch.nn.RNN, 'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU}[kind]
        kwargs = dict(input_size=core_input, hidden_size=hidden_size,
                      num_layers=1, batch_first=True)
        if kind == 'rnn':
            kwargs['nonlinearity'] = spec.get('nonlinearity', 'tanh')
        core = cls(**kwargs)

    elif kind == 'sequential2d':
        W_xh = make_block(spec.get('W_xh', 'linear'), core_input, hidden_size, spec)
        W_hh = make_block(spec.get('W_hh', 'linear'), hidden_size, hidden_size, spec)

        # Orthogonal init of W_hh.  The motivating argument -- that it offsets the
        # contraction tanh' < 1 introduces at every internal iteration -- comes from
        # examples/rnn_internal_iterations.py, which measured it at seq_len 20 and
        # *at initialisation*.  It has since failed to transfer twice at seq_len
        # 1024, at gain 1.2 and at gain 1.0 (see the image_full and image_wide
        # tables).  Treat it as an untested hypothesis, not a fix.
        if spec.get('orthogonal_hh'):
            # Assert rather than skip: MonarchNoViews and nested Sequential2D have
            # no `.weight`, and silently ignoring the request would report a row as
            # orthogonally initialised when it was not.
            assert hasattr(W_hh, 'weight'), (
                f'orthogonal_hh was requested but W_hh is a '
                f'{type(W_hh).__name__}, which has no .weight to orthogonalise. '
                f'Orthogonal init is only meaningful for a dense W_hh.')
            torch.nn.init.orthogonal_(W_hh.weight, gain=spec.get('gain', 1.2))

        activation = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}[
            spec.get('nonlinearity', 'tanh')]()

        # No y-slot: classification reads h_n, and the head is the observation map.
        # blocks[i][j] maps slot i to slot j; slots are x = 0, h = 1.  M_xx = I
        # holds the injected token across all K internal iterations.
        core = Sequential2DRNN(
            features_list=[core_input, hidden_size],
            blocks=[[Identity(in_features=core_input, out_features=core_input), W_xh],
                    [None,                                                      W_hh]],
            bias=[None, torch.zeros(hidden_size)],
            activation=[torch.nn.Identity(), activation],
            inject_slot=0, hidden_slot=1, output_slot=1,
            K=spec.get('K', 1), batch_first=True)
    else:
        raise ValueError(f'unknown model type {kind!r}')

    return SequenceClassifier(core, hidden_size, num_classes, embedding)


# =============================================================================
# Training
# =============================================================================

def evaluate(model, X, y, batch_size, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in batches(X, y, batch_size, shuffle=False, device=device):
            logits = model(xb)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, yb, reduction='sum').item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def train(model, data, train_cfg, device, lr=None):
    """Train one model, returning its per-epoch history and timings.

    `lr` overrides `train_cfg['lr']` so a config can cross width with learning
    rate.  Necessary because a single lr across widths confounds the comparison:
    the same step size moves further in function space for a wider model.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr if lr is not None else train_cfg['lr'])
    generator = torch.Generator().manual_seed(train_cfg['seed'])

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best = {'val_acc': -1.0, 'epoch': -1, 'state': None}
    diverged_at = None
    started = time.time()

    for epoch in range(train_cfg['epochs']):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in batches(*data['train'], train_cfg['batch_size'],
                              shuffle=True, device=device, generator=generator):
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            # Recurrent models on long sequences diverge without this.
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            optimizer.step()

            running += loss.item() * len(yb)
            seen += len(yb)

        val_loss, val_acc = evaluate(model, *data['val'],
                                     train_cfg['batch_size'], device)
        history['train_loss'].append(running / seen)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Keep the best-on-validation weights, and report test accuracy from
        # those rather than from the last epoch.  Only ever from a finite epoch:
        # a diverged run must not be able to contribute a "best" result.
        finite = (np.isfinite(history['train_loss'][-1])
                  and np.isfinite(val_loss))
        if finite and val_acc > best['val_acc']:
            best = {'val_acc': val_acc, 'epoch': epoch,
                    'state': {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}

        print(f'    epoch {epoch + 1:3d}/{train_cfg["epochs"]}  '
              f'train {history["train_loss"][-1]:.4f}  '
              f'val {val_loss:.4f}  val acc {val_acc:.4f}', flush=True)

        # Once a single NaN appears anywhere it poisons every parameter through
        # the optimiser, so the remaining epochs are guaranteed wasted.  Stop, and
        # record it -- a diverged run reported as its best pre-divergence epoch
        # would look like a legitimate (merely bad) result, which is worse than a
        # visible failure.
        if not finite:
            diverged_at = epoch + 1
            print(f'    DIVERGED at epoch {diverged_at}; stopping this model',
                  flush=True)
            break

    train_seconds = time.time() - started

    if best['state'] is None:
        # Never had a finite epoch, so there is nothing meaningful to test.
        test_loss, test_acc = float('nan'), float('nan')
    else:
        model.load_state_dict(best['state'])
        test_loss, test_acc = evaluate(model, *data['test'],
                                       train_cfg['batch_size'], device)

    return {'history': history, 'train_seconds': train_seconds,
            'best_val_acc': best['val_acc'], 'best_epoch': best['epoch'] + 1,
            'test_acc': test_acc, 'test_loss': test_loss,
            'diverged_at': diverged_at,
            'parameters': sum(p.numel() for p in model.parameters()
                              if p.requires_grad)}


# =============================================================================
# Reporting
# =============================================================================

def write_report(directory, cfg, shape, results, num_classes):
    seq_len, step_size = int(shape[0]), int(shape[1])
    chance = 1.0 / num_classes

    lines = [f'# {cfg["dataset"]["name"]} — recurrent core comparison', '',
             f'Generated by `examples/lra_benchmark.py` from `{directory.name}/config.yaml`.', '',
             '## Setup', '',
             '| | |', '| --- | --- |',
             f'| dataset | `{cfg["dataset"]["name"]}` |',
             f'| sequence length | {seq_len} |',
             f'| features per timestep (`step_size`) | {step_size} |',
             f'| classes | {num_classes} (chance = {chance:.3f}) |',
             f'| train / val / test | {" / ".join(str(len(results["_splits"][s])) for s in ("train", "val", "test"))} |',
             f'| epochs | {cfg["training"]["epochs"]} |',
             f'| batch size | {cfg["training"]["batch_size"]} |',
             f'| learning rate | {cfg["training"]["lr"]} (per-model overrides shown in the table) |',
             f'| gradient clip | {cfg["training"]["grad_clip"]} |',
             f'| device | {cfg["training"]["device"]} |',
             '',
             '## Results', '',
             'Test accuracy is from the epoch with the best validation accuracy.',
             f'One seed. Validation and test splits are '
             f'{len(results["_splits"]["val"])} and {len(results["_splits"]["test"])} '
             f'rows, so the standard error on any accuracy near {chance * 2:.2f} is '
             f'about {(chance * 2 * (1 - chance * 2) / len(results["_splits"]["val"])) ** 0.5:.3f}.',
             '',
             '| model | params | lr | best val acc | test acc | epoch | train time | s / epoch |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']

    for name, r in results['models'].items():
        epochs_run = len(r['history']['train_loss'])
        seconds_per_epoch = r['train_seconds'] / max(epochs_run, 1)
        mark = f' DIVERGED@{r["diverged_at"]}' if r.get('diverged_at') else ''
        lines.append(
            f'| {name}{mark} | {r["parameters"]:,} | {r.get("lr", cfg["training"]["lr"]):g} | '
            f'{r["best_val_acc"]:.4f} | '
            f'{r["test_acc"]:.4f} | {r["best_epoch"]} | '
            f'{r["train_seconds"]:.0f} s | {seconds_per_epoch:.1f} |')

    lines += ['',
              '## Per-epoch curves',
              '',
              '![curves](curves.png)',
              '',
              '## Config', '',
              '```yaml',
              yaml.safe_dump(cfg, sort_keys=False).rstrip(),
              '```']

    (directory / 'results.md').write_text('\n'.join(lines) + '\n')


def write_curves(directory, cfg, results, num_classes):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for name, r in results['models'].items():
        epochs = range(1, len(r['history']['train_loss']) + 1)
        axes[0].plot(epochs, r['history']['train_loss'], label=name)
        axes[1].plot(epochs, r['history']['val_loss'], label=name)
        axes[2].plot(epochs, r['history']['val_acc'], label=name)

    axes[2].axhline(1.0 / num_classes, color='grey', linestyle=':', label='chance')

    for axis, title, ylabel in zip(
            axes,
            ['training loss', 'validation loss', 'validation accuracy'],
            ['cross-entropy', 'cross-entropy', 'accuracy']):
        axis.set_title(title)
        axis.set_xlabel('epoch')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    figure.suptitle(f'{cfg["dataset"]["name"]} '
                    f'(step_size={cfg["dataset"]["step_size"]})')
    figure.tight_layout()
    figure.savefig(directory / 'curves.png', dpi=130)
    plt.close(figure)


# =============================================================================
# Entry point
# =============================================================================

def main(directory):
    directory = Path(directory)
    cfg = yaml.safe_load((directory / 'config.yaml').read_text())

    device = cfg['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('cuda requested but unavailable; falling back to cpu')
        device = cfg['training']['device'] = 'cpu'

    print(f'loading {cfg["dataset"]["name"]} ...', flush=True)
    data, shape = load_task(cfg['dataset'])
    seq_len, step_size = int(shape[0]), int(shape[1])
    num_classes = int(data['train'][1].max().item()) + 1
    print(f'  seq_len={seq_len} step_size={step_size} classes={num_classes} '
          f'train/val/test={len(data["train"][0])}/{len(data["val"][0])}/'
          f'{len(data["test"][0])}', flush=True)

    results = {'models': {},
               '_splits': {s: data[s][1] for s in ('train', 'val', 'test')}}

    for spec in cfg['models']:
        # A per-model embedding config falls back to the dataset-level one, since
        # whether tokens need embedding is a property of the task.
        spec = {**spec}
        spec.setdefault('embedding', cfg['dataset'].get('embedding'))

        lr = spec.get('lr', cfg['training']['lr'])
        print(f'\n  {spec["name"]}  (lr={lr:g})', flush=True)
        model = build_model(spec, step_size, num_classes, cfg['training']['seed'])
        results['models'][spec['name']] = train(model, data, cfg['training'],
                                                device, lr=lr)
        results['models'][spec['name']]['lr'] = lr

    write_curves(directory, cfg, results, num_classes)
    write_report(directory, cfg, shape, results, num_classes)

    # Machine-readable alongside the markdown, for replotting without retraining.
    serialisable = {name: {k: v for k, v in r.items()}
                    for name, r in results['models'].items()}
    (directory / 'results.json').write_text(json.dumps(serialisable, indent=2))

    print(f'\nwrote {directory / "results.md"}, {directory / "curves.png"}, '
          f'{directory / "results.json"}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'usage: {sys.argv[0]} <directory containing config.yaml>')
    main(sys.argv[1])
