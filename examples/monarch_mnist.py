#!/usr/bin/env python
import time

import click
import numpy as np
import torch
import torch.nn as nn

from generatedata.load_data import data_names, load_data_as_sequence, load_data

from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "quick": {
        "h_sizes":     [64],
        "num_blocks":  4,
        "iterations":  3,
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "lr":          1e-3,
        "epochs":      10,
        "batch_size":  256,
    },
    "small": {
        "h_sizes":     [128, 128],
        "num_blocks":  4,
        "iterations":  4,
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "lr":          1e-3,
        "epochs":      20,
        "batch_size":  256,
    },
    "medium": {
        "h_sizes":     [8192, 8192],
        "num_blocks":  8,
        "iterations":  5,
        "lstm_hidden": 0,
        "lstm_layers": 0,
        "lr":          5e-4,
        "epochs":      40,
        "batch_size":  32,
    },
    "large": {
        "h_sizes":     [512, 512, 512],
        "num_blocks":  8,
        "iterations":  6,
        "lstm_hidden": 0,
        "lstm_layers": 0,
        "lr":          2e-4,
        "epochs":      60,
        "batch_size":  256,
    },
}


# ---------------------------------------------------------------------------
# Data loading  (delegates to generatedata)
# ---------------------------------------------------------------------------

def load_split(
    name: str,
    step_size: int,
    local: bool,
    label_every_step: bool,
    val_fraction: float = 1 / 7,
    seed: int = 42,
):
    """
    Load a generatedata dataset as a pixel-chunk sequence and split train/val.

    Returns
    -------
    X_train, y_train, X_val, y_val : tensors
    label_dim                       : number of output classes
    input_dim                       : features per time-step
                                      (= step_size + label_dim if label_every_step)
    """
    X_seq, labels = load_data_as_sequence(
        name,
        step_size=step_size,
        local=local,
        label_every_step=label_every_step,
    )
    # X_seq  : (N, seq_len, step_size [+ label_dim])  numpy float
    # labels : (N, label_dim)                          numpy float one-hot
    X_seq  = torch.from_numpy(X_seq.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.float32))

    label_dim = labels.shape[1]
    input_dim = X_seq.shape[2]       # step_size or step_size + label_dim
    y_cls     = labels.argmax(dim=1) # (N,) integer class indices

    N   = len(X_seq)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return (
        X_seq[train_idx], y_cls[train_idx],
        X_seq[val_idx],   y_cls[val_idx],
        label_dim, input_dim,
    )


def supervised_step_set(seq_len: int, supervise_every: int) -> set:
    """
    Return the set of time-step indices at which to compute a loss.

    supervise_every == 0  ->  last step only
    supervise_every == N  ->  every Nth step (last step always included)
    """
    if supervise_every == 0:
        return {seq_len - 1}
    steps = set(range(supervise_every - 1, seq_len, supervise_every))
    steps.add(seq_len - 1)
    return steps


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _make_monarch_block(in_f, out_f, num_blocks, activation=None):
    nb = num_blocks
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    layer = MonarchLinear.from_uniform_blocks(
        in_features=in_f, out_features=out_f, num_blocks=nb, bias=True)
    act_map = {"ReLU": nn.ReLU, "ELU": nn.ELU, "Tanh": nn.Tanh}
    seq = nn.Sequential(act_map[activation](), layer) if activation in act_map \
          else nn.Sequential(layer)
    return Sequential1D(seq, in_features=in_f, out_features=out_f)


def _build_seq2d(input_dim, h_sizes, label_dim, num_blocks, activation="ReLU"):
    """
    Build the Sequential2D map.

    State: [x(input_dim) | h1 | ... | hk | y(label_dim)]

    blocks[0][0]   = Identity  -- x frozen during inner iterations
    blocks[i][i+1] = MonarchLinear  -- feed-forward chain
    """
    sizes  = [input_dim] + list(h_sizes) + [label_dim]
    n      = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)
    for i in range(n - 1):
        act = None if i == 0 else activation
        blocks[i][i + 1] = _make_monarch_block(sizes[i], sizes[i + 1], num_blocks, act)
    return Sequential2D(sizes, sizes, blocks)


class MonarchSequenceClassifier(nn.Module):
    """
    Monarch INN used as a recurrent sequence classifier.

    The sequence is processed one chunk at a time.  At each step:
      - Overwrite the x-slot of the state with the current input chunk.
      - Apply the Monarch map `iterations` times.
    A CrossEntropy loss is computed at each step in `sup_steps` and summed.
    """

    def __init__(self, input_dim, h_sizes, label_dim, num_blocks, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.h_sizes   = list(h_sizes)
        self.label_dim = label_dim
        self.seq2d     = _build_seq2d(input_dim, h_sizes, label_dim, num_blocks, activation)

    def forward(self, x_seq, iterations, sup_steps, criterion, targets):
        """
        Parameters
        ----------
        x_seq      : (B, T, input_dim)
        iterations : Monarch map applications per time-step
        sup_steps  : set of time-step indices at which to compute loss
        criterion  : loss function  (logits, class_indices)
        targets    : (B,) integer class labels

        Returns
        -------
        loss   : scalar -- sum of losses at all supervised steps
        logits : (B, label_dim) -- y-slot at the final time-step
        """
        B, T, _ = x_seq.shape
        total_h = sum(self.h_sizes)
        state   = torch.zeros(
            B, self.input_dim + total_h + self.label_dim, device=x_seq.device)
        state[:, -self.label_dim:] = 1.0 / self.label_dim  # uniform y-init

        loss = torch.tensor(0.0, device=x_seq.device)

        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(iterations):
                state = self.seq2d(state)
            if t in sup_steps:
                loss = loss + criterion(state[:, -self.label_dim:], targets)

        return loss, state[:, -self.label_dim:]

    def number_of_trainable_parameters(self):
        return self.seq2d.number_of_trainable_parameters()


class LSTMClassifier(nn.Module):
    """
    Standard LSTM sequence classifier (baseline).

    Processes the sequence step by step so that intermediate-step losses can
    be computed at the same sup_steps as the Monarch model.
    """

    def __init__(self, input_dim, hidden_size, num_layers, label_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, label_dim)

    def forward(self, x_seq, sup_steps, criterion, targets):
        """
        Parameters
        ----------
        x_seq     : (B, T, input_dim)
        sup_steps : set of time-step indices at which to compute loss
        criterion : loss function
        targets   : (B,) integer class labels

        Returns
        -------
        loss   : scalar -- sum of losses at supervised steps
        logits : (B, label_dim) -- output at the final supervised step
        """
        B, T, _ = x_seq.shape
        h_state = None
        loss    = torch.tensor(0.0, device=x_seq.device)
        logits  = torch.zeros(B, self.fc.out_features, device=x_seq.device)

        for t in range(T):
            out, h_state = self.lstm(x_seq[:, t:t + 1, :], h_state)
            if t in sup_steps:
                logits = self.fc(out[:, 0, :])
                loss   = loss + criterion(logits, targets)

        # If the last step wasn't supervised, still produce final logits
        if (T - 1) not in sup_steps:
            logits = self.fc(out[:, 0, :])  # out is still valid from last iteration

        return loss, logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, criterion, device, train,
               sup_steps, is_monarch, monarch_iters=0):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x_seq, targets in loader:
            x_seq, targets = x_seq.to(device), targets.to(device)

            if is_monarch:
                loss, logits = model(x_seq, monarch_iters, sup_steps, criterion, targets)
            else:
                loss, logits = model(x_seq, sup_steps, criterion, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(x_seq)
            correct    += (logits.detach().argmax(1) == targets).sum().item()
            total      += len(x_seq)

    return total_loss / total, correct / total * 100


def train_and_eval(model, train_loader, val_loader, cfg, device,
                   sup_steps, is_monarch, label="model"):
    """Train for cfg['epochs'] epochs.  Returns (best_val_acc, best_epoch, elapsed_s)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_acc, best_epoch = 0.0, 0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, sup_steps=sup_steps,
            is_monarch=is_monarch, monarch_iters=cfg.get("iterations", 0))
        _, val_acc = _run_epoch(
            model, val_loader, None, criterion, device,
            train=False, sup_steps=sup_steps,
            is_monarch=is_monarch, monarch_iters=cfg.get("iterations", 0))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch

        print(f"  [{label}] epoch {epoch:3d}/{cfg['epochs']}  "
              f"loss={tr_loss:.4f}  train={tr_acc:.1f}%  val={val_acc:.2f}%")

    return best_val_acc, best_epoch, int(time.time() - t0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--list-datasets", is_flag=True, default=False,
              help="Print available dataset names and exit.")
@click.option("--local", is_flag=True, default=False,
              help="Load from local processed directory instead of remote.")
# Dataset
@click.option("--dataset", default=None,
              help="generatedata dataset name (see --list-datasets).")
@click.option("--step-size", type=int, default=1, show_default=True,
              help="Feature values per time-step (must divide x_y_index).")
@click.option("--label-every-step/--no-label-every-step", default=True, show_default=True,
              help="Append one-hot ground-truth label to every input chunk.")
@click.option("--supervise-every", type=int, default=0, show_default=True,
              help="Backprop on loss every N steps "
                   "(0 = last step only; 1 = every step; N = every Nth step).")
# Model
@click.option("--model",
              type=click.Choice(["monarch", "lstm", "both"], case_sensitive=False),
              default="both", show_default=True,
              help="Which model(s) to train and compare.")
@click.option("--preset",
              type=click.Choice(list(PRESETS)), default="small", show_default=True,
              help="Architecture / training preset.")
# Monarch overrides
@click.option("--h-sizes",    type=int, multiple=True,
              help="Monarch hidden-layer widths (overrides preset).")
@click.option("--num-blocks", type=int, default=None,
              help="Monarch blocks per layer (overrides preset).")
@click.option("--iterations", type=int, default=None,
              help="Monarch map applications per time-step (overrides preset).")
# LSTM overrides
@click.option("--lstm-hidden", type=int, default=None,
              help="LSTM hidden size (overrides preset).")
@click.option("--lstm-layers", type=int, default=None,
              help="LSTM number of layers (overrides preset).")
# Training
@click.option("--lr",           type=float, default=None, help="Learning rate.")
@click.option("--epochs",       type=int,   default=None, help="Training epochs.")
@click.option("--batch-size",   type=int,   default=None, help="Mini-batch size.")
@click.option("--val-fraction", type=float, default=1 / 7, show_default=True,
              help="Fraction of data held out for validation.")
@click.option("--seed",         type=int,   default=42, show_default=True)
def main(
    list_datasets, local,
    dataset, step_size, label_every_step, supervise_every,
    model, preset,
    h_sizes, num_blocks, iterations,
    lstm_hidden, lstm_layers,
    lr, epochs, batch_size, val_fraction, seed,
):
    # ── List mode ─────────────────────────────────────────────────────────────
    if list_datasets:
        names = data_names(local=local)
        print(f"Available datasets ({'local' if local else 'remote'}):")
        for n in sorted(names):
            print(f"  {n}")
        return

    if dataset is None:
        raise click.UsageError(
            "Please specify --dataset NAME.  "
            "Use --list-datasets to see available options.")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = dict(PRESETS[preset])
    if h_sizes:     cfg["h_sizes"]     = list(h_sizes)
    if num_blocks:  cfg["num_blocks"]  = num_blocks
    if iterations:  cfg["iterations"]  = iterations
    if lstm_hidden: cfg["lstm_hidden"] = lstm_hidden
    if lstm_layers: cfg["lstm_layers"] = lstm_layers
    if lr:          cfg["lr"]          = lr
    if epochs:      cfg["epochs"]      = epochs
    if batch_size:  cfg["batch_size"]  = batch_size

    # ── Load data ─────────────────────────────────────────────────────────────
    info      = load_data(dataset, local=local)["info"]
    x_y_index = info["x_y_index"]
    seq_len   = x_y_index // step_size

    print(f"Dataset : {dataset}")
    print(f"  x_y_index={x_y_index}  step_size={step_size}  seq_len={seq_len}")
    print(f"  label_every_step={label_every_step}")

    X_train, y_train, X_val, y_val, label_dim, input_dim = load_split(
        dataset, step_size=step_size, local=local,
        label_every_step=label_every_step,
        val_fraction=val_fraction, seed=seed,
    )

    sup_steps = supervised_step_set(seq_len, supervise_every)
    sup_preview = sorted(sup_steps)[:8]
    sup_suffix  = "..." if len(sup_steps) > 8 else ""

    print(f"  train={len(y_train):,}  val={len(y_val):,}")
    print(f"  label_dim={label_dim}  input_dim={input_dim}")
    print(f"  supervise_every={supervise_every}  "
          f"-> {len(sup_steps)} supervised step(s)  "
          f"(steps: {sup_preview}{sup_suffix})")
    print(f"  device={device}  preset={preset}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=cfg["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=cfg["batch_size"])

    results = {}

    # ── Monarch ───────────────────────────────────────────────────────────────
    if model in ("monarch", "both"):
        monarch = MonarchSequenceClassifier(
            input_dim=input_dim,
            h_sizes=cfg["h_sizes"],
            label_dim=label_dim,
            num_blocks=cfg["num_blocks"],
        ).to(device)

        n_params    = monarch.number_of_trainable_parameters()
        sizes       = [input_dim] + cfg["h_sizes"] + [label_dim]
        dense_equiv = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
        sparsity    = 1.0 - n_params / dense_equiv

        print(f"\n{'='*60}")
        print(f"Monarch INN  h_sizes={cfg['h_sizes']}  "
              f"num_blocks={cfg['num_blocks']}  iterations={cfg['iterations']}")
        print(f"  params={n_params:,}  dense_equiv={dense_equiv:,}  sparsity={sparsity:.2f}")

        best_acc, best_ep, elapsed = train_and_eval(
            monarch, train_loader, val_loader, cfg, device,
            sup_steps=sup_steps, is_monarch=True, label="Monarch")

        results["Monarch"] = dict(
            acc=best_acc, epoch=best_ep, elapsed=elapsed,
            params=n_params, sparsity=sparsity)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if model in ("lstm", "both"):
        lstm = LSTMClassifier(
            input_dim=input_dim,
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            label_dim=label_dim,
        ).to(device)

        n_params_lstm = sum(p.numel() for p in lstm.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"LSTM baseline  hidden={cfg['lstm_hidden']}  layers={cfg['lstm_layers']}")
        print(f"  params={n_params_lstm:,}")

        best_acc, best_ep, elapsed = train_and_eval(
            lstm, train_loader, val_loader, cfg, device,
            sup_steps=sup_steps, is_monarch=False, label="LSTM")

        results["LSTM"] = dict(
            acc=best_acc, epoch=best_ep, elapsed=elapsed,
            params=n_params_lstm, sparsity=None)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  dataset         : {dataset}")
    print(f"  step_size       : {step_size}   seq_len={seq_len}")
    print(f"  label_every_step: {label_every_step}   input_dim={input_dim}")
    print(f"  supervised steps: {len(sup_steps)} / {seq_len}  "
          f"(supervise_every={supervise_every})")
    print(f"  random baseline : {100 / label_dim:.1f}%")
    print()
    hdr = (f"{'Model':<14} {'Val Acc':>8} {'BestEp':>7} "
           f"{'Params':>10} {'Sparsity':>9} {'Time':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        mins, secs = divmod(r["elapsed"], 60)
        sp = f"{r['sparsity']:.2f}" if r["sparsity"] is not None else "  n/a"
        print(f"{name:<14} {r['acc']:>7.2f}%  {r['epoch']:>6}  "
              f"{r['params']:>10,}  {sp:>8}  {mins}m{secs:02d}s")


if __name__ == "__main__":
    main()
