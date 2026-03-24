#!/usr/bin/env python
"""
Monarch INN vs LSTM on Fashion-MNIST with rotations.

Treats image pixels as a sequence of chunks and classifies using an
iterative Monarch sparse map, with an LSTM baseline for comparison.

Dataset: Fashion-MNIST with 0-45° rotations and 0.75-1× scaling
(10 classes, random baseline = 10%).

Usage
-----
    uv run examples/monarch_mnist.py                              # both models
    uv run examples/monarch_mnist.py --model monarch              # Monarch only
    uv run examples/monarch_mnist.py --dataset MNIST --epochs 5   # quick MNIST
    uv run examples/monarch_mnist.py --list-datasets              # show all datasets
"""

import time

import click
import numpy as np
import torch
import torch.nn as nn

from generatedata.load_data import data_names, load_data_as_sequence, load_data

from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity

# -- Default dataset (Fashion-MNIST with moderate rotation + scaling) --------
DEFAULT_DATASET = (
    "FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0"
)

# -- Hyperparameters ---------------------------------------------------------
HIDDEN_SIZES = [128, 128]   # two hidden slots in the Monarch state vector
NUM_BLOCKS   = 4            # Monarch blocks per layer (controls sparsity)
ITERATIONS   = 4            # map applications per time-step
STEP_SIZE    = 4            # pixels per time-step (must divide image size)
LSTM_HIDDEN  = 128          # LSTM hidden size (comparable param budget)
LSTM_LAYERS  = 2            # LSTM layers
LR           = 1e-3
EPOCHS       = 20
BATCH_SIZE   = 256


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(name, step_size, local=False, val_frac=1 / 7, seed=42):
    """Load a generatedata dataset, split into train/val torch tensors."""
    X_seq, labels = load_data_as_sequence(
        name, step_size=step_size, local=local, label_every_step=True,
    )
    X = torch.from_numpy(X_seq.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32)).argmax(dim=1)

    # Reproducible train/val split
    N   = len(X)
    idx = np.random.default_rng(seed).permutation(N)
    n_val = max(1, int(N * val_frac))

    return (
        X[idx[n_val:]], y[idx[n_val:]],   # train
        X[idx[:n_val]], y[idx[:n_val]],    # val
        labels.shape[1],                   # label_dim  (number of classes)
        X.shape[2],                        # input_dim  (features per step)
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# ---- Monarch INN ----------------------------------------------------------

def _monarch_block(in_f, out_f, num_blocks, activation=None):
    """Single MonarchLinear block, optionally preceded by an activation."""
    nb = num_blocks
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    layer = MonarchLinear.from_uniform_blocks(
        in_features=in_f, out_features=out_f, num_blocks=nb, bias=True,
    )
    modules = [activation, layer] if activation else [layer]
    return Sequential1D(nn.Sequential(*modules), in_features=in_f, out_features=out_f)


def _build_map(input_dim, hidden_sizes, label_dim, num_blocks):
    """
    Build the Sequential2D iterative map.

    State vector: [ x (input) | h1 | ... | hk | y (output) ]

    The map is a feed-forward chain  x -> h1 -> ... -> hk -> y
    with x held fixed (Identity) so inputs persist across iterations.
    """
    sizes  = [input_dim] + list(hidden_sizes) + [label_dim]
    n      = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)
    for i in range(n - 1):
        act = nn.ReLU() if i > 0 else None
        blocks[i][i + 1] = _monarch_block(sizes[i], sizes[i + 1], num_blocks, act)
    return Sequential2D(sizes, sizes, blocks)


class MonarchClassifier(nn.Module):
    """
    Monarch INN sequence classifier.

    At each time-step: write the chunk into the x-slot, then apply the
    Monarch map ``iterations`` times.  Returns logits from the y-slot.
    """

    def __init__(self, input_dim, hidden_sizes, label_dim, num_blocks):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_sizes = list(hidden_sizes)
        self.map = _build_map(input_dim, hidden_sizes, label_dim, num_blocks)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        state_dim = self.input_dim + sum(self.hidden_sizes) + self.label_dim
        state = torch.zeros(B, state_dim, device=x_seq.device)
        state[:, -self.label_dim:] = 1.0 / self.label_dim  # uniform prior

        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(ITERATIONS):
                state = self.map(state)

        return state[:, -self.label_dim:]


# ---- LSTM baseline --------------------------------------------------------

class LSTMClassifier(nn.Module):
    """Standard LSTM sequence classifier (baseline)."""

    def __init__(self, input_dim, hidden_size, num_layers, label_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, label_dim)

    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :])  # classify from last hidden state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    """Run one epoch.  Returns (avg_loss, accuracy%)."""
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(X)
            correct += (logits.detach().argmax(1) == y).sum().item()
            count += len(X)

    return total_loss / count, correct / count * 100


def train_model(model, train_loader, val_loader, device, epochs, label="model"):
    """Train for ``epochs`` epochs.  Returns (best_val_acc, elapsed_seconds)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
        )
        _, val_acc = run_epoch(
            model, val_loader, None, criterion, device, train=False,
        )
        best_val_acc = max(best_val_acc, val_acc)
        epoch_s = time.time() - t_epoch

        print(f"  [{label}] epoch {epoch:3d}/{epochs}  "
              f"loss {tr_loss:.4f}  train {tr_acc:.1f}%  val {val_acc:.1f}%  "
              f"({epoch_s:.1f}s)")

    return best_val_acc, time.time() - t0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--list-datasets", is_flag=True, help="Print available datasets and exit.")
@click.option("--local", is_flag=True, help="Load from local cache instead of remote.")
@click.option("--dataset", default=DEFAULT_DATASET, show_default=False,
              help="Dataset name (default: Fashion-MNIST with rotations).")
@click.option("--model", type=click.Choice(["monarch", "lstm", "both"], case_sensitive=False),
              default="both", show_default=True, help="Which model(s) to train.")
@click.option("--compile/--no-compile", default=False, show_default=True,
              help="Use torch.compile to fuse kernels (reduces GPU launch overhead).")
@click.option("--epochs", default=EPOCHS, show_default=True, help="Training epochs.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
def main(list_datasets, local, dataset, model, compile, epochs, seed):
    if list_datasets:
        for name in sorted(data_names(local=local)):
            print(name)
        return

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Data ----------------------------------------------------------------
    info = load_data(dataset, local=local)["info"]
    seq_len = info["x_y_index"] // STEP_SIZE

    X_tr, y_tr, X_va, y_va, label_dim, input_dim = load_dataset(
        dataset, step_size=STEP_SIZE, local=local,
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_va, y_va),
        batch_size=BATCH_SIZE,
    )

    print(f"Dataset      : {dataset}")
    print(f"  samples    : {len(y_tr):,} train / {len(y_va):,} val")
    print(f"  seq_len    : {seq_len}  (step_size={STEP_SIZE})")
    print(f"  classes    : {label_dim}  (random baseline = {100/label_dim:.0f}%)")
    print(f"  device     : {device}")

    results = {}

    # -- Monarch -------------------------------------------------------------
    if model in ("monarch", "both"):
        monarch = MonarchClassifier(
            input_dim=input_dim,
            hidden_sizes=HIDDEN_SIZES,
            label_dim=label_dim,
            num_blocks=NUM_BLOCKS,
        ).to(device)

        n_params    = monarch.map.number_of_trainable_parameters()
        sizes       = [input_dim] + HIDDEN_SIZES + [label_dim]
        dense_equiv = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
        sparsity    = 1.0 - n_params / dense_equiv

        print(f"\nMonarch INN  state=[{' + '.join(str(s) for s in sizes)}]  "
              f"blocks={NUM_BLOCKS}  iters={ITERATIONS}")
        print(f"  params={n_params:,}  (dense equiv {dense_equiv:,}, sparsity {sparsity:.0%})")

        if compile:
            monarch = torch.compile(monarch, mode="reduce-overhead")
            print("  torch.compile enabled (first epoch will be slower due to compilation)")

        acc, elapsed = train_model(monarch, train_loader, val_loader, device, epochs, "Monarch")
        results["Monarch"] = {"acc": acc, "params": n_params, "sparsity": sparsity, "time": elapsed}

    # -- LSTM ----------------------------------------------------------------
    if model in ("lstm", "both"):
        lstm = LSTMClassifier(
            input_dim=input_dim,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            label_dim=label_dim,
        ).to(device)

        n_params_lstm = sum(p.numel() for p in lstm.parameters() if p.requires_grad)

        print(f"\nLSTM baseline  hidden={LSTM_HIDDEN}  layers={LSTM_LAYERS}")
        print(f"  params={n_params_lstm:,}")

        if compile:
            lstm = torch.compile(lstm, mode="reduce-overhead")
            print("  torch.compile enabled (first epoch will be slower due to compilation)")

        acc, elapsed = train_model(lstm, train_loader, val_loader, device, epochs, "LSTM")
        results["LSTM"] = {"acc": acc, "params": n_params_lstm, "sparsity": None, "time": elapsed}

    # -- Summary -------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"{'Model':<12} {'Val Acc':>8} {'Params':>10} {'Sparsity':>9} {'Time':>8}")
    print(f"{'-'*55}")
    for name, r in results.items():
        sp = f"{r['sparsity']:.0%}" if r['sparsity'] is not None else "  n/a"
        print(f"{name:<12} {r['acc']:>7.1f}%  {r['params']:>10,}  {sp:>8}  {r['time']:>6.0f}s")


if __name__ == "__main__":
    main()
