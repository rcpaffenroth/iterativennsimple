#!/usr/bin/env python
"""Benchmark comparison: 6 model architectures on a hard classification task.

Compares:
  1. Monarch INN        — Sequential2D + MonarchLinear, iterated (sparse structured)
  2. MaskedLinear INN   — Sequential2D + MaskedLinear, iterated (same sparsity pattern)
  3. L+S (LSLinear)     — Single-pass L+S (Robust PCA: low-rank + sparse Monarch)
  4. L+S Iterated       — L+S with iteration on the sparse component
  5. MLP                — Standard dense feedforward (nn.Linear), non-iterated
  6. LSTM               — Standard LSTM sequence model
  7. Transformer        — nn.TransformerEncoder sequence model

Dataset: Uses generatedata library. Defaults to FashionMNIST with aggressive
augmentation (large rotations, scaling, random erasing). Loaded as a sequence
for fair comparison across all architectures.

Usage
-----
    uv run examples/benchmark_comparison.py
    uv run examples/benchmark_comparison.py --dataset MNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0
    uv run examples/benchmark_comparison.py --models monarch masked_inn
    uv run examples/benchmark_comparison.py --device cuda --epochs 20
    uv run examples/benchmark_comparison.py --list-datasets
"""

import time

import click
import numpy as np
import torch
import torch.nn as nn

from generatedata.load_data import data_names, load_data_as_sequence, load_data

from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity


# -- Default dataset (FashionMNIST with hard augmentation) -------------------
DEFAULT_DATASET = (
    "FashionMNIST_custom_degrees0_90_translate0_0.5_scale0.5_1_randomerasing_0.5"
)

# -- Hyperparameters ---------------------------------------------------------
# Dimensions chosen to target ~2-3M trainable params for the INN models.
# With dim=1536, k=4: Monarch INN ~ 1.8M, L+S(r=64) ~ 2.4M params.
HIDDEN_SIZES   = [1536, 1536, 1536]  # three hidden slots in the INN state vector
NUM_BLOCKS     = 4            # Monarch blocks per layer (controls sparsity)
LS_RANK        = 64           # rank of the low-rank L component in L+S
ITERATIONS     = 4            # map applications per time-step (INN models)
STEP_SIZE      = 4            # pixels per time-step (must divide image size)
LSTM_HIDDEN    = 512          # LSTM hidden size (comparable param budget)
LSTM_LAYERS    = 2            # LSTM layers
TRANSFORMER_D  = 512          # Transformer model dimension
TRANSFORMER_NH = 8            # Transformer attention heads
TRANSFORMER_NL = 4            # Transformer encoder layers
LR             = 1e-3
EPOCHS         = 20
BATCH_SIZE     = 128

ALL_MODELS = ["monarch", "masked_inn", "ls", "ls_iter", "mlp", "lstm", "transformer"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(name, step_size, local=False, val_frac=1/7, seed=42):
    """Load a generatedata dataset, split into train/val torch tensors."""
    X_seq, labels = load_data_as_sequence(
        name, step_size=step_size, local=local, label_every_step=True,
    )
    X = torch.from_numpy(X_seq.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32)).argmax(dim=1)

    N   = len(X)
    idx = np.random.default_rng(seed).permutation(N)
    n_val = max(1, int(N * val_frac))

    return (
        X[idx[n_val:]], y[idx[n_val:]],   # train
        X[idx[:n_val]], y[idx[:n_val]],    # val
        labels.shape[1],                   # label_dim (number of classes)
        X.shape[2],                        # input_dim (features per step)
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _find_num_blocks(dim, desired):
    """Largest divisor of dim that is <= desired."""
    nb = desired
    while nb > 1 and dim % nb != 0:
        nb -= 1
    return nb


# -- Shared INN building blocks ---------------------------------------------

def _monarch_block(in_f, out_f, num_blocks, activation=None):
    """Single MonarchLinear block, optionally preceded by an activation."""
    nb = _find_num_blocks(min(in_f, out_f), num_blocks)
    layer = MonarchLinear.from_uniform_blocks(
        in_features=in_f, out_features=out_f, num_blocks=nb, bias=True,
    )
    modules = [activation, layer] if activation else [layer]
    return Sequential1D(nn.Sequential(*modules), in_features=in_f, out_features=out_f)


def _masked_block(in_f, out_f, num_blocks, activation=None):
    """MaskedLinear block with same sparsity pattern as Monarch (via to_MaskedLinear)."""
    nb = _find_num_blocks(min(in_f, out_f), num_blocks)
    monarch = MonarchLinear.from_uniform_blocks(
        in_features=in_f, out_features=out_f, num_blocks=nb, bias=True, seed=42,
    )
    masked = monarch.to_MaskedLinear()
    modules = [activation, masked] if activation else [masked]
    return Sequential1D(nn.Sequential(*modules), in_features=in_f, out_features=out_f)


def _build_inn_map(input_dim, hidden_sizes, label_dim, block_fn, num_blocks):
    """Build a Sequential2D iterative map using the given block_fn."""
    sizes  = [input_dim] + list(hidden_sizes) + [label_dim]
    n      = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)
    for i in range(n - 1):
        act = nn.ReLU() if i > 0 else None
        blocks[i][i + 1] = block_fn(sizes[i], sizes[i + 1], num_blocks, act)
    return Sequential2D(sizes, sizes, blocks)


# -- 1. Monarch INN ---------------------------------------------------------

class MonarchINNClassifier(nn.Module):
    """Monarch INN: iterated sparse structured map."""
    def __init__(self, input_dim, hidden_sizes, label_dim, num_blocks, iterations):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_sizes = list(hidden_sizes)
        self.iterations = iterations
        self.map = _build_inn_map(input_dim, hidden_sizes, label_dim, _monarch_block, num_blocks)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        state_dim = self.input_dim + sum(self.hidden_sizes) + self.label_dim
        state = torch.zeros(B, state_dim, device=x_seq.device)
        state[:, -self.label_dim:] = 1.0 / self.label_dim
        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(self.iterations):
                state = self.map(state)
        return state[:, -self.label_dim:]


# -- 2. MaskedLinear INN ----------------------------------------------------

class MaskedLinearINNClassifier(nn.Module):
    """MaskedLinear INN: iterated sparse unstructured map (same sparsity as Monarch)."""
    def __init__(self, input_dim, hidden_sizes, label_dim, num_blocks, iterations):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_sizes = list(hidden_sizes)
        self.iterations = iterations
        self.map = _build_inn_map(input_dim, hidden_sizes, label_dim, _masked_block, num_blocks)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        state_dim = self.input_dim + sum(self.hidden_sizes) + self.label_dim
        state = torch.zeros(B, state_dim, device=x_seq.device)
        state[:, -self.label_dim:] = 1.0 / self.label_dim
        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(self.iterations):
                state = self.map(state)
        return state[:, -self.label_dim:]


# -- 3. L+S Single-Pass Classifier ------------------------------------------

class LSClassifier(nn.Module):
    """L+S (Robust PCA) classifier: single forward pass through LSLinear layers.

    This is NOT an INN — it's a feedforward network using L+S layers.
    The sparse Monarch component IS the INN-like structured part; the low-rank
    component adds global expressivity without iteration.
    """
    def __init__(self, input_dim, hidden_sizes, label_dim, num_blocks, rank):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [label_dim]
        layers = []
        for i in range(len(sizes) - 1):
            nb = _find_num_blocks(min(sizes[i], sizes[i + 1]), num_blocks)
            layers.append(LSLinear.from_uniform_blocks(
                sizes[i], sizes[i + 1], num_blocks=nb, rank=rank, bias=True,
            ))
            if i < len(sizes) - 2:  # no activation after last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq):
        # Process sequence: use the last timestep's features
        B, T, D = x_seq.shape
        x = x_seq[:, -1, :]  # (B, D) — use final timestep
        return self.net(x)


# -- 4. L+S Iterated Classifier ---------------------------------------------

class LSIteratedClassifier(nn.Module):
    """L+S with iteration on the sparse component.

    Uses LSLinear layers in a Sequential2D map, iterated like the INNs.
    The low-rank L component provides global structure; the sparse S component
    is the iterated INN part.
    """
    def __init__(self, input_dim, hidden_sizes, label_dim, num_blocks, rank, iterations):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_sizes = list(hidden_sizes)
        self.iterations = iterations

        sizes  = [input_dim] + list(hidden_sizes) + [label_dim]
        n      = len(sizes)
        blocks = [[None] * n for _ in range(n)]
        blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)
        for i in range(n - 1):
            nb = _find_num_blocks(min(sizes[i], sizes[i + 1]), num_blocks)
            act = nn.ReLU() if i > 0 else None
            layer = LSLinear.from_uniform_blocks(
                sizes[i], sizes[i + 1], num_blocks=nb, rank=rank, bias=True,
            )
            modules = [act, layer] if act else [layer]
            blocks[i][i + 1] = Sequential1D(
                nn.Sequential(*modules), in_features=sizes[i], out_features=sizes[i + 1],
            )
        self.map = Sequential2D(sizes, sizes, blocks)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        state_dim = self.input_dim + sum(self.hidden_sizes) + self.label_dim
        state = torch.zeros(B, state_dim, device=x_seq.device)
        state[:, -self.label_dim:] = 1.0 / self.label_dim
        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(self.iterations):
                state = self.map(state)
        return state[:, -self.label_dim:]


# -- 5. MLP Baseline --------------------------------------------------------

class MLPClassifier(nn.Module):
    """Standard dense MLP. Non-iterated, non-sequential baseline."""
    def __init__(self, input_dim, hidden_sizes, label_dim):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [label_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq):
        B, T, D = x_seq.shape
        x = x_seq[:, -1, :]  # use final timestep
        return self.net(x)


# -- 6. LSTM Baseline -------------------------------------------------------

class LSTMClassifier(nn.Module):
    """Standard LSTM sequence classifier."""
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
        return self.fc(out[:, -1, :])


# -- 7. Transformer Baseline ------------------------------------------------

class TransformerClassifier(nn.Module):
    """Transformer encoder sequence classifier."""
    def __init__(self, input_dim, d_model, nhead, num_layers, label_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, label_dim)

    def forward(self, x_seq):
        x = self.input_proj(x_seq)          # (B, T, d_model)
        x = self.encoder(x)                 # (B, T, d_model)
        return self.fc(x[:, -1, :])         # classify from last position


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, train=True):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        _, val_acc = run_epoch(model, val_loader, None, criterion, device, train=False)
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
              help="Dataset name (default: FashionMNIST with hard augmentation).")
@click.option("--models", multiple=True, type=click.Choice(ALL_MODELS, case_sensitive=False),
              default=ALL_MODELS, show_default=True,
              help="Which model(s) to train. Repeat for multiple.")
@click.option("--epochs", default=EPOCHS, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(list_datasets, local, dataset, models, epochs, device, seed):
    if list_datasets:
        for name in sorted(data_names(local=local)):
            print(name)
        return

    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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
    print(f"  input_dim  : {input_dim}  (features per step)")
    print(f"  classes    : {label_dim}  (random baseline = {100/label_dim:.0f}%)")
    print(f"  device     : {device}")
    print()

    results = {}

    # -- 1. Monarch INN ------------------------------------------------------
    if "monarch" in models:
        model = MonarchINNClassifier(
            input_dim, HIDDEN_SIZES, label_dim, NUM_BLOCKS, ITERATIONS,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Monarch INN  blocks={NUM_BLOCKS}  iters={ITERATIONS}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "Monarch INN")
        results["Monarch INN"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 2. MaskedLinear INN -------------------------------------------------
    if "masked_inn" in models:
        model = MaskedLinearINNClassifier(
            input_dim, HIDDEN_SIZES, label_dim, NUM_BLOCKS, ITERATIONS,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nMaskedLinear INN  blocks={NUM_BLOCKS}  iters={ITERATIONS}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "Masked INN")
        results["MaskedLinear INN"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 3. L+S Single-Pass --------------------------------------------------
    if "ls" in models:
        model = LSClassifier(
            input_dim, HIDDEN_SIZES, label_dim, NUM_BLOCKS, LS_RANK,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nL+S (single-pass)  blocks={NUM_BLOCKS}  rank={LS_RANK}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "L+S")
        results["L+S (single)"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 4. L+S Iterated -----------------------------------------------------
    if "ls_iter" in models:
        model = LSIteratedClassifier(
            input_dim, HIDDEN_SIZES, label_dim, NUM_BLOCKS, LS_RANK, ITERATIONS,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nL+S (iterated)  blocks={NUM_BLOCKS}  rank={LS_RANK}  iters={ITERATIONS}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "L+S iter")
        results["L+S (iterated)"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 5. MLP --------------------------------------------------------------
    if "mlp" in models:
        model = MLPClassifier(input_dim, HIDDEN_SIZES, label_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nMLP (dense)  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "MLP")
        results["MLP"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 6. LSTM -------------------------------------------------------------
    if "lstm" in models:
        model = LSTMClassifier(
            input_dim, LSTM_HIDDEN, LSTM_LAYERS, label_dim,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nLSTM  hidden={LSTM_HIDDEN}  layers={LSTM_LAYERS}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "LSTM")
        results["LSTM"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- 7. Transformer ------------------------------------------------------
    if "transformer" in models:
        model = TransformerClassifier(
            input_dim, TRANSFORMER_D, TRANSFORMER_NH, TRANSFORMER_NL, label_dim,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTransformer  d={TRANSFORMER_D}  heads={TRANSFORMER_NH}  layers={TRANSFORMER_NL}  params={n_params:,}")
        acc, elapsed = train_model(model, train_loader, val_loader, device, epochs, "Transformer")
        results["Transformer"] = {"acc": acc, "params": n_params, "time": elapsed}

    # -- Summary -------------------------------------------------------------
    if results:
        print(f"\n{'='*65}")
        print(f"{'Model':<20} {'Val Acc':>8} {'Params':>10} {'Time':>8}")
        print(f"{'-'*65}")
        for name, r in results.items():
            print(f"{name:<20} {r['acc']:>7.1f}%  {r['params']:>10,}  {r['time']:>6.0f}s")


if __name__ == "__main__":
    main()
