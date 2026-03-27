#!/usr/bin/env python
"""
Full training comparison across all architectures on real data.

Trains every model architecture on the same dataset and reports:
  - Final train/val accuracy
  - Convergence speed (epochs to reach X% accuracy)
  - Total training time
  - Parameter count and memory usage
  - Accuracy per parameter (efficiency metric)

Target: ~2-3M trainable parameters per model for fair comparison.

Usage:
    # Quick run (5 epochs, all models)
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_training.py

    # Full academic run
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_training.py \
        --epochs 50 --dataset FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0

    # Specific models only
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_training.py \
        --models monarch_inn lstm transformer --epochs 30

    # List available datasets
    uv run examples/advanced/comparisons/compare_training.py --list-datasets
"""

import argparse
import gc
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

# Add parent to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from models import build_model, ALL_MODEL_NAMES, _count_params

from generatedata.load_data import data_names, load_data_as_sequence, load_data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name, step_size, local=False, val_frac=1/7, seed=42):
    """Load dataset as sequential chunks, split into train/val."""
    X_seq, labels = load_data_as_sequence(
        name, step_size=step_size, local=local, label_every_step=True,
    )
    X = torch.from_numpy(X_seq.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32)).argmax(dim=1)

    N = len(X)
    idx = np.random.default_rng(seed).permutation(N)
    n_val = max(1, int(N * val_frac))

    return dict(
        X_train=X[idx[n_val:]], y_train=y[idx[n_val:]],
        X_val=X[idx[:n_val]],   y_val=y[idx[:n_val]],
        label_dim=labels.shape[1],
        input_dim=X.shape[2],
        seq_len=X.shape[1],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(X)
    return total_loss / count, correct / count * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(X)
        correct += (logits.argmax(1) == y).sum().item()
        count += len(X)
    return total_loss / count, correct / count * 100


def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3,
                label="model", verbose=True):
    """Train and track metrics per epoch. Returns results dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_acc = 0.0
    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        best_val_acc = max(best_val_acc, va_acc)
        elapsed = time.time() - t_epoch

        history.append(dict(
            epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss, val_acc=va_acc, epoch_time=elapsed,
        ))

        if verbose:
            print(f"  [{label:>20s}] epoch {epoch:3d}/{epochs}  "
                  f"loss {tr_loss:.4f}  train {tr_acc:5.1f}%  val {va_acc:5.1f}%  "
                  f"({elapsed:.1f}s)")

    total_time = time.time() - t0
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return dict(
        best_val_acc=best_val_acc,
        final_train_acc=history[-1]["train_acc"],
        final_val_acc=history[-1]["val_acc"],
        total_time=total_time,
        peak_mem_mb=peak_mem,
        history=history,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_MODEL_STYLES = {
    "monarch_inn":  dict(color="#1f77b4", marker="o"),
    "masked_inn":   dict(color="#2ca02c", marker="D"),
    "ls_inn":       dict(color="#17becf", marker="P"),
    "lstm":         dict(color="#d62728", marker="s"),
    "gru":          dict(color="#ff7f0e", marker="^"),
    "rnn_tanh":     dict(color="#e377c2", marker="v"),
    "transformer":  dict(color="#9467bd", marker="X"),
    "mlp_flat":     dict(color="#8c564b", marker="<"),
}


def _style_for(name):
    s = _MODEL_STYLES.get(name, {})
    return s.get("color", "#333333"), s.get("marker", "x")


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def generate_plots(all_results, model_names, save_dir):
    """Generate training comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    completed = [n for n in model_names if n in all_results]
    if not completed:
        return

    # --- Plot 1: Val accuracy vs epoch (learning curves) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in completed:
        r = all_results[name]
        epochs = [h["epoch"] for h in r["history"]]
        val_accs = [h["val_acc"] for h in r["history"]]
        color, marker = _style_for(name)
        ax.plot(epochs, val_accs, marker=marker, color=color, label=name,
                linewidth=1.5, markersize=4, markevery=max(1, len(epochs) // 10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Learning Curves: Validation Accuracy vs Epoch")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'training_curves.png')}")

    # --- Plot 2: Bar chart of best val accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = completed
    accs = [all_results[n]["best_val_acc"] for n in names]
    colors = [_style_for(n)[0] for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Best Validation Accuracy by Model")
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'training_accuracy.png')}")

    # --- Plot 3: Scatter of val accuracy vs params (efficiency) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in completed:
        r = all_results[name]
        color, marker = _style_for(name)
        ax.scatter(r["params"], r["best_val_acc"], color=color, marker=marker,
                   s=100, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Parameter Efficiency: Accuracy vs Parameter Count")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_efficiency.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'training_efficiency.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full training comparison across architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-datasets", action="store_true",
                        help="Print available datasets and exit.")
    parser.add_argument("--local", action="store_true",
                        help="Load from local cache.")
    parser.add_argument("--dataset", default="FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_NAMES,
                        choices=ALL_MODEL_NAMES,
                        help="Which models to compare.")
    # --- Sizing for ~2-3M param target ---
    # INN models: hidden_dim=1536, num_blocks=16 -> Monarch params ~ 1.8M
    # LSTM: hidden_dim=512 (2 layers) -> ~4.2M params (LSTM needs 4x for gates)
    # Transformer: hidden_dim=512 (2 layers) -> ~6.3M params
    # These are in the same ballpark for meaningful comparison.
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=4,
                        help="Pixels per timestep (must divide image size).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save results as JSON.")
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save PNG plots (default: no plots)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list_datasets:
        for name in sorted(data_names(local=args.local)):
            print(name)
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    data = load_dataset(args.dataset, step_size=args.step_size, local=args.local)
    input_dim = data["input_dim"]
    seq_len = data["seq_len"]
    label_dim = data["label_dim"]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=args.batch_size,
    )

    print(f"{'=' * 80}")
    print(f" Training Comparison — {args.dataset}")
    print(f"{'=' * 80}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
    print(f"  Samples    : {len(data['y_train']):,} train / {len(data['y_val']):,} val")
    print(f"  Sequence   : {seq_len} steps x {input_dim} features")
    print(f"  Classes    : {label_dim}  (random baseline = {100/label_dim:.0f}%)")
    print(f"  Hidden dim : {args.hidden_dim}")
    print(f"  Blocks (k) : {args.num_blocks}  Rank: {args.rank}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}  (cosine annealing)")
    print()

    # --- Train each model ---
    all_results = {}

    for model_name in args.models:
        print(f"--- {model_name} ---")
        torch.manual_seed(args.seed)

        try:
            model = build_model(
                model_name, input_dim, args.hidden_dim, label_dim, seq_len,
                num_blocks=args.num_blocks, rank=args.rank,
                iterations=args.iterations, num_layers=2, nhead=4,
            ).to(device)

            n_params = _count_params(model)
            print(f"  Parameters: {n_params:,}")

            result = train_model(
                model, train_loader, val_loader, device, args.epochs,
                lr=args.lr, label=model_name, verbose=not args.quiet,
            )
            result["params"] = n_params
            result["model_name"] = model_name
            all_results[model_name] = result

            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            err = str(e).split("\n")[0][:100]
            print(f"  ** ERROR: {err} **")

        print()

    if not all_results:
        print("No models completed training.")
        return

    # --- Summary table ---
    print(f"\n{'=' * 110}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 110}")
    print(f"{'Model':<22s} {'Params':>12s}  {'Best Val%':>9s}  {'Final Val%':>10s}  "
          f"{'Time (s)':>9s}  {'Mem (MB)':>9s}  {'Acc/1K-Param':>12s}")
    print(f"{'-' * 110}")

    for name in args.models:
        if name not in all_results:
            continue
        r = all_results[name]
        acc_per_kparam = r["best_val_acc"] / (r["params"] / 1000) if r["params"] > 0 else 0
        print(f"{name:<22s} {r['params']:>12,}  {r['best_val_acc']:>8.2f}%  "
              f"{r['final_val_acc']:>9.2f}%  {r['total_time']:>8.1f}s  "
              f"{r['peak_mem_mb']:>8.1f}  {acc_per_kparam:>11.4f}")

    # --- Convergence speed ---
    thresholds = [50.0, 70.0, 80.0, 85.0, 90.0]
    print(f"\nConvergence Speed (epochs to reach accuracy threshold):")
    header = f"{'Model':<22s}" + "".join(f"  {t:>5.0f}%" for t in thresholds)
    print(header)
    print("-" * len(header))
    for name in args.models:
        if name not in all_results:
            continue
        r = all_results[name]
        cols = []
        for thresh in thresholds:
            reached = None
            for h in r["history"]:
                if h["val_acc"] >= thresh:
                    reached = h["epoch"]
                    break
            cols.append(f"  {reached:>5d} " if reached else "     -- ")
        print(f"{name:<22s}{''.join(cols)}")

    # --- Parameter efficiency ranking ---
    print(f"\nParameter Efficiency Ranking (best val accuracy / 1K parameters):")
    ranked = sorted(all_results.values(),
                    key=lambda r: r["best_val_acc"] / max(r["params"] / 1000, 1),
                    reverse=True)
    for i, r in enumerate(ranked, 1):
        eff = r["best_val_acc"] / (r["params"] / 1000)
        print(f"  {i}. {r['model_name']:<22s}  {eff:>8.4f} acc%/kParam  "
              f"(val={r['best_val_acc']:.1f}%, params={r['params']:,})")

    # --- Save JSON ---
    if args.save_json:
        out = {
            "config": vars(args),
            "dataset": args.dataset,
            "results": {
                name: {k: v for k, v in r.items() if k != "history"}
                for name, r in all_results.items()
            },
            "history": {
                name: r["history"] for name, r in all_results.items()
            },
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    # --- Plots ---
    if args.save_plots and all_results:
        generate_plots(all_results, args.models, args.save_plots)


if __name__ == "__main__":
    main()
