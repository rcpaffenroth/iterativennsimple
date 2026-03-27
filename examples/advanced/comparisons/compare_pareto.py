#!/usr/bin/env python
"""
Pareto frontier analysis: accuracy vs parameter count.

Trains all models with MULTIPLE configurations (varying hidden dim, blocks,
rank) to find the Pareto-optimal frontier of accuracy vs. parameters.

This is the key academic plot: for a given parameter budget, which
architecture achieves the best accuracy?

Uses larger hidden dims (256-2048) to produce models in the ~100K-10M
parameter range for meaningful comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_pareto.py --epochs 15

    # Quick test with fewer configs
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_pareto.py --epochs 5 --quick
"""

import argparse
import csv
import gc
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model, _count_params

from generatedata.load_data import load_data_as_sequence, load_data


def load_dataset(name, step_size, local=False, val_frac=1/7, seed=42):
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
        X_val=X[idx[:n_val]], y_val=y[idx[:n_val]],
        label_dim=labels.shape[1], input_dim=X.shape[2], seq_len=X.shape[1],
    )


def quick_train(model, train_loader, val_loader, device, epochs, lr=1e-3):
    """Lean training loop returning best val acc."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = 0.0
    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total += len(y)
        best = max(best, correct / total * 100)
    return best


# ---------------------------------------------------------------------------
# Configurations to sweep
# ---------------------------------------------------------------------------

def get_configs(quick=False):
    """Return list of (model_name, kwargs) to evaluate.

    Dims chosen to produce models in the ~100K to ~10M param range.
    """
    configs = []

    # Monarch INN at various hidden dims and block counts
    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 8, 16, 32, 64]:
            if dim % k == 0:
                configs.append(("monarch_inn", dict(hidden_dim=dim, num_blocks=k)))

    # MaskedLinear INN at various configs (same sparsity as Monarch)
    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 8, 16, 32, 64]:
            if dim % k == 0:
                configs.append(("masked_inn", dict(hidden_dim=dim, num_blocks=k)))

    # L+S INN at various configs
    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 16, 64]:
            for r in [8, 16, 32]:
                if dim % k == 0:
                    configs.append(("ls_inn", dict(hidden_dim=dim, num_blocks=k, rank=r)))

    # LSTM at various hidden dims
    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("lstm", dict(hidden_dim=dim)))

    # GRU at various hidden dims
    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("gru", dict(hidden_dim=dim)))

    # Vanilla RNN
    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("rnn_tanh", dict(hidden_dim=dim)))

    # Transformer at various dims
    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("transformer", dict(hidden_dim=dim)))

    # Flat MLP
    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        configs.append(("mlp_flat", dict(hidden_dim=dim)))

    return configs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Architecture family colors: INN variants in blues/greens, baselines in reds/oranges
_ARCH_STYLES = {
    "monarch_inn":  dict(color="#1f77b4", marker="o",  label="Monarch INN"),
    "masked_inn":   dict(color="#2ca02c", marker="D",  label="Masked INN"),
    "ls_inn":       dict(color="#17becf", marker="P",  label="L+S INN"),
    "lstm":         dict(color="#d62728", marker="s",  label="LSTM"),
    "gru":          dict(color="#ff7f0e", marker="^",  label="GRU"),
    "rnn_tanh":     dict(color="#e377c2", marker="v",  label="RNN (tanh)"),
    "transformer":  dict(color="#9467bd", marker="X",  label="Transformer"),
    "mlp_flat":     dict(color="#8c564b", marker="<",  label="MLP (flat)"),
}


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def generate_plots(results, pareto, save_dir):
    """Generate Pareto frontier plot."""
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all points, colored by architecture
    plotted_labels = set()
    for r in results:
        arch = r["model"]
        style = _ARCH_STYLES.get(arch, dict(color="#333333", marker="x", label=arch))
        label = style["label"] if arch not in plotted_labels else None
        plotted_labels.add(arch)
        ax.scatter(r["params"], r["val_acc"], color=style["color"], marker=style["marker"],
                   s=80, alpha=0.7, label=label, zorder=3)

    # Draw Pareto frontier line
    if len(pareto) >= 2:
        pareto_params = [r["params"] for r in pareto]
        pareto_accs = [r["val_acc"] for r in pareto]
        ax.plot(pareto_params, pareto_accs, "k--", linewidth=1.5, alpha=0.6,
                label="Pareto frontier", zorder=2)
        # Highlight Pareto-optimal points
        ax.scatter(pareto_params, pareto_accs, facecolors="none", edgecolors="black",
                   s=200, linewidths=2, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Pareto Frontier: Accuracy vs Parameter Count")
    ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "pareto_frontier.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'pareto_frontier.png')}")


def main():
    parser = argparse.ArgumentParser(description="Pareto frontier: accuracy vs parameters")
    parser.add_argument("--dataset", default="FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Fewer configurations")
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save PNG plots (default: no plots)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(args.dataset, step_size=args.step_size, local=args.local)
    input_dim, seq_len, label_dim = data["input_dim"], data["seq_len"], data["label_dim"]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=args.batch_size,
    )

    configs = get_configs(args.quick)

    print(f"{'=' * 90}")
    print(f" Pareto Frontier Analysis — {len(configs)} configurations")
    print(f"{'=' * 90}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print()

    results = []
    print(f"{'#':>3s}  {'Model':<18s} {'Config':<40s} {'Params':>12s}  {'Val Acc%':>8s}  {'Time':>6s}")
    print("-" * 95)

    for i, (model_name, kwargs) in enumerate(configs, 1):
        dim = kwargs.get("hidden_dim", 128)
        try:
            torch.manual_seed(args.seed)
            model = build_model(
                model_name, input_dim, dim, label_dim, seq_len,
                iterations=args.iterations, **kwargs,
            ).to(device)
            params = _count_params(model)

            t0 = time.time()
            acc = quick_train(model, train_loader, val_loader, device, args.epochs, args.lr)
            elapsed = time.time() - t0

            cfg_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"{i:>3d}  {model_name:<18s} {cfg_str:<40s} {params:>12,}  {acc:>7.2f}%  {elapsed:>5.0f}s")

            results.append(dict(
                model=model_name, config=cfg_str, params=params,
                val_acc=acc, time_s=elapsed, **kwargs,
            ))

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            cfg_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            err = str(e).split("\n")[0][:80]
            print(f"{i:>3d}  {model_name:<18s} {cfg_str:<40s}  ** ERROR: {err} **")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        print("No configurations completed.")
        return

    # --- Pareto frontier ---
    print(f"\n{'=' * 90}")
    print(f" Pareto-Optimal Configurations (no other config is both cheaper AND more accurate)")
    print(f"{'=' * 90}")

    # Sort by params ascending
    results.sort(key=lambda r: r["params"])
    pareto = []
    best_acc = -1
    for r in results:
        if r["val_acc"] > best_acc:
            pareto.append(r)
            best_acc = r["val_acc"]

    print(f"{'Model':<18s} {'Config':<40s} {'Params':>12s}  {'Val Acc%':>8s}")
    print("-" * 85)
    for r in pareto:
        print(f"{r['model']:<18s} {r['config']:<40s} {r['params']:>12,}  {r['val_acc']:>7.2f}%")

    # --- Per-architecture best ---
    print(f"\n{'=' * 90}")
    print(f" Best Configuration Per Architecture")
    print(f"{'=' * 90}")
    arch_best = {}
    for r in results:
        name = r["model"]
        if name not in arch_best or r["val_acc"] > arch_best[name]["val_acc"]:
            arch_best[name] = r

    for name, r in sorted(arch_best.items(), key=lambda kv: -kv[1]["val_acc"]):
        print(f"  {name:<18s}  val={r['val_acc']:.2f}%  params={r['params']:>12,}  "
              f"({r['config']})")

    if args.save_csv and results:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.save_csv}")

    if args.save_plots and results:
        generate_plots(results, pareto, args.save_plots)


if __name__ == "__main__":
    main()
