#!/usr/bin/env python
"""
Throughput & memory comparison across all architectures.

Measures pure computational performance (no data loading):
  - Forward throughput (samples/sec)
  - Forward + backward throughput
  - Peak GPU memory
  - Parameter count

Sweeps across hidden dimensions to show scaling behaviour.
All models are attempted at all dims; OOM errors are caught gracefully.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_throughput.py

    # Specific dims and models
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_throughput.py \
        --dims 1024 4096 16384 --models monarch_inn masked_inn lstm transformer
"""

import argparse
import csv
import gc
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model, ALL_MODEL_NAMES, _count_params


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(model, x, warmup=5, rounds=15):
    """Return (fwd_ms, total_ms, peak_mem_mb)."""
    for _ in range(warmup):
        y = model(x)
        y.sum().backward()

    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fwd_times, total_times = [], []
    for _ in range(rounds):
        _sync()
        t0 = time.perf_counter()
        y = model(x)
        _sync()
        t1 = time.perf_counter()
        y.sum().backward()
        _sync()
        t2 = time.perf_counter()
        fwd_times.append(t1 - t0)
        total_times.append(t2 - t0)

    fwd_ms = sum(fwd_times) / len(fwd_times) * 1e3
    total_ms = sum(total_times) / len(total_times) * 1e3
    peak_mb = 0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return fwd_ms, total_ms, peak_mb


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# INN variants in blues/greens, baselines in reds/oranges
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


def _color_for(name):
    return _MODEL_STYLES.get(name, {}).get("color", "#333333")


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def generate_plots(rows, dims, model_names, save_dir):
    """Generate throughput comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    # Organize data: {dim: {model: row}}
    data = defaultdict(dict)
    for r in rows:
        data[r["dim"]][r["model"]] = r

    dims_present = sorted(set(r["dim"] for r in rows))
    models_present = [m for m in model_names if any(m in data[d] for d in dims_present)]

    if not models_present or not dims_present:
        return

    x_pos = np.arange(len(dims_present))
    n_models = len(models_present)
    width = 0.8 / n_models

    # --- Plot 1: F+B Throughput grouped bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models_present):
        vals = []
        for d in dims_present:
            r = data[d].get(model)
            vals.append(r["total_throughput"] if r else 0)
        ax.bar(x_pos + i * width, vals, width, label=model, color=_color_for(model))
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("F+B Throughput (samples/s)")
    ax.set_title("Forward+Backward Throughput by Model and Dimension")
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"{d:,}" for d in dims_present])
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "throughput_speed.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'throughput_speed.png')}")

    # --- Plot 2: Peak memory grouped bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models_present):
        vals = []
        for d in dims_present:
            r = data[d].get(model)
            vals.append(r["peak_mb"] if r else 0)
        ax.bar(x_pos + i * width, vals, width, label=model, color=_color_for(model))
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak Memory by Model and Dimension")
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"{d:,}" for d in dims_present])
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "throughput_memory.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'throughput_memory.png')}")

    # --- Plot 3: Params grouped bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models_present):
        vals = []
        for d in dims_present:
            r = data[d].get(model)
            vals.append(r["params"] if r else 0)
        ax.bar(x_pos + i * width, vals, width, label=model, color=_color_for(model))
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count by Model and Dimension")
    ax.set_xticks(x_pos + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"{d:,}" for d in dims_present])
    ax.legend(fontsize=8, loc="best")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "throughput_params.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'throughput_params.png')}")


def main():
    parser = argparse.ArgumentParser(description="Throughput comparison across architectures")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_NAMES, choices=ALL_MODEL_NAMES)
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save PNG plots (default: no plots)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dim = 10
    rows = []

    W = 120
    print(f"{'=' * W}")
    print(f" Throughput & Memory Comparison")
    print(f"{'=' * W}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Seq length : {args.seq_len}")
    print(f"  Blocks (k) : {args.num_blocks}")
    print(f"  L+S rank   : {args.rank}")
    print(f"  Warmup     : {args.warmup}  Rounds: {args.rounds}")
    print()

    for dim in args.dims:
        print(f"{'=' * W}")
        print(f" Hidden dim = {dim:,}")
        print(f"{'=' * W}")
        print(f"{'Model':<22s} {'Params':>12s}  {'Fwd ms':>8s}  {'F+B ms':>8s}  "
              f"{'Fwd samp/s':>12s}  {'F+B samp/s':>12s}  {'Mem MB':>8s}")
        print("-" * W)

        x = torch.randn(args.batch_size, args.seq_len, dim, device=device)
        first_params = None

        for model_name in args.models:
            try:
                torch.manual_seed(42)
                model = build_model(
                    model_name, dim, dim, label_dim, args.seq_len,
                    num_blocks=args.num_blocks, rank=args.rank,
                    iterations=args.iterations,
                ).to(device)

                params = _count_params(model)
                if first_params is None:
                    first_params = params

                fwd_ms, total_ms, peak_mb = benchmark_model(
                    model, x, args.warmup, args.rounds
                )
                fwd_tput = args.batch_size / (fwd_ms / 1e3)
                total_tput = args.batch_size / (total_ms / 1e3)

                print(f"{model_name:<22s} {params:>12,}  {fwd_ms:>8.2f}  {total_ms:>8.2f}  "
                      f"{fwd_tput:>12,.0f}  {total_tput:>12,.0f}  {peak_mb:>8.1f}")

                rows.append(dict(
                    dim=dim, model=model_name, params=params,
                    fwd_ms=fwd_ms, total_ms=total_ms,
                    fwd_throughput=fwd_tput, total_throughput=total_tput,
                    peak_mb=peak_mb, batch_size=args.batch_size,
                    seq_len=args.seq_len,
                ))

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                err = str(e).split("\n")[0][:80]
                print(f"{model_name:<22s}  ** ERROR: {err} **")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    if args.save_csv and rows:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.save_csv}")

    if args.save_plots and rows:
        generate_plots(rows, args.dims, args.models, args.save_plots)


if __name__ == "__main__":
    main()
