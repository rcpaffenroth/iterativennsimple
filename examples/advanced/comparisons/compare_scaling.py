#!/usr/bin/env python
"""
Scaling analysis: how each architecture scales with hidden dimension.

For each model, measures throughput and memory at increasing hidden dims,
producing data suitable for log-log scaling plots.  This is the key figure
for demonstrating Monarch's sub-quadratic parameter scaling.

All models are attempted at all dims; OOM errors are caught gracefully.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_scaling.py
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_scaling.py --save-csv results.csv
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_scaling.py --save-plots ./plots
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

import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model, ALL_MODEL_NAMES, _count_params


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(model, x, warmup=3, rounds=10):
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

    return dict(
        fwd_ms=sum(fwd_times) / len(fwd_times) * 1e3,
        total_ms=sum(total_times) / len(total_times) * 1e3,
        peak_mb=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
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


def generate_plots(rows, save_dir):
    """Generate scaling analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    by_model = defaultdict(lambda: {"dims": [], "params": [], "total_throughput": [], "peak_mb": []})
    for r in rows:
        d = by_model[r["model"]]
        d["dims"].append(r["dim"])
        d["params"].append(r["params"])
        d["total_throughput"].append(r["total_throughput"])
        d["peak_mb"].append(r["peak_mb"])

    # --- Plot 1: Params vs Dim (THE key scaling plot) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_model.items()):
        color, marker = _style_for(name)
        ax.loglog(d["dims"], d["params"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Scaling vs Hidden Dimension")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scaling_params.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'scaling_params.png')}")

    # --- Plot 2: F+B Throughput vs Dim ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_model.items()):
        color, marker = _style_for(name)
        ax.loglog(d["dims"], d["total_throughput"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("F+B Throughput (samples/s)")
    ax.set_title("Forward+Backward Throughput vs Hidden Dimension")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scaling_throughput.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'scaling_throughput.png')}")

    # --- Plot 3: Peak Memory vs Dim ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_model.items()):
        color, marker = _style_for(name)
        ax.loglog(d["dims"], d["peak_mb"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak Memory vs Hidden Dimension")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scaling_memory.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'scaling_memory.png')}")


def main():
    parser = argparse.ArgumentParser(description="Scaling analysis across dimensions")
    parser.add_argument("--models", nargs="+",
                        default=["monarch_inn", "masked_inn", "ls_inn",
                                 "lstm", "gru", "rnn_tanh", "transformer", "mlp_flat"],
                        choices=ALL_MODEL_NAMES)
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save PNG plots (default: no plots)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dim = 10
    rows = []

    W = 120
    print(f"{'=' * W}")
    print(f" Scaling Analysis")
    print(f"{'=' * W}")
    print(f"  Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name()}")
    print(f"  Models  : {', '.join(args.models)}")
    print(f"  Dims    : {args.dims}")
    print(f"  Blocks  : {args.num_blocks}  Rank: {args.rank}")
    print(f"  Batch   : {args.batch_size}  Seq: {args.seq_len}")
    print()

    # Header
    print(f"{'Model':<22s} {'Dim':>7s}  {'Params':>12s}  {'Fwd ms':>8s}  {'F+B ms':>8s}  "
          f"{'Fwd samp/s':>12s}  {'F+B samp/s':>12s}  {'Mem MB':>8s}")
    print("-" * W)

    for model_name in args.models:
        for dim in args.dims:
            try:
                torch.manual_seed(42)
                x = torch.randn(args.batch_size, args.seq_len, dim, device=device)
                model = build_model(
                    model_name, dim, dim, label_dim, args.seq_len,
                    num_blocks=args.num_blocks, rank=args.rank,
                    iterations=args.iterations,
                ).to(device)

                params = _count_params(model)
                r = benchmark(model, x, args.warmup, args.rounds)

                fwd_tput = args.batch_size / (r["fwd_ms"] / 1e3)
                total_tput = args.batch_size / (r["total_ms"] / 1e3)

                print(f"{model_name:<22s} {dim:>7,}  {params:>12,}  {r['fwd_ms']:>8.2f}  "
                      f"{r['total_ms']:>8.2f}  {fwd_tput:>12,.0f}  {total_tput:>12,.0f}  "
                      f"{r['peak_mb']:>8.1f}")

                rows.append(dict(
                    model=model_name, dim=dim, params=params,
                    fwd_ms=r["fwd_ms"], total_ms=r["total_ms"],
                    fwd_throughput=fwd_tput, total_throughput=total_tput,
                    peak_mb=r["peak_mb"],
                    batch_size=args.batch_size, seq_len=args.seq_len,
                ))

                del model, x
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                err = str(e).split("\n")[0][:80]
                print(f"{model_name:<22s} {dim:>7,}  ** ERROR: {err} **")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print()  # blank line between models

    # --- Summary: parameter scaling ---
    print(f"\n{'=' * 80}")
    print(f" Parameter Scaling Summary (params vs hidden dim)")
    print(f"{'=' * 80}")
    print(f"{'Model':<22s}", end="")
    for dim in args.dims:
        print(f"  {f'd={dim:,}':>12s}", end="")
    print()
    print("-" * (22 + 14 * len(args.dims)))

    for model_name in args.models:
        print(f"{model_name:<22s}", end="")
        for dim in args.dims:
            matching = [r for r in rows if r["model"] == model_name and r["dim"] == dim]
            if matching:
                p = matching[0]["params"]
                if p >= 1_000_000:
                    print(f"  {p/1e6:>11.2f}M", end="")
                elif p >= 1_000:
                    print(f"  {p/1e3:>11.1f}K", end="")
                else:
                    print(f"  {p:>12d}", end="")
            else:
                print(f"  {'---':>12s}", end="")
        print()

    # --- CSV output ---
    if args.save_csv and rows:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.save_csv}")

    if args.save_plots and rows:
        generate_plots(rows, args.save_plots)


if __name__ == "__main__":
    main()
