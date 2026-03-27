#!/usr/bin/env python
"""
Layer-level scaling comparison: single-layer forward+backward across dims.

This isolates the raw computational efficiency of each layer type WITHOUT
the INN iteration overhead.  Compares:
  - nn.Linear (dense)
  - MonarchLinear (various k)  — structured sparse: k dense block matmuls via torch.bmm
  - MaskedLinear (various k)   — unstructured sparse: full dense matmul then mask
  - LSLinear (L+S)             — low-rank + sparse Monarch (Robust PCA)
  - nn.LSTM (single-step)
  - nn.TransformerEncoderLayer (single-step)

All layer types are attempted at all dims.  If construction or benchmarking
causes an OOM, the error is caught gracefully and the layer is skipped.

Designed to produce the "layer efficiency at scale" figure.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run examples/advanced/comparisons/compare_layer_scaling.py
    CUDA_VISIBLE_DEVICES=0 uv run examples/advanced/comparisons/compare_layer_scaling.py --dims 1024 4096 16384 65536
    CUDA_VISIBLE_DEVICES=0 uv run examples/advanced/comparisons/compare_layer_scaling.py --save-plots ./plots
"""

import argparse
import csv
import gc
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear


def _effective_params(module):
    """Count effective trainable params (respects MaskedLinear mask)."""
    if hasattr(module, "number_of_trainable_parameters"):
        n = module.number_of_trainable_parameters()
        return int(n) if hasattr(n, 'item') else int(n)
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_free_mb():
    """Approximate free GPU memory in MB."""
    if not torch.cuda.is_available():
        return float("inf")
    free, total = torch.cuda.mem_get_info()
    return free / 1024 / 1024


def bench_layer(layer_fn, x, warmup=5, rounds=20):
    """Benchmark a layer callable. Returns (fwd_ms, bwd_ms)."""
    for _ in range(warmup):
        y = layer_fn(x)
        y.sum().backward()

    _sync()
    fwd_times, bwd_times = [], []
    for _ in range(rounds):
        _sync()
        t0 = time.perf_counter()
        y = layer_fn(x)
        _sync()
        t1 = time.perf_counter()
        y.sum().backward()
        _sync()
        t2 = time.perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    return (
        sum(fwd_times) / len(fwd_times) * 1e3,
        sum(bwd_times) / len(bwd_times) * 1e3,
    )


def _try_add_layer(layers, name, build_fn, input_tensor):
    """Try to build a layer; skip gracefully on OOM."""
    try:
        module = build_fn()
        params = _effective_params(module)
        if hasattr(module, "forward"):
            # Quick smoke test — can we even do one forward pass?
            with torch.no_grad():
                _ = module(input_tensor[:1] if input_tensor.dim() >= 2 else input_tensor)
        layers.append((name, module, module, input_tensor, params))
        return True
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        err = str(e).split("\n")[0][:80]
        print(f"  {name:<35s}  [SKIP] {err}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Color/marker scheme: INN variants in blues/greens, baselines in reds/oranges
_LAYER_STYLES = {
    "nn.Linear":           dict(color="#d62728", marker="s"),
    "LSTM(1-step)":        dict(color="#ff7f0e", marker="^"),
    "Transformer(1-step)": dict(color="#e377c2", marker="v"),
}

_MONARCH_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#4477aa", "#66ccee", "#228833"]
_MASKED_COLORS  = ["#ff9896", "#f7b6d2", "#c49c94", "#c5b0d5", "#dbdb8d", "#9edae5"]
_LS_COLORS      = ["#98df8a", "#aec7e8", "#c7c7c7", "#bcbd22", "#7f7f7f", "#393b79"]


def _get_style(name):
    """Return (color, marker) for a layer name."""
    if name in _LAYER_STYLES:
        s = _LAYER_STYLES[name]
        return s["color"], s["marker"]
    if name.startswith("Monarch"):
        idx = hash(name) % len(_MONARCH_COLORS)
        return _MONARCH_COLORS[idx], "o"
    if name.startswith("Masked"):
        idx = hash(name) % len(_MASKED_COLORS)
        return _MASKED_COLORS[idx], "D"
    if name.startswith("L+S"):
        idx = hash(name) % len(_LS_COLORS)
        return _LS_COLORS[idx], "P"
    return "#333333", "x"


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def generate_plots(rows, save_dir):
    """Generate and save layer scaling plots."""
    os.makedirs(save_dir, exist_ok=True)
    _setup_style()

    # Group by layer name
    from collections import defaultdict
    by_layer = defaultdict(lambda: {"dims": [], "params": [], "throughput": [], "total_ms": []})
    for r in rows:
        d = by_layer[r["layer"]]
        d["dims"].append(r["dim"])
        d["params"].append(r["params"])
        d["throughput"].append(r["throughput"])
        d["total_ms"].append(r["total_ms"])

    # --- Plot 1: Params vs Dim ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_layer.items()):
        color, marker = _get_style(name)
        ax.loglog(d["dims"], d["params"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count vs Hidden Dimension")
    ax.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "layer_scaling_params.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'layer_scaling_params.png')}")

    # --- Plot 2: Throughput vs Dim ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_layer.items()):
        color, marker = _get_style(name)
        ax.loglog(d["dims"], d["throughput"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Throughput vs Hidden Dimension")
    ax.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "layer_scaling_throughput.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'layer_scaling_throughput.png')}")

    # --- Plot 3: Total time vs Dim ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in sorted(by_layer.items()):
        color, marker = _get_style(name)
        ax.loglog(d["dims"], d["total_ms"], marker=marker, color=color, label=name,
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Forward+Backward Time vs Hidden Dimension")
    ax.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "layer_scaling_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(save_dir, 'layer_scaling_time.png')}")


def main():
    parser = argparse.ArgumentParser(description="Layer-level scaling comparison")
    parser.add_argument(
        "--dims", type=int, nargs="+",
        default=[1024, 4096, 16384, 65536, 262144, 1048576],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--blocks", type=int, nargs="+",
                        default=[4, 16, 64, 256, 1024, 4096, 16384])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save PNG plots (default: no plots)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    W = 130  # output width
    print(f"{'=' * W}")
    print(" Layer-Level Scaling Comparison")
    print(f"{'=' * W}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
        print(f"  GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Blocks (k) : {args.blocks}")
    print(f"  L+S rank   : {args.rank}")
    print()

    for dim in args.dims:
        print(f"── dim={dim:,} {'─' * (W - len(f'── dim={dim:,} '))}")

        x = torch.randn(args.batch_size, dim, device=device)
        x_3d = x.unsqueeze(1)  # (B, 1, dim) for RNN/Transformer

        layers = []  # list of (name, module, callable, input, params)

        # --- Dense baseline ---
        dense_params = dim * dim + dim  # theoretical dense param count for % column
        try:
            dense = nn.Linear(dim, dim, bias=True).to(device)
            layers.append((
                "nn.Linear",
                dense,
                lambda xx, l=dense: l(xx),
                x,
                _effective_params(dense),
            ))
            dense_params = layers[-1][4]
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print(f"  {'nn.Linear':<35s}  [SKIP] OOM")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Monarch at various k (structured sparse) ---
        for k in args.blocks:
            if dim % k != 0:
                continue
            try:
                m = MonarchLinear.from_uniform_blocks(dim, dim, num_blocks=k, bias=True).to(device)
                layers.append((
                    f"Monarch(k={k})",
                    m,
                    lambda xx, l=m: l(xx),
                    x,
                    _effective_params(m),
                ))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                err = str(e).split("\n")[0][:80]
                print(f"  {'Monarch(k=' + str(k) + ')':<35s}  [SKIP] {err}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- MaskedLinear at various k ---
        for k in args.blocks:
            if dim % k != 0:
                continue
            try:
                monarch_for_mask = MonarchLinear.from_uniform_blocks(
                    dim, dim, num_blocks=k, bias=True, seed=42,
                )
                masked = monarch_for_mask.to_MaskedLinear().to(device)
                layers.append((
                    f"Masked(k={k})",
                    masked,
                    lambda xx, l=masked: l(xx),
                    x,
                    _effective_params(masked),
                ))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                err = str(e).split("\n")[0][:80]
                print(f"  {'Masked(k=' + str(k) + ')':<35s}  [SKIP] {err}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- L+S (LSLinear) at various k ---
        for k in args.blocks:
            if dim % k != 0:
                continue
            try:
                ls = LSLinear.from_uniform_blocks(
                    dim, dim, num_blocks=k, rank=args.rank, bias=True, seed=42,
                ).to(device)
                layers.append((
                    f"L+S(k={k},r={args.rank})",
                    ls,
                    lambda xx, l=ls: l(xx),
                    x,
                    _effective_params(ls),
                ))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                err = str(e).split("\n")[0][:80]
                print(f"  {'L+S(k=' + str(k) + ',r=' + str(args.rank) + ')':<35s}  [SKIP] {err}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- LSTM cell ---
        try:
            lstm = nn.LSTM(dim, dim, 1, batch_first=True).to(device)
            layers.append((
                "LSTM(1-step)",
                lstm,
                lambda xx, l=lstm: l(xx)[0].squeeze(1),
                x_3d,
                _effective_params(lstm),
            ))
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print(f"  {'LSTM(1-step)':<35s}  [SKIP] OOM")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Transformer layer ---
        try:
            nh = 4
            while dim % nh != 0:
                nh -= 1
            tlayer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=nh, dim_feedforward=dim * 4,
                batch_first=True,
            ).to(device)
            layers.append((
                "Transformer(1-step)",
                tlayer,
                lambda xx, l=tlayer: l(xx).squeeze(1),
                x_3d,
                _effective_params(tlayer),
            ))
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print(f"  {'Transformer(1-step)':<35s}  [SKIP] OOM")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Benchmark all ---
        if not layers:
            print(f"  (no layers could be constructed at dim={dim:,})")
            print()
            continue

        for name, module, fn, inp, params in layers:
            try:
                fwd, bwd = bench_layer(fn, inp, args.warmup, args.rounds)
                total = fwd + bwd
                tput = args.batch_size / (total / 1e3)
                pct = params / dense_params * 100

                print(
                    f"  {name:<35s} params={params:>15,}  fwd={fwd:8.2f}ms  "
                    f"bwd={bwd:8.2f}ms  total={total:8.2f}ms  "
                    f"tput={tput:>10,.0f} samp/s  ({pct:7.2f}% of dense)"
                )

                rows.append(
                    dict(
                        dim=dim,
                        layer=name,
                        params=params,
                        fwd_ms=fwd,
                        bwd_ms=bwd,
                        total_ms=total,
                        throughput=tput,
                        pct_dense_params=pct,
                        batch_size=args.batch_size,
                    )
                )
            except Exception as e:
                err = str(e).split("\n")[0][:80]
                print(f"  {name:<35s}  ** ERROR: {err} **")

            del module
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()

    # --- Summary table ---
    if rows:
        print(f"\n{'=' * W}")
        print(" Summary: Monarch speedup over Dense (where both measured)")
        print(f"{'=' * W}")
        by_dim = {}
        for r in rows:
            by_dim.setdefault(r["dim"], {})[r["layer"]] = r
        for dim in sorted(by_dim):
            dense_row = by_dim[dim].get("nn.Linear")
            if not dense_row:
                continue
            dense_total = dense_row["total_ms"]
            for layer_name, r in sorted(by_dim[dim].items()):
                if layer_name.startswith("Monarch"):
                    speedup = dense_total / r["total_ms"]
                    print(f"  dim={dim:<8,}  {layer_name:<25s}  "
                          f"{speedup:5.2f}× faster  "
                          f"({r['pct_dense_params']:5.1f}% params)")
        print()

    if args.save_csv and rows:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.save_csv}")

    if args.save_plots and rows:
        generate_plots(rows, args.save_plots)


if __name__ == "__main__":
    main()
