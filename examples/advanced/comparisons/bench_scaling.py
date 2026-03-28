#!/usr/bin/env python
"""Scaling benchmark: grow hidden dimension until OOM for every architecture.

Replaces the former compare_scaling, compare_throughput, compare_layer_scaling,
benchmark_layers, and benchmark_models scripts with one unified entry point.

Two modes:
  --level model   Full INN / RNN / Transformer models via build_model()
  --level layer   Individual layers (nn.Linear, Monarch, Masked, LSLinear, LSTM, Transformer)

Each architecture independently scales from --start-dim upward (geometric ×2)
until GPU OOM, then moves on to the next.  Results are appended to a single
JSONL log via ResultLog.

Usage:
    uv run examples/advanced/comparisons/bench_scaling.py --level model --log results.jsonl
    uv run examples/advanced/comparisons/bench_scaling.py --level layer --log results.jsonl
    uv run examples/advanced/comparisons/bench_scaling.py --level model --start-dim 512 --factor 2
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from bench_utils import (
    ResultLog, benchmark, count_stored, count_surface,
    safe_cleanup, sync_cuda,
)
from models import build_model, ALL_MODEL_NAMES, _count_params

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ── Dimension generator ───────────────────────────────────────────────

def dim_sequence(start=256, factor=2.0):
    """Yield dimensions in geometric progression: start, start*f, start*f², …"""
    dim = start
    while True:
        yield int(dim)
        dim = int(dim * factor)


# ── Model-level benchmarking ─────────────────────────────────────────

def run_model_scaling(args):
    """Scale full models (INN, RNN, Transformer, MLP) until OOM per model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = ResultLog(args.log) if args.log else None
    label_dim = 10

    W = 120
    print(f"{'=' * W}")
    print(" Model-Level Scaling Benchmark")
    print(f"{'=' * W}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
        print(f"  GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print(f"  Models     : {', '.join(args.models)}")
    print(f"  Start dim  : {args.start_dim}")
    print(f"  Factor     : {args.factor}")
    print(f"  Batch      : {args.batch_size}  Seq: {args.seq_len}")
    print()

    print(f"{'Model':<22s} {'Dim':>7s}  {'Stored':>12s}  {'Surface':>12s}  "
          f"{'Fwd ms':>8s}  {'F+B ms':>8s}  {'Peak MB':>8s}")
    print("-" * W)

    for model_name in args.models:
        # Full cleanup between models so the next one starts with a clean slate
        safe_cleanup()

        for dim in dim_sequence(args.start_dim, args.factor):
            # Pre-flight memory check to avoid OS SIGKILL
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                if free / total < 0.15:
                    print(f"{model_name:<22s} {dim:>7,}  "
                          f"** {free/1024**3:.1f}/{total/1024**3:.1f} GiB free — stopping **")
                    break

            model, x = None, None
            try:
                torch.manual_seed(42)
                x = torch.randn(args.batch_size, args.seq_len, dim, device=device)
                model = build_model(
                    model_name, dim, dim, label_dim, args.seq_len,
                    num_blocks=args.num_blocks, rank=args.rank,
                    iterations=args.iterations,
                ).to(device)

                stored = count_stored(model)
                surface = count_surface(model)
                r = benchmark(model, x, args.warmup, args.rounds)

                print(f"{model_name:<22s} {dim:>7,}  {stored:>12,}  {surface:>12,}  "
                      f"{r['fwd_ms']:>8.2f}  {r['total_ms']:>8.2f}  {r['peak_mb']:>8.1f}")

                if log:
                    log.log("model_scaling",
                            model=model_name, dim=dim,
                            stored_params=stored, surface_params=surface,
                            batch_size=args.batch_size, seq_len=args.seq_len,
                            device=str(device),
                            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "",
                            **r)

                del model, x
                model, x = None, None
                safe_cleanup()

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                print(f"{model_name:<22s} {dim:>7,}  ** OOM — stopping **")
                del model, x
                safe_cleanup()
                break

        print()  # blank line between models


# ── Layer-level benchmarking ──────────────────────────────────────────

class _LSTMWrapper(nn.Module):
    """Wrap nn.LSTM so forward() returns a plain 2-D tensor."""
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
    def forward(self, x):
        return self.lstm(x)[0].squeeze(1)


class _TransformerWrapper(nn.Module):
    """Wrap TransformerEncoderLayer so forward() returns a plain 2-D tensor."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x).squeeze(1)


def _wrap_layer(module):
    """Wrap RNN/Transformer layers so benchmark() gets a simple tensor back."""
    if isinstance(module, nn.LSTM):
        return _LSTMWrapper(module)
    if isinstance(module, nn.TransformerEncoderLayer):
        return _TransformerWrapper(module)
    return module


def _find_num_blocks(dim, desired):
    nb = desired
    while nb > 1 and dim % nb != 0:
        nb -= 1
    return nb


def run_layer_scaling(args):
    """Scale individual layers (Linear, Monarch, Masked, LS, LSTM, Transformer) until OOM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = ResultLog(args.log) if args.log else None

    W = 130
    print(f"{'=' * W}")
    print(" Layer-Level Scaling Benchmark")
    print(f"{'=' * W}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
        print(f"  GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print(f"  Start dim  : {args.start_dim}")
    print(f"  Factor     : {args.factor}")
    print(f"  Batch      : {args.batch_size}")
    print(f"  Blocks (k) : {args.num_blocks}")
    print(f"  L+S rank   : {args.rank}")
    print()

    # All layer names we attempt to build (used for OOM tracking)
    ALL_LAYER_NAMES = [
        "nn.Linear", "Monarch", "Masked", "L+S", "LSTM(1-step)", "Transformer(1-step)",
    ]

    def _layer_key(name):
        """Normalize layer name to a stable key for OOM tracking."""
        for prefix in ALL_LAYER_NAMES:
            if name.startswith(prefix):
                return prefix
        return name

    def build_layers(dim, device, oom_set):
        """Return list of (name, module, input_tensor) for a given dim.

        Layers that OOM during construction are added to oom_set and reported.
        """
        layers = []
        k = _find_num_blocks(dim, args.num_blocks)

        def _try_build(name, build_fn):
            key = _layer_key(name)
            if key in oom_set:
                return
            try:
                m, inp = build_fn()
                layers.append((name, m, inp))
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                print(f"  {name:<35s} ** OOM during construction — dropping **")
                oom_set.add(key)
                safe_cleanup()

        # Dense
        def _build_linear():
            m = nn.Linear(dim, dim, bias=True).to(device)
            return m, torch.randn(args.batch_size, dim, device=device)
        _try_build("nn.Linear", _build_linear)

        # Monarch
        def _build_monarch():
            m = MonarchLinear.from_uniform_blocks(dim, dim, num_blocks=k, bias=True).to(device)
            return m, torch.randn(args.batch_size, dim, device=device)
        _try_build(f"Monarch(k={k})", _build_monarch)

        # MaskedLinear (same sparsity as Monarch)
        def _build_masked():
            base = MonarchLinear.from_uniform_blocks(dim, dim, num_blocks=k, bias=True, seed=42)
            m = base.to_MaskedLinear().to(device)
            return m, torch.randn(args.batch_size, dim, device=device)
        _try_build(f"Masked(k={k})", _build_masked)

        # LSLinear
        def _build_ls():
            m = LSLinear.from_uniform_blocks(
                dim, dim, num_blocks=k, rank=args.rank, bias=True, seed=42,
            ).to(device)
            return m, torch.randn(args.batch_size, dim, device=device)
        _try_build(f"L+S(k={k},r={args.rank})", _build_ls)

        # LSTM cell (single step)
        def _build_lstm():
            lstm = nn.LSTM(dim, dim, 1, batch_first=True).to(device)
            x3d = torch.randn(args.batch_size, 1, dim, device=device)
            return lstm, x3d
        _try_build("LSTM(1-step)", _build_lstm)

        # Transformer layer (single step)
        def _build_transformer():
            nh = 4
            while dim % nh != 0:
                nh -= 1
            tlayer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=nh, dim_feedforward=dim * 4, batch_first=True,
            ).to(device)
            x3d = torch.randn(args.batch_size, 1, dim, device=device)
            return tlayer, x3d
        _try_build("Transformer(1-step)", _build_transformer)

        return layers

    # Track which layer types have OOM'd — stop scaling them
    oom_layers = set()

    for dim in dim_sequence(args.start_dim, args.factor):
        # Clean slate before each dimension
        safe_cleanup()

        # Pre-flight memory check: bail if <15% GPU memory free to avoid OS SIGKILL
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            pct_free = free / total * 100
            print(f"── dim={dim:,} (GPU: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free) "
                  f"{'─' * max(1, W - len(f'── dim={dim:,} (GPU: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free) ') - 1)}")
            if pct_free < 15:
                print(f"  ⚠ Only {pct_free:.0f}% GPU memory free — stopping to avoid OS kill.")
                break
        else:
            print(f"── dim={dim:,} {'─' * (W - len(f'── dim={dim:,} ') - 1)}")

        layers = build_layers(dim, device, oom_layers)

        # Check if all layer types have OOM'd (build-time + runtime)
        if len(oom_layers) >= len(ALL_LAYER_NAMES):
            print("  All layer types OOM'd — done.")
            break

        if not layers:
            print("  (no layers survived construction at this dim)")
            print()
            continue

        any_success = False
        for name, module, inp in layers:
            key = _layer_key(name)
            if key in oom_layers:
                continue
            try:
                # Wrap RNN/Transformer so benchmark() gets a simple tensor back
                fn_model = _wrap_layer(module)

                stored = count_stored(module)
                surface = count_surface(module)
                r = benchmark(fn_model, inp, args.warmup, args.rounds)

                print(f"  {name:<35s} stored={stored:>12,}  surface={surface:>12,}  "
                      f"fwd={r['fwd_ms']:8.2f}ms  F+B={r['total_ms']:8.2f}ms  "
                      f"mem={r['peak_mb']:8.1f}MB")

                if log:
                    log.log("layer_scaling",
                            layer=name, dim=dim,
                            stored_params=stored, surface_params=surface,
                            batch_size=args.batch_size,
                            device=str(device),
                            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "",
                            **r)

                any_success = True

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                print(f"  {name:<35s} ** OOM during benchmark — dropping **")
                oom_layers.add(key)

            # Clean up each layer before moving to the next
            safe_cleanup()

        # Drop all references from this dim before moving on
        del layers
        safe_cleanup()
        print()

        if not any_success:
            print("  All layers OOM'd — done.")
            break


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scaling benchmark: grow dims until OOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--level", choices=["model", "layer"], default="model",
                        help="Benchmark full models or individual layers (default: model)")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_NAMES,
                        choices=ALL_MODEL_NAMES,
                        help="Which models to benchmark (model level only)")
    parser.add_argument("--start-dim", type=int, default=256,
                        help="Starting hidden dimension (default: 256)")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Geometric growth factor (default: 2.0)")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length (model level only, default: 32)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--num-blocks", type=int, default=16,
                        help="Monarch block count (default: 16)")
    parser.add_argument("--rank", type=int, default=16,
                        help="L+S rank (default: 16)")
    parser.add_argument("--iterations", type=int, default=4,
                        help="INN iterations per timestep (default: 4)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--rounds", type=int, default=15,
                        help="Timed iterations — median reported (default: 15)")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to JSONL log file (default: no logging)")
    args = parser.parse_args()

    if args.level == "model":
        run_model_scaling(args)
    else:
        run_layer_scaling(args)


if __name__ == "__main__":
    main()
