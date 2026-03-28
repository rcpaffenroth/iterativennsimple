#!/usr/bin/env python
"""Kernel-level benchmark: MonarchLinear and LSLinear forward/backward paths.

Compares wall-clock time of every compute path at the kernel level:
  1. MonarchLinear — Fused Triton kernel
  2. MonarchLinear — BMM path (torch.bmm, no views)
  3. MonarchLinear — View-loop path (gather/scatter per block)
  4. LSLinear      — Full forward (sparse S + low-rank L)
  5. nn.Linear     — Dense baseline

Sweeps over block sizes, number of blocks, and batch sizes.
Reports median forward, backward, and combined (F+B) times in milliseconds.
Results are appended to a JSONL log via ResultLog.

Usage:
    uv run examples/advanced/comparisons/bench_kernels.py
    uv run examples/advanced/comparisons/bench_kernels.py --log results.jsonl
    uv run examples/advanced/comparisons/bench_kernels.py --block-sizes 64 128 --num-blocks 4 8 16
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from bench_utils import ResultLog, benchmark, count_stored, safe_cleanup, sync_cuda

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ── Helpers ──────────────────────────────────────────────────────────

class _ForwardWrapper(nn.Module):
    """Wrap a layer + kwargs so benchmark() can call model(x) directly."""
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer
        self._kwargs = kwargs

    def forward(self, x):
        return self.layer(x, **self._kwargs)

    def parameters(self, recurse=True):
        return self.layer.parameters(recurse=recurse)


def _try_benchmark(name, build_fn, x, warmup, rounds):
    """Build a layer, benchmark it, return results dict or None on failure."""
    try:
        layer = build_fn()
        r = benchmark(layer, x, warmup=warmup, rounds=rounds)
        stored = count_stored(layer.layer if isinstance(layer, _ForwardWrapper) else layer)
        r["stored_params"] = stored
        safe_cleanup(layer)
        return r
    except Exception as e:
        print(f"    {name:<30s}  ERROR: {e!s:.60s}")
        safe_cleanup()
        return None


# ── Main sweep ───────────────────────────────────────────────────────

def run_kernel_bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = ResultLog(args.log) if args.log else None

    W = 140
    print(f"{'=' * W}")
    print(" Kernel-Level Benchmark: MonarchLinear & LSLinear Forward/Backward Paths")
    print(f"{'=' * W}")
    print(f"  Device      : {device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name()}")
        mem_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory  : {mem_gib:.1f} GiB")
    print(f"  Block sizes : {args.block_sizes}")
    print(f"  Num blocks  : {args.num_blocks}")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  L+S rank    : {args.rank}")
    print(f"  Warmup      : {args.warmup}   Rounds: {args.rounds}")
    print()

    # Header
    print(f"{'blk':>5} {'nblk':>5} {'batch':>6} {'dim':>7} | "
          f"{'Path':<20s} {'Fwd ms':>9} {'Bwd ms':>9} {'F+B ms':>9} "
          f"{'Peak MB':>9} {'Params':>10} {'vs Dense':>9}")
    print("-" * W)

    for block_size, num_blocks, batch in itertools.product(
        args.block_sizes, args.num_blocks, args.batch_sizes
    ):
        dim = block_size * num_blocks
        if dim > 65536:
            continue

        x = torch.randn(batch, dim, device=device, dtype=torch.float32)

        # Collect results for this config to compute speedups
        results = {}

        # 1. Dense baseline
        def _build_dense():
            return nn.Linear(dim, dim, bias=True, device=device).train()
        r = _try_benchmark("Dense", _build_dense, x, args.warmup, args.rounds)
        if r:
            results["Dense"] = r

        # 2. Monarch — Fused Triton
        def _build_fused():
            m = MonarchLinear.from_uniform_blocks(
                dim, dim, num_blocks=num_blocks, bias=True, seed=0,
            ).to(device).train()
            return _ForwardWrapper(m, use_fused=True)
        r = _try_benchmark("Fused", _build_fused, x, args.warmup, args.rounds)
        if r:
            results["Fused"] = r

        # 3. Monarch — BMM
        def _build_bmm():
            m = MonarchLinear.from_uniform_blocks(
                dim, dim, num_blocks=num_blocks, bias=True, seed=0,
            ).to(device).train()
            return _ForwardWrapper(m, use_fused=False, use_views=False)
        r = _try_benchmark("BMM", _build_bmm, x, args.warmup, args.rounds)
        if r:
            results["BMM"] = r

        # 4. Monarch — Views
        def _build_views():
            m = MonarchLinear.from_uniform_blocks(
                dim, dim, num_blocks=num_blocks, bias=True, seed=0,
            ).to(device).train()
            return _ForwardWrapper(m, use_fused=False, use_views=True)
        r = _try_benchmark("Views", _build_views, x, args.warmup, args.rounds)
        if r:
            results["Views"] = r

        # 5. Monarch — Auto (default dispatch: picks Fused or BMM automatically)
        def _build_auto():
            m = MonarchLinear.from_uniform_blocks(
                dim, dim, num_blocks=num_blocks, bias=True, seed=0,
            ).to(device).train()
            return _ForwardWrapper(m)  # use_fused=None (default auto-detect)
        r = _try_benchmark("Auto", _build_auto, x, args.warmup, args.rounds)
        if r:
            results["Auto"] = r

        # 6. LSLinear
        def _build_ls():
            return LSLinear.from_uniform_blocks(
                dim, dim, num_blocks=num_blocks, rank=args.rank,
                bias=True, seed=0,
            ).to(device).train()
        r = _try_benchmark("L+S", _build_ls, x, args.warmup, args.rounds)
        if r:
            results["L+S"] = r

        # Print results for this config
        dense_total = results.get("Dense", {}).get("total_ms", float("inf"))
        first = True
        for path_name in ["Dense", "Fused", "BMM", "Auto", "L+S"]:
            if path_name not in results:
                continue
            r = results[path_name]
            speedup = dense_total / r["total_ms"] if r["total_ms"] > 0 else 0

            prefix = (f"{block_size:>5} {num_blocks:>5} {batch:>6} {dim:>7} | "
                      if first else f"{'':>5} {'':>5} {'':>6} {'':>7} | ")
            first = False

            print(f"{prefix}{path_name:<20s} "
                  f"{r['fwd_ms']:>9.3f} {r['bwd_ms']:>9.3f} {r['total_ms']:>9.3f} "
                  f"{r['peak_mb']:>9.1f} {r['stored_params']:>10,} "
                  f"{speedup:>8.2f}x")

            if log:
                log.log("kernel_bench",
                        block_size=block_size, num_blocks=num_blocks,
                        batch_size=batch, dim=dim, path=path_name,
                        stored_params=r["stored_params"],
                        device=str(device),
                        gpu=(torch.cuda.get_device_name()
                             if torch.cuda.is_available() else ""),
                        **{k: v for k, v in r.items() if k != "stored_params"})

        if results:
            print()

        del x
        safe_cleanup()

    print(f"{'=' * W}")
    print("Done.")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kernel-level benchmark for MonarchLinear & LSLinear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--block-sizes", nargs="+", type=int,
                        default=[32, 64, 128, 256],
                        help="Block sizes to sweep (default: 32 64 128 256)")
    parser.add_argument("--num-blocks", nargs="+", type=int,
                        default=[4, 8, 16, 32],
                        help="Number of blocks to sweep (default: 4 8 16 32)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[32, 128, 512],
                        help="Batch sizes to sweep (default: 32 128 512)")
    parser.add_argument("--rank", type=int, default=16,
                        help="L+S rank (default: 16)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--rounds", type=int, default=15,
                        help="Timed iterations — median reported (default: 15)")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to JSONL log file (default: no logging)")
    args = parser.parse_args()
    run_kernel_bench(args)


if __name__ == "__main__":
    main()
