#!/usr/bin/env python
"""Layer-level microbenchmarks: forward + backward timing for individual layers.

Compares nn.Linear, MonarchLinear, MaskedLinear (same sparsity), and LSLinear
at various sizes.  Default dims target 2-3M params at the model level.

Usage
-----
    uv run examples/advanced/benchmarks/benchmark_layers.py
    uv run examples/advanced/benchmarks/benchmark_layers.py --device cpu
    uv run examples/advanced/benchmarks/benchmark_layers.py --dims 1024,1536,2048 --batch-size 128
    uv run examples/advanced/benchmarks/benchmark_layers.py --csv results.csv
"""
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import click
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _timer_fn(device: str) -> Callable[[], float]:
    """Return a zero-argument callable that returns current time in seconds."""
    if device == "cuda" and torch.cuda.is_available():
        def t():
            torch.cuda.synchronize()
            return time.perf_counter()
    else:
        def t():
            return time.perf_counter()
    return t


def measure(
    fn: Callable,
    warmup: int = 5,
    rounds: int = 20,
    device: str = "cpu",
) -> float:
    """Return median wall-clock time in milliseconds over ``rounds`` calls."""
    timer = _timer_fn(device)

    for _ in range(warmup):
        fn()

    times = []
    for _ in range(rounds):
        t0 = timer()
        fn()
        t1 = timer()
        times.append((t1 - t0) * 1000.0)

    return statistics.median(times)


# ---------------------------------------------------------------------------
# Layer factories
# ---------------------------------------------------------------------------

def _find_num_blocks(dim: int, desired: int) -> int:
    """Largest divisor of dim that is <= desired."""
    nb = desired
    while nb > 1 and dim % nb != 0:
        nb -= 1
    return nb


def make_linear(dim: int, bias: bool = True, device: str = "cpu") -> nn.Linear:
    return nn.Linear(dim, dim, bias=bias).to(device)


def make_monarch(dim: int, num_blocks: int, bias: bool = True, device: str = "cpu") -> MonarchLinear:
    nb = _find_num_blocks(dim, num_blocks)
    return MonarchLinear.from_uniform_blocks(
        dim, dim, num_blocks=nb, bias=bias, seed=42
    ).to(device)


def make_masked(dim: int, num_blocks: int, bias: bool = True, device: str = "cpu") -> MaskedLinear:
    """MaskedLinear with same sparsity pattern as Monarch (via to_MaskedLinear)."""
    nb = _find_num_blocks(dim, num_blocks)
    monarch = MonarchLinear.from_uniform_blocks(
        dim, dim, num_blocks=nb, bias=bias, seed=42
    )
    return monarch.to_MaskedLinear().to(device)


def make_ls(dim: int, num_blocks: int, rank: int, bias: bool = True, device: str = "cpu") -> LSLinear:
    nb = _find_num_blocks(dim, num_blocks)
    return LSLinear.from_uniform_blocks(
        dim, dim, num_blocks=nb, rank=rank, bias=bias, seed=42
    ).to(device)


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    dim: int
    batch_size: int
    params: int
    fwd_ms: float
    bwd_ms: float

    @property
    def total_ms(self) -> float:
        return self.fwd_ms + self.bwd_ms

    @property
    def throughput(self) -> float:
        """Samples per second (forward+backward)."""
        return self.batch_size / (self.total_ms / 1000.0)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def benchmark_layer(
    label: str,
    layer: nn.Module,
    dim: int,
    batch_size: int,
    warmup: int,
    rounds: int,
    device: str,
) -> Result:
    layer.eval()
    x = torch.randn(batch_size, dim, device=device)

    # --- forward ---
    def fwd():
        with torch.no_grad():
            layer(x)

    fwd_ms = measure(fwd, warmup=warmup, rounds=rounds, device=device)

    # --- forward + backward ---
    x_grad = x.detach().requires_grad_(True)

    def fwd_bwd():
        y = layer(x_grad)
        y.sum().backward()
        if x_grad.grad is not None:
            x_grad.grad = None

    bwd_ms = measure(fwd_bwd, warmup=warmup, rounds=rounds, device=device) - fwd_ms

    params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    return Result(
        label=label,
        dim=dim,
        batch_size=batch_size,
        params=params,
        fwd_ms=fwd_ms,
        bwd_ms=max(bwd_ms, 0.0),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--dims", default="1024,1536,2048", show_default=True,
              help="Comma-separated list of square layer dimensions to benchmark.")
@click.option("--batch-size", default=64, show_default=True,
              help="Batch size for all benchmarks.")
@click.option("--num-blocks", default=4, show_default=True,
              help="Number of Monarch diagonal blocks.")
@click.option("--rank", default=64, show_default=True,
              help="Low-rank component rank for LSLinear.")
@click.option("--warmup", default=5, show_default=True,
              help="Number of warmup iterations before timing.")
@click.option("--rounds", default=20, show_default=True,
              help="Number of timed iterations (median reported).")
@click.option("--device", default="auto", show_default=True,
              help="Device: 'cpu', 'cuda', or 'auto' (cuda if available).")
@click.option("--csv", "csv_path", default=None,
              help="Optional path to write results as CSV.")
def main(dims, batch_size, num_blocks, rank, warmup, rounds, device, csv_path):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dim_list = [int(d.strip()) for d in dims.split(",")]

    print(f"Device: {device}  |  batch_size={batch_size}  |  "
          f"num_blocks={num_blocks}  rank={rank}  "
          f"warmup={warmup}  rounds={rounds}")
    print()

    results: list[Result] = []

    for dim in dim_list:
        print(f"── dim={dim} " + "─" * 60)

        layers = [
            ("nn.Linear (dense)", make_linear(dim, device=device)),
            (f"Monarch(k={num_blocks})", make_monarch(dim, num_blocks, device=device)),
            (f"MaskedLinear(k={num_blocks})", make_masked(dim, num_blocks, device=device)),
            (f"L+S(k={num_blocks},r={rank})", make_ls(dim, num_blocks, rank, device=device)),
        ]

        for label, layer in layers:
            r = benchmark_layer(
                label=label,
                layer=layer,
                dim=dim,
                batch_size=batch_size,
                warmup=warmup,
                rounds=rounds,
                device=device,
            )
            results.append(r)
            print(
                f"  {label:<30}  params={r.params:>10,}  "
                f"fwd={r.fwd_ms:6.2f}ms  bwd={r.bwd_ms:6.2f}ms  "
                f"total={r.total_ms:6.2f}ms  "
                f"throughput={r.throughput:,.0f} samp/s"
            )
        print()

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "label", "dim", "batch_size", "params",
                "fwd_ms", "bwd_ms", "total_ms", "throughput"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "label": r.label,
                    "dim": r.dim,
                    "batch_size": r.batch_size,
                    "params": r.params,
                    "fwd_ms": f"{r.fwd_ms:.4f}",
                    "bwd_ms": f"{r.bwd_ms:.4f}",
                    "total_ms": f"{r.total_ms:.4f}",
                    "throughput": f"{r.throughput:.0f}",
                })
        print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
