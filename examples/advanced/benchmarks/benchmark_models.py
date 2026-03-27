#!/usr/bin/env python
"""End-to-end model training throughput benchmarks.

Builds full Sequential2D models and compares:
  - Monarch INN     (MonarchLinear — sparse structured)
  - MaskedLinear INN (same sparsity pattern via to_MaskedLinear — sparse unstructured)
  - L+S INN         (LSLinear — Robust PCA: low-rank + sparse Monarch)
  - LSTM baseline

NOTE: Dense MLP as an INN makes no architectural sense — an INN should use
sparse layers.  Dense is included only as a non-INN feed-forward baseline
(single pass, no iteration).

Default dimensions target 2-3M trainable parameters for the INN models.

Usage
-----
    uv run examples/advanced/benchmarks/benchmark_models.py
    uv run examples/advanced/benchmarks/benchmark_models.py --device cpu
    uv run examples/advanced/benchmarks/benchmark_models.py --dim 2048 --seq-len 64 --batch-size 128
    uv run examples/advanced/benchmarks/benchmark_models.py --csv results.csv
"""
import csv
import statistics
import sys
import time
from dataclasses import dataclass

import click
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_ms(fn, warmup: int, rounds: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(rounds):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _find_num_blocks(dim: int, desired: int) -> int:
    """Largest divisor of dim that is <= desired."""
    nb = desired
    while nb > 1 and dim % nb != 0:
        nb -= 1
    return nb


def build_monarch_model(sizes: list[int], num_blocks: int) -> Sequential2D:
    """Sequential2D using MonarchLinear blocks (sparse structured INN)."""
    n = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=sizes[0], out_features=sizes[0])
    for i in range(n - 1):
        nb = _find_num_blocks(min(sizes[i], sizes[i + 1]), num_blocks)
        act = nn.ReLU() if i > 0 else None
        layer = MonarchLinear.from_uniform_blocks(
            sizes[i], sizes[i + 1], num_blocks=nb, bias=True, seed=42
        )
        modules = [act, layer] if act else [layer]
        blocks[i][i + 1] = Sequential1D(
            nn.Sequential(*modules),
            in_features=sizes[i],
            out_features=sizes[i + 1],
        )
    return Sequential2D(sizes, sizes, blocks)


def build_masked_model(sizes: list[int], num_blocks: int) -> Sequential2D:
    """Sequential2D using MaskedLinear blocks (same sparsity as Monarch via to_MaskedLinear)."""
    n = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=sizes[0], out_features=sizes[0])
    for i in range(n - 1):
        nb = _find_num_blocks(min(sizes[i], sizes[i + 1]), num_blocks)
        act = nn.ReLU() if i > 0 else None
        monarch = MonarchLinear.from_uniform_blocks(
            sizes[i], sizes[i + 1], num_blocks=nb, bias=True, seed=42
        )
        masked = monarch.to_MaskedLinear()
        modules = [act, masked] if act else [masked]
        blocks[i][i + 1] = Sequential1D(
            nn.Sequential(*modules),
            in_features=sizes[i],
            out_features=sizes[i + 1],
        )
    return Sequential2D(sizes, sizes, blocks)


def build_ls_model(sizes: list[int], num_blocks: int, rank: int) -> Sequential2D:
    """Sequential2D using LSLinear blocks (L+S = Robust PCA INN)."""
    n = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=sizes[0], out_features=sizes[0])
    for i in range(n - 1):
        nb = _find_num_blocks(min(sizes[i], sizes[i + 1]), num_blocks)
        act = nn.ReLU() if i > 0 else None
        layer = LSLinear.from_uniform_blocks(
            sizes[i], sizes[i + 1], num_blocks=nb, rank=rank, bias=True, seed=42
        )
        modules = [act, layer] if act else [layer]
        blocks[i][i + 1] = Sequential1D(
            nn.Sequential(*modules),
            in_features=sizes[i],
            out_features=sizes[i + 1],
        )
    return Sequential2D(sizes, sizes, blocks)


class LSTMModel(nn.Module):
    """Sequence-processing LSTM baseline."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :])


class IterativeMapModel(nn.Module):
    """Wrapper that applies a Sequential2D map over a sequence."""
    def __init__(self, map_module: Sequential2D, input_dim: int, iterations: int = 4):
        super().__init__()
        self.map = map_module
        self.input_dim = input_dim
        self.iterations = iterations

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        state = torch.zeros(B, self.map.in_features, device=x_seq.device)
        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(self.iterations):
                state = self.map(state)
        return state[:, -x_seq.shape[2]:]


# ---------------------------------------------------------------------------
# Dense MLP (non-INN, single-pass baseline)
# ---------------------------------------------------------------------------

class DenseMLPBaseline(nn.Module):
    """Standard dense MLP — NOT iterated. Just a feed-forward baseline."""
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # Use last timestep (non-sequential baseline)
        return self.net(x_seq[:, -1, :])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    params: int
    fwd_ms: float
    fwd_bwd_ms: float
    batch_size: int

    @property
    def fwd_throughput(self) -> float:
        return self.batch_size / (self.fwd_ms / 1000.0)

    @property
    def fwd_bwd_throughput(self) -> float:
        return self.batch_size / (self.fwd_bwd_ms / 1000.0)


# ---------------------------------------------------------------------------
# Peak memory (GPU only)
# ---------------------------------------------------------------------------

def peak_memory_mb(fn, device: str) -> float | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--dim", default=1536, show_default=True,
              help="Hidden state dimension (target ~2-3M params for INN models).")
@click.option("--seq-len", default=32, show_default=True,
              help="Sequence length for iterative models and LSTM.")
@click.option("--batch-size", default=64, show_default=True,
              help="Batch size.")
@click.option("--num-blocks", default=4, show_default=True,
              help="Number of Monarch diagonal blocks.")
@click.option("--rank", default=64, show_default=True,
              help="Rank for LSLinear low-rank component.")
@click.option("--iterations", default=4, show_default=True,
              help="Map iterations per time-step (INN models only).")
@click.option("--warmup", default=5, show_default=True,
              help="Warmup iterations before timing.")
@click.option("--rounds", default=20, show_default=True,
              help="Timed iterations (median reported).")
@click.option("--device", default="auto", show_default=True,
              help="Device: 'cpu', 'cuda', or 'auto'.")
@click.option("--csv", "csv_path", default=None,
              help="Optional CSV output path.")
def main(dim, seq_len, batch_size, num_blocks, rank, iterations, warmup, rounds, device, csv_path):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4-slot state: [input | hidden1 | hidden2 | output]
    sizes = [dim, dim, dim, dim]
    input_dim = dim

    print(f"Device: {device}  |  dim={dim}  seq_len={seq_len}  "
          f"batch_size={batch_size}  num_blocks={num_blocks}  rank={rank}")
    print(f"State: [{' + '.join(str(s) for s in sizes)}] = {sum(sizes)} total")
    print()

    # Build all models
    monarch_map = build_monarch_model(sizes, num_blocks).to(device)
    masked_map  = build_masked_model(sizes, num_blocks).to(device)
    ls_map      = build_ls_model(sizes, num_blocks, rank).to(device)
    lstm_model  = LSTMModel(input_dim, dim, input_dim).to(device)
    dense_mlp   = DenseMLPBaseline(input_dim, [dim, dim], input_dim).to(device)

    models = [
        ("Monarch INN",      IterativeMapModel(monarch_map, input_dim, iterations).to(device)),
        ("MaskedLinear INN", IterativeMapModel(masked_map,  input_dim, iterations).to(device)),
        ("L+S INN",          IterativeMapModel(ls_map,      input_dim, iterations).to(device)),
        ("LSTM",             lstm_model),
        ("Dense MLP (1-pass)", dense_mlp),
    ]

    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    target = torch.randn(batch_size, input_dim, device=device)
    criterion = nn.MSELoss()

    results: list[ModelResult] = []

    print(f"{'Model':<24} {'Params':>10}  {'Fwd (ms)':>9}  {'Fwd+Bwd (ms)':>13}  "
          f"{'Fwd samp/s':>11}  {'Fwd+Bwd samp/s':>15}")
    print("-" * 95)

    for name, model in models:
        model.train()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def fwd():
            with torch.no_grad():
                model(x)

        def fwd_bwd():
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()

        fwd_ms = measure_ms(fwd, warmup, rounds, device)
        fwd_bwd_ms = measure_ms(fwd_bwd, warmup, rounds, device)

        r = ModelResult(
            name=name,
            params=params,
            fwd_ms=fwd_ms,
            fwd_bwd_ms=fwd_bwd_ms,
            batch_size=batch_size,
        )
        results.append(r)

        mem_str = ""
        mem = peak_memory_mb(fwd, device)
        if mem is not None:
            mem_str = f"  peak_mem={mem:.0f}MB"

        print(
            f"{name:<24} {params:>10,}  {fwd_ms:>9.2f}  {fwd_bwd_ms:>13.2f}  "
            f"{r.fwd_throughput:>11,.0f}  {r.fwd_bwd_throughput:>15,.0f}"
            f"{mem_str}"
        )

    print()

    # Parameter efficiency summary
    monarch_params = results[0].params
    print("Parameter counts (relative to Monarch INN):")
    for r in results:
        ratio = r.params / monarch_params
        print(f"  {r.name:<24}  {r.params:>10,}  ({ratio:.2%} of Monarch)")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "params", "fwd_ms", "fwd_bwd_ms",
                "fwd_throughput", "fwd_bwd_throughput",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "name": r.name,
                    "params": r.params,
                    "fwd_ms": f"{r.fwd_ms:.4f}",
                    "fwd_bwd_ms": f"{r.fwd_bwd_ms:.4f}",
                    "fwd_throughput": f"{r.fwd_throughput:.0f}",
                    "fwd_bwd_throughput": f"{r.fwd_bwd_throughput:.0f}",
                })
        print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
