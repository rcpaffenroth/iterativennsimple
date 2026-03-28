"""Shared utilities for the benchmark and comparison suite.

Single source of truth for: styling, timing, parameter counting,
dataset loading, and result logging.  Every other script imports from here.
"""

import gc
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import torch
import torch.nn as nn

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ── Style constants (single source of truth) ─────────────────────────

MODEL_STYLES = {
    "monarch_inn":  dict(color="#1f77b4", marker="o",  label="Monarch INN"),
    "masked_inn":   dict(color="#2ca02c", marker="D",  label="Masked INN"),
    "ls_inn":       dict(color="#17becf", marker="P",  label="L+S INN"),
    "lstm":         dict(color="#d62728", marker="s",  label="LSTM"),
    "gru":          dict(color="#ff7f0e", marker="^",  label="GRU"),
    "rnn_tanh":     dict(color="#e377c2", marker="v",  label="RNN (tanh)"),
    "transformer":  dict(color="#9467bd", marker="X",  label="Transformer"),
    "mlp_flat":     dict(color="#8c564b", marker="<",  label="MLP (flat)"),
}

LAYER_STYLES = {
    "nn.Linear":           dict(color="#d62728", marker="s",  label="Dense"),
    "LSTM(1-step)":        dict(color="#ff7f0e", marker="^",  label="LSTM"),
    "Transformer(1-step)": dict(color="#e377c2", marker="v",  label="Transformer"),
}

MONARCH_BLUE = "#1f77b4"
MASKED_GREEN = "#2ca02c"
LS_CYAN      = "#17becf"


def style_for(name):
    """Return (color, marker, label) for a model or layer name."""
    if name in MODEL_STYLES:
        s = MODEL_STYLES[name]
        return s["color"], s["marker"], s["label"]
    if name in LAYER_STYLES:
        s = LAYER_STYLES[name]
        return s["color"], s["marker"], s["label"]
    if name.startswith("Monarch"):
        return MONARCH_BLUE, "o", name
    if name.startswith("Masked"):
        return MASKED_GREEN, "D", name
    if name.startswith("L+S"):
        return LS_CYAN, "P", name
    return "#333333", "x", name


def setup_plot_style():
    """Configure matplotlib for publication-quality output."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (5.5, 4.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ── CUDA helpers ──────────────────────────────────────────────────────

def sync_cuda():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_cleanup(*objs):
    """Delete objects, run GC, and free CUDA cache."""
    for o in objs:
        del o
    # Two GC passes: first collects cycles, second catches anything freed by __del__
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ── Benchmarking ──────────────────────────────────────────────────────

def benchmark(model, x, warmup=5, rounds=15):
    """Benchmark forward + backward, returning median timings and peak memory.

    Returns dict with keys: fwd_ms, bwd_ms, total_ms, peak_mb.
    """
    # Warmup
    for _ in range(warmup):
        y = model(x)
        y.sum().backward()

    sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fwd_times, bwd_times = [], []
    for _ in range(rounds):
        sync_cuda()
        t0 = time.perf_counter()
        y = model(x)
        sync_cuda()
        t1 = time.perf_counter()
        y.sum().backward()
        sync_cuda()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1e3)
        bwd_times.append((t2 - t1) * 1e3)

    fwd = statistics.median(fwd_times)
    bwd = statistics.median(bwd_times)
    peak = (torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available() else 0.0)

    return dict(fwd_ms=fwd, bwd_ms=bwd, total_ms=fwd + bwd, peak_mb=peak)


# ── Parameter counting ────────────────────────────────────────────────

def count_stored(model):
    """Count stored (compressed) parameters — the actual floats in memory.

    Uses number_of_trainable_parameters() when available, which returns:
      - MonarchLinear: block entries only (d²/k)
      - MaskedLinear: nnz(mask) entries
      - LSLinear: sparse blocks + A + B factors
    """
    # Re-use the existing _count_params logic from models.py
    if hasattr(model, "number_of_trainable_parameters"):
        n = model.number_of_trainable_parameters()
        return int(n) if hasattr(n, "item") else int(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _leaf_surface(module):
    """Surface params for a single leaf module (non-recursive)."""
    if isinstance(module, LSLinear):
        # Low-rank L=AB is dense → covers full in×out surface
        # S adds sparse corrections on the same surface
        s = module.in_features * module.out_features
        if module.bias is not None:
            s += module.out_features
        return s
    if isinstance(module, MonarchLinear):
        # Sparse: surface = stored (only block entries are non-zero)
        return module.number_of_trainable_parameters()
    if isinstance(module, MaskedLinear):
        # Sparse: surface = nnz(mask) + bias
        return module.number_of_trainable_parameters()
    if isinstance(module, nn.Linear):
        # Dense: surface = in×out + bias
        s = module.in_features * module.out_features
        if module.bias is not None:
            s += module.out_features
        return s
    if isinstance(module, nn.RNNBase):
        # LSTM/GRU/RNN: dense gate weights, no factorization → surface = stored
        return int(sum(p.numel() for p in module.parameters()))
    if isinstance(module, nn.TransformerEncoderLayer):
        # Dense attention + FFN, no factorization → surface = stored
        return int(sum(p.numel() for p in module.parameters()))
    return None  # Not a linear-like layer


def count_surface(model):
    """Count surface parameters — the matrix entries actually reached.

    For dense/sparse-only layers, surface ≈ stored.
    For LSLinear, surface = full in×out (low-rank L=AB is dense).
    """
    total = 0
    counted_modules = set()

    for module in model.modules():
        if id(module) in counted_modules:
            continue
        s = _leaf_surface(module)
        if s is not None:
            total += s
            counted_modules.add(id(module))
            # Don't double-count children (e.g., LSLinear contains a MonarchLinear)
            for child in module.modules():
                counted_modules.add(id(child))

    # Fall back: if no linear-like leaves found (e.g., RNN internals),
    # count via stored params (dense gates, no factorization).
    if total == 0:
        total = count_stored(model)

    return total


# ── Dataset loading ───────────────────────────────────────────────────

def load_dataset(name, step_size, local=False, val_frac=1 / 7, seed=42):
    """Load dataset as sequential chunks, split into train/val."""
    import numpy as np
    from generatedata.load_data import load_data_as_sequence

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
        label_dim=labels.shape[1],
        input_dim=X.shape[2],
        seq_len=X.shape[1],
    )


# ── Unified JSONL logger ─────────────────────────────────────────────

class ResultLog:
    """Append-only JSONL logger.  All benchmark scripts write to one file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, kind, **data):
        """Append one JSON line with a kind tag and timestamp."""
        record = {"kind": kind, "ts": datetime.now(timezone.utc).isoformat()}
        record.update(data)
        # Ensure all values are JSON-serializable (torch tensors → Python scalars)
        for k, v in record.items():
            if hasattr(v, "item"):
                record[k] = v.item()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self, kind=None):
        """Read back records, optionally filtered by kind."""
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if kind is None or r.get("kind") == kind:
                    records.append(r)
        return records
