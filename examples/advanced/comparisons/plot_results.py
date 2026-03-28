#!/usr/bin/env python
"""Generate publication-quality plots from the unified JSONL benchmark log.

Decoupled from benchmarking: re-run plots without re-running experiments.

Produces three plots:
  1. Stored vs Surface Parameters  — the factored-efficiency plot
  2. Stored Parameters vs Memory   — memory overhead beyond raw storage
  3. Compute Time vs Dimension     — wall-clock scaling

Usage:
    uv run examples/advanced/comparisons/plot_results.py --log results.jsonl --output plots/
    uv run examples/advanced/comparisons/plot_results.py --log results.jsonl --kind model_scaling
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import ResultLog, setup_plot_style, style_for


# ── Helpers ───────────────────────────────────────────────────────────

def _name_key(record):
    """Extract the series name from a record (model or layer)."""
    return record.get("model") or record.get("layer", "unknown")


def _group_by_name(records):
    """Group records by model/layer name, collecting lists of metrics."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        name = _name_key(r)
        for key in ("dim", "stored_params", "surface_params",
                     "fwd_ms", "total_ms", "peak_mb"):
            if key in r:
                groups[name][key].append(r[key])
    return groups


def _save(fig, path_stem):
    """Save figure as both PNG and PDF."""
    fig.savefig(path_stem + ".png")
    fig.savefig(path_stem + ".pdf")
    plt.close(fig)
    print(f"  Saved {path_stem}.png / .pdf")


# ── Plot 1: Stored vs Surface Parameters ─────────────────────────────

def plot_stored_vs_surface(groups, output_dir):
    """Parameter efficiency: stored (compressed) vs surface (reached) params."""
    fig, ax = plt.subplots()

    for name in sorted(groups):
        g = groups[name]
        if "dim" not in g or "stored_params" not in g:
            continue
        color, marker, label = style_for(name)

        dims = g["dim"]
        stored = g["stored_params"]
        surface = g.get("surface_params", stored)

        # Surface — solid line, filled marker
        ax.loglog(dims, surface, color=color, marker=marker, markersize=5,
                  linewidth=1.5, label=f"{label} (surface)")

        # Stored — dashed line, open marker
        ax.loglog(dims, stored, color=color, marker=marker, markersize=5,
                  linewidth=1.2, linestyle="--", markerfacecolor="none",
                  label=f"{label} (stored)")

    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Parameter Count")
    ax.set_title("Stored vs Surface Parameters")

    # De-duplicate legend: only show entries where stored ≠ surface
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", ncol=1, frameon=True,
              edgecolor="#cccccc", fancybox=False)

    fig.tight_layout(pad=0.5)
    _save(fig, os.path.join(output_dir, "params_stored_vs_surface"))


# ── Plot 2: Stored Parameters vs Memory Footprint ────────────────────

def plot_params_vs_memory(groups, output_dir):
    """Memory footprint vs stored parameter count."""
    fig, ax = plt.subplots()

    for name in sorted(groups):
        g = groups[name]
        if "stored_params" not in g or "peak_mb" not in g:
            continue
        color, marker, label = style_for(name)

        ax.loglog(g["stored_params"], g["peak_mb"],
                  color=color, marker=marker, markersize=5,
                  linewidth=1.5, label=label)

    ax.set_xlabel("Stored Parameters")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Parameter Count vs Memory Footprint")
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout(pad=0.5)
    _save(fig, os.path.join(output_dir, "params_vs_memory"))


# ── Plot 3: Compute Time vs Dimension ────────────────────────────────

def plot_compute_time(groups, output_dir):
    """Forward+backward compute time scaling."""
    fig, ax = plt.subplots()

    for name in sorted(groups):
        g = groups[name]
        if "dim" not in g or "total_ms" not in g:
            continue
        color, marker, label = style_for(name)

        ax.loglog(g["dim"], g["total_ms"],
                  color=color, marker=marker, markersize=5,
                  linewidth=1.5, label=label)

    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Forward + Backward Time (ms)")
    ax.set_title("Compute Time Scaling")
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout(pad=0.5)
    _save(fig, os.path.join(output_dir, "compute_time"))


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSONL log")
    parser.add_argument("--log", required=True, help="Path to JSONL log file")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    parser.add_argument("--kind", default=None,
                        choices=["model_scaling", "layer_scaling"],
                        help="Filter by record kind (default: all scaling records)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    setup_plot_style()

    log = ResultLog(args.log)

    # Collect records — combine model_scaling and layer_scaling if no filter
    if args.kind:
        records = log.read(kind=args.kind)
    else:
        records = log.read(kind="model_scaling") + log.read(kind="layer_scaling")

    if not records:
        print(f"No records found in {args.log}")
        return

    groups = _group_by_name(records)
    print(f"Loaded {len(records)} records across {len(groups)} architectures")

    plot_stored_vs_surface(groups, args.output)
    plot_params_vs_memory(groups, args.output)
    plot_compute_time(groups, args.output)

    print(f"\nAll plots saved to {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
