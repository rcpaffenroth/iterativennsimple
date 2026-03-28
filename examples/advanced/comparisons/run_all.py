#!/usr/bin/env python
"""Run the complete comparison suite.  All results go to one JSONL log.

Usage:
    uv run examples/advanced/comparisons/run_all.py --output-dir results/
    uv run examples/advanced/comparisons/run_all.py --skip-training --skip-pareto
    uv run examples/advanced/comparisons/run_all.py --quick --output-dir results/
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(label, cmd):
    """Run a subprocess with live output.  Returns True on success."""
    print(f"\n{'#' * 80}")
    print(f"# {label}")
    print(f"# CMD: {' '.join(cmd)}")
    print(f"{'#' * 80}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} completed in {elapsed:.0f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run the full comparison suite")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller sweeps and fewer epochs")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training comparisons (just benchmarks)")
    parser.add_argument("--skip-pareto", action="store_true",
                        help="Skip Pareto sweep (longest running)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for log and plots (default: results/)")
    args = parser.parse_args()

    py = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "benchmark_log.jsonl")
    plot_dir = os.path.join(args.output_dir, "plots")
    results = []

    # Quick-mode overrides
    quick_bench = ["--start-dim", "512", "--factor", "4",
                   "--warmup", "3", "--rounds", "5"] if args.quick else []

    # ── 1. Model-level scaling (scales until OOM) ─────────────────
    cmd = [py, "bench_scaling.py", "--level", "model", "--log", log_path] + quick_bench
    results.append(("Model Scaling", run("1/4  Model-Level Scaling", cmd)))

    # ── 2. Layer-level scaling (scales until OOM) ─────────────────
    cmd = [py, "bench_scaling.py", "--level", "layer", "--log", log_path] + quick_bench
    results.append(("Layer Scaling", run("2/4  Layer-Level Scaling", cmd)))

    # ── 3. Training comparison (optional) ─────────────────────────
    if not args.skip_training:
        cmd = [py, "compare_training.py", "--log", log_path]
        if args.quick:
            cmd += ["--epochs", "5", "--hidden-dim", "512",
                    "--models", "monarch_inn", "masked_inn", "ls_inn",
                    "lstm", "gru", "transformer"]
        else:
            cmd += ["--epochs", "20", "--hidden-dim", "1024",
                    "--num-blocks", "16", "--rank", "16", "--batch-size", "128"]
        results.append(("Training", run("3/4  Training Comparison", cmd)))
    else:
        print("\n  [SKIP] Training comparison")

    # ── 4. Pareto frontier (optional) ─────────────────────────────
    if not args.skip_training and not args.skip_pareto:
        cmd = [py, "compare_pareto.py", "--log", log_path]
        if args.quick:
            cmd += ["--epochs", "5", "--quick"]
        else:
            cmd += ["--epochs", "15"]
        results.append(("Pareto", run("4/4  Pareto Frontier", cmd)))
    else:
        print("\n  [SKIP] Pareto frontier")

    # ── 5. Generate plots from the unified log ────────────────────
    cmd = [py, "plot_results.py", "--log", log_path, "--output", plot_dir]
    run("Plotting", cmd)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(" SUITE COMPLETE")
    print(f"{'=' * 60}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':>4s}  {name}")
    print(f"\n  Log   : {os.path.abspath(log_path)}")
    print(f"  Plots : {os.path.abspath(plot_dir)}/")
    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail:
        print(f"\n  {n_fail} step(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
