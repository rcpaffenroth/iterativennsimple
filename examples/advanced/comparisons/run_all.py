#!/usr/bin/env python
"""
Run the full comparison suite in one go.

Usage:
    # Full suite (takes ~30-60 min on GPU)
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/run_all.py

    # Quick mode (smaller sweeps, fewer epochs)
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/run_all.py --quick

    # Skip training (just computational benchmarks)
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/run_all.py --skip-training

    # Save all results to a directory
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/run_all.py --output-dir results/
"""

import argparse
import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(label, cmd):
    """Run a subprocess, printing its output live."""
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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save CSV/JSON results")
    args = parser.parse_args()

    python = sys.executable
    results = []

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    def csv_path(name):
        return os.path.join(args.output_dir, name) if args.output_dir else None

    # ---------------------------------------------------------------
    # 1. Layer-level scaling (fast, ~5 min)
    # ---------------------------------------------------------------
    cmd = [python, "compare_layer_scaling.py"]
    if args.quick:
        cmd += ["--dims", "1024", "4096", "16384",
                "--blocks", "4", "16", "64",
                "--batch-size", "16", "--warmup", "3", "--rounds", "10"]
    else:
        cmd += ["--dims", "1024", "4096", "16384", "65536",
                "--blocks", "4", "16", "64", "256", "1024",
                "--batch-size", "16", "--warmup", "3", "--rounds", "10",
                "--max-dense-mb", "4096"]
    p = csv_path("layer_scaling.csv")
    if p:
        cmd += ["--save-csv", p]
    results.append(("Layer Scaling", run("1/5  Layer-Level Scaling", cmd)))

    # ---------------------------------------------------------------
    # 2. Model throughput comparison (fast, ~5 min)
    # ---------------------------------------------------------------
    cmd = [python, "compare_throughput.py"]
    if args.quick:
        cmd += ["--dims", "1024", "4096",
                "--batch-size", "32", "--warmup", "3", "--rounds", "8"]
    else:
        cmd += ["--dims", "1024", "4096", "16384",
                "--batch-size", "32", "--num-blocks", "16", "--rank", "16",
                "--warmup", "3", "--rounds", "10",
                "--max-dense-mb", "2048"]
    p = csv_path("throughput.csv")
    if p:
        cmd += ["--save-csv", p]
    results.append(("Model Throughput", run("2/5  Model Throughput", cmd)))

    # ---------------------------------------------------------------
    # 3. Scaling analysis across dims (fast, ~5 min)
    # ---------------------------------------------------------------
    cmd = [python, "compare_scaling.py"]
    if args.quick:
        cmd += ["--dims", "1024", "4096",
                "--models", "monarch_inn", "masked_inn", "ls_inn", "lstm", "transformer",
                "--warmup", "2", "--rounds", "5"]
    else:
        cmd += ["--dims", "1024", "2048", "4096", "8192", "16384",
                "--num-blocks", "16", "--rank", "16",
                "--max-dense-mb", "2048"]
    p = csv_path("scaling.csv")
    if p:
        cmd += ["--save-csv", p]
    results.append(("Scaling Analysis", run("3/5  Scaling Analysis", cmd)))

    # ---------------------------------------------------------------
    # 4. Training comparison (medium, ~10-30 min)
    # ---------------------------------------------------------------
    if not args.skip_training:
        cmd = [python, "compare_training.py"]
        if args.quick:
            cmd += ["--epochs", "5", "--hidden-dim", "512",
                    "--num-blocks", "16", "--rank", "16",
                    "--models", "monarch_inn", "masked_inn", "ls_inn",
                    "lstm", "gru", "transformer"]
        else:
            cmd += ["--epochs", "20", "--hidden-dim", "1024",
                    "--num-blocks", "16", "--rank", "16",
                    "--batch-size", "128"]
        p = csv_path("training.json") if args.output_dir else None
        if p:
            cmd += ["--save-json", p]
        results.append(("Training", run("4/5  Training Comparison", cmd)))
    else:
        print("\n  [SKIP] Training comparison")

    # ---------------------------------------------------------------
    # 5. Pareto frontier (slow, ~15-45 min)
    # ---------------------------------------------------------------
    if not args.skip_training and not args.skip_pareto:
        cmd = [python, "compare_pareto.py"]
        if args.quick:
            cmd += ["--epochs", "5", "--quick"]
        else:
            cmd += ["--epochs", "15"]
        p = csv_path("pareto.csv")
        if p:
            cmd += ["--save-csv", p]
        results.append(("Pareto", run("5/5  Pareto Frontier", cmd)))
    else:
        print("\n  [SKIP] Pareto frontier")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f" SUITE COMPLETE")
    print(f"{'=' * 60}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':>4s}  {name}")
    if args.output_dir:
        print(f"\n  Results saved to: {os.path.abspath(args.output_dir)}/")
    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail:
        print(f"\n  {n_fail} test(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
