#!/usr/bin/env python
"""Pareto frontier analysis: accuracy vs parameter count.

Trains all models with MULTIPLE configurations (varying hidden dim, blocks,
rank) to find the Pareto-optimal frontier of accuracy vs. parameters.

This is the key academic plot: for a given parameter budget, which
architecture achieves the best accuracy?

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_pareto.py --epochs 15
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_pareto.py --epochs 5 --quick
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import ResultLog, count_stored, count_surface, load_dataset, safe_cleanup
from models import build_model


# ── Training ──────────────────────────────────────────────────────────

def quick_train(model, train_loader, val_loader, device, epochs, lr=1e-3):
    """Lean training loop returning best val acc."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = 0.0
    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total += len(y)
        best = max(best, correct / total * 100)
    return best


# ── Configurations ────────────────────────────────────────────────────

def get_configs(quick=False):
    """Return list of (model_name, kwargs) to evaluate."""
    configs = []

    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 8, 16, 32, 64]:
            if dim % k == 0:
                configs.append(("monarch_inn", dict(hidden_dim=dim, num_blocks=k)))

    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 8, 16, 32, 64]:
            if dim % k == 0:
                configs.append(("masked_inn", dict(hidden_dim=dim, num_blocks=k)))

    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        for k in [4, 16, 64]:
            for r in [8, 16, 32]:
                if dim % k == 0:
                    configs.append(("ls_inn", dict(hidden_dim=dim, num_blocks=k, rank=r)))

    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("lstm", dict(hidden_dim=dim)))

    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("gru", dict(hidden_dim=dim)))

    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("rnn_tanh", dict(hidden_dim=dim)))

    for dim in ([128, 256, 512] if quick else [128, 256, 512, 1024]):
        configs.append(("transformer", dict(hidden_dim=dim)))

    for dim in ([256, 512, 1024] if quick else [256, 512, 1024, 2048]):
        configs.append(("mlp_flat", dict(hidden_dim=dim)))

    return configs


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pareto frontier: accuracy vs parameters")
    parser.add_argument("--dataset", default="FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Fewer configurations")
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--log", type=str, default=None,
                        help="Path to unified JSONL log (appends).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = ResultLog(args.log) if args.log else None

    data = load_dataset(args.dataset, step_size=args.step_size, local=args.local)
    input_dim, seq_len, label_dim = data["input_dim"], data["seq_len"], data["label_dim"]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_val"], data["y_val"]),
        batch_size=args.batch_size,
    )

    configs = get_configs(args.quick)

    print(f"{'=' * 100}")
    print(f" Pareto Frontier Analysis — {len(configs)} configurations")
    print(f"{'=' * 100}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print()

    results = []
    print(f"{'#':>3s}  {'Model':<18s} {'Config':<40s} {'Stored':>12s}  {'Surface':>12s}  "
          f"{'Val Acc%':>8s}  {'Time':>6s}")
    print("-" * 110)

    for i, (model_name, kwargs) in enumerate(configs, 1):
        dim = kwargs.get("hidden_dim", 128)
        try:
            torch.manual_seed(args.seed)
            model = build_model(
                model_name, input_dim, dim, label_dim, seq_len,
                iterations=args.iterations, **kwargs,
            ).to(device)
            stored = count_stored(model)
            surface = count_surface(model)

            t0 = time.time()
            acc = quick_train(model, train_loader, val_loader, device, args.epochs, args.lr)
            elapsed = time.time() - t0

            cfg_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"{i:>3d}  {model_name:<18s} {cfg_str:<40s} {stored:>12,}  {surface:>12,}  "
                  f"{acc:>7.2f}%  {elapsed:>5.0f}s")

            row = dict(
                model=model_name, config=cfg_str,
                stored_params=stored, surface_params=surface,
                val_acc=acc, time_s=elapsed, **kwargs,
            )
            results.append(row)

            if log:
                log.log("pareto", **row)

            safe_cleanup(model)

        except Exception as e:
            cfg_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            err = str(e).split("\n")[0][:80]
            print(f"{i:>3d}  {model_name:<18s} {cfg_str:<40s}  ** ERROR: {err} **")
            safe_cleanup()

    if not results:
        print("No configurations completed.")
        return

    # ── Pareto frontier ───────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f" Pareto-Optimal Configurations")
    print(f"{'=' * 100}")

    results.sort(key=lambda r: r["stored_params"])
    pareto = []
    best_acc = -1
    for r in results:
        if r["val_acc"] > best_acc:
            pareto.append(r)
            best_acc = r["val_acc"]

    print(f"{'Model':<18s} {'Config':<40s} {'Stored':>12s}  {'Surface':>12s}  {'Val Acc%':>8s}")
    print("-" * 100)
    for r in pareto:
        print(f"{r['model']:<18s} {r['config']:<40s} {r['stored_params']:>12,}  "
              f"{r['surface_params']:>12,}  {r['val_acc']:>7.2f}%")

    # ── Per-architecture best ─────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f" Best Configuration Per Architecture")
    print(f"{'=' * 100}")
    arch_best = {}
    for r in results:
        name = r["model"]
        if name not in arch_best or r["val_acc"] > arch_best[name]["val_acc"]:
            arch_best[name] = r

    for name, r in sorted(arch_best.items(), key=lambda kv: -kv[1]["val_acc"]):
        print(f"  {name:<18s}  val={r['val_acc']:.2f}%  stored={r['stored_params']:>12,}  "
              f"surface={r['surface_params']:>12,}  ({r['config']})")

    # ── CSV output ────────────────────────────────────────────────
    if args.save_csv and results:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.save_csv}")


if __name__ == "__main__":
    main()
