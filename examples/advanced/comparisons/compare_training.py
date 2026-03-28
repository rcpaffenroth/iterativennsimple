#!/usr/bin/env python
"""Full training comparison across all architectures on real data.

Trains every model architecture on the same dataset and reports:
  - Final train/val accuracy
  - Convergence speed (epochs to reach X% accuracy)
  - Total training time
  - Parameter count (stored + surface) and memory usage

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_training.py
    CUDA_VISIBLE_DEVICES=1 uv run examples/advanced/comparisons/compare_training.py \
        --epochs 50 --models monarch_inn lstm transformer
    uv run examples/advanced/comparisons/compare_training.py --list-datasets
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import (
    ResultLog, count_stored, count_surface, load_dataset, safe_cleanup,
)
from models import build_model, ALL_MODEL_NAMES

from generatedata.load_data import data_names


# ── Training loop ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(X)
    return total_loss / count, correct / count * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(X)
        correct += (logits.argmax(1) == y).sum().item()
        count += len(X)
    return total_loss / count, correct / count * 100


def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3,
                label="model", verbose=True):
    """Train and track metrics per epoch.  Returns results dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_acc = 0.0
    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        best_val_acc = max(best_val_acc, va_acc)
        elapsed = time.time() - t_epoch

        history.append(dict(
            epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss, val_acc=va_acc, epoch_time=elapsed,
        ))

        if verbose:
            print(f"  [{label:>20s}] epoch {epoch:3d}/{epochs}  "
                  f"loss {tr_loss:.4f}  train {tr_acc:5.1f}%  val {va_acc:5.1f}%  "
                  f"({elapsed:.1f}s)")

    total_time = time.time() - t0
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return dict(
        best_val_acc=best_val_acc,
        final_train_acc=history[-1]["train_acc"],
        final_val_acc=history[-1]["val_acc"],
        total_time=total_time,
        peak_mem_mb=peak_mem,
        history=history,
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full training comparison across architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-datasets", action="store_true",
                        help="Print available datasets and exit.")
    parser.add_argument("--local", action="store_true",
                        help="Load from local cache.")
    parser.add_argument("--dataset", default="FashionMNIST_custom_degrees0_45_translate0_0.0_scale0.75_1_randomerasing_0.0")
    parser.add_argument("--models", nargs="+", default=ALL_MODEL_NAMES,
                        choices=ALL_MODEL_NAMES)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save results as JSON.")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to unified JSONL log (appends).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list_datasets:
        for name in sorted(data_names(local=args.local)):
            print(name)
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = ResultLog(args.log) if args.log else None

    # Load data
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

    print(f"{'=' * 80}")
    print(f" Training Comparison — {args.dataset}")
    print(f"{'=' * 80}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name()}")
    print(f"  Samples    : {len(data['y_train']):,} train / {len(data['y_val']):,} val")
    print(f"  Sequence   : {seq_len} steps x {input_dim} features")
    print(f"  Classes    : {label_dim}  (random baseline = {100/label_dim:.0f}%)")
    print(f"  Hidden dim : {args.hidden_dim}")
    print(f"  Blocks (k) : {args.num_blocks}  Rank: {args.rank}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}  (cosine annealing)")
    print()

    all_results = {}

    for model_name in args.models:
        print(f"--- {model_name} ---")
        torch.manual_seed(args.seed)

        try:
            model = build_model(
                model_name, input_dim, args.hidden_dim, label_dim, seq_len,
                num_blocks=args.num_blocks, rank=args.rank,
                iterations=args.iterations, num_layers=2, nhead=4,
            ).to(device)

            stored = count_stored(model)
            surface = count_surface(model)
            print(f"  Stored params : {stored:,}")
            print(f"  Surface params: {surface:,}")

            result = train_model(
                model, train_loader, val_loader, device, args.epochs,
                lr=args.lr, label=model_name, verbose=not args.quiet,
            )
            result["stored_params"] = stored
            result["surface_params"] = surface
            result["model_name"] = model_name
            all_results[model_name] = result

            if log:
                log.log("training",
                        model=model_name,
                        stored_params=stored, surface_params=surface,
                        best_val_acc=result["best_val_acc"],
                        final_val_acc=result["final_val_acc"],
                        total_time=result["total_time"],
                        peak_mem_mb=result["peak_mem_mb"],
                        epochs=args.epochs, hidden_dim=args.hidden_dim)

            safe_cleanup(model)

        except Exception as e:
            err = str(e).split("\n")[0][:100]
            print(f"  ** ERROR: {err} **")

        print()

    if not all_results:
        print("No models completed training.")
        return

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 120}")
    print(f"{'Model':<22s} {'Stored':>12s}  {'Surface':>12s}  {'Best Val%':>9s}  "
          f"{'Final Val%':>10s}  {'Time (s)':>9s}  {'Mem (MB)':>9s}")
    print(f"{'-' * 120}")

    for name in args.models:
        if name not in all_results:
            continue
        r = all_results[name]
        print(f"{name:<22s} {r['stored_params']:>12,}  {r['surface_params']:>12,}  "
              f"{r['best_val_acc']:>8.2f}%  {r['final_val_acc']:>9.2f}%  "
              f"{r['total_time']:>8.1f}s  {r['peak_mem_mb']:>8.1f}")

    # ── Convergence speed ─────────────────────────────────────────
    thresholds = [50.0, 70.0, 80.0, 85.0, 90.0]
    print(f"\nConvergence Speed (epochs to reach accuracy threshold):")
    header = f"{'Model':<22s}" + "".join(f"  {t:>5.0f}%" for t in thresholds)
    print(header)
    print("-" * len(header))
    for name in args.models:
        if name not in all_results:
            continue
        r = all_results[name]
        cols = []
        for thresh in thresholds:
            reached = None
            for h in r["history"]:
                if h["val_acc"] >= thresh:
                    reached = h["epoch"]
                    break
            cols.append(f"  {reached:>5d} " if reached else "     -- ")
        print(f"{name:<22s}{''.join(cols)}")

    # ── Parameter efficiency ranking ──────────────────────────────
    print(f"\nParameter Efficiency Ranking (best val accuracy / 1K stored parameters):")
    ranked = sorted(all_results.values(),
                    key=lambda r: r["best_val_acc"] / max(r["stored_params"] / 1000, 1),
                    reverse=True)
    for i, r in enumerate(ranked, 1):
        eff = r["best_val_acc"] / (r["stored_params"] / 1000)
        print(f"  {i}. {r['model_name']:<22s}  {eff:>8.4f} acc%/kParam  "
              f"(val={r['best_val_acc']:.1f}%, stored={r['stored_params']:,}, "
              f"surface={r['surface_params']:,})")

    # ── Save JSON ─────────────────────────────────────────────────
    if args.save_json:
        out = {
            "config": vars(args),
            "dataset": args.dataset,
            "results": {
                name: {k: v for k, v in r.items() if k != "history"}
                for name, r in all_results.items()
            },
            "history": {
                name: r["history"] for name, r in all_results.items()
            },
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
