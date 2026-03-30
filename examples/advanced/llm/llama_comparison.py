#!/usr/bin/env python
"""Llama 3 training comparison: LSLinear Llama vs Standard Llama.

Demonstrates LSLinear's parameter efficiency on language modeling by training
two Llama 3 architecture variants on WikiText and comparing perplexity,
throughput, and memory usage.

Llama 3 features:  RMSNorm · RoPE · GQA · SwiGLU · no bias

Variants:
  1. Standard Llama  — decoder-only transformer with nn.Linear (no bias)
  2. LSLinear Llama   — identical Llama, LSLinear replaces all linear projections

Usage
-----
Quick smoke test (minutes):
    uv run --extra llm examples/llm/llama_comparison.py --config small --epochs 2

Full small comparison:
    uv run --extra llm examples/llm/llama_comparison.py --config small --epochs 20

Large-scale:
    uv run --extra llm examples/llm/llama_comparison.py \\
        --config large --dataset wikitext-103 --epochs 10

Single model only:
    uv run --extra llm examples/llm/llama_comparison.py --models ls --config medium
"""

import gc
import json
import math
import os
import sys
import time

import click
import torch
import torch.nn as nn

# Allow imports from this directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# bench_utils lives in the comparisons directory (sibling to this dir)
_comparisons_dir = os.path.join(_this_dir, "..", "comparisons")
_comparisons_dir = os.path.normpath(_comparisons_dir)
if _comparisons_dir not in sys.path:
    sys.path.insert(1, _comparisons_dir)  # insert early to beat any pip-installed bench_utils

from data import load_wikitext, create_dataloaders
from llama_models import (
    build_llama_model, LLAMA_ALL_MODEL_NAMES, LLAMA_CONFIGS, LLAMA_LS_CONFIGS,
    LLAMA_FACTORED_CONFIGS, LSLinear3D, MonarchLinear3D,
)

# Reuse bench_utils from advanced comparisons
from bench_utils import count_stored, count_surface, safe_cleanup, ResultLog


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def count_params(model):
    """Count stored parameters, using model's own method if available."""
    if hasattr(model, "number_of_trainable_parameters"):
        return model.number_of_trainable_parameters()
    return count_stored(model)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    device, grad_clip=1.0, use_amp=False):
    """Train one epoch. Returns (avg_loss, tokens_per_sec)."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids)  # (B, T, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    elapsed = time.time() - t0
    avg_loss = total_loss / total_tokens
    tok_per_sec = total_tokens / elapsed
    return avg_loss, tok_per_sec


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    """Evaluate on a DataLoader. Returns (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow
    return avg_loss, ppl


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, device, epochs, lr=3e-4,
                weight_decay=0.1, grad_clip=1.0, warmup_frac=0.05,
                label="model", use_amp=False, verbose=True):
    """Full training loop with cosine schedule and evaluation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_ppl = float("inf")
    history = []
    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        train_loss, tok_per_sec = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, grad_clip, use_amp,
        )
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, use_amp)
        train_ppl = math.exp(min(train_loss, 100))
        best_val_ppl = min(best_val_ppl, val_ppl)
        epoch_time = time.time() - t_epoch

        history.append(dict(
            epoch=epoch, train_loss=train_loss, train_ppl=train_ppl,
            val_loss=val_loss, val_ppl=val_ppl, tok_per_sec=tok_per_sec,
            epoch_time=epoch_time,
        ))

        if verbose:
            print(f"  [{label:>16s}] epoch {epoch:3d}/{epochs}  "
                  f"loss {train_loss:.4f}  train_ppl {train_ppl:8.2f}  "
                  f"val_ppl {val_ppl:8.2f}  "
                  f"tok/s {tok_per_sec:,.0f}  ({epoch_time:.1f}s)")

    total_time = time.time() - t0
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return dict(
        best_val_ppl=best_val_ppl,
        final_val_ppl=history[-1]["val_ppl"],
        final_train_ppl=history[-1]["train_ppl"],
        avg_tok_per_sec=sum(h["tok_per_sec"] for h in history) / len(history),
        total_time=total_time,
        peak_mem_mb=peak_mem,
        history=history,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_name",
              type=click.Choice(list(LLAMA_CONFIGS.keys())),
              default="small", show_default=True,
              help="Preset model configuration.")
@click.option("--models", multiple=True, type=click.Choice(LLAMA_ALL_MODEL_NAMES),
              default=LLAMA_ALL_MODEL_NAMES, show_default=True,
              help="Which model(s) to train. Repeat for multiple.")
@click.option("--dataset", type=click.Choice(["wikitext-2", "wikitext-103"]),
              default="wikitext-2", show_default=True)
@click.option("--d-model", type=int, default=None, help="Override d_model from config.")
@click.option("--n-layers", type=int, default=None, help="Override n_layers from config.")
@click.option("--n-heads", type=int, default=None, help="Override n_heads from config.")
@click.option("--n-kv-heads", type=int, default=None, help="Override n_kv_heads from config.")
@click.option("--d-ff", type=int, default=None, help="Override d_ff from config.")
@click.option("--context-len", type=int, default=None, help="Override context_len from config.")
@click.option("--num-blocks", type=int, default=None, help="Override LSLinear num_blocks.")
@click.option("--rank", type=int, default=None, help="Override LSLinear rank.")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=10, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=0.1, show_default=True)
@click.option("--dropout", type=float, default=0.0, show_default=True,
              help="Dropout probability (0.0 = Llama convention).")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--amp/--no-amp", default=False, show_default=True,
              help="Enable mixed-precision (bf16) training.")
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--save-json", type=str, default=None,
              help="Path to save results as JSON.")
@click.option("--log", type=str, default=None,
              help="Path to unified JSONL log (appends).")
@click.option("--quiet", is_flag=True)
def main(config_name, models, dataset, d_model, n_layers, n_heads, n_kv_heads,
         d_ff, context_len, num_blocks, rank, batch_size, epochs, lr,
         weight_decay, dropout, grad_clip, amp, device, seed, save_json, log, quiet):
    """Train and compare Standard Llama vs LSLinear Llama on WikiText."""

    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    use_amp = amp and device.type == "cuda"

    # Build overrides dict
    overrides = {}
    if d_model is not None:
        overrides["d_model"] = d_model
    if n_layers is not None:
        overrides["n_layers"] = n_layers
    if n_heads is not None:
        overrides["n_heads"] = n_heads
    if n_kv_heads is not None:
        overrides["n_kv_heads"] = n_kv_heads
    if d_ff is not None:
        overrides["d_ff"] = d_ff
    if context_len is not None:
        overrides["context_len"] = context_len
    if num_blocks is not None:
        overrides["num_blocks"] = num_blocks
    if rank is not None:
        overrides["rank"] = rank
    if dropout > 0:
        overrides["dropout"] = dropout

    # Resolve effective config for display
    effective_cfg = dict(LLAMA_CONFIGS[config_name])
    effective_cfg.update(overrides)
    ctx_len = effective_cfg["context_len"]

    # -- Data ---------------------------------------------------------------
    version = dataset.split("-")[1]  # "2" or "103"
    train_tokens, val_tokens, _ = load_wikitext(version, verbose=not quiet)
    train_loader, val_loader, vocab_size = create_dataloaders(
        train_tokens, val_tokens, context_len=ctx_len, batch_size=batch_size,
    )

    n_train_tokens = len(train_tokens)
    n_val_tokens = len(val_tokens)

    # -- Header -------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f" Llama 3 Comparison -- {dataset} (GPT-2 tokenizer, context={ctx_len})")
    print(f"{'=' * 80}")
    print(f"  Device      : {device}", end="")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f" ({torch.cuda.get_device_name()})")
    else:
        print()
    print(f"  Dataset     : {dataset} ({n_train_tokens:,} train / {n_val_tokens:,} val tokens)")
    print(f"  Config      : {config_name} "
          f"(d={effective_cfg['d_model']}, layers={effective_cfg['n_layers']}, "
          f"heads={effective_cfg['n_heads']}, kv_heads={effective_cfg['n_kv_heads']}, "
          f"ff={effective_cfg['d_ff']})")
    print(f"  Context len : {ctx_len}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}  (cosine w/ warmup)")
    print(f"  AMP         : {'bf16' if use_amp else 'off'}")
    print(f"  Vocab       : {vocab_size:,} (GPT-2)")
    print()

    result_log = ResultLog(log) if log else None
    all_results = {}

    # -- Train each model ---------------------------------------------------
    for model_name in models:
        print(f"--- {model_name} (Llama 3) ---")
        torch.manual_seed(seed)

        try:
            model = build_llama_model(
                model_name, vocab_size=vocab_size,
                config_name=config_name, **overrides,
            ).to(device)

            stored = count_params(model)
            surface = count_surface(model)

            print(f"  Stored params : {stored:>12,}")
            print(f"  Surface params: {surface:>12,}")
            if surface > stored:
                ratio = surface / stored
                print(f"  Compression   : {ratio:>11.1f}x  (surface / stored)")

            result = train_model(
                model, train_loader, val_loader, device, epochs,
                lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
                label=model_name, use_amp=use_amp, verbose=not quiet,
            )
            result["stored_params"] = stored
            result["surface_params"] = surface
            result["model_name"] = model_name
            all_results[model_name] = result

            if result_log:
                result_log.log(
                    "llama_training",
                    model=model_name, config=config_name, dataset=dataset,
                    stored_params=stored, surface_params=surface,
                    best_val_ppl=result["best_val_ppl"],
                    final_val_ppl=result["final_val_ppl"],
                    avg_tok_per_sec=result["avg_tok_per_sec"],
                    total_time=result["total_time"],
                    peak_mem_mb=result["peak_mem_mb"],
                    epochs=epochs,
                )

            safe_cleanup(model)

        except Exception as e:
            import traceback
            print(f"  ** ERROR: {e} **")
            if not quiet:
                traceback.print_exc()

        print()

    if not all_results:
        print("No models completed training.")
        return

    # -- Summary table ------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f" RESULTS SUMMARY (Llama 3)")
    print(f"{'=' * 100}")
    print(f"{'Model':<18s} {'Stored':>10s}  {'Surface':>10s}  "
          f"{'Val PPL':>9s}  {'Tok/s':>10s}  {'Time':>8s}  {'Mem (MB)':>9s}")
    print(f"{'-' * 100}")

    for name in models:
        if name not in all_results:
            continue
        r = all_results[name]
        print(f"{name:<18s} {r['stored_params']:>10,}  {r['surface_params']:>10,}  "
              f"{r['best_val_ppl']:>9.2f}  {r['avg_tok_per_sec']:>10,.0f}  "
              f"{r['total_time']:>7.0f}s  {r['peak_mem_mb']:>8.1f}")

    # -- Parameter efficiency -----------------------------------------------
    print(f"\n  Parameter Efficiency (lower perplexity per stored param is better):")
    for name in models:
        if name not in all_results:
            continue
        r = all_results[name]
        stored_m = r["stored_params"] / 1e6
        print(f"    {name:<18s}  PPL={r['best_val_ppl']:.2f}  "
              f"stored={stored_m:.2f}M  surface={r['surface_params']/1e6:.2f}M"
              + (f"  ({r['surface_params']/r['stored_params']:.1f}x compression)"
                 if r['surface_params'] > r['stored_params'] else ""))

    # -- Save JSON ----------------------------------------------------------
    if save_json:
        out = {
            "config": {
                "config_name": config_name,
                "dataset": dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "context_len": ctx_len,
                "effective_cfg": effective_cfg,
            },
            "results": {
                name: {k: v for k, v in r.items() if k != "history"}
                for name, r in all_results.items()
            },
            "history": {
                name: r["history"] for name, r in all_results.items()
            },
        }
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {save_json}")


if __name__ == "__main__":
    main()
