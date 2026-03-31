#!/usr/bin/env python
"""Llama 3 training comparison: structured-sparse variants.

Demonstrates parameter efficiency and throughput of structured-sparse linear
layers on language modeling by training multiple Llama 3 architecture variants
on WikiText and comparing perplexity, throughput, and memory usage.

Llama 3 features:  RMSNorm · RoPE · GQA · SwiGLU · no bias

Variants:
  1. Standard Llama        — decoder-only transformer with nn.Linear (no bias)
  2. LS Llama              — LSLinear (unfactored Monarch + low-rank, with perms)
  3. LS-Factored Llama     — LSLinear (factored Monarch + low-rank, with perms)
  4. LS-BlockDiag Llama    — LSBlockDiagLinear (unfactored block-diag + low-rank, no perms)
  5. LS-BlockDiag-Factored — LSBlockDiagLinear (factored block-diag + low-rank, no perms)

Usage
-----
Quick smoke test (minutes):
    uv run --extra llm examples/advanced/llm/llama_comparison.py --config small --epochs 2

Full small comparison:
    uv run --extra llm examples/advanced/llm/llama_comparison.py --config small --epochs 20

Large-scale with gradient checkpointing + AMP:
    uv run --extra llm examples/advanced/llm/llama_comparison.py \\
        --config 7b --dataset wikitext-103 --epochs 3 \\
        --gradient-checkpointing --amp --batch-size 2 --grad-accum-steps 16

Generate comparison plots from logs:
    uv run --extra llm examples/advanced/llm/llama_comparison.py --plot-from results.jsonl

Single model only:
    uv run --extra llm examples/advanced/llm/llama_comparison.py --models ls --config medium
"""

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone

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
    sys.path.insert(1, _comparisons_dir)

from data import load_wikitext, create_dataloaders
from llama_models import (
    build_llama_model, LLAMA_ALL_MODEL_NAMES, LLAMA_CONFIGS,
)

# Reuse bench_utils from advanced comparisons
from bench_utils import count_stored, count_surface, safe_cleanup


# ---------------------------------------------------------------------------
# JSONL Logger with full per-step history
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Append-only JSONL logger with structured records for curve generation.

    Record kinds:
      - run_config: hyperparameters and system info (1 per model run)
      - step:       per-step metrics (loss, lr, grad_norm, throughput, memory)
      - epoch:      per-epoch aggregate metrics
      - run_summary: final summary for the model run
    """

    def __init__(self, path):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, kind, **data):
        if not self.path:
            return
        record = {"kind": kind, "ts": datetime.now(timezone.utc).isoformat()}
        record.update(data)
        # Ensure JSON-serializable
        for k, v in record.items():
            if hasattr(v, "item"):
                record[k] = v.item()
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                record[k] = str(v)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def read(path, kind=None):
        if not os.path.exists(path):
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if kind is None or r.get("kind") == kind:
                    records.append(r)
        return records


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def get_cuda_memory_breakdown(device):
    """Return detailed CUDA memory breakdown in MB.

    Returns dict with:
      allocated_mb:    currently allocated tensor memory
      reserved_mb:     total memory reserved by caching allocator
      peak_allocated_mb: peak allocated since last reset
      free_mb:         free memory on the device
      total_mb:        total device memory
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return {}

    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    free, total = torch.cuda.mem_get_info(device)
    free_mb = free / 1024 / 1024
    total_mb = total / 1024 / 1024

    return dict(
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        peak_allocated_mb=round(peak, 1),
        free_mb=round(free_mb, 1),
        total_mb=round(total_mb, 1),
    )


def measure_model_memory(model, device):
    """Measure memory used by model parameters alone (MB)."""
    param_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    return param_bytes / 1024 / 1024


def measure_optimizer_memory(optimizer):
    """Estimate optimizer state memory (MB). AdamW stores 2 states per param."""
    total_bytes = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.numel()
    return total_bytes / 1024 / 1024


def compute_grad_norm(model):
    """Compute total gradient L2 norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.float().norm().item() ** 2
    return math.sqrt(total_norm_sq)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_ppl, model_name, history=None):
    """Save training checkpoint for resumption.

    Saves:
      - {model_name}_latest.pt  — always overwritten (for --resume auto-detection)
      - {model_name}_best.pt    — only when best_val_ppl improves
      - {model_name}_final.pt   — call manually at end of training
    """
    if not checkpoint_dir:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_ppl": best_val_ppl,
        "history": history or [],
    }
    # Save latest (always)
    latest_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pt")
    torch.save(state, latest_path)
    return latest_path


def save_best_checkpoint(checkpoint_dir, model, model_name):
    """Save just the model weights as the best checkpoint."""
    if not checkpoint_dir:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    torch.save({"model": model.state_dict()}, path)
    return path


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and return (start_epoch, global_step, best_val_ppl, history).

    Restores model, optimizer, scheduler, and scaler states.
    """
    print(f"  Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
    history = ckpt.get("history", [])
    print(f"  Resumed at epoch {epoch}, global_step {global_step}, "
          f"best_val_ppl {best_val_ppl:.2f}")
    return epoch, global_step, best_val_ppl, history


def find_resume_checkpoint(checkpoint_dir, model_name):
    """Return path to latest checkpoint if it exists, else None."""
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, f"{model_name}_latest.pt")
    if os.path.exists(path):
        return path
    return None


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def count_params(model):
    """Count stored parameters, using model's own method if available."""
    if hasattr(model, "number_of_trainable_parameters"):
        return model.number_of_trainable_parameters()
    return count_stored(model)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    device, grad_clip=1.0, use_amp=False,
                    grad_accum_steps=1, logger=None, model_name="",
                    epoch=0, global_step=0):
    """Train one epoch with gradient accumulation and per-step logging.

    Returns (avg_loss, tokens_per_sec, global_step).
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    accum_loss = 0.0
    accum_tokens = 0
    t0 = time.time()
    t_step = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step_in_epoch, (input_ids, targets) in enumerate(loader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids)  # (B, T, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        batch_tokens = input_ids.numel()
        accum_loss += loss.item() * batch_tokens
        accum_tokens += batch_tokens
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # Accumulate gradients, step every grad_accum_steps
        if (step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = compute_grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

            # Per-step logging (every optimizer step)
            step_elapsed = time.time() - t_step
            step_tok_per_sec = accum_tokens / step_elapsed if step_elapsed > 0 else 0
            step_loss = accum_loss / accum_tokens if accum_tokens > 0 else 0

            if logger:
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
                mem = get_cuda_memory_breakdown(device)
                logger.log(
                    "step",
                    model=model_name, epoch=epoch, global_step=global_step,
                    step_in_epoch=step_in_epoch + 1,
                    loss=step_loss, ppl=math.exp(min(step_loss, 100)),
                    grad_norm=grad_norm, lr=current_lr,
                    tok_per_sec=step_tok_per_sec, step_time_s=step_elapsed,
                    tokens=accum_tokens,
                    **mem,
                )

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_tokens = 0
            t_step = time.time()

    elapsed = time.time() - t0
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    return avg_loss, tok_per_sec, global_step


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

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
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
                grad_accum_steps=1, label="model", use_amp=False,
                verbose=True, logger=None, compile_model=False,
                checkpoint_dir=None):
    """Full training loop with comprehensive monitoring and checkpointing.

    Features:
      - Cosine LR schedule with warmup
      - Gradient accumulation for large-batch training
      - Per-step JSONL logging (loss, ppl, grad_norm, lr, throughput, memory)
      - Per-epoch evaluation with memory breakdown
      - Optional torch.compile for extra speed
      - Checkpoint save/resume (auto-detects existing checkpoints)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  fused=torch.cuda.is_available())

    effective_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = epochs * effective_steps_per_epoch
    warmup_steps = int(total_steps * warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Try to resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_ppl = float("inf")
    history = []

    resume_path = find_resume_checkpoint(checkpoint_dir, label)
    if resume_path:
        start_epoch, global_step, best_val_ppl, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )
        if start_epoch >= epochs:
            print(f"  Already completed {start_epoch}/{epochs} epochs. Skipping.")
            return _build_result(history, best_val_ppl, 0.0, device,
                                 measure_model_memory(model, device),
                                 measure_optimizer_memory(optimizer))

    # Optional torch.compile (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        if verbose:
            print(f"  Compiling model with torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Measure memory components before training
    param_mem_mb = measure_model_memory(model, device)

    if logger:
        logger.log(
            "training_start",
            model=label, param_mem_mb=round(param_mem_mb, 1),
            total_steps=total_steps, warmup_steps=warmup_steps,
            effective_steps_per_epoch=effective_steps_per_epoch,
            grad_accum_steps=grad_accum_steps,
            resumed_from_epoch=start_epoch,
        )

    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch + 1, epochs + 1):
        t_epoch = time.time()

        train_loss, tok_per_sec, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, grad_clip, use_amp, grad_accum_steps,
            logger=logger, model_name=label, epoch=epoch,
            global_step=global_step,
        )

        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, use_amp)
        train_ppl = math.exp(min(train_loss, 100))
        is_best = val_ppl < best_val_ppl
        best_val_ppl = min(best_val_ppl, val_ppl)
        epoch_time = time.time() - t_epoch

        # Memory breakdown at end of epoch
        mem = get_cuda_memory_breakdown(device)
        optimizer_mem_mb = measure_optimizer_memory(optimizer)

        epoch_record = dict(
            epoch=epoch, train_loss=train_loss, train_ppl=train_ppl,
            val_loss=val_loss, val_ppl=val_ppl, tok_per_sec=tok_per_sec,
            epoch_time=epoch_time,
            param_mem_mb=round(param_mem_mb, 1),
            optimizer_mem_mb=round(optimizer_mem_mb, 1),
            **mem,
        )
        history.append(epoch_record)

        if logger:
            logger.log("epoch", model=label, **epoch_record)

        if verbose:
            mem_str = ""
            if mem:
                mem_str = (f"  mem: {mem.get('allocated_mb', 0):.0f}/"
                           f"{mem.get('peak_allocated_mb', 0):.0f}MB "
                           f"(param={param_mem_mb:.0f} opt={optimizer_mem_mb:.0f})")
            ckpt_str = " ★" if is_best else ""
            print(f"  [{label:>22s}] epoch {epoch:3d}/{epochs}  "
                  f"loss {train_loss:.4f}  train_ppl {train_ppl:8.2f}  "
                  f"val_ppl {val_ppl:8.2f}  "
                  f"tok/s {tok_per_sec:,.0f}  ({epoch_time:.1f}s)"
                  f"{mem_str}{ckpt_str}")

        # Save checkpoints
        if checkpoint_dir:
            # Unwrap compiled model for state_dict
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(checkpoint_dir, raw_model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_ppl, label, history)
            if is_best:
                save_best_checkpoint(checkpoint_dir, raw_model, label)
                if verbose:
                    print(f"  ↳ Saved best checkpoint (val_ppl={val_ppl:.2f})")

    total_time = time.time() - t0

    # Save final checkpoint
    if checkpoint_dir:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        final_path = os.path.join(checkpoint_dir, f"{label}_final.pt")
        torch.save({"model": raw_model.state_dict()}, final_path)
        if verbose:
            print(f"  Saved final model to {final_path}")

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    result = _build_result(history, best_val_ppl, total_time, device,
                           param_mem_mb, measure_optimizer_memory(optimizer),
                           peak_mem)

    if logger:
        logger.log(
            "run_summary",
            model=label,
            best_val_ppl=best_val_ppl,
            final_val_ppl=result["final_val_ppl"],
            final_train_ppl=result["final_train_ppl"],
            avg_tok_per_sec=result["avg_tok_per_sec"],
            total_time=total_time,
            peak_mem_mb=peak_mem,
            param_mem_mb=param_mem_mb,
            optimizer_mem_mb=result["optimizer_mem_mb"],
        )

    return result


def _build_result(history, best_val_ppl, total_time, device,
                  param_mem_mb, optimizer_mem_mb, peak_mem=None):
    """Build the result dict from training history."""
    if peak_mem is None:
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return dict(
        best_val_ppl=best_val_ppl,
        final_val_ppl=history[-1]["val_ppl"] if history else float("inf"),
        final_train_ppl=history[-1]["train_ppl"] if history else float("inf"),
        avg_tok_per_sec=(sum(h["tok_per_sec"] for h in history) / len(history)
                         if history else 0),
        total_time=total_time,
        peak_mem_mb=peak_mem,
        param_mem_mb=param_mem_mb,
        optimizer_mem_mb=optimizer_mem_mb,
        history=history,
    )


# ---------------------------------------------------------------------------
# Plot generation from logs
# ---------------------------------------------------------------------------

def plot_from_logs(log_path, output_dir=None):
    """Generate comparison plots from a JSONL log file.

    Creates:
      - loss_curves.png:      training loss over steps for all models
      - val_ppl_curves.png:   validation perplexity over epochs
      - throughput_curves.png: tokens/sec over epochs
      - memory_breakdown.png: stacked bar chart of memory components
      - param_efficiency.png: perplexity vs stored parameters
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return

    if output_dir is None:
        output_dir = os.path.dirname(log_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    records = TrainingLogger.read(log_path)
    if not records:
        print(f"No records found in {log_path}")
        return

    # Collect per-model data
    step_data = {}    # model -> list of step records
    epoch_data = {}   # model -> list of epoch records
    summaries = {}    # model -> summary record
    configs = {}      # model -> config record

    for r in records:
        kind = r.get("kind")
        model = r.get("model", "unknown")
        if kind == "step":
            step_data.setdefault(model, []).append(r)
        elif kind == "epoch":
            epoch_data.setdefault(model, []).append(r)
        elif kind == "run_summary":
            summaries[model] = r
        elif kind == "run_config":
            configs[model] = r

    if not step_data and not epoch_data:
        print("No training data found in logs.")
        return

    # Style setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    colors = {
        "standard": "#d62728",
        "ls": "#1f77b4",
        "ls-factored": "#17becf",
        "ls-blockdiag": "#2ca02c",
        "ls-blockdiag-factored": "#ff7f0e",
    }

    def _color(name):
        return colors.get(name, "#333333")

    # 1. Training loss curves (per step)
    if step_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, steps in sorted(step_data.items()):
            gs = [s["global_step"] for s in steps]
            losses = [s["loss"] for s in steps]
            ax.plot(gs, losses, label=model, color=_color(model), alpha=0.7, linewidth=0.8)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Curves")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "loss_curves.png"))
        plt.close(fig)
        print(f"  Saved loss_curves.png")

    # 2. Validation perplexity curves (per epoch)
    if epoch_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, epochs_list in sorted(epoch_data.items()):
            ep = [e["epoch"] for e in epochs_list]
            vppl = [e["val_ppl"] for e in epochs_list]
            ax.plot(ep, vppl, "o-", label=model, color=_color(model), markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Perplexity")
        ax.set_title("Validation Perplexity")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "val_ppl_curves.png"))
        plt.close(fig)
        print(f"  Saved val_ppl_curves.png")

    # 3. Throughput curves (per epoch)
    if epoch_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, epochs_list in sorted(epoch_data.items()):
            ep = [e["epoch"] for e in epochs_list]
            tps = [e["tok_per_sec"] for e in epochs_list]
            ax.plot(ep, tps, "o-", label=model, color=_color(model), markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Tokens / sec")
        ax.set_title("Training Throughput")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "throughput_curves.png"))
        plt.close(fig)
        print(f"  Saved throughput_curves.png")

    # 4. Memory breakdown (stacked bar)
    if summaries:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted(summaries.keys())
        param_mems = [summaries[m].get("param_mem_mb", 0) for m in models]
        opt_mems = [summaries[m].get("optimizer_mem_mb", 0) for m in models]
        peak_mems = [summaries[m].get("peak_mem_mb", 0) for m in models]
        # Activations ≈ peak - params - optimizer
        act_mems = [max(0, p - pm - om) for p, pm, om in zip(peak_mems, param_mems, opt_mems)]

        x = range(len(models))
        ax.bar(x, param_mems, label="Parameters", color="#1f77b4")
        ax.bar(x, opt_mems, bottom=param_mems, label="Optimizer", color="#ff7f0e")
        ax.bar(x, act_mems, bottom=[p + o for p, o in zip(param_mems, opt_mems)],
               label="Activations + Other", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("GPU Memory Breakdown")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "memory_breakdown.png"))
        plt.close(fig)
        print(f"  Saved memory_breakdown.png")

    # 5. Parameter efficiency scatter
    if summaries and configs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in sorted(summaries.keys()):
            cfg = configs.get(model, {})
            stored = cfg.get("stored_params", 0)
            ppl = summaries[model].get("best_val_ppl", 0)
            if stored > 0 and ppl > 0:
                ax.scatter(stored / 1e6, ppl, label=model, color=_color(model), s=80, zorder=5)
                ax.annotate(model, (stored / 1e6, ppl), fontsize=7,
                            xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Stored Parameters (M)")
        ax.set_ylabel("Best Validation Perplexity")
        ax.set_title("Parameter Efficiency")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "param_efficiency.png"))
        plt.close(fig)
        print(f"  Saved param_efficiency.png")

    # 6. Gradient norm curves (per step)
    if step_data:
        has_grad = any("grad_norm" in s for steps in step_data.values() for s in steps)
        if has_grad:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model, steps in sorted(step_data.items()):
                gs = [s["global_step"] for s in steps if "grad_norm" in s]
                gn = [s["grad_norm"] for s in steps if "grad_norm" in s]
                ax.plot(gs, gn, label=model, color=_color(model), alpha=0.7, linewidth=0.8)
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norms")
            ax.legend()
            fig.savefig(os.path.join(output_dir, "grad_norm_curves.png"))
            plt.close(fig)
            print(f"  Saved grad_norm_curves.png")

    # 7. Learning rate schedule
    if step_data:
        has_lr = any("lr" in s for steps in step_data.values() for s in steps)
        if has_lr:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model, steps in sorted(step_data.items()):
                gs = [s["global_step"] for s in steps if "lr" in s]
                lrs = [s["lr"] for s in steps if "lr" in s]
                ax.plot(gs, lrs, label=model, color=_color(model), alpha=0.7, linewidth=0.8)
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.legend()
            fig.savefig(os.path.join(output_dir, "lr_schedule.png"))
            plt.close(fig)
            print(f"  Saved lr_schedule.png")

    print(f"\nAll plots saved to {output_dir}/")


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
@click.option("--num-blocks", type=int, default=None, help="Override num_blocks for sparse layers.")
@click.option("--rank", type=int, default=None, help="Override low-rank component rank.")
@click.option("--chain-length", type=int, default=None,
              help="Override chain_length for factored variants (higher = more param savings).")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=10, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=0.1, show_default=True)
@click.option("--dropout", type=float, default=0.0, show_default=True,
              help="Dropout probability (0.0 = Llama convention).")
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--grad-accum-steps", type=int, default=1, show_default=True,
              help="Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps.")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=False,
              show_default=True,
              help="Trade ~30%% slower training for major activation memory savings. Essential for 7B+.")
@click.option("--compile/--no-compile", "compile_model", default=False, show_default=True,
              help="Use torch.compile for extra speed (PyTorch 2.0+).")
@click.option("--amp/--no-amp", default=False, show_default=True,
              help="Enable mixed-precision (bf16) training.")
@click.option("--tf32/--no-tf32", default=True, show_default=True,
              help="Enable TF32 matmul precision (faster on Ampere+).")
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--save-json", type=str, default=None,
              help="Path to save results summary as JSON.")
@click.option("--log", type=str, default=None,
              help="Path to JSONL log file (appends). Used for curve generation.")
@click.option("--checkpoint-dir", type=str, default=None,
              help="Directory for saving/resuming checkpoints. Auto-resumes if checkpoint exists.")
@click.option("--plot-from", type=str, default=None,
              help="Generate plots from an existing JSONL log file and exit.")
@click.option("--plot-dir", type=str, default=None,
              help="Output directory for plots (default: same dir as log file).")
@click.option("--quiet", is_flag=True)
def main(config_name, models, dataset, d_model, n_layers, n_heads, n_kv_heads,
         d_ff, context_len, num_blocks, rank, chain_length, batch_size, epochs, lr,
         weight_decay, dropout, grad_clip, grad_accum_steps, gradient_checkpointing,
         compile_model, amp, tf32, device, seed, save_json, log, checkpoint_dir,
         plot_from, plot_dir, quiet):
    """Train and compare structured-sparse Llama variants on WikiText."""

    # -- Plot mode: generate plots from existing logs and exit ---------------
    if plot_from:
        print(f"Generating plots from {plot_from}...")
        plot_from_logs(plot_from, output_dir=plot_dir)
        return

    # -- Performance settings -----------------------------------------------
    if tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    if chain_length is not None:
        overrides["chain_length"] = chain_length
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
    effective_batch = batch_size * grad_accum_steps

    # -- Logger -------------------------------------------------------------
    logger = TrainingLogger(log)

    # -- Header -------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print(f" Llama 3 Comparison -- {dataset} (GPT-2 tokenizer, context={ctx_len})")
    print(f"{'=' * 90}")
    print(f"  Device          : {device}", end="")
    if torch.cuda.is_available() and device.type == "cuda":
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024
        print(f" ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print()
    print(f"  Dataset         : {dataset} ({n_train_tokens:,} train / {n_val_tokens:,} val tokens)")
    print(f"  Config          : {config_name} "
          f"(d={effective_cfg['d_model']}, layers={effective_cfg['n_layers']}, "
          f"heads={effective_cfg['n_heads']}, kv_heads={effective_cfg['n_kv_heads']}, "
          f"ff={effective_cfg['d_ff']})")
    print(f"  Context len     : {ctx_len}")
    print(f"  Batch size      : {batch_size} × {grad_accum_steps} accum = {effective_batch} effective")
    print(f"  Epochs          : {epochs}")
    print(f"  LR              : {lr}  (cosine w/ warmup)")
    print(f"  AMP             : {'bf16' if use_amp else 'off'}")
    print(f"  TF32            : {'on' if tf32 and torch.cuda.is_available() else 'off'}")
    print(f"  Grad checkpoint : {'on' if gradient_checkpointing else 'off'}")
    print(f"  torch.compile   : {'on' if compile_model else 'off'}")
    print(f"  Vocab           : {vocab_size:,} (GPT-2)")
    if checkpoint_dir:
        print(f"  Checkpoints     : {checkpoint_dir}")
    if log:
        print(f"  Log file        : {log}")
    print()

    all_results = {}

    # -- Train each model ---------------------------------------------------
    for model_name in models:
        print(f"--- {model_name} (Llama 3) ---")
        torch.manual_seed(seed)

        try:
            model = build_llama_model(
                model_name, vocab_size=vocab_size,
                config_name=config_name,
                gradient_checkpointing=gradient_checkpointing,
                **overrides,
            ).to(device)

            stored = count_params(model)
            surface = count_surface(model)
            param_mem_mb = measure_model_memory(model, device)

            print(f"  Stored params   : {stored:>12,}")
            print(f"  Surface params  : {surface:>12,}")
            print(f"  Param memory    : {param_mem_mb:>11.1f} MB")
            if surface > stored:
                ratio = surface / stored
                print(f"  Compression     : {ratio:>11.1f}x  (surface / stored)")

            # Log run config
            if logger:
                logger.log(
                    "run_config",
                    model=model_name, config=config_name, dataset=dataset,
                    stored_params=stored, surface_params=surface,
                    param_mem_mb=round(param_mem_mb, 1),
                    d_model=effective_cfg["d_model"],
                    n_layers=effective_cfg["n_layers"],
                    n_heads=effective_cfg["n_heads"],
                    d_ff=effective_cfg["d_ff"],
                    context_len=ctx_len,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    effective_batch=effective_batch,
                    epochs=epochs, lr=lr,
                    use_amp=use_amp, gradient_checkpointing=gradient_checkpointing,
                    compile_model=compile_model,
                )

            result = train_model(
                model, train_loader, val_loader, device, epochs,
                lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
                grad_accum_steps=grad_accum_steps,
                label=model_name, use_amp=use_amp, verbose=not quiet,
                logger=logger, compile_model=compile_model,
                checkpoint_dir=checkpoint_dir,
            )
            result["stored_params"] = stored
            result["surface_params"] = surface
            result["model_name"] = model_name
            all_results[model_name] = result

            safe_cleanup(model)

        except Exception as e:
            print(f"  ** ERROR: {e} **")
            if not quiet:
                traceback.print_exc()

        print()

    if not all_results:
        print("No models completed training.")
        return

    # -- Summary table ------------------------------------------------------
    print(f"\n{'=' * 120}")
    print(f" RESULTS SUMMARY (Llama 3)")
    print(f"{'=' * 120}")
    print(f"{'Model':<24s} {'Stored':>10s}  {'Surface':>10s}  "
          f"{'Val PPL':>9s}  {'Tok/s':>10s}  {'Time':>8s}  "
          f"{'Peak Mem':>9s}  {'Params':>8s}  {'Optim':>8s}")
    print(f"{'-' * 120}")

    for name in models:
        if name not in all_results:
            continue
        r = all_results[name]
        print(f"{name:<24s} {r['stored_params']:>10,}  {r['surface_params']:>10,}  "
              f"{r['best_val_ppl']:>9.2f}  {r['avg_tok_per_sec']:>10,.0f}  "
              f"{r['total_time']:>7.0f}s  {r['peak_mem_mb']:>8.1f}  "
              f"{r['param_mem_mb']:>7.1f}  {r['optimizer_mem_mb']:>7.1f}")

    # -- Parameter efficiency -----------------------------------------------
    print(f"\n  Parameter Efficiency (lower perplexity per stored param is better):")
    for name in models:
        if name not in all_results:
            continue
        r = all_results[name]
        stored_m = r["stored_params"] / 1e6
        print(f"    {name:<24s}  PPL={r['best_val_ppl']:.2f}  "
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
                "grad_accum_steps": grad_accum_steps,
                "effective_batch": effective_batch,
                "lr": lr,
                "context_len": ctx_len,
                "effective_cfg": effective_cfg,
                "gradient_checkpointing": gradient_checkpointing,
                "use_amp": use_amp,
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

    # -- Auto-generate plots if log was specified ---------------------------
    if log and os.path.exists(log):
        print(f"\nGenerating plots from {log}...")
        plot_from_logs(log, output_dir=plot_dir or os.path.dirname(log) or ".")


if __name__ == "__main__":
    main()
