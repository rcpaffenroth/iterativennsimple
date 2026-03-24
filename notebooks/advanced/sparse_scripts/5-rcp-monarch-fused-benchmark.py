"""Fused Triton Kernel Benchmark for MonarchLinear.

Compares wall-clock time of:
  1. Fused Triton kernel (use_fused=True)
  2. Existing bmm path  (use_views=False, uniform blocks)
  3. Existing view-based loop (use_views=True)
  4. Dense nn.Linear baseline

Sweeps over block sizes, number of blocks, and batch sizes.
Reports per-iteration time (forward + backward) in milliseconds.

Usage:
    python notebooks/advanced/sparse_scripts/5-rcp-monarch-fused-benchmark.py
"""

import time
import itertools

import torch
import torch.nn as nn

from iterativennsimple.MonarchLinear import MonarchLinear

# ============================================================================
# Configuration
# ============================================================================

BLOCK_SIZES = [16, 32, 64, 128]
NUM_BLOCKS_LIST = [4, 8, 16, 32]
BATCH_SIZES = [32, 128]
DTYPE = torch.float32
WARMUP = 3
ITERS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def benchmark_one(layer, x, forward_fn=None, warmup=WARMUP, iters=ITERS):
    """Time forward + backward for a layer. Returns avg ms per iteration.

    Args:
        layer: nn.Module (used for .parameters() to clear grads).
        x: Input tensor.
        forward_fn: If provided, called as forward_fn(x) instead of layer(x).
                    Use this to pass custom kwargs like use_fused=True.
    """
    call = forward_fn if forward_fn is not None else layer

    # Warmup
    for _ in range(warmup):
        y = call(x)
        y.sum().backward()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = call(x)
        y.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
        for p in layer.parameters():
            if p.grad is not None:
                p.grad = None

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


# ============================================================================
# Sweep
# ============================================================================

print(f"\n{'='*90}")
print(f"{'block_size':>10} {'num_blocks':>10} {'batch':>6} {'size':>8} | "
      f"{'Fused (ms)':>12} {'BMM (ms)':>12} {'Views (ms)':>12} {'Dense (ms)':>12} | "
      f"{'Speedup':>8}")
print(f"{'='*90}")

for block_size, num_blocks, batch in itertools.product(BLOCK_SIZES, NUM_BLOCKS_LIST, BATCH_SIZES):
    n = block_size * num_blocks
    # Skip very large layers that would OOM
    if n > 8192:
        continue

    x = torch.randn(batch, n, device=device, dtype=DTYPE)

    # --- Fused ---
    try:
        layer_fused = MonarchLinear.from_uniform_blocks(
            n, n, num_blocks=num_blocks, bias=True, seed=0
        ).to(device).to(DTYPE).train()
        t_fused, s_fused = benchmark_one(
            layer_fused, x,
            forward_fn=lambda inp, _l=layer_fused: _l(inp, use_fused=True),
        )
        fused_str = f"{t_fused:8.3f}+-{s_fused:.2f}"
    except Exception as e:
        fused_str = f"ERR: {e!s:.20s}"
        t_fused = float("inf")

    # --- BMM (use_views=False) ---
    try:
        layer_bmm = MonarchLinear.from_uniform_blocks(
            n, n, num_blocks=num_blocks, bias=True, seed=0
        ).to(device).to(DTYPE).train()
        t_bmm, s_bmm = benchmark_one(
            layer_bmm, x,
            forward_fn=lambda inp, _l=layer_bmm: _l(inp, use_fused=False, use_views=False),
        )
        bmm_str = f"{t_bmm:8.3f}+-{s_bmm:.2f}"
    except Exception as e:
        bmm_str = f"ERR: {e!s:.20s}"
        t_bmm = float("inf")

    # --- Views loop ---
    try:
        layer_views = MonarchLinear.from_uniform_blocks(
            n, n, num_blocks=num_blocks, bias=True, seed=0
        ).to(device).to(DTYPE).train()
        t_views, s_views = benchmark_one(
            layer_views, x,
            forward_fn=lambda inp, _l=layer_views: _l(inp, use_fused=False, use_views=True),
        )
        views_str = f"{t_views:8.3f}+-{s_views:.2f}"
    except Exception as e:
        views_str = f"ERR: {e!s:.20s}"
        t_views = float("inf")

    # --- Dense baseline ---
    try:
        layer_dense = nn.Linear(n, n, bias=True, device=device, dtype=DTYPE).train()
        t_dense, s_dense = benchmark_one(layer_dense, x)
        dense_str = f"{t_dense:8.3f}+-{s_dense:.2f}"
    except Exception as e:
        dense_str = f"ERR: {e!s:.20s}"
        t_dense = float("inf")

    # Speedup vs best non-fused Monarch path
    best_nonfused = min(t_bmm, t_views)
    speedup = best_nonfused / t_fused if t_fused > 0 else 0

    print(f"{block_size:>10} {num_blocks:>10} {batch:>6} {n:>8} | "
          f"{fused_str:>12} {bmm_str:>12} {views_str:>12} {dense_str:>12} | "
          f"{speedup:>7.2f}x")

    # Clean up GPU memory
    del x
    for v in [layer_fused, layer_bmm, layer_views, layer_dense]:
        try:
            del v
        except NameError:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

print(f"{'='*90}")
print("Done.")
