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


def benchmark_one_cuda_graph(layer, x, forward_fn=None, warmup=WARMUP, iters=ITERS):
    """Time forward + backward using a captured CUDA graph. Returns avg ms per replay.

    The capture overhead is excluded from the reported timings.
    """
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph benchmarking requires a CUDA device")

    call = forward_fn if forward_fn is not None else layer
    static_x = x.detach().clone()

    # Warm up eager execution first so parameter grads and kernels are initialized
    # before capture. This avoids allocations during capture/replay.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            y = call(static_x)
            y.sum().backward()
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = call(static_x)
        y.sum().backward()

    # One replay to ensure grads are materialized on their steady-state buffers.
    graph.replay()
    for p in layer.parameters():
        if p.grad is not None:
            p.grad.zero_()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


# ============================================================================
# Sweep
# ============================================================================

print(f"\n{'='*142}")
print(f"{'block_size':>10} {'num_blocks':>10} {'batch':>6} {'size':>8} | "
      f"{'Fused (ms)':>12} {'Graph (ms)':>12} {'Fused+Graph (ms)':>18} {'BMM (ms)':>12} {'Views (ms)':>12} {'Dense (ms)':>12} | "
      f"{'FusedSpd':>8} {'GraphSpd':>8} {'FGSpd':>8}")
print(f"{'='*142}")

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

    # --- Graph only (non-fused BMM path) ---
    try:
        if device.type != "cuda":
            raise RuntimeError("n/a")
        layer_graph = MonarchLinear.from_uniform_blocks(
            n, n, num_blocks=num_blocks, bias=True, seed=0
        ).to(device).to(DTYPE).train()
        t_graph, s_graph = benchmark_one_cuda_graph(
            layer_graph,
            x,
            forward_fn=lambda inp, _l=layer_graph: _l(inp, use_fused=False, use_views=False),
        )
        graph_str = f"{t_graph:8.3f}+-{s_graph:.2f}"
    except Exception as e:
        graph_str = f"ERR: {e!s:.20s}"
        t_graph = float("inf")

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

    # --- Fused + CUDA Graph ---
    try:
        if device.type != "cuda":
            raise RuntimeError("n/a")
        layer_fused_graph = MonarchLinear.from_uniform_blocks(
            n, n, num_blocks=num_blocks, bias=True, seed=0
        ).to(device).to(DTYPE).train()
        t_fused_graph, s_fused_graph = benchmark_one_cuda_graph(
            layer_fused_graph,
            x,
            forward_fn=lambda inp, _l=layer_fused_graph: _l(inp, use_fused=True),
        )
        fused_graph_str = f"{t_fused_graph:8.3f}+-{s_fused_graph:.2f}"
    except Exception as e:
        fused_graph_str = f"ERR: {e!s:.20s}"
        t_fused_graph = float("inf")

    # Speedup vs best non-fused Monarch path
    best_nonfused = min(t_bmm, t_views)
    fused_speedup = best_nonfused / t_fused if t_fused > 0 else 0
    graph_speedup = best_nonfused / t_graph if t_graph > 0 else 0
    fused_graph_speedup = best_nonfused / t_fused_graph if t_fused_graph > 0 else 0

    print(f"{block_size:>10} {num_blocks:>10} {batch:>6} {n:>8} | "
          f"{fused_str:>12} {graph_str:>12} {fused_graph_str:>18} {bmm_str:>12} {views_str:>12} {dense_str:>12} | "
          f"{fused_speedup:>7.2f}x {graph_speedup:>7.2f}x {fused_graph_speedup:>7.2f}x")

    # Clean up GPU memory
    del x
    for v in [layer_fused, layer_graph, layer_fused_graph, layer_bmm, layer_views, layer_dense]:
        try:
            del v
        except NameError:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

print(f"{'='*142}")
print("Done.")
