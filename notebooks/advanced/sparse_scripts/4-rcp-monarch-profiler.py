"""
MonarchLinear PyTorch Profiler Analysis

Kernel-level profiling of `MonarchLinear` using the PyTorch Profiler.
Records every CPU and CUDA kernel call to identify bottlenecks and
exports a Chrome trace JSON for visualisation in
Perfetto (https://ui.perfetto.dev/) or chrome://tracing.
Converted from: notebooks/advanced/8a-rcp-monarch-performance.ipynb
"""

# ==============================================================================
# 1. Import Required Libraries
# ==============================================================================

import warnings

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from iterativennsimple.MonarchLinear import MonarchLinear

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU memory: {total_mem / 1024**3:.2f} GB")


# ==============================================================================
# 2. Configuration
# ==============================================================================

PROF_SCALE      = 4
PROF_SIZE       = PROF_SCALE * 16 * 1024  # layer in_features / out_features
PROF_BATCH      = 128                     # batch size
PROF_NUM_BLOCKS = PROF_SCALE * 8          # Monarch blocks 
PROF_WARMUP     = 5
PROF_ITERS      = 20
PROF_DTYPE      = torch.bfloat16          # set to None to keep float32, or try torch.float16

print(f"Layer size : {PROF_SIZE:,}  x  {PROF_SIZE:,}")
print(f"Num blocks : {PROF_NUM_BLOCKS}")
print(f"Batch size : {PROF_BATCH}")
print(f"dtype      : {PROF_DTYPE}")


# ==============================================================================
# 3. First-Principles Memory Estimates
#
# For a square layer of width `n = PROF_SIZE` with `dtype = PROF_DTYPE`:
#
# - **MonarchLinear** (Monarch): `k` blocks, each of size `(n/k) x (n/k)`
#   -> total = `k * (n/k)^2 = n^2/k` elements (a `k`-fold reduction vs. dense)
#
# Both layers also store two permutation index vectors of length `n` (int64),
# which are negligible for large `n`.
# ==============================================================================

_dtype = PROF_DTYPE if PROF_DTYPE is not None else torch.float32
_bytes = torch.finfo(_dtype).bits // 8 if _dtype.is_floating_point else torch.iinfo(_dtype).bits // 8
n   = PROF_SIZE
k   = PROF_NUM_BLOCKS
bs  = PROF_BATCH

dense_params   = n * n
monarch_params = n * n // k
act_elements   = bs * n

def _fmt(nbytes: int) -> str:
    """Human-readable byte count: B / KB / MB / GB."""
    for unit, thr in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if nbytes >= thr:
            return f"{nbytes / thr:.2f} {unit}"
    return f"{nbytes} B"

print(f"dtype : {_dtype}  ({_bytes} bytes/element)")
print(f"n     = {n:,}   k (num_blocks) = {k}   batch = {bs:,}")
print()
print("Weight matrices (parameters only, excludes bias & permutations):")
print(f"  Dense reference  n^2     = {dense_params:>15,} elements  ->  {_fmt(dense_params * _bytes)}")
print(f"  MonarchLinear    n^2/k   = {monarch_params:>15,} elements  ->  {_fmt(monarch_params * _bytes)}"
      f"  (reduction {k}x)")
print()
print("Activation tensor (batch x n, forward pass):")
print(f"  {bs} x {n} = {act_elements:>15,} elements  ->  {_fmt(act_elements * _bytes)}")
print()
# Rough total peak memory: weights + gradients (~same size) + input + output activation
peak = (monarch_params * 2 + act_elements * 2) * _bytes
print(f"  Rough peak (weights + grads + activations) - MonarchLinear: {_fmt(peak)}")


# ==============================================================================
# 4. PyTorch Profiler: Kernel-Level Analysis
#
# The PyTorch Profiler (https://pytorch.org/docs/stable/profiler.html)
# records every CPU and CUDA kernel call, making it easy to identify bottlenecks.
#
# `profile_layer` wraps a forward+backward loop in `torch.profiler.profile`.  After
# `iters` profiled steps it returns a finished `Profile` object whose `.key_averages()`
# method summarises self-CPU time, self-CUDA time, and memory allocations per operator.
#
# A Chrome trace JSON is also exported — open it at `chrome://tracing` or in
# Perfetto (https://ui.perfetto.dev/) for a graphical flame-chart view.
# ==============================================================================

def profile_layer(
    layer: nn.Module,
    x: torch.Tensor,
    layer_name: str = "layer",
    warmup: int = 5,
    iters: int = 20,
    export_trace: bool = True,
    dtype: torch.dtype | None = None,
) -> "torch.profiler.profile":
    """Profile a single layer's forward + backward pass with the PyTorch profiler.

    Args:
        layer:        Layer to profile (already on the correct device).
        x:            Input tensor (already on the correct device).
        layer_name:   Label used in record_function scopes and the trace filename.
        warmup:       Warmup steps before recording starts.
        iters:        Number of profiled steps.
        export_trace: If True, write a Chrome-format JSON trace to
                      ``{layer_name}_{dtype}_trace.json``.
        dtype:        Optional dtype to cast both the layer parameters and the
                      input tensor before profiling (e.g. ``torch.bfloat16``,
                      ``torch.float16``, ``torch.float32``).  If ``None``, the
                      layer and input are used as-is.

    Returns:
        The finished ``torch.profiler.profile`` object.
    """
    use_cuda = x.device.type == "cuda"
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    if dtype is not None:
        layer = layer.to(dtype=dtype)
        x = x.to(dtype=dtype)
        dtype_name = str(dtype).replace("torch.", "")
    else:
        dtype_name = str(x.dtype).replace("torch.", "")

    # Warmup — run outside the profiler to avoid cold-start artefacts
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad = None
    if use_cuda:
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            with record_function(f"{layer_name}_fwd"):
                y = layer(x)
            with record_function(f"{layer_name}_bwd"):
                y.sum().backward()
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad = None

    if export_trace:
        trace_path = f"{layer_name}_{dtype_name}_trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"  Chrome trace saved -> {trace_path}  (open in chrome://tracing or Perfetto)")

    return prof


print("profile_layer() defined.")


# ==============================================================================
# 5. Run Profiler on MonarchLinear
# ==============================================================================

# Print the current memory use on the device before starting the profiler (useful for debugging OOMs)
if device.type == "cuda":
    mem_alloc = torch.cuda.memory_allocated(device)
    mem_reserved = torch.cuda.memory_reserved(device)
    print(f"Initial GPU memory: allocated {_fmt(mem_alloc)}, reserved {_fmt(mem_reserved)}")

layer = MonarchLinear.from_uniform_blocks(
    PROF_SIZE, PROF_SIZE, num_blocks=PROF_NUM_BLOCKS, bias=True, seed=0, 
    force_loop_matmul=True
).to(device).train()

x_prof = torch.randn(PROF_BATCH, PROF_SIZE, device=device)

sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
dtype_label = str(PROF_DTYPE).replace("torch.", "") if PROF_DTYPE is not None else "float32"

print(f"{'='*60}")
print(f"Profiling MonarchLinear  (size={PROF_SIZE}, batch={PROF_BATCH}, dtype={dtype_label})")
print("=" * 60)

if device.type == "cuda":
    mem_alloc = torch.cuda.memory_allocated(device)
    mem_reserved = torch.cuda.memory_reserved(device)
    print(f"Before profiling: allocated {_fmt(mem_alloc)}, reserved {_fmt(mem_reserved)}")

prof = profile_layer(
    layer, x_prof,
    layer_name="MonarchLinear",
    warmup=PROF_WARMUP,
    iters=PROF_ITERS,
    export_trace=True,
    dtype=PROF_DTYPE,
)

if device.type == "cuda":
    mem_alloc = torch.cuda.memory_allocated(device)
    mem_reserved = torch.cuda.memory_reserved(device)
    print(f"After profiling: allocated {_fmt(mem_alloc)}, reserved {_fmt(mem_reserved)}")

print(
    prof.key_averages().table(
        sort_by=sort_key,
        row_limit=20,
    )
)

del layer, x_prof
if device.type == "cuda":
    torch.cuda.empty_cache()
