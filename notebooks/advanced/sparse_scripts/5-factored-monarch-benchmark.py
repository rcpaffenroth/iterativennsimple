# Benchmark: Factored vs Standard MonarchLinear block-diagonal storage.
#
# Measures parameter counts, forward/backward timing, and memory usage.
# Works both before and after the factored block-diagonal change —
# when factored mode is unavailable, reports "N/A" for factored columns.
#
# Usage:
#   python notebooks/advanced/sparse_scripts/5-factored-monarch-benchmark.py

import time
import torch
from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_factored_support() -> bool:
    """Check whether MonarchLinear has factored block-diagonal support."""
    return hasattr(MonarchLinear, '_factorization')


def time_forward_backward(layer, x, warmup: int = 5, repeats: int = 20) -> dict:
    """Time forward and backward passes, returning millisecond averages."""
    # Warmup
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad()

    # Timed runs
    fwd_ms, bwd_ms = 0.0, 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = layer(x)
        t1 = time.perf_counter()
        y.sum().backward()
        t2 = time.perf_counter()
        layer.zero_grad()
        fwd_ms += (t1 - t0) * 1000
        bwd_ms += (t2 - t1) * 1000

    return {
        "forward_ms": fwd_ms / repeats,
        "backward_ms": bwd_ms / repeats,
        "total_ms": (fwd_ms + bwd_ms) / repeats,
    }


def param_bytes(layer) -> int:
    """Total bytes of stored parameter tensors."""
    return sum(p.numel() * p.element_size() for p in layer.parameters())


# ---------------------------------------------------------------------------
# 1. Parameter count comparison
# ---------------------------------------------------------------------------

def benchmark_param_counts():
    print("=" * 80)
    print("1. PARAMETER COUNT COMPARISON")
    print("=" * 80)
    header = f"{'features':>8} {'num_blocks':>10} {'standard':>10} {'factored':>10} {'ratio':>8} {'factored?':>10}"
    print(header)
    print("-" * len(header))

    # Include non-factorable num_blocks (2, 3, 5, 6, 7) alongside factorable ones
    # (4, 8, 9, 16, 27) to show the contrast.
    for features in [256, 512, 1024]:
        for num_blocks in [2, 3, 4, 5, 6, 7, 8, 9, 16, 27, 32]:
            if features % num_blocks != 0:
                continue

            # Standard: always create independent blocks (even if factoring available)
            standard_params = num_blocks * (features // num_blocks) ** 2

            if has_factored_support():
                layer = MonarchLinear.from_uniform_blocks(features, features, num_blocks, seed=0)
                factored_params = layer.number_of_trainable_parameters()
                is_factored = layer.num_factors is not None
                ratio = f"{factored_params / standard_params:.3f}"
                f_str = f"{factored_params:>10}"
                mode = "yes" if is_factored else "no"
            else:
                f_str = "N/A".rjust(10)
                ratio = "N/A".rjust(8)
                mode = "N/A".rjust(10)

            print(f"{features:>8} {num_blocks:>10} {standard_params:>10} {f_str} {ratio:>8} {mode:>10}")
    print()


# ---------------------------------------------------------------------------
# 2. Forward / backward timing
# ---------------------------------------------------------------------------

def benchmark_timing():
    print("=" * 80)
    print("2. FORWARD / BACKWARD TIMING (milliseconds)")
    print("=" * 80)

    if not has_factored_support():
        print("  Factored mode not available — skipping timing comparison.\n")
        return

    batch = 64
    configs = [
        (512, 2),    # NOT factorable — baseline
        (512, 4),    # factored: 2^2
        (512, 8),    # factored: 2^3
        (512, 16),   # factored: 2^4
        (1024, 2),   # NOT factorable — baseline
        (1024, 4),   # factored: 2^2
        (1024, 8),   # factored: 2^3
        (1024, 16),  # factored: 2^4
        (1024, 32),  # factored: 2^5
    ]

    header = f"{'features':>8} {'blocks':>6} {'factored?':>9}  {'fwd_ms':>8} {'bwd_ms':>8} {'total_ms':>8}  {'params':>8}"
    print(header)
    print("-" * len(header))

    for features, num_blocks in configs:
        if features % num_blocks != 0:
            continue
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = MonarchLinear.from_uniform_blocks(features, features, num_blocks, bias=True, seed=42, device=device)
        x = torch.randn(batch, features, device=device, requires_grad=True)
        t = time_forward_backward(layer, x, warmup=10, repeats=50)
        is_f = "yes" if layer.num_factors is not None else "no"
        p = layer.number_of_trainable_parameters()
        print(f"{features:>8} {num_blocks:>6} {is_f:>9}  {t['forward_ms']:>8.3f} {t['backward_ms']:>8.3f} {t['total_ms']:>8.3f}  {p:>8}")
    print()


# ---------------------------------------------------------------------------
# 3. Memory comparison
# ---------------------------------------------------------------------------

def benchmark_memory():
    print("=" * 80)
    print("3. MEMORY USAGE (parameter tensor bytes)")
    print("=" * 80)

    if not has_factored_support():
        print("  Factored mode not available — skipping memory comparison.\n")
        return

    header = f"{'features':>8} {'blocks':>6} {'standard_KB':>12} {'factored_KB':>12} {'savings':>8}"
    print(header)
    print("-" * len(header))

    for features in [512, 1024]:
        for num_blocks in [2, 3, 4, 5, 8, 9, 16, 27, 32]:
            if features % num_blocks != 0:
                continue

            # Standard param count (what it would be without factoring)
            d = features // num_blocks
            standard_bytes = num_blocks * d * d * 4  # float32 = 4 bytes

            layer = MonarchLinear.from_uniform_blocks(features, features, num_blocks, seed=0)
            actual_bytes = param_bytes(layer)

            std_kb = standard_bytes / 1024
            act_kb = actual_bytes / 1024
            savings = f"{1 - actual_bytes / standard_bytes:.1%}" if actual_bytes < standard_bytes else "none"
            print(f"{features:>8} {num_blocks:>6} {std_kb:>12.1f} {act_kb:>12.1f} {savings:>8}")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Factored block-diagonal support: {has_factored_support()}\n")
    benchmark_param_counts()
    benchmark_timing()
    benchmark_memory()
