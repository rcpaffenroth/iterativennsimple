# Benchmark for MonarchLinear: parameter counts, timing, and memory.
#
# Run this BEFORE and AFTER changes to compare. It only uses the public API
# (from_uniform_blocks, number_of_trainable_parameters, forward, backward)
# so it works on any version of the code.
#
# Usage:
#   python notebooks/advanced/sparse_scripts/5-factored-monarch-benchmark.py

import time
import torch
from iterativennsimple.MonarchLinear import MonarchLinear


def time_forward_backward(layer, x, warmup=10, repeats=50):
    """Time forward and backward passes, return millisecond averages."""
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad()

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

    return fwd_ms / repeats, bwd_ms / repeats


def param_bytes(layer):
    """Total bytes stored in parameter tensors."""
    return sum(p.numel() * p.element_size() for p in layer.parameters())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    factored = hasattr(MonarchLinear, '_factorization')
    print(f"Factored support: {factored}\n")

    # -----------------------------------------------------------------------
    # 1. Parameter counts and memory
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("STORED PARAMETERS & MEMORY")
    print("=" * 90)
    hdr = f"{'feat':>6} {'blocks':>6} {'blk_sz':>6} {'params':>10} {'KB':>10} {'factored':>8}"
    print(hdr)
    print("-" * len(hdr))

    for features in [512, 1024]:
        for num_blocks in [2, 3, 4, 5, 8, 9, 16, 27, 32]:
            if features % num_blocks != 0:
                continue
            layer = MonarchLinear.from_uniform_blocks(
                features, features, num_blocks, bias=True, seed=42,
            )
            params = layer.number_of_trainable_parameters()
            kb = param_bytes(layer) / 1024
            is_f = str(getattr(layer, 'num_factors', None) is not None)
            d = features // num_blocks
            print(f"{features:>6} {num_blocks:>6} {d:>6} {params:>10} {kb:>10.1f} {is_f:>8}")
    print()

    # -----------------------------------------------------------------------
    # 2. Forward / backward timing
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("FORWARD / BACKWARD TIMING (ms)")
    print("=" * 90)
    hdr = f"{'feat':>6} {'blocks':>6} {'params':>10}  {'fwd':>8} {'bwd':>8} {'total':>8}"
    print(hdr)
    print("-" * len(hdr))

    batch = 64
    configs = [
        (512, 2), (512, 4), (512, 8), (512, 16),
        (1024, 2), (1024, 4), (1024, 8), (1024, 16), (1024, 32),
    ]
    for features, num_blocks in configs:
        layer = MonarchLinear.from_uniform_blocks(
            features, features, num_blocks, bias=True, seed=42, device=device,
        )
        x = torch.randn(batch, features, device=device)
        fwd, bwd = time_forward_backward(layer, x)
        params = layer.number_of_trainable_parameters()
        print(f"{features:>6} {num_blocks:>6} {params:>10}  {fwd:>8.3f} {bwd:>8.3f} {fwd+bwd:>8.3f}")
    print()
