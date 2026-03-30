#!/usr/bin/env python
"""Diagnose where factored BlockDiagLinear overhead comes from."""

import os, sys, time, gc
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_this_dir, "..", "..")
sys.path.insert(0, _root)

import torch
import torch.nn as nn

from iterativennsimple.BlockDiagLinear import BlockDiagLinear

torch.set_float32_matmul_precision('high')
device = torch.device("cuda")

def time_fn(fn, warmup=20, iters=200, label=""):
    """Time a function with CUDA sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    print(f"  {label:50s} {ms:7.3f} ms")
    return ms

batch = 32 * 512  # 16384

configs = [
    (768, 768, "attn_proj (square)"),
    (768, 2048, "ffn_up (rect)"),
    (2048, 768, "ffn_down (rect)"),
]

num_blocks = 8

for in_f, out_f, name in configs:
    print(f"\n{'='*70}")
    print(f" {name}: ({in_f} -> {out_f}), batch={batch}, num_blocks={num_blocks}")
    print(f"{'='*70}")

    x = torch.randn(batch, in_f, device=device)

    # --- Unfactored baseline ---
    bd = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=False).to(device)

    def fwd_unfactored():
        return bd(x)

    def fwd_bwd_unfactored():
        y = bd(x)
        y.sum().backward()
        if x.grad is not None:
            x.grad = None

    x.requires_grad_(False)
    time_fn(fwd_unfactored, label="Unfactored forward only")
    x.requires_grad_(True)
    time_fn(fwd_bwd_unfactored, label="Unfactored forward+backward")

    del bd; gc.collect(); torch.cuda.empty_cache()

    # --- Factored m=2 ---
    bd_f = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=True, chain_length=2).to(device)

    # Time just the materialization
    def materialize_only():
        return bd_f._materialize_weight_stack()

    def fwd_factored():
        return bd_f(x)

    def fwd_bwd_factored():
        y = bd_f(x)
        y.sum().backward()
        if x.grad is not None:
            x.grad = None

    x.requires_grad_(False)
    time_fn(materialize_only, label="Factored materialize only")
    time_fn(fwd_factored, label="Factored forward only")
    x.requires_grad_(True)
    time_fn(fwd_bwd_factored, label="Factored forward+backward")

    # Time with pre-materialized weights (simulating cached path)
    W_cached = bd_f._materialize_weight_stack().detach()

    def fwd_prematerialized():
        batch_size = x.shape[0]
        xr = x.reshape(batch_size, num_blocks, in_f // num_blocks)
        return torch.einsum('koi,bki->bko', W_cached, xr).reshape(batch_size, out_f)

    x.requires_grad_(False)
    time_fn(fwd_prematerialized, label="Pre-materialized einsum (no autograd)")

    # Break down: autograd overhead of materialization
    def materialize_with_grad():
        W = bd_f._materialize_weight_stack()
        return W.sum()  # force grad computation

    def materialize_bwd():
        s = materialize_with_grad()
        s.backward()

    time_fn(materialize_bwd, label="Materialize + backward (grad through chain)")

    del bd_f, W_cached; gc.collect(); torch.cuda.empty_cache()

    # --- Test: unfactored but with an extra detached bmm to simulate overhead ---
    bd2 = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=False).to(device)
    dummy = torch.randn(num_blocks, min(in_f,out_f)//num_blocks, min(in_f,out_f)//num_blocks, device=device)

    def fwd_with_dummy_bmm():
        _ = torch.bmm(dummy, dummy)  # simulate materialization cost
        return bd2(x)

    x.requires_grad_(False)
    time_fn(fwd_with_dummy_bmm, label="Unfactored + dummy bmm (overhead estimate)")

    del bd2, dummy; gc.collect(); torch.cuda.empty_cache()

    # --- Test: does torch.compile help factored? ---
    bd_fc = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=True, chain_length=2).to(device)
    bd_fc_compiled = torch.compile(bd_fc)

    def fwd_factored_compiled():
        return bd_fc_compiled(x)

    def fwd_bwd_factored_compiled():
        y = bd_fc_compiled(x)
        y.sum().backward()
        if x.grad is not None:
            x.grad = None

    x.requires_grad_(False)
    time_fn(fwd_factored_compiled, label="Factored compiled forward only")
    x.requires_grad_(True)
    time_fn(fwd_bwd_factored_compiled, label="Factored compiled forward+backward")

    del bd_fc, bd_fc_compiled; gc.collect(); torch.cuda.empty_cache()
