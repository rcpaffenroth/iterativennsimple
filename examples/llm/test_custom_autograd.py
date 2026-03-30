#!/usr/bin/env python
"""Verify custom autograd Function matches autograd-traced materialization."""

import os, sys
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _root)

import torch
from iterativennsimple.BlockDiagLinear import BlockDiagLinear

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_case(in_f, out_f, num_blocks, chain_length, label):
    print(f"\n--- {label}: ({in_f}->{out_f}), K={num_blocks}, m={chain_length} ---")

    # Create factored layer
    bd = BlockDiagLinear(in_f, out_f, num_blocks, bias=False,
                         factored=True, chain_length=chain_length).to(device)

    x = torch.randn(16, in_f, device=device, requires_grad=True)

    # --- Custom autograd path (default) ---
    y_custom = bd(x)
    loss_custom = y_custom.sum()
    loss_custom.backward()

    grad_fs_custom = bd.factor_stack.grad.clone()
    grad_x_custom = x.grad.clone()
    grad_adapter_custom = bd.adapter.grad.clone() if bd.adapter is not None else None

    # Zero grads
    bd.factor_stack.grad = None
    if bd.adapter is not None:
        bd.adapter.grad = None
    x.grad = None

    # --- Autograd-traced path (reference) ---
    # Manually run the old path: materialize then einsum
    W = bd._materialize_weight_stack()
    batch = x.shape[0]
    x_blocked = x.reshape(batch, num_blocks, in_f // num_blocks)
    y_ref = torch.einsum('koi,bki->bko', W, x_blocked).reshape(batch, out_f)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    grad_fs_ref = bd.factor_stack.grad.clone()
    grad_x_ref = x.grad.clone()
    grad_adapter_ref = bd.adapter.grad.clone() if bd.adapter is not None else None

    # Compare
    y_match = torch.allclose(y_custom.reshape(batch, out_f), y_ref, atol=1e-5, rtol=1e-4)
    fs_match = torch.allclose(grad_fs_custom, grad_fs_ref, atol=1e-4, rtol=1e-3)
    x_match = torch.allclose(grad_x_custom, grad_x_ref, atol=1e-5, rtol=1e-4)

    print(f"  Forward match:          {y_match}  (max diff: {(y_custom.reshape(batch,out_f) - y_ref).abs().max():.2e})")
    print(f"  factor_stack grad match:{fs_match}  (max diff: {(grad_fs_custom - grad_fs_ref).abs().max():.2e})")
    print(f"  x grad match:           {x_match}  (max diff: {(grad_x_custom - grad_x_ref).abs().max():.2e})")

    if grad_adapter_custom is not None:
        ad_match = torch.allclose(grad_adapter_custom, grad_adapter_ref, atol=1e-4, rtol=1e-3)
        print(f"  adapter grad match:     {ad_match}  (max diff: {(grad_adapter_custom - grad_adapter_ref).abs().max():.2e})")
    else:
        ad_match = True

    ok = y_match and fs_match and x_match and ad_match
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok

all_pass = True
# Square, chain=2
all_pass &= test_case(768, 768, 8, 2, "square m=2")
# Square, chain=3
all_pass &= test_case(768, 768, 8, 3, "square m=3")
# Square, chain=1
all_pass &= test_case(768, 768, 8, 1, "square m=1")
# Rectangular (wide), chain=2 — adapter at "end"
all_pass &= test_case(768, 2048, 8, 2, "rect wide m=2")
# Rectangular (tall), chain=2 — adapter at "start"
all_pass &= test_case(2048, 768, 8, 2, "rect tall m=2")
# Rectangular, chain=3
all_pass &= test_case(768, 2048, 8, 3, "rect wide m=3")
all_pass &= test_case(2048, 768, 8, 3, "rect tall m=3")

print(f"\n{'='*50}")
print(f"ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
