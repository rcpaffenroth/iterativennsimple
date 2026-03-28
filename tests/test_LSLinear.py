"""Tests for LSLinear (L+S decomposition: low-rank + sparse Monarch).

Follows the conventions established in test_MonarchLinear.py.
"""
import copy

import pytest
import torch
import torch.nn as nn

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(
    in_f: int = 16,
    out_f: int = 16,
    num_blocks: int = 4,
    rank: int = 4,
    bias: bool = True,
    seed: int = 42,
) -> LSLinear:
    """Create a small uniform-block LSLinear for testing."""
    return LSLinear.from_uniform_blocks(
        in_features=in_f,
        out_features=out_f,
        num_blocks=num_blocks,
        rank=rank,
        bias=bias,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Test 1: Construction
# ---------------------------------------------------------------------------

def test_construction_uniform():
    in_f, out_f, k, r = 16, 24, 4, 6
    layer = LSLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=k, rank=r, seed=0
    )

    assert layer.in_features == in_f
    assert layer.out_features == out_f
    assert layer.rank == r

    # L component parameter shapes
    assert layer.A.shape == (out_f, r)
    assert layer.B.shape == (r, in_f)

    # S component (inner monarch) must have bias=None
    assert layer.sparse.bias is None
    assert layer.sparse.in_features == in_f
    assert layer.sparse.out_features == out_f
    assert layer.sparse.num_blocks == k

    # Outer bias
    assert layer.bias is not None
    assert layer.bias.shape == (out_f,)


def test_construction_no_bias():
    layer = make_layer(bias=False)
    assert layer.bias is None


def test_construction_block_config():
    block_in = [4, 6, 6]
    block_out = [5, 5, 6]
    in_f = sum(block_in)   # 16
    out_f = sum(block_out)  # 16

    layer = LSLinear.from_block_config(
        in_features=in_f,
        out_features=out_f,
        block_in_features=block_in,
        block_out_features=block_out,
        rank=3,
        seed=1,
    )

    assert layer.in_features == in_f
    assert layer.out_features == out_f
    assert layer.rank == 3
    assert layer.sparse.num_blocks == 3
    assert layer.A.shape == (out_f, 3)
    assert layer.B.shape == (3, in_f)


# ---------------------------------------------------------------------------
# Test 2: Reject inner monarch with bias
# ---------------------------------------------------------------------------

def test_inner_monarch_with_bias_rejected():
    monarch_with_bias = MonarchLinear.from_uniform_blocks(
        in_features=16, out_features=16, num_blocks=4, bias=True, seed=0
    )
    with pytest.raises(ValueError, match="bias=False"):
        LSLinear(monarch_with_bias, rank=4)


# ---------------------------------------------------------------------------
# Test 3: Forward output shape
# ---------------------------------------------------------------------------

def test_forward_shape_batched():
    layer = make_layer(in_f=16, out_f=24)
    x = torch.randn(32, 16)
    y = layer(x)
    assert y.shape == (32, 24)


def test_forward_shape_unbatched():
    layer = make_layer(in_f=16, out_f=16)
    x = torch.randn(16)
    y = layer(x)
    assert y.shape == (16,)


# ---------------------------------------------------------------------------
# Test 4: Forward matches explicit dense computation
# ---------------------------------------------------------------------------

def test_forward_matches_dense():
    layer = make_layer(in_f=16, out_f=16, seed=7)
    x = torch.randn(8, 16)

    y = layer(x)
    W = layer.to_dense()
    y_dense = x @ W.T + layer.bias

    assert torch.allclose(y, y_dense, atol=1e-5), \
        f"max diff: {(y - y_dense).abs().max().item()}"


def test_forward_no_bias_matches_dense():
    layer = make_layer(in_f=16, out_f=16, bias=False, seed=8)
    x = torch.randn(8, 16)

    y = layer(x)
    W = layer.to_dense()
    y_dense = x @ W.T

    assert torch.allclose(y, y_dense, atol=1e-5), \
        f"max diff: {(y - y_dense).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 5: Initialization starts as pure sparse S (A is zero, so L=AB=0)
# ---------------------------------------------------------------------------

def test_init_a_is_zero():
    layer = make_layer()
    assert torch.allclose(layer.A, torch.zeros_like(layer.A)), \
        "A should be zero at initialization"


def test_init_output_equals_sparse():
    """At init (A=0), output should equal S(x) + bias."""
    layer = make_layer(in_f=16, out_f=16, seed=11)
    x = torch.randn(8, 16)

    y_combined = layer(x)
    # S has no bias; layer.bias is the combined bias
    y_sparse_only = layer.sparse(x) + layer.bias

    assert torch.allclose(y_combined, y_sparse_only, atol=1e-6), \
        "At init, combined layer should equal S(x) + bias (since A=0 -> L=AB=0)"


# ---------------------------------------------------------------------------
# Test 6: to_dense
# ---------------------------------------------------------------------------

def test_to_dense_shape():
    layer = make_layer(in_f=16, out_f=24)
    W = layer.to_dense()
    assert W.shape == (24, 16)


def test_to_dense_equals_sparse_plus_low_rank():
    layer = make_layer(in_f=16, out_f=16, seed=3)
    W = layer.to_dense()
    W_expected = layer.sparse.to_dense() + layer.A @ layer.B
    assert torch.allclose(W, W_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    layer = make_layer()
    x = torch.randn(8, 16)

    loss = layer(x).sum()
    loss.backward()

    assert layer.A.grad is not None, "A.grad is None"
    assert layer.B.grad is not None, "B.grad is None"
    assert layer.bias.grad is not None, "bias.grad is None"

    for i in range(layer.sparse.num_blocks):
        assert layer.sparse.block_grad(i) is not None, f"sparse.block_grad({i}) is None"

    # Permutation buffers must never accumulate gradients
    assert layer.sparse.perm_in.grad is None
    assert layer.sparse.perm_out.grad is None


def test_gradient_flow_no_bias():
    layer = make_layer(bias=False)
    x = torch.randn(8, 16)
    layer(x).sum().backward()

    assert layer.A.grad is not None
    assert layer.B.grad is not None
    assert layer.bias is None


# ---------------------------------------------------------------------------
# Test 8: Gradients match equivalent dense nn.Linear
# ---------------------------------------------------------------------------

def test_gradient_matches_dense():
    torch.manual_seed(0)
    layer = make_layer(in_f=16, out_f=16, seed=5)
    x = torch.randn(8, 16)
    target = torch.randn(8, 16)

    # -- LSLinear backward --
    y = layer(x)
    loss = nn.functional.mse_loss(y, target)
    loss.backward()
    ls_bias_grad = layer.bias.grad.clone()

    # -- Equivalent dense nn.Linear --
    W = layer.to_dense().detach()
    dense = nn.Linear(16, 16, bias=True)
    with torch.no_grad():
        dense.weight.copy_(W)
        dense.bias.copy_(layer.bias.detach())

    y_dense = dense(x.detach().clone())
    loss_dense = nn.functional.mse_loss(y_dense, target)
    loss_dense.backward()

    # Bias gradients must match exactly
    assert torch.allclose(ls_bias_grad, dense.bias.grad, atol=1e-5), \
        f"bias grad mismatch: {(ls_bias_grad - dense.bias.grad).abs().max().item()}"

    # Input gradients: x.grad from LSLinear should match dense
    x2 = x.detach().clone().requires_grad_(True)
    x2_dense = x.detach().clone().requires_grad_(True)

    layer.zero_grad()
    loss2 = nn.functional.mse_loss(layer(x2), target)
    loss2.backward()

    layer_dense = nn.Linear(16, 16, bias=True)
    with torch.no_grad():
        layer_dense.weight.copy_(layer.to_dense().detach())
        layer_dense.bias.copy_(layer.bias.detach())
    loss2_dense = nn.functional.mse_loss(layer_dense(x2_dense), target)
    loss2_dense.backward()

    assert torch.allclose(x2.grad, x2_dense.grad, atol=1e-5), \
        f"input grad mismatch: {(x2.grad - x2_dense.grad).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 9: number_of_trainable_parameters
# ---------------------------------------------------------------------------

def test_number_of_trainable_parameters():
    in_f, out_f, k, r = 16, 16, 4, 4
    # S (sparse): k blocks of (out_f/k) x (in_f/k) = 4 x 4 x 4 = 64
    # L (low-rank): A: out_f x r = 64, B: r x in_f = 64
    # bias: out_f = 16
    # total = 64 + 64 + 64 + 16 = 208
    layer = make_layer(in_f=in_f, out_f=out_f, num_blocks=k, rank=r, bias=True)
    expected = (k * (out_f // k) * (in_f // k)) + r * out_f + r * in_f + out_f
    assert layer.number_of_trainable_parameters() == expected, \
        f"got {layer.number_of_trainable_parameters()}, expected {expected}"


def test_number_of_trainable_parameters_no_bias():
    in_f, out_f, k, r = 16, 16, 4, 4
    layer = make_layer(in_f=in_f, out_f=out_f, num_blocks=k, rank=r, bias=False)
    expected = (k * (out_f // k) * (in_f // k)) + r * out_f + r * in_f
    assert layer.number_of_trainable_parameters() == expected


# ---------------------------------------------------------------------------
# Test 10: Save and reload via state_dict
# ---------------------------------------------------------------------------

def test_save_load(tmp_path):
    layer = make_layer(seed=5)
    x = torch.randn(8, 16)
    y_before = layer(x)

    path = tmp_path / "ls.pt"
    torch.save(layer.state_dict(), str(path))

    # Build a fresh instance with different seed and load weights
    layer2 = make_layer(seed=99)
    layer2.load_state_dict(torch.load(str(path), weights_only=True))

    y_after = layer2(x)
    assert torch.allclose(y_before, y_after), \
        f"max diff after reload: {(y_before - y_after).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 11: Optimizer step
# ---------------------------------------------------------------------------

def test_optimizer_step():
    """Verify that parameters update correctly after a gradient step.

    At initialization A=0, so the gradient chain is:
      dL/dA = (dL/dy).T @ (x @ B.T)  -- non-zero because B!=0  -> A updates
      dL/dB = (A @ dL/dy.T).T @ x    -- zero because A=0        -> B is silent on step 1

    This is correct and desirable: the network starts as pure sparse S, and
    the low-rank component L grows in gradually as A develops non-zero values.
    On subsequent steps, B will also update.
    """
    layer = make_layer(seed=99)

    A_before = layer.A.data.clone()
    B_before = layer.B.data.clone()
    blocks_before = [layer.sparse.blocks[i].data.clone() for i in range(layer.sparse.num_blocks)]
    perm_in_before = layer.sparse.perm_in.clone()
    perm_out_before = layer.sparse.perm_out.clone()

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(8, 16)
    layer(x).sum().backward()
    optimizer.step()

    # A receives a gradient (B!=0 so x @ B.T != 0) and must change
    assert not torch.equal(layer.A.data, A_before), "A unchanged after optimizer step"

    # B has zero gradient on step 1 (A=0 means no path back to B) -- this is correct
    assert torch.equal(layer.B.data, B_before), \
        "B should not change on step 1 when A=0 (no gradient path back to B)"

    # After a second step (now A!=0), B should update
    optimizer.zero_grad()
    layer(x).sum().backward()
    optimizer.step()
    assert not torch.equal(layer.B.data, B_before), "B unchanged after second optimizer step"

    # Sparse S blocks must also have changed
    for i in range(layer.sparse.num_blocks):
        assert not torch.equal(layer.sparse.blocks[i].data, blocks_before[i]), \
            f"sparse.blocks[{i}] unchanged after optimizer step"

    # Permutations must not change (buffers, not parameters)
    assert torch.equal(layer.sparse.perm_in, perm_in_before)
    assert torch.equal(layer.sparse.perm_out, perm_out_before)


# ---------------------------------------------------------------------------
# Test 12: Rectangular (in_features != out_features)
# ---------------------------------------------------------------------------

def test_rectangular():
    layer = LSLinear.from_uniform_blocks(
        in_features=32, out_features=16, num_blocks=4, rank=6, seed=11, bias=True
    )
    x = torch.randn(7, 32)

    y = layer(x)
    assert y.shape == (7, 16)

    W = layer.to_dense()
    assert W.shape == (16, 32)

    y_dense = x @ W.T + layer.bias
    assert torch.allclose(y, y_dense, atol=1e-5), \
        f"rectangular: max diff {(y - y_dense).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 13: Deterministic seed
# ---------------------------------------------------------------------------

def test_deterministic_seed():
    a = LSLinear.from_uniform_blocks(16, 16, num_blocks=4, rank=4, seed=42)
    b = LSLinear.from_uniform_blocks(16, 16, num_blocks=4, rank=4, seed=42)
    c = LSLinear.from_uniform_blocks(16, 16, num_blocks=4, rank=4, seed=99)

    assert torch.equal(a.sparse.perm_in, b.sparse.perm_in), \
        "Same seed: perm_in should match"
    assert torch.equal(a.sparse.perm_out, b.sparse.perm_out), \
        "Same seed: perm_out should match"

    assert not (
        torch.equal(a.sparse.perm_in, c.sparse.perm_in)
        and torch.equal(a.sparse.perm_out, c.sparse.perm_out)
    ), "Different seeds produced identical permutations -- very unlikely"


# ---------------------------------------------------------------------------
# Test 14: from_sparsity_target
# ---------------------------------------------------------------------------

def test_from_sparsity_target():
    layer = LSLinear.from_sparsity_target(
        in_features=256,
        out_features=256,
        target_sparsity=0.75,
        rank=16,
        seed=2,
    )
    assert layer.in_features == 256
    assert layer.out_features == 256
    assert layer.rank == 16

    # The sparse S component should have ~75% sparsity
    S = layer.sparse.to_dense()
    total = S.numel()
    nonzero = S.count_nonzero().item()
    achieved = 1.0 - nonzero / total
    assert abs(achieved - 0.75) < 0.1, \
        f"achieved S sparsity {achieved:.3f} too far from 0.75"


# ---------------------------------------------------------------------------
# Test 15: Device move
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_to_device():
    layer_cpu = make_layer()
    x_cpu = torch.randn(8, 16)

    layer_gpu = copy.deepcopy(layer_cpu).to("cuda")
    x_gpu = x_cpu.to("cuda")
    # Force use_fused=False so CPU and GPU use the same algorithm
    # (the Triton fused kernel has small numerical differences from the
    # pure-PyTorch path, which is expected but confuses this test).
    y_gpu = layer_gpu(x_gpu, use_fused=False)

    assert y_gpu.device.type == "cuda"
    assert layer_gpu.A.device.type == "cuda"
    assert layer_gpu.B.device.type == "cuda"
    assert layer_gpu.bias.device.type == "cuda"
    assert layer_gpu.sparse.perm_in.device.type == "cuda"

    y_cpu_via_gpu = y_gpu.cpu()
    y_cpu_direct = layer_cpu(x_cpu)
    assert torch.allclose(y_cpu_via_gpu, y_cpu_direct, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 16: sparse_kwargs forwarded correctly (use_views)
# ---------------------------------------------------------------------------

def test_sparse_kwargs_forwarded():
    """use_views=True and use_views=False should produce the same result."""
    layer = make_layer(seed=17)
    x = torch.randn(8, 16)

    y_views = layer(x, use_views=True)
    y_no_views = layer(x, use_views=False)

    assert torch.allclose(y_views, y_no_views, atol=1e-5), \
        f"sparse_kwargs not forwarded: max diff {(y_views - y_no_views).abs().max().item()}"
