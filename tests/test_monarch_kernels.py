"""Tests for the fused Triton kernel path of MonarchLinear.

All tests compare the fused kernel output against the existing pure-PyTorch
path (use_views=True) to verify numerical correctness of the forward and
backward passes.

These tests require an NVIDIA GPU with CUDA and the ``triton`` package.
They are automatically skipped when either is unavailable.

Note on tolerances:
    Triton's ``tl.dot`` uses TF32 tensor cores by default on Ampere/Ada GPUs
    (e.g. RTX 4090), which provides ~10-bit mantissa precision instead of
    float32's 23 bits.  Combined with different accumulation order between
    the fused kernel and the reference PyTorch path, this typically yields
    max absolute differences of ~1e-3.  Tolerances are set accordingly.
"""

import pytest
import torch
import torch.nn as nn

from iterativennsimple.MonarchLinear import MonarchLinear

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA GPU available"
)

try:
    import triton  # noqa: F401
    _has_triton = True
except ImportError:
    _has_triton = False

_skip_no_triton = pytest.mark.skipif(
    not _has_triton, reason="Triton not installed"
)

# Combine both skips for convenience
requires_fused = pytest.mark.usefixtures()  # placeholder
requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_triton),
    reason="Requires CUDA + Triton",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monarch(in_f=64, out_f=64, num_blocks=4, bias=True, seed=42, device="cuda",
                 factored=False):
    """Create a MonarchLinear on the specified device.

    Uses factored=False by default so that non-factored kernel tests remain
    isolated. Factored-specific tests use make_factored() instead.
    """
    return MonarchLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=num_blocks, bias=bias, seed=seed, factored=factored,
    ).to(device)


# ---------------------------------------------------------------------------
# Forward correctness tests
# ---------------------------------------------------------------------------

@requires_fused
class TestFusedForward:
    """Verify that the fused forward matches the pure-PyTorch forward."""

    def test_basic_square(self):
        layer = make_monarch(64, 64, num_blocks=4, seed=1)
        x = torch.randn(32, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False, use_views=True)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_rectangular(self):
        layer = make_monarch(64, 32, num_blocks=4, seed=2)
        x = torch.randn(16, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_no_bias(self):
        layer = make_monarch(64, 64, num_blocks=4, bias=False, seed=3)
        x = torch.randn(8, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_unbatched(self):
        layer = make_monarch(64, 64, num_blocks=4, seed=4)
        x = torch.randn(64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert y_fused.shape == (64,)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    @pytest.mark.parametrize("in_f,out_f,k,batch", [
        (32, 32, 2, 8),
        (64, 64, 8, 16),
        (128, 64, 4, 32),
        (64, 128, 4, 32),
        (256, 256, 16, 64),
    ])
    def test_various_sizes(self, in_f, out_f, k, batch):
        layer = make_monarch(in_f, out_f, num_blocks=k, seed=10)
        x = torch.randn(batch, in_f, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"({in_f},{out_f},k={k},batch={batch}) max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_matches_dense(self):
        """Fused forward matches explicit dense computation x @ S.T + b."""
        layer = make_monarch(64, 64, num_blocks=4, seed=7)
        x = torch.randn(8, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        S = layer.to_dense()
        y_dense = x @ S.T
        if layer.bias is not None:
            y_dense = y_dense + layer.bias
        assert torch.allclose(y_fused, y_dense, atol=3e-3), \
            f"max diff: {(y_fused - y_dense).abs().max().item()}"


# ---------------------------------------------------------------------------
# Backward correctness tests
# ---------------------------------------------------------------------------

@requires_fused
class TestFusedBackward:
    """Verify that gradients from the fused path match the pure-PyTorch path."""

    def _compare_grads(self, layer_fused, layer_ref, x):
        """Run forward+backward on both paths and compare gradients."""
        # Fused path
        x_f = x.clone().detach()
        y_f = layer_fused(x_f, use_fused=True)
        loss_f = y_f.sum()
        loss_f.backward()

        # Reference path
        x_r = x.clone().detach()
        y_r = layer_ref(x_r, use_fused=False)
        loss_r = y_r.sum()
        loss_r.backward()

        # Compare block gradients — backward involves two matmul steps
        # (forward + weight gradient) so TF32 error accumulates more.
        for i in range(layer_fused.num_blocks):
            gf = layer_fused.block_grad(i)
            gr = layer_ref.block_grad(i)
            assert gf is not None, f"fused block_grad({i}) is None"
            assert gr is not None, f"ref block_grad({i}) is None"
            assert torch.allclose(gf, gr, atol=1e-2), \
                f"block[{i}] grad max diff: {(gf - gr).abs().max().item()}"

        # Compare bias gradients
        if layer_fused.bias is not None:
            assert torch.allclose(layer_fused.bias.grad, layer_ref.bias.grad, atol=1e-2), \
                f"bias grad max diff: {(layer_fused.bias.grad - layer_ref.bias.grad).abs().max().item()}"

    def test_grad_blocks_and_bias(self):
        layer = make_monarch(64, 64, num_blocks=4, seed=20)
        # Use the same layer for both paths (no weight update between them)
        # We need separate layers with same weights to avoid gradient accumulation
        layer2 = make_monarch(64, 64, num_blocks=4, seed=20)
        layer2.load_state_dict(layer.state_dict())

        x = torch.randn(16, 64, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_grad_no_bias(self):
        layer = make_monarch(64, 64, num_blocks=4, bias=False, seed=21)
        layer2 = make_monarch(64, 64, num_blocks=4, bias=False, seed=21)
        layer2.load_state_dict(layer.state_dict())

        x = torch.randn(16, 64, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_grad_rectangular(self):
        layer = make_monarch(128, 64, num_blocks=4, seed=22)
        layer2 = make_monarch(128, 64, num_blocks=4, seed=22)
        layer2.load_state_dict(layer.state_dict())

        x = torch.randn(8, 128, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_gradient_matches_dense(self):
        """Fused backward matches dense nn.Linear backward at Monarch positions."""
        torch.manual_seed(0)
        layer = make_monarch(64, 64, num_blocks=4, seed=3)
        x = torch.randn(8, 64, device="cuda")
        target = torch.randn(8, 64, device="cuda")

        # Fused backward
        y = layer(x, use_fused=True)
        loss = nn.functional.mse_loss(y, target)
        loss.backward()

        # Reconstruct S_grad from block gradients
        M_grad = torch.block_diag(*[layer.block_grad(i).clone() for i in range(layer.num_blocks)])
        S_grad_temp = torch.zeros(layer.out_features, layer.in_features, device="cuda")
        S_grad_temp[layer.perm_out] = M_grad
        S_grad = torch.zeros_like(S_grad_temp)
        S_grad[:, layer.perm_in] = S_grad_temp
        monarch_bias_grad = layer.bias.grad.clone()

        # Build mask for Monarch positions
        monarch_mask = torch.zeros(layer.out_features, layer.in_features, dtype=torch.bool, device="cuda")
        row_off, col_off = 0, 0
        for k in range(layer.num_blocks):
            bor = layer.block_out_features[k]
            bir = layer.block_in_features[k]
            rows = layer.perm_out[row_off:row_off + bor]
            cols = layer.perm_in[col_off:col_off + bir]
            monarch_mask[rows[:, None], cols[None, :]] = True
            row_off += bor
            col_off += bir

        # Dense reference
        S = layer.to_dense().detach()
        dense = nn.Linear(64, 64, bias=True, device="cuda")
        with torch.no_grad():
            dense.weight.copy_(S)
            dense.bias.copy_(layer.bias.detach())
        y_dense = dense(x.detach().clone())
        loss_dense = nn.functional.mse_loss(y_dense, target)
        loss_dense.backward()

        assert torch.allclose(S_grad[monarch_mask], dense.weight.grad[monarch_mask], atol=3e-3), \
            f"weight grad mismatch: {(S_grad[monarch_mask] - dense.weight.grad[monarch_mask]).abs().max().item()}"
        assert torch.allclose(monarch_bias_grad, dense.bias.grad, atol=3e-3), \
            f"bias grad mismatch: {(monarch_bias_grad - dense.bias.grad).abs().max().item()}"

    def test_optimizer_step(self):
        """SGD step updates blocks but not permutations when using fused path."""
        layer = make_monarch(64, 64, num_blocks=4, seed=99)
        blocks_before = [layer.blocks[i].data.clone() for i in range(layer.num_blocks)]
        perm_in_before = layer.perm_in.clone()
        perm_out_before = layer.perm_out.clone()

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(8, 64, device="cuda")
        loss = layer(x, use_fused=True).sum()
        loss.backward()
        optimizer.step()

        for i in range(layer.num_blocks):
            assert not torch.equal(layer.blocks[i].data, blocks_before[i]), \
                f"block[{i}] unchanged"

        assert torch.equal(layer.perm_in, perm_in_before)
        assert torch.equal(layer.perm_out, perm_out_before)


# ---------------------------------------------------------------------------
# Half precision tests
# ---------------------------------------------------------------------------

@requires_fused
class TestFusedHalfPrecision:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_half(self, dtype):
        layer = make_monarch(64, 64, num_blocks=4, seed=50).to(dtype)
        x = torch.randn(16, 64, device="cuda", dtype=dtype)
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused.float(), y_ref.float(), atol=1e-2), \
            f"max diff ({dtype}): {(y_fused - y_ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# Fallback test
# ---------------------------------------------------------------------------

def test_fallback_on_cpu():
    """_can_use_fused returns False on CPU, so forward still works."""
    layer = MonarchLinear.from_uniform_blocks(32, 32, num_blocks=4, seed=1, factored=False)
    x = torch.randn(8, 32)
    y = layer(x)  # should use Python path, no error
    assert y.shape == (8, 32)


@requires_fused
def test_auto_detect_uses_fused_on_cuda():
    """When use_fused=None (default), CUDA input with ≥8 blocks should auto-select fused path."""
    # num_blocks=8 — enough GPU parallelism for the fused path to be preferred.
    layer = make_monarch(64, 64, num_blocks=8, seed=1)
    assert layer._can_use_fused(torch.randn(1, 64, device="cuda"))


@requires_fused
def test_auto_detect_uses_bmm_for_few_blocks():
    """With < 8 blocks, auto-detect should prefer BMM (not fused) for better perf."""
    layer = make_monarch(64, 64, num_blocks=4, seed=1)
    assert not layer._can_use_fused(torch.randn(1, 64, device="cuda"))
    # But explicit use_fused=True should still work
    x = torch.randn(8, 64, device="cuda")
    y = layer(x, use_fused=True)
    y_ref = layer(x, use_fused=False)
    assert torch.allclose(y, y_ref, atol=3e-3)


# ===========================================================================
# Factored block tests
# ===========================================================================

def make_factored(dim=64, num_blocks=4, bias=True, seed=42, device="cuda"):
    """Create a factored MonarchLinear on the specified device."""
    return MonarchLinear.from_uniform_blocks(
        dim, dim, num_blocks=num_blocks, bias=bias, seed=seed, factored=True
    ).to(device)


@requires_fused
class TestFactoredForward:
    """Verify factored fused forward matches the views fallback."""

    def test_basic_4_blocks(self):
        layer = make_factored(64, num_blocks=4, seed=1)
        x = torch.randn(32, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False, use_views=True)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_9_blocks(self):
        layer = make_factored(9 * 16, num_blocks=9, seed=2)
        x = torch.randn(16, 9 * 16, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False, use_views=True)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_16_blocks(self):
        layer = make_factored(256, num_blocks=16, seed=3)
        x = torch.randn(8, 256, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False, use_views=True)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_no_bias(self):
        layer = make_factored(64, num_blocks=4, bias=False, seed=4)
        x = torch.randn(8, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False, use_views=True)
        assert torch.allclose(y_fused, y_ref, atol=3e-3), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_matches_dense(self):
        """Factored fused forward matches dense x @ S.T + b."""
        layer = make_factored(64, num_blocks=4, seed=7)
        x = torch.randn(8, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        S = layer.to_dense()
        y_dense = x @ S.T
        if layer.bias is not None:
            y_dense = y_dense + layer.bias
        assert torch.allclose(y_fused, y_dense, atol=3e-3), \
            f"max diff: {(y_fused - y_dense).abs().max().item()}"

    def test_parameter_savings(self):
        """Factored mode stores fewer parameters than unfactored."""
        factored = make_factored(64, num_blocks=4, seed=1, device="cpu")
        unfactored = MonarchLinear.from_uniform_blocks(
            64, 64, num_blocks=4, seed=1, factored=False,
        )
        # 4 blocks -> 2 factors: 2x savings in weight params
        assert factored.num_factors == 2
        assert factored.factor_stack.shape == (2, 16, 16)
        assert factored.number_of_trainable_parameters() < unfactored.number_of_trainable_parameters()


@requires_fused
class TestFactoredBackward:
    """Verify factored fused backward matches the views fallback."""

    def _compare_grads(self, layer_fused, layer_ref, x):
        """Run forward+backward on both paths and compare factor_stack gradients."""
        x_f = x.clone().detach()
        y_f = layer_fused(x_f, use_fused=True)
        y_f.sum().backward()

        x_r = x.clone().detach()
        y_r = layer_ref(x_r, use_fused=False, use_views=True)
        y_r.sum().backward()

        # Compare factor_stack gradients.
        # Factored backward involves extra matmul steps (dW reduction through
        # factor products), so TF32 error accumulates more than unfactored.
        gf = layer_fused.factor_stack.grad
        gr = layer_ref.factor_stack.grad
        assert gf is not None, "fused factor_stack.grad is None"
        assert gr is not None, "ref factor_stack.grad is None"
        assert torch.allclose(gf, gr, atol=5e-2), \
            f"factor_stack grad max diff: {(gf - gr).abs().max().item()}"

        if layer_fused.bias is not None:
            assert torch.allclose(layer_fused.bias.grad, layer_ref.bias.grad, atol=5e-2), \
                f"bias grad max diff: {(layer_fused.bias.grad - layer_ref.bias.grad).abs().max().item()}"

    def test_grad_4_blocks(self):
        layer = make_factored(64, num_blocks=4, seed=20)
        layer2 = make_factored(64, num_blocks=4, seed=20)
        layer2.load_state_dict(layer.state_dict())
        x = torch.randn(16, 64, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_grad_9_blocks(self):
        layer = make_factored(9 * 16, num_blocks=9, seed=21)
        layer2 = make_factored(9 * 16, num_blocks=9, seed=21)
        layer2.load_state_dict(layer.state_dict())
        x = torch.randn(8, 9 * 16, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_grad_no_bias(self):
        layer = make_factored(64, num_blocks=4, bias=False, seed=22)
        layer2 = make_factored(64, num_blocks=4, bias=False, seed=22)
        layer2.load_state_dict(layer.state_dict())
        x = torch.randn(16, 64, device="cuda")
        self._compare_grads(layer, layer2, x)

    def test_optimizer_step(self):
        """SGD step updates factor_stack but not permutations."""
        layer = make_factored(64, num_blocks=4, seed=99)
        factors_before = layer.factor_stack.data.clone()
        perm_in_before = layer.perm_in.clone()
        perm_out_before = layer.perm_out.clone()

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(8, 64, device="cuda")
        loss = layer(x, use_fused=True).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(layer.factor_stack.data, factors_before), \
            "factor_stack unchanged after optimizer step"
        assert torch.equal(layer.perm_in, perm_in_before)
        assert torch.equal(layer.perm_out, perm_out_before)


def test_factored_fallback_on_cpu():
    """Factored mode works on CPU via views path."""
    layer = MonarchLinear.from_uniform_blocks(
        64, 64, num_blocks=4, seed=1, factored=True
    )
    x = torch.randn(8, 64)
    y = layer(x)
    assert y.shape == (8, 64)


# ===========================================================================
# Chain-factored GPU tests (chain_length > 2 and rectangular)
# ===========================================================================

@requires_fused
def test_chain_length_3_bmm_fallback():
    """chain_length=3 should fall back from fused to BMM, results still correct."""
    dim = 8 * 16  # 128: 8 blocks of 16x16, 2 factors with chain_length=3
    layer = MonarchLinear.from_uniform_blocks(
        dim, dim, num_blocks=8, factored=True, chain_length=3, seed=1,
    ).to("cuda")
    x = torch.randn(16, dim, device="cuda")
    # chain_length > 2 → fused disabled
    assert not layer._can_use_fused(x)
    y = layer(x)
    S = layer.to_dense()
    y_dense = x @ S.T
    assert torch.allclose(y, y_dense, atol=3e-3), \
        f"chain_length=3 GPU mismatch: {(y - y_dense).abs().max().item()}"


@requires_fused
def test_rectangular_factored_gpu():
    """Rectangular factored blocks work on GPU via BMM materialization."""
    layer = MonarchLinear.from_uniform_blocks(
        128, 64, num_blocks=4, factored=True, chain_length=2, bias=True, seed=1,
    ).to("cuda")
    x = torch.randn(16, 128, device="cuda")
    # non-square factored → fused disabled
    assert not layer._can_use_fused(x)
    y = layer(x)
    S = layer.to_dense()
    y_dense = x @ S.T + layer.bias
    assert torch.allclose(y, y_dense, atol=3e-3), \
        f"rectangular GPU mismatch: {(y - y_dense).abs().max().item()}"


@requires_fused
def test_rectangular_gradient_gpu():
    """Gradients flow through adapter on GPU."""
    layer = MonarchLinear.from_uniform_blocks(
        128, 64, num_blocks=4, factored=True, chain_length=2, bias=True, seed=1,
    ).to("cuda")
    x = torch.randn(16, 128, device="cuda")
    loss = layer(x).sum()
    loss.backward()
    assert layer.factor_stack.grad is not None
    assert layer.adapter.grad is not None
    assert layer.bias.grad is not None


@requires_fused
def test_chain3_rectangular_gpu():
    """chain_length=3 with rectangular blocks on GPU."""
    # 128→64, 8 blocks: 2 factors, chain_length=3
    layer = MonarchLinear.from_uniform_blocks(
        128, 64, num_blocks=8, factored=True, chain_length=3, bias=True, seed=1,
    ).to("cuda")
    x = torch.randn(8, 128, device="cuda")
    y = layer(x)
    S = layer.to_dense()
    y_dense = x @ S.T + layer.bias
    assert torch.allclose(y, y_dense, atol=3e-3)
    # Gradient flow
    y.sum().backward()
    assert layer.factor_stack.grad is not None
    assert layer.adapter.grad is not None
