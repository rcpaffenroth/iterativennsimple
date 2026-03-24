"""Tests for the fused Triton kernel path of MonarchLinear.

All tests compare the fused kernel output against the existing pure-PyTorch
path (use_views=True) to verify numerical correctness of the forward and
backward passes.

These tests require an NVIDIA GPU with CUDA and the ``triton`` package.
They are automatically skipped when either is unavailable.
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

def make_monarch(in_f=64, out_f=64, num_blocks=4, bias=True, seed=42, device="cuda"):
    """Create a MonarchLinear on the specified device."""
    return MonarchLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=num_blocks, bias=bias, seed=seed
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
        assert torch.allclose(y_fused, y_ref, atol=1e-4), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_rectangular(self):
        layer = make_monarch(64, 32, num_blocks=4, seed=2)
        x = torch.randn(16, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused, y_ref, atol=1e-4), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_no_bias(self):
        layer = make_monarch(64, 64, num_blocks=4, bias=False, seed=3)
        x = torch.randn(8, 64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert torch.allclose(y_fused, y_ref, atol=1e-4), \
            f"max diff: {(y_fused - y_ref).abs().max().item()}"

    def test_unbatched(self):
        layer = make_monarch(64, 64, num_blocks=4, seed=4)
        x = torch.randn(64, device="cuda")
        y_fused = layer(x, use_fused=True)
        y_ref = layer(x, use_fused=False)
        assert y_fused.shape == (64,)
        assert torch.allclose(y_fused, y_ref, atol=1e-4), \
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
        assert torch.allclose(y_fused, y_ref, atol=1e-4), \
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
        assert torch.allclose(y_fused, y_dense, atol=1e-4), \
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

        # Compare block gradients
        for i, (bf, br) in enumerate(zip(layer_fused.blocks, layer_ref.blocks)):
            assert bf.grad is not None, f"fused block[{i}].grad is None"
            assert br.grad is not None, f"ref block[{i}].grad is None"
            assert torch.allclose(bf.grad, br.grad, atol=1e-4), \
                f"block[{i}] grad max diff: {(bf.grad - br.grad).abs().max().item()}"

        # Compare bias gradients
        if layer_fused.bias is not None:
            assert torch.allclose(layer_fused.bias.grad, layer_ref.bias.grad, atol=1e-4), \
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
        M_grad = torch.block_diag(*[b.grad.clone() for b in layer.blocks])
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

        assert torch.allclose(S_grad[monarch_mask], dense.weight.grad[monarch_mask], atol=1e-4), \
            f"weight grad mismatch: {(S_grad[monarch_mask] - dense.weight.grad[monarch_mask]).abs().max().item()}"
        assert torch.allclose(monarch_bias_grad, dense.bias.grad, atol=1e-4), \
            f"bias grad mismatch: {(monarch_bias_grad - dense.bias.grad).abs().max().item()}"

    def test_optimizer_step(self):
        """SGD step updates blocks but not permutations when using fused path."""
        layer = make_monarch(64, 64, num_blocks=4, seed=99)
        blocks_before = [b.data.clone() for b in layer.blocks]
        perm_in_before = layer.perm_in.clone()
        perm_out_before = layer.perm_out.clone()

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(8, 64, device="cuda")
        loss = layer(x, use_fused=True).sum()
        loss.backward()
        optimizer.step()

        for i, (block, before) in enumerate(zip(layer.blocks, blocks_before)):
            assert not torch.equal(block.data, before), f"block[{i}] unchanged"

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
    layer = MonarchLinear.from_uniform_blocks(32, 32, num_blocks=4, seed=1)
    x = torch.randn(8, 32)
    y = layer(x)  # should use Python path, no error
    assert y.shape == (8, 32)


@requires_fused
def test_auto_detect_uses_fused_on_cuda():
    """When use_fused=None (default), CUDA input should auto-select fused path."""
    layer = make_monarch(64, 64, num_blocks=4, seed=1)
    assert layer._can_use_fused(torch.randn(1, 64, device="cuda"))
