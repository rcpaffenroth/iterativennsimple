import copy
import os

import pytest
import torch
import torch.nn as nn

from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_monarch(in_f: int = 16, out_f: int = 16, num_blocks: int = 4,
                        bias: bool = True, seed: int = 42) -> MonarchLinear:
    """Create a small uniform-block MonarchLinear for testing."""
    return MonarchLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=num_blocks, bias=bias, seed=seed
    )


def is_valid_permutation(t: torch.Tensor, n: int) -> bool:
    """Return True if t is a permutation of [0, n)."""
    return t.shape == (n,) and torch.equal(t.sort().values, torch.arange(n))


# ---------------------------------------------------------------------------
# Test 1: uniform construction
# ---------------------------------------------------------------------------

def test_construction_uniform():
    in_f, out_f, k = 16, 24, 4
    layer = MonarchLinear.from_uniform_blocks(in_f, out_f, num_blocks=k, seed=0)

    assert layer.in_features == in_f
    assert layer.out_features == out_f
    assert layer.num_blocks == k

    assert is_valid_permutation(layer.perm_in, in_f)
    assert is_valid_permutation(layer.perm_out, out_f)

    assert len(layer.blocks) == k
    for i, block in enumerate(layer.blocks):
        assert block.shape == (out_f // k, in_f // k), f"block {i} shape mismatch"


# ---------------------------------------------------------------------------
# Test 2: non-uniform construction
# ---------------------------------------------------------------------------

def test_construction_nonuniform():
    block_in  = [4, 6, 6]
    block_out = [5, 5, 6]
    in_f  = sum(block_in)   # 16
    out_f = sum(block_out)  # 16

    layer = MonarchLinear.from_block_config(
        in_features=in_f,
        out_features=out_f,
        block_in_features=block_in,
        block_out_features=block_out,
        seed=1,
    )

    assert layer.in_features == in_f
    assert layer.out_features == out_f
    assert layer.num_blocks == 3
    assert layer.block_in_features == block_in
    assert layer.block_out_features == block_out

    for i, block in enumerate(layer.blocks):
        assert block.shape == (block_out[i], block_in[i])


# ---------------------------------------------------------------------------
# Test 3: forward output shape
# ---------------------------------------------------------------------------

def test_forward_shape():
    batch = 32
    layer = make_simple_monarch(in_f=16, out_f=24)
    x = torch.randn(batch, 16)
    y = layer(x)
    assert y.shape == (batch, 24)


def test_forward_shape_unbatched():
    layer = make_simple_monarch(in_f=16, out_f=16)
    x = torch.randn(16)
    y = layer(x)
    assert y.shape == (16,)

def test_to_dense_matches_to_dense_slow():
    layer = make_simple_monarch(in_f=16, out_f=16)
    S = layer.to_dense()
    S_slow = layer.to_dense_slow()
    assert torch.allclose(S, S_slow, atol=1e-5), \
        f"to_dense mismatch: max diff {(S - S_slow).abs().max().item()}"


def test_forward_use_views_match():
    """forward() with use_views=True and use_views=False must produce identical results."""
    # Test with uniform blocks (batched input)
    layer = make_simple_monarch(in_f=16, out_f=24, num_blocks=4, seed=42)
    x_batched = torch.randn(8, 16)
    y_with_views = layer(x_batched, use_views=True)
    y_without_views = layer(x_batched, use_views=False)
    assert torch.allclose(y_with_views, y_without_views, atol=1e-5), \
        f"batched: max diff {(y_with_views - y_without_views).abs().max().item()}"

    # Test with unbatched input (1D)
    x_unbatched = torch.randn(16)
    y_with_views = layer(x_unbatched, use_views=True)
    y_without_views = layer(x_unbatched, use_views=False)
    assert torch.allclose(y_with_views, y_without_views, atol=1e-5), \
        f"unbatched: max diff {(y_with_views - y_without_views).abs().max().item()}"

    # Test with non-uniform blocks
    layer_nonuniform = MonarchLinear.from_block_config(
        in_features=16,
        out_features=16,
        block_in_features=[4, 6, 6],
        block_out_features=[5, 5, 6],
        seed=43,
    )
    x = torch.randn(8, 16)
    y_with_views = layer_nonuniform(x, use_views=True)
    y_without_views = layer_nonuniform(x, use_views=False)
    assert torch.allclose(y_with_views, y_without_views, atol=1e-5), \
        f"non-uniform: max diff {(y_with_views - y_without_views).abs().max().item()}"


def test_bmm_and_loop_matmul_agree():
    """forward() with bmm fast path and loop path must produce identical results."""
    x = torch.randn(8, 16)
    layer_bmm = make_simple_monarch(in_f=16, out_f=16, seed=13)
    # Build a loop-path layer with identical weights/permutations via state_dict.
    layer_loop = MonarchLinear.from_uniform_blocks(
        16, 16, num_blocks=4, force_loop_matmul=True, seed=13, bias=True
    )
    layer_loop.load_state_dict(layer_bmm.state_dict())
    y_bmm  = layer_bmm(x)
    y_loop = layer_loop(x)
    assert torch.allclose(y_bmm, y_loop, atol=1e-5), \
        f"bmm vs loop mismatch: max diff {(y_bmm - y_loop).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 4: forward matches explicit dense computation
# ---------------------------------------------------------------------------

def test_forward_matches_dense():
    layer = make_simple_monarch(in_f=16, out_f=16, seed=7)
    x = torch.randn(8, 16)

    y = layer(x)
    S = layer.to_dense()
    y_dense = x @ S.T + layer.bias

    assert torch.allclose(y, y_dense, atol=1e-5), \
        f"max diff: {(y - y_dense).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 5: forward without bias
# ---------------------------------------------------------------------------

def test_forward_no_bias():
    layer = make_simple_monarch(in_f=16, out_f=16, bias=False, seed=8)
    assert layer.bias is None

    x = torch.randn(8, 16)
    y = layer(x)
    S = layer.to_dense()
    y_dense = x @ S.T

    assert torch.allclose(y, y_dense, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: gradients flow to blocks and bias
# ---------------------------------------------------------------------------

def test_gradient_flow():
    layer = make_simple_monarch()
    x = torch.randn(8, 16)

    loss = layer(x).sum()
    loss.backward()

    for i in range(layer.num_blocks):
        g = layer.block_grad(i)
        assert g is not None, f"block_grad({i}) is None"
        assert g.shape == layer.blocks[i].shape

    assert layer.bias.grad is not None
    # Permutation buffers must never accumulate gradients
    assert layer.perm_in.grad is None
    assert layer.perm_out.grad is None

# ---------------------------------------------------------------------------
# Test 7: gradients match a dense nn.Linear with weight = to_dense()
# ---------------------------------------------------------------------------

def test_gradient_matches_dense():
    torch.manual_seed(0)
    layer = make_simple_monarch(in_f=16, out_f=16, seed=3)
    x = torch.randn(8, 16)
    target = torch.randn(8, 16)

    # --- MonarchLinear backward ---
    y_monarch = layer(x)
    loss_monarch = nn.functional.mse_loss(y_monarch, target)
    loss_monarch.backward()

    # Reconstruct dL/dS at Monarch positions from block gradients.
    # Derivation: blocks[k][a,b] parametrizes S[perm_out[row_start+a], perm_in[col_start+b]],
    # so dL/dblocks[k][a,b] == dL/dS[perm_out[row_start+a], perm_in[col_start+b]].
    # The reconstruction below mirrors to_dense(): assemble M_grad as block-diag,
    # then apply the same row/column permutation to get S_grad.
    M_grad = torch.block_diag(*[layer.block_grad(i).clone() for i in range(layer.num_blocks)])
    S_grad_temp = torch.zeros(layer.out_features, layer.in_features)
    S_grad_temp[layer.perm_out] = M_grad          # permute rows
    S_grad = torch.zeros_like(S_grad_temp)
    S_grad[:, layer.perm_in] = S_grad_temp        # permute columns
    monarch_bias_grad = layer.bias.grad.clone()

    # Build a boolean mask for the Monarch positions in S (the entries that are
    # actually parametrized).  Non-Monarch entries are not parametrized, so their
    # gradient is 0 in S_grad but non-zero in the dense-layer gradient — they
    # need not match.
    monarch_mask = torch.zeros(layer.out_features, layer.in_features, dtype=torch.bool)
    row_offset = 0
    col_offset = 0
    for k in range(layer.num_blocks):
        bor = layer.block_out_features[k]
        bir = layer.block_in_features[k]
        rows = layer.perm_out[row_offset: row_offset + bor]
        cols = layer.perm_in[col_offset: col_offset + bir]
        monarch_mask[rows[:, None], cols[None, :]] = True
        row_offset += bor
        col_offset += bir

    # --- Equivalent dense nn.Linear backward with the same effective S ---
    S = layer.to_dense().detach()
    dense = nn.Linear(16, 16, bias=True)
    with torch.no_grad():
        dense.weight.copy_(S)
        dense.bias.copy_(layer.bias.detach())

    y_dense = dense(x.detach().clone())
    loss_dense = nn.functional.mse_loss(y_dense, target)
    loss_dense.backward()

    # At Monarch positions the reconstructed S_grad must match dense.weight.grad.
    assert torch.allclose(S_grad[monarch_mask], dense.weight.grad[monarch_mask], atol=1e-5), \
        f"weight grad mismatch at Monarch positions: " \
        f"{(S_grad[monarch_mask] - dense.weight.grad[monarch_mask]).abs().max().item()}"

    # Bias gradient is unconstrained — must match fully.
    assert torch.allclose(monarch_bias_grad, dense.bias.grad, atol=1e-5), \
        f"bias grad mismatch: {(monarch_bias_grad - dense.bias.grad).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 8: optimizer step updates blocks but not permutations
# ---------------------------------------------------------------------------

def test_optimizer_step():
    layer = make_simple_monarch(seed=99)
    # Snapshot block values before step
    blocks_before = [layer.blocks[i].data.clone() for i in range(layer.num_blocks)]
    perm_in_before  = layer.perm_in.clone()
    perm_out_before = layer.perm_out.clone()

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(8, 16)
    loss = layer(x).sum()
    loss.backward()
    optimizer.step()

    # Blocks must have changed
    for i in range(layer.num_blocks):
        assert not torch.equal(layer.blocks[i].data, blocks_before[i]), \
            f"block[{i}] unchanged after optimizer step"

    # Permutations must be unchanged (they are buffers, not parameters)
    assert torch.equal(layer.perm_in, perm_in_before)
    assert torch.equal(layer.perm_out, perm_out_before)


# ---------------------------------------------------------------------------
# Test 9: save and reload via state_dict
# ---------------------------------------------------------------------------

def test_save_load(tmp_path):
    layer = make_simple_monarch(seed=5)
    x = torch.randn(8, 16)
    y_before = layer(x)

    path = tmp_path / "monarch.pt"
    torch.save(layer.state_dict(), str(path))

    # Build a fresh (randomly initialized) instance with identical architecture
    layer2 = make_simple_monarch(seed=0)  # different seed → different init
    layer2.load_state_dict(torch.load(str(path), weights_only=True))

    y_after = layer2(x)
    assert torch.allclose(y_before, y_after), \
        f"max diff after reload: {(y_before - y_after).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 10: move to CUDA and back
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_to_device():
    layer_cpu = make_simple_monarch()
    x_cpu = torch.randn(8, 16)

    # Deep-copy so moving to GPU doesn't mutate the CPU reference
    layer_gpu = copy.deepcopy(layer_cpu).to("cuda")
    x_gpu = x_cpu.to("cuda")
    y_gpu = layer_gpu(x_gpu)

    assert y_gpu.device.type == "cuda"
    for block in layer_gpu.blocks:
        assert block.device.type == "cuda"
    assert layer_gpu.perm_in.device.type == "cuda"
    assert layer_gpu.perm_out.device.type == "cuda"
    assert layer_gpu.bias.device.type == "cuda"

    # Bring back to CPU and compare with CPU result
    y_cpu_via_gpu = y_gpu.cpu()
    y_cpu_direct  = layer_cpu(x_cpu)
    # TF32 tensor cores on Ampere/Ada GPUs reduce mantissa precision,
    # so CPU vs GPU results differ by ~1e-3.
    assert torch.allclose(y_cpu_via_gpu, y_cpu_direct, atol=2e-3)


# ---------------------------------------------------------------------------
# Test 11: number_of_trainable_parameters
# ---------------------------------------------------------------------------

def test_number_of_trainable_parameters():
    # 4 blocks, each 4×4 = 16 params, plus bias of length 16 → total 80
    layer = make_simple_monarch(in_f=16, out_f=16, num_blocks=4, bias=True)
    assert layer.number_of_trainable_parameters() == 4 * 4 * 4 + 16

    # Without bias
    layer_nb = make_simple_monarch(in_f=16, out_f=16, num_blocks=4, bias=False)
    assert layer_nb.number_of_trainable_parameters() == 4 * 4 * 4


# ---------------------------------------------------------------------------
# Test 12: sparsity target
# ---------------------------------------------------------------------------

def test_sparsity_target():
    layer = MonarchLinear.from_sparsity_target(256, 256, target_sparsity=0.75, seed=2)
    S = layer.to_dense()
    total = S.numel()
    nonzero = S.count_nonzero().item()
    achieved_sparsity = 1.0 - nonzero / total
    assert abs(achieved_sparsity - 0.75) < 0.1, \
        f"achieved sparsity {achieved_sparsity:.3f} too far from 0.75"


# ---------------------------------------------------------------------------
# Test 13: from_block_config validation errors
# ---------------------------------------------------------------------------

def test_from_block_config_validation_sum_mismatch():
    with pytest.raises(ValueError, match="sum\\(block_in_features\\)"):
        MonarchLinear.from_block_config(
            in_features=10,
            out_features=10,
            block_in_features=[4, 4],   # sums to 8, not 10
            block_out_features=[5, 5],
        )


def test_from_block_config_validation_length_mismatch():
    with pytest.raises(ValueError):
        MonarchLinear.from_block_config(
            in_features=10,
            out_features=10,
            block_in_features=[5, 5],
            block_out_features=[5, 3, 2],  # different length
        )


def test_from_uniform_blocks_validation_not_divisible():
    with pytest.raises(ValueError):
        MonarchLinear.from_uniform_blocks(in_features=10, out_features=10, num_blocks=3)


# ---------------------------------------------------------------------------
# Test 14: deterministic seed for permutations
# ---------------------------------------------------------------------------

def test_deterministic_seed():
    a = MonarchLinear.from_uniform_blocks(16, 16, num_blocks=4, seed=42)
    b = MonarchLinear.from_uniform_blocks(16, 16, num_blocks=4, seed=42)
    c = MonarchLinear.from_uniform_blocks(16, 16, num_blocks=4, seed=99)

    assert torch.equal(a.perm_in, b.perm_in),  "Same seed: perm_in should match"
    assert torch.equal(a.perm_out, b.perm_out), "Same seed: perm_out should match"

    # Different seed → at least one permutation should differ (with overwhelming probability)
    assert not (torch.equal(a.perm_in, c.perm_in) and torch.equal(a.perm_out, c.perm_out)), \
        "Different seeds produced identical permutations — very unlikely"


# ---------------------------------------------------------------------------
# Test 15: rectangular (in_features != out_features)
# ---------------------------------------------------------------------------

def test_entry_target():
    in_f, out_f = 256, 256
    total = in_f * out_f  # 65536

    # Ask for 1/4 of all entries → k=4 → 16384 entries
    target = total // 4
    layer = MonarchLinear.from_entry_target(in_f, out_f, target_entries=target, seed=3)
    achieved = layer.number_of_trainable_parameters()
    assert abs(achieved - target) <= 0.05 * target, \
        f"achieved {achieved} entries, expected ~{target}"

    # Exact match: target_entries == total → k=1 → dense layer
    layer_dense = MonarchLinear.from_entry_target(in_f, out_f, target_entries=total, seed=3)
    assert layer_dense.num_blocks == 1

    # Exact match: target_entries == total/in_f → k=in_f → smallest blocks
    layer_min = MonarchLinear.from_entry_target(in_f, out_f, target_entries=out_f, seed=3)
    assert layer_min.num_blocks == in_f

    # Invalid target raises ValueError
    with pytest.raises(ValueError):
        MonarchLinear.from_entry_target(in_f, out_f, target_entries=0)
    with pytest.raises(ValueError):
        MonarchLinear.from_entry_target(in_f, out_f, target_entries=total + 1)


# ---------------------------------------------------------------------------
# Test: to_MaskedLinear
# ---------------------------------------------------------------------------

def test_to_masked_linear_type():
    """to_MaskedLinear returns a MaskedLinear instance."""
    from iterativennsimple.MaskedLinear import MaskedLinear
    layer = make_simple_monarch()
    masked = layer.to_MaskedLinear()
    assert isinstance(masked, MaskedLinear)


def test_to_masked_linear_forward_matches():
    """MaskedLinear produced by to_MaskedLinear gives identical outputs to MonarchLinear."""
    layer = make_simple_monarch(bias=True)
    masked = layer.to_MaskedLinear()
    x = torch.randn(32, 16)
    assert torch.allclose(layer(x), masked(x), atol=1e-5), \
        f"max diff: {(layer(x) - masked(x)).abs().max().item()}"


def test_to_masked_linear_no_bias():
    """Works correctly when bias=False."""
    layer = make_simple_monarch(bias=False)
    masked = layer.to_MaskedLinear()
    assert masked.bias is None
    x = torch.randn(8, 16)
    assert torch.allclose(layer(x), masked(x), atol=1e-5)


def test_to_masked_linear_sparsity_pattern():
    """The mask of the returned MaskedLinear matches the non-zero pattern of to_dense()."""
    layer = make_simple_monarch()
    masked = layer.to_MaskedLinear()
    S = layer.to_dense()
    expected_mask = (S != 0).to(S.dtype)
    assert torch.equal(masked.mask, expected_mask)


def test_to_masked_linear_weight_0():
    """weight_0 of the returned MaskedLinear equals to_dense()."""
    layer = make_simple_monarch()
    masked = layer.to_MaskedLinear()
    assert torch.allclose(masked.weight_0, layer.to_dense(), atol=1e-6)


def test_to_masked_linear_u_is_zero():
    """Trainable update U is initialised to zero."""
    layer = make_simple_monarch()
    masked = layer.to_MaskedLinear()
    assert torch.all(masked.U == 0)


def test_to_masked_linear_rectangular():
    """to_MaskedLinear works for non-square (rectangular) layers."""
    layer = MonarchLinear.from_uniform_blocks(32, 16, num_blocks=4, seed=7, bias=True)
    masked = layer.to_MaskedLinear()
    x = torch.randn(5, 32)
    assert torch.allclose(layer(x), masked(x), atol=1e-5)


def test_to_masked_linear_bias_values():
    """Bias values are copied correctly."""
    layer = make_simple_monarch(bias=True)
    masked = layer.to_MaskedLinear()
    assert torch.equal(masked.bias, layer.bias)


def test_rectangular():
    # 32 inputs → 16 outputs, 4 blocks of (4 × 8)
    layer = MonarchLinear.from_uniform_blocks(32, 16, num_blocks=4, seed=11, bias=True)
    x = torch.randn(7, 32)

    y = layer(x)
    assert y.shape == (7, 16)

    S = layer.to_dense()
    assert S.shape == (16, 32)

    y_dense = x @ S.T + layer.bias
    assert torch.allclose(y, y_dense, atol=1e-5), \
        f"rectangular: max diff {(y - y_dense).abs().max().item()}"
