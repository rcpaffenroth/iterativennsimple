"""Fused Triton kernels for MonarchLinear.

This module provides GPU-accelerated forward and backward passes for the
Monarch matrix linear layer.  The kernels fuse the full
permute -> block-diagonal matmul -> inverse-permute -> bias pipeline into
single kernel launches, eliminating intermediate global-memory traffic
and Python-loop overhead.

Requirements:
    - NVIDIA GPU with CUDA
    - ``triton >= 3.0.0``  (ships with PyTorch 2.x on Linux)

When Triton is not available the module can still be imported -- the
public helpers simply return ``False`` / raise ``RuntimeError`` so that
``MonarchLinear`` falls back to its pure-PyTorch path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Lazy Triton import -- keeps the package installable on CPU-only machines.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def triton_is_available() -> bool:
    """Return True if Triton is importable and a CUDA device exists."""
    return HAS_TRITON and torch.cuda.is_available()


# =========================================================================== #
#  Autotuning configurations                                                   #
# =========================================================================== #

# Maximum tile dimension.  128 works on Ampere/Hopper GPUs with >=164KB
# shared memory per SM.  The autotune will find optimal tile sizes.
_MAX_TILE = 128


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n (min 16 for Triton tl.dot)."""
    return max(16, 1 << (n - 1).bit_length())


if HAS_TRITON:

    def _forward_configs():
        """Curated autotuning configs for the forward kernel.

        Tile memory budget: x_tile(BLOCK_BATCH × BLOCK_K) + w_tile(BLOCK_N × BLOCK_K)
        + acc(BLOCK_BATCH × BLOCK_N).  Must fit in ~164KB shared memory on Ada/Ampere.
        The 128×128 tiles (~192KB) exceed this, so we use asymmetric tiles for large
        block sizes: BLOCK_K=64 with BLOCK_N=128 or vice versa.
        """
        return [
            # Small tiles — best for block_in/out ≤ 32
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 16, "BLOCK_N": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
            # Medium tiles — sweet spot for 32-64 blocks
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            # Large tiles — for block_in/out = 64-128
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            # Asymmetric tiles — for block_in=128 (loop K=64 twice, wide N)
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        ]

    def _backward_input_configs():
        """Curated autotuning configs for the backward input kernel."""
        return [
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 16, "BLOCK_N": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            # Asymmetric: wide output tile, moderate K
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        ]

    def _backward_weight_configs():
        """Curated autotuning configs for the backward weight kernel."""
        return [
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            # Asymmetric: wide M (block_out), moderate N (block_in)
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_BATCH": 128, "BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        ]

    # =================================================================== #
    #  Forward kernel                                                       #
    # =================================================================== #

    @triton.autotune(
        configs=_forward_configs(),
        key=["block_in", "block_out", "batch"],
    )
    @triton.jit
    def _monarch_forward_kernel(
        # Pointers
        input_ptr,
        output_ptr,
        weight_ptr,
        perm_in_ptr,
        perm_out_ptr,
        bias_ptr,
        # Dimensions
        batch,
        in_features,
        out_features,
        block_in: tl.constexpr,
        block_out: tl.constexpr,
        num_blocks: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        # Flags
        HAS_BIAS: tl.constexpr,
    ):
        """Fused forward: permute -> block matmul -> inverse-permute -> bias.

        Grid: (num_blocks, cdiv(batch, BLOCK_BATCH), cdiv(block_out, BLOCK_N))
        """
        # Which Monarch block and which output / batch tile
        block_id = tl.program_id(0)
        batch_tile = tl.program_id(1)
        out_tile = tl.program_id(2)

        # ---- Batch offsets ------------------------------------------------
        batch_offs = batch_tile * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch

        # ---- Output-column offsets for this block via perm_out ------------
        out_local = out_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        out_mask = out_local < block_out
        # Global index into perm_out
        perm_out_idx = block_id * block_out + out_local
        # Actual output column indices (scattered)
        out_cols = tl.load(perm_out_ptr + perm_out_idx, mask=out_mask, other=0)

        # ---- Accumulator --------------------------------------------------
        acc = tl.zeros((BLOCK_BATCH, BLOCK_N), dtype=tl.float32)

        # ---- Inner loop over block_in in tiles of BLOCK_K ----------------
        for k_start in range(0, block_in, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < block_in

            # Load input columns via perm_in (gathered reads)
            perm_in_idx = block_id * block_in + k_offs
            in_cols = tl.load(perm_in_ptr + perm_in_idx, mask=k_mask, other=0)

            # x_tile: (BLOCK_BATCH, BLOCK_K)
            x_tile = tl.load(
                input_ptr + batch_offs[:, None] * in_features + in_cols[None, :],
                mask=batch_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # w_tile: (BLOCK_N, BLOCK_K) -- weight[block_id, out_local, k_offs]
            w_ptr_base = weight_ptr + block_id * block_out * block_in
            w_tile = tl.load(
                w_ptr_base + out_local[:, None] * block_in + k_offs[None, :],
                mask=out_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Matmul tile: (BLOCK_BATCH, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            acc += tl.dot(x_tile, tl.trans(w_tile))

        # ---- Bias ---------------------------------------------------------
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + out_cols, mask=out_mask, other=0.0)
            acc += bias_vals[None, :]

        # ---- Scattered write to output ------------------------------------
        # The fp32 accumulator is truncated to the output dtype on store.
        tl.store(
            output_ptr + batch_offs[:, None] * out_features + out_cols[None, :],
            acc,
            mask=batch_mask[:, None] & out_mask[None, :],
        )

    # =================================================================== #
    #  Backward kernel -- grad_input                                        #
    # =================================================================== #

    @triton.autotune(
        configs=_backward_input_configs(),
        key=["block_in", "block_out", "batch"],
    )
    @triton.jit
    def _monarch_backward_input_kernel(
        # Pointers
        grad_output_ptr,
        grad_input_ptr,
        weight_ptr,
        perm_in_ptr,
        perm_out_ptr,
        # Dimensions
        batch,
        in_features,
        out_features,
        block_in: tl.constexpr,
        block_out: tl.constexpr,
        num_blocks: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_K: tl.constexpr,   # tile over block_out (the reduction dim)
        BLOCK_N: tl.constexpr,   # tile over block_in  (the output dim)
    ):
        """grad_input[b, perm_in[i]] += sum_j grad_output[b, perm_out[j]] * W[block, j, i]

        Grid: (num_blocks, cdiv(batch, BLOCK_BATCH), cdiv(block_in, BLOCK_N))
        """
        block_id = tl.program_id(0)
        batch_tile = tl.program_id(1)
        in_tile = tl.program_id(2)

        batch_offs = batch_tile * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch

        # Which input columns this tile writes to
        in_local = in_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_mask = in_local < block_in
        perm_in_idx = block_id * block_in + in_local
        in_cols = tl.load(perm_in_ptr + perm_in_idx, mask=in_mask, other=0)

        acc = tl.zeros((BLOCK_BATCH, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, block_out, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < block_out

            # grad_output columns via perm_out
            perm_out_idx = block_id * block_out + k_offs
            out_cols = tl.load(perm_out_ptr + perm_out_idx, mask=k_mask, other=0)

            # go_tile: (BLOCK_BATCH, BLOCK_K)
            go_tile = tl.load(
                grad_output_ptr + batch_offs[:, None] * out_features + out_cols[None, :],
                mask=batch_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # w_tile: (BLOCK_K, BLOCK_N) -- W[block_id, k_offs, in_local]
            w_ptr_base = weight_ptr + block_id * block_out * block_in
            w_tile = tl.load(
                w_ptr_base + k_offs[:, None] * block_in + in_local[None, :],
                mask=k_mask[:, None] & in_mask[None, :],
                other=0.0,
            )

            # (BLOCK_BATCH, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            acc += tl.dot(go_tile, w_tile)

        # Scatter to grad_input
        tl.store(
            grad_input_ptr + batch_offs[:, None] * in_features + in_cols[None, :],
            acc,
            mask=batch_mask[:, None] & in_mask[None, :],
        )

    # =================================================================== #
    #  Backward kernel -- grad_weight                                       #
    # =================================================================== #

    @triton.autotune(
        configs=_backward_weight_configs(),
        key=["block_in", "block_out", "batch"],
    )
    @triton.jit
    def _monarch_backward_weight_kernel(
        # Pointers
        grad_output_ptr,
        input_ptr,
        grad_weight_ptr,
        perm_in_ptr,
        perm_out_ptr,
        # Dimensions
        batch,
        in_features,
        out_features,
        block_in: tl.constexpr,
        block_out: tl.constexpr,
        num_blocks: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_M: tl.constexpr,  # tile over block_out
        BLOCK_N: tl.constexpr,  # tile over block_in
    ):
        """grad_W[block, j, i] = sum_b grad_output[b, perm_out[j]] * input[b, perm_in[i]]

        Grid: (num_blocks, cdiv(block_out, BLOCK_M), cdiv(block_in, BLOCK_N))
        """
        block_id = tl.program_id(0)
        out_tile = tl.program_id(1)
        in_tile = tl.program_id(2)

        out_local = out_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        out_mask = out_local < block_out
        perm_out_idx = block_id * block_out + out_local
        out_cols = tl.load(perm_out_ptr + perm_out_idx, mask=out_mask, other=0)

        in_local = in_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_mask = in_local < block_in
        perm_in_idx = block_id * block_in + in_local
        in_cols = tl.load(perm_in_ptr + perm_in_idx, mask=in_mask, other=0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for b_start in range(0, batch, BLOCK_BATCH):
            b_offs = b_start + tl.arange(0, BLOCK_BATCH)
            b_mask = b_offs < batch

            # go_tile: (BLOCK_BATCH, BLOCK_M)
            go_tile = tl.load(
                grad_output_ptr + b_offs[:, None] * out_features + out_cols[None, :],
                mask=b_mask[:, None] & out_mask[None, :],
                other=0.0,
            )

            # x_tile: (BLOCK_BATCH, BLOCK_N)
            x_tile = tl.load(
                input_ptr + b_offs[:, None] * in_features + in_cols[None, :],
                mask=b_mask[:, None] & in_mask[None, :],
                other=0.0,
            )

            # (BLOCK_M, BLOCK_BATCH) @ (BLOCK_BATCH, BLOCK_N)
            acc += tl.dot(tl.trans(go_tile), x_tile)

        # Store grad_weight
        gw_ptr_base = grad_weight_ptr + block_id * block_out * block_in
        tl.store(
            gw_ptr_base + out_local[:, None] * block_in + in_local[None, :],
            acc,
            mask=out_mask[:, None] & in_mask[None, :],
        )

    # =================================================================== #
    #  Factored block kernels                                               #
    # =================================================================== #
    # Instead of storing num_blocks independent weight matrices, factored
    # mode stores num_factors = ceil(sqrt(num_blocks)) factor matrices and
    # constructs each block on-the-fly as factor[left] @ factor[right].

    def _factored_forward_configs():
        """Autotuning configs for the factored forward kernel."""
        return [
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 16, "BLOCK_N": 16, "BLOCK_F": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 16, "BLOCK_N": 16, "BLOCK_F": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_F": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_F": 32}, num_warps=8, num_stages=2),
        ]

    @triton.autotune(
        configs=_factored_forward_configs(),
        key=["block_size", "batch"],
    )
    @triton.jit
    def _monarch_factored_forward_kernel(
        # Pointers
        input_ptr,
        output_ptr,
        factor_ptr,
        recipe_ptr,
        perm_in_ptr,
        perm_out_ptr,
        bias_ptr,
        # Dimensions
        batch,
        in_features,
        out_features,
        block_size: tl.constexpr,
        num_blocks: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        # Flags
        HAS_BIAS: tl.constexpr,
    ):
        """Fused factored forward: permute -> factored block matmul -> inv-permute -> bias.

        Each block weight is reconstructed on-the-fly as factor[left] @ factor[right].
        Grid: (num_blocks, cdiv(batch, BLOCK_BATCH), cdiv(block_size, BLOCK_N))
        """
        block_id = tl.program_id(0)
        batch_tile = tl.program_id(1)
        out_tile = tl.program_id(2)

        # Load recipe for this block
        left_idx = tl.load(recipe_ptr + block_id * 2).to(tl.int64)
        right_idx = tl.load(recipe_ptr + block_id * 2 + 1).to(tl.int64)
        bs2 = block_size * block_size

        batch_offs = batch_tile * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch

        out_local = out_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        out_mask = out_local < block_size
        perm_out_idx = block_id * block_size + out_local
        out_cols = tl.load(perm_out_ptr + perm_out_idx, mask=out_mask, other=0)

        acc = tl.zeros((BLOCK_BATCH, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, block_size, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < block_size

            # Load input columns via perm_in (gathered reads)
            perm_in_idx = block_id * block_size + k_offs
            in_cols = tl.load(perm_in_ptr + perm_in_idx, mask=k_mask, other=0)

            x_tile = tl.load(
                input_ptr + batch_offs[:, None] * in_features + in_cols[None, :],
                mask=batch_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Reconstruct w_tile = factor[left, out_local, :] @ factor[right, :, k_offs]
            # w_tile: (BLOCK_N, BLOCK_K)
            w_tile = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for f_start in range(0, block_size, BLOCK_F):
                f_offs = f_start + tl.arange(0, BLOCK_F)
                f_mask = f_offs < block_size

                l_tile = tl.load(
                    factor_ptr + left_idx * bs2
                    + out_local[:, None] * block_size + f_offs[None, :],
                    mask=out_mask[:, None] & f_mask[None, :],
                    other=0.0,
                )
                r_tile = tl.load(
                    factor_ptr + right_idx * bs2
                    + f_offs[:, None] * block_size + k_offs[None, :],
                    mask=f_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                w_tile += tl.dot(l_tile, r_tile)

            # Cast w_tile to match x_tile dtype (e.g. bf16) so tl.dot is happy
            w_tile = w_tile.to(x_tile.dtype)
            # Matmul: (BLOCK_BATCH, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            acc += tl.dot(x_tile, tl.trans(w_tile))

        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + out_cols, mask=out_mask, other=0.0)
            acc += bias_vals[None, :]

        tl.store(
            output_ptr + batch_offs[:, None] * out_features + out_cols[None, :],
            acc,
            mask=batch_mask[:, None] & out_mask[None, :],
        )

    # ---- Factored backward input kernel --------------------------------

    def _factored_backward_input_configs():
        """Autotuning configs for the factored backward input kernel."""
        return [
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 16, "BLOCK_N": 16, "BLOCK_F": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 16, "BLOCK_N": 16, "BLOCK_F": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 32, "BLOCK_N": 32, "BLOCK_F": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_BATCH": 32, "BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_F": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_BATCH": 64, "BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_F": 32}, num_warps=8, num_stages=2),
        ]

    @triton.autotune(
        configs=_factored_backward_input_configs(),
        key=["block_size", "batch"],
    )
    @triton.jit
    def _monarch_factored_backward_input_kernel(
        # Pointers
        grad_output_ptr,
        grad_input_ptr,
        factor_ptr,
        recipe_ptr,
        perm_in_ptr,
        perm_out_ptr,
        # Dimensions
        batch,
        in_features,
        out_features,
        block_size: tl.constexpr,
        num_blocks: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        """Factored grad_input: reconstruct W from factors on-the-fly.

        Grid: (num_blocks, cdiv(batch, BLOCK_BATCH), cdiv(block_size, BLOCK_N))
        """
        block_id = tl.program_id(0)
        batch_tile = tl.program_id(1)
        in_tile = tl.program_id(2)

        left_idx = tl.load(recipe_ptr + block_id * 2).to(tl.int64)
        right_idx = tl.load(recipe_ptr + block_id * 2 + 1).to(tl.int64)
        bs2 = block_size * block_size

        batch_offs = batch_tile * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch

        in_local = in_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_mask = in_local < block_size
        perm_in_idx = block_id * block_size + in_local
        in_cols = tl.load(perm_in_ptr + perm_in_idx, mask=in_mask, other=0)

        acc = tl.zeros((BLOCK_BATCH, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, block_size, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < block_size

            perm_out_idx = block_id * block_size + k_offs
            out_cols = tl.load(perm_out_ptr + perm_out_idx, mask=k_mask, other=0)

            go_tile = tl.load(
                grad_output_ptr + batch_offs[:, None] * out_features + out_cols[None, :],
                mask=batch_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Reconstruct w_tile = factor[left, k_offs, :] @ factor[right, :, in_local]
            # w_tile: (BLOCK_K, BLOCK_N)
            w_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
            for f_start in range(0, block_size, BLOCK_F):
                f_offs = f_start + tl.arange(0, BLOCK_F)
                f_mask = f_offs < block_size

                l_tile = tl.load(
                    factor_ptr + left_idx * bs2
                    + k_offs[:, None] * block_size + f_offs[None, :],
                    mask=k_mask[:, None] & f_mask[None, :],
                    other=0.0,
                )
                r_tile = tl.load(
                    factor_ptr + right_idx * bs2
                    + f_offs[:, None] * block_size + in_local[None, :],
                    mask=f_mask[:, None] & in_mask[None, :],
                    other=0.0,
                )
                w_tile += tl.dot(l_tile, r_tile)

            # Cast w_tile to match go_tile dtype for tl.dot compatibility
            w_tile = w_tile.to(go_tile.dtype)
            # (BLOCK_BATCH, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            acc += tl.dot(go_tile, w_tile)

        tl.store(
            grad_input_ptr + batch_offs[:, None] * in_features + in_cols[None, :],
            acc,
            mask=batch_mask[:, None] & in_mask[None, :],
        )

    # ---- Factored backward factor gradient kernel ----------------------

    def _factored_grad_factor_configs():
        """Autotuning configs for the factor gradient reduction kernel."""
        return [
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_P": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_P": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_P": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_P": 32}, num_warps=8, num_stages=2),
        ]

    @triton.autotune(
        configs=_factored_grad_factor_configs(),
        key=["block_size", "num_blocks"],
    )
    @triton.jit
    def _monarch_factored_grad_factor_kernel(
        # Pointers
        dw_ptr,
        factor_ptr,
        grad_factor_ptr,
        recipe_ptr,
        # Dimensions
        block_size: tl.constexpr,
        num_blocks: tl.constexpr,
        num_factors: tl.constexpr,
        # Tile sizes (autotuned)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """Reduce per-block dW into factor gradients.

        For each factor f:
          grad_f += sum over blocks b where recipe[b,0]==f: dW_b @ factor[recipe[b,1]]^T
          grad_f += sum over blocks b where recipe[b,1]==f: factor[recipe[b,0]]^T @ dW_b

        Grid: (num_factors, cdiv(block_size, BLOCK_M), cdiv(block_size, BLOCK_N))
        """
        factor_idx = tl.program_id(0)
        m_tile = tl.program_id(1)
        n_tile = tl.program_id(2)

        bs2 = block_size * block_size

        m_offs = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < block_size
        n_mask = n_offs < block_size

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for b in range(num_blocks):
            left_idx = tl.load(recipe_ptr + b * 2).to(tl.int64)
            right_idx = tl.load(recipe_ptr + b * 2 + 1).to(tl.int64)

            # Scalar masks: 1.0 if this factor participates, else 0.0
            left_scale = tl.where(left_idx == factor_idx, 1.0, 0.0)
            right_scale = tl.where(right_idx == factor_idx, 1.0, 0.0)

            for p_start in range(0, block_size, BLOCK_P):
                p_offs = p_start + tl.arange(0, BLOCK_P)
                p_mask = p_offs < block_size

                # Left factor: grad_L[m, n] += dW[b, m, p] @ R[right, n, p]^T
                dw_mp = tl.load(
                    dw_ptr + b * bs2 + m_offs[:, None] * block_size + p_offs[None, :],
                    mask=m_mask[:, None] & p_mask[None, :],
                    other=0.0,
                )
                r_np = tl.load(
                    factor_ptr + right_idx * bs2
                    + n_offs[:, None] * block_size + p_offs[None, :],
                    mask=n_mask[:, None] & p_mask[None, :],
                    other=0.0,
                )
                # (BLOCK_M, BLOCK_P) @ (BLOCK_P, BLOCK_N)
                acc += tl.dot(dw_mp, tl.trans(r_np)) * left_scale

                # Right factor: grad_R[m, n] += L[left, p, m]^T @ dW[b, p, n]
                l_pm = tl.load(
                    factor_ptr + left_idx * bs2
                    + p_offs[:, None] * block_size + m_offs[None, :],
                    mask=p_mask[:, None] & m_mask[None, :],
                    other=0.0,
                )
                dw_pn = tl.load(
                    dw_ptr + b * bs2 + p_offs[:, None] * block_size + n_offs[None, :],
                    mask=p_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                # (BLOCK_M, BLOCK_P) @ (BLOCK_P, BLOCK_N)
                acc += tl.dot(tl.trans(l_pm), dw_pn) * right_scale

        tl.store(
            grad_factor_ptr + factor_idx * bs2
            + m_offs[:, None] * block_size + n_offs[None, :],
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


# =========================================================================== #
#  Autograd Function                                                           #
# =========================================================================== #

class MonarchLinearFusedFn(torch.autograd.Function):
    """Custom autograd function that dispatches to fused Triton kernels."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight_stack: torch.Tensor,
        perm_in: torch.Tensor,
        perm_out: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_blocks: int,
        block_in: int,
        block_out: int,
    ) -> torch.Tensor:
        """
        Args:
            input:        (batch, in_features)
            weight_stack: (num_blocks, block_out, block_in)
            perm_in:      (in_features,) int64
            perm_out:     (out_features,) int64
            bias:         (out_features,) or None
            num_blocks, block_in, block_out: scalar metadata
        """
        # Triton kernels use raw pointer arithmetic assuming contiguous layout.
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight_stack.is_contiguous():
            weight_stack = weight_stack.contiguous()

        batch, in_features = input.shape
        out_features = num_blocks * block_out

        output = torch.empty(batch, out_features, device=input.device, dtype=input.dtype)

        # Grid uses lambda for autotuned tile sizes.
        grid = lambda META: (
            num_blocks,
            math.ceil(batch / META["BLOCK_BATCH"]),
            math.ceil(block_out / META["BLOCK_N"]),
        )

        _monarch_forward_kernel[grid](
            input, output, weight_stack,
            perm_in, perm_out,
            bias if bias is not None else input,  # dummy ptr when no bias
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
            HAS_BIAS=bias is not None,
        )

        ctx.save_for_backward(input, weight_stack, perm_in, perm_out, bias)
        ctx.num_blocks = num_blocks
        ctx.block_in = block_in
        ctx.block_out = block_out

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight_stack, perm_in, perm_out, bias = ctx.saved_tensors
        num_blocks = ctx.num_blocks
        block_in = ctx.block_in
        block_out = ctx.block_out
        batch, in_features = input.shape
        out_features = num_blocks * block_out

        # Ensure contiguity for raw pointer arithmetic.
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not input.is_contiguous():
            input = input.contiguous()

        # For uniform blocks, every input/output column is written exactly once
        # (the permutation is a bijection), so empty_like is safe (no memset).
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight_stack)
        grad_bias = None

        # --- grad_input kernel ---
        grid_gi = lambda META: (
            num_blocks,
            math.ceil(batch / META["BLOCK_BATCH"]),
            math.ceil(block_in / META["BLOCK_N"]),
        )

        _monarch_backward_input_kernel[grid_gi](
            grad_output, grad_input, weight_stack,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
        )

        # --- grad_weight kernel ---
        grid_gw = lambda META: (
            num_blocks,
            math.ceil(block_out / META["BLOCK_M"]),
            math.ceil(block_in / META["BLOCK_N"]),
        )

        _monarch_backward_weight_kernel[grid_gw](
            grad_output, input, grad_weight,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
        )

        # --- grad_bias ---
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)

        # Return grads for: input, weight_stack, perm_in, perm_out, bias,
        #                    num_blocks, block_in, block_out
        return grad_input, grad_weight, None, None, grad_bias, None, None, None


# =========================================================================== #
#  Public convenience function                                                 #
# =========================================================================== #

def monarch_linear_fused(
    input: torch.Tensor,
    weight_stack: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_blocks: int,
    block_in: int,
    block_out: int,
) -> torch.Tensor:
    """Fused MonarchLinear forward (and backward via autograd).

    Args:
        input:        (batch, in_features) or (in_features,).
        weight_stack: (num_blocks, block_out, block_in).
        perm_in:      (in_features,) int64.
        perm_out:     (out_features,) int64.
        bias:         (out_features,) or None.
        num_blocks:   Number of diagonal blocks.
        block_in:     Column size of each block.
        block_out:    Row size of each block.

    Returns:
        (batch, out_features) tensor.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "monarch_linear_fused requires Triton (pip install triton)"
        )
    return MonarchLinearFusedFn.apply(
        input, weight_stack, perm_in, perm_out, bias,
        num_blocks, block_in, block_out,
    )


# =========================================================================== #
#  Factored Autograd Function                                                  #
# =========================================================================== #

class MonarchLinearFactoredFusedFn(torch.autograd.Function):
    """Custom autograd for factored Monarch: blocks = factor[left] @ factor[right]."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        factor_stack: torch.Tensor,
        block_recipe: torch.Tensor,
        perm_in: torch.Tensor,
        perm_out: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_blocks: int,
        num_factors: int,
        block_size: int,
    ) -> torch.Tensor:
        """
        Args:
            input:        (batch, in_features)
            factor_stack: (num_factors, block_size, block_size)
            block_recipe: (num_blocks * 2,) int32 flat [left0, right0, left1, right1, ...]
            perm_in:      (in_features,) int32
            perm_out:     (out_features,) int32
            bias:         (out_features,) or None
        """
        if not input.is_contiguous():
            input = input.contiguous()
        if not factor_stack.is_contiguous():
            factor_stack = factor_stack.contiguous()

        batch, in_features = input.shape
        out_features = num_blocks * block_size

        output = torch.empty(batch, out_features, device=input.device, dtype=input.dtype)

        grid = lambda META: (
            num_blocks,
            math.ceil(batch / META["BLOCK_BATCH"]),
            math.ceil(block_size / META["BLOCK_N"]),
        )

        _monarch_factored_forward_kernel[grid](
            input, output, factor_stack, block_recipe,
            perm_in, perm_out,
            bias if bias is not None else input,
            batch, in_features, out_features,
            block_size=block_size,
            num_blocks=num_blocks,
            HAS_BIAS=bias is not None,
        )

        ctx.save_for_backward(input, factor_stack, block_recipe, perm_in, perm_out, bias)
        ctx.num_blocks = num_blocks
        ctx.num_factors = num_factors
        ctx.block_size = block_size

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, factor_stack, block_recipe, perm_in, perm_out, bias = ctx.saved_tensors
        num_blocks = ctx.num_blocks
        num_factors = ctx.num_factors
        block_size = ctx.block_size
        batch, in_features = input.shape
        out_features = num_blocks * block_size

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not input.is_contiguous():
            input = input.contiguous()

        grad_input = torch.empty_like(input)
        grad_bias = None

        # --- grad_input: reconstruct W from factors on-the-fly ---
        grid_gi = lambda META: (
            num_blocks,
            math.ceil(batch / META["BLOCK_BATCH"]),
            math.ceil(block_size / META["BLOCK_N"]),
        )

        _monarch_factored_backward_input_kernel[grid_gi](
            grad_output, grad_input, factor_stack, block_recipe,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_size=block_size,
            num_blocks=num_blocks,
        )

        # --- grad_factor: first compute per-block dW, then reduce ---
        # Step 1: Compute dW into temp buffer using the standard backward weight kernel
        grad_weight_temp = torch.empty(
            num_blocks, block_size, block_size,
            device=input.device, dtype=torch.float32,
        )

        grid_gw = lambda META: (
            num_blocks,
            math.ceil(block_size / META["BLOCK_M"]),
            math.ceil(block_size / META["BLOCK_N"]),
        )

        _monarch_backward_weight_kernel[grid_gw](
            grad_output, input, grad_weight_temp,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_in=block_size,
            block_out=block_size,
            num_blocks=num_blocks,
        )

        # Step 2: Reduce dW into factor gradients
        grad_factor = torch.zeros_like(factor_stack)

        grid_gf = lambda META: (
            num_factors,
            math.ceil(block_size / META["BLOCK_M"]),
            math.ceil(block_size / META["BLOCK_N"]),
        )

        _monarch_factored_grad_factor_kernel[grid_gf](
            grad_weight_temp, factor_stack, grad_factor, block_recipe,
            block_size=block_size,
            num_blocks=num_blocks,
            num_factors=num_factors,
        )

        del grad_weight_temp

        # --- grad_bias ---
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)

        # Return grads for: input, factor_stack, block_recipe, perm_in, perm_out,
        #                    bias, num_blocks, num_factors, block_size
        return grad_input, grad_factor, None, None, None, grad_bias, None, None, None


# =========================================================================== #
#  Factored public convenience function                                        #
# =========================================================================== #

def monarch_linear_factored_fused(
    input: torch.Tensor,
    factor_stack: torch.Tensor,
    block_recipe: torch.Tensor,
    perm_in: torch.Tensor,
    perm_out: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_blocks: int,
    num_factors: int,
    block_size: int,
) -> torch.Tensor:
    """Fused factored MonarchLinear forward (and backward via autograd).

    Args:
        input:        (batch, in_features) or (in_features,).
        factor_stack: (num_factors, block_size, block_size).
        block_recipe: (num_blocks * 2,) int32 flat.
        perm_in:      (in_features,) int32.
        perm_out:     (out_features,) int32.
        bias:         (out_features,) or None.
        num_blocks:   Number of diagonal blocks.
        num_factors:  Number of factor matrices.
        block_size:   Size of each square block.

    Returns:
        (batch, out_features) tensor.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "monarch_linear_factored_fused requires Triton (pip install triton)"
        )
    return MonarchLinearFactoredFusedFn.apply(
        input, factor_stack, block_recipe, perm_in, perm_out, bias,
        num_blocks, num_factors, block_size,
    )
