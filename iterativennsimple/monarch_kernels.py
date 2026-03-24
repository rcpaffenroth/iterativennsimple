"""Fused Triton kernels for MonarchLinear.

This module provides GPU-accelerated forward and backward passes for the
Monarch matrix linear layer.  The kernels fuse the full
permute → block-diagonal matmul → inverse-permute → bias pipeline into
single kernel launches, eliminating intermediate global-memory traffic
and Python-loop overhead.

Requirements:
    - NVIDIA GPU with CUDA
    - ``triton >= 3.0.0``  (ships with PyTorch 2.x on Linux)

When Triton is not available the module can still be imported — the
public helpers simply return ``False`` / raise ``RuntimeError`` so that
``MonarchLinear`` falls back to its pure-PyTorch path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Lazy Triton import — keeps the package installable on CPU-only machines.
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
#  Forward kernel                                                              #
# =========================================================================== #

if HAS_TRITON:

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
        # Tile sizes (auto-tuned)
        BLOCK_BATCH: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        # Flags
        HAS_BIAS: tl.constexpr,
    ):
        """Fused forward: permute → block matmul → inverse-permute → bias.

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

            # w_tile: (BLOCK_N, BLOCK_K) — weight[block_id, out_local, k_offs]
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

    # ----------------------------------------------------------------------- #
    #  Backward kernel – grad_input                                            #
    # ----------------------------------------------------------------------- #

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
        # Tile sizes
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

            # w_tile: (BLOCK_K, BLOCK_N) — W[block_id, k_offs, in_local]
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

    # ----------------------------------------------------------------------- #
    #  Backward kernel – grad_weight                                           #
    # ----------------------------------------------------------------------- #

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
        # Tile sizes
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


# =========================================================================== #
#  Autograd Function                                                           #
# =========================================================================== #

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n (min 16 for Triton tl.dot)."""
    return max(16, 1 << (n - 1).bit_length())


# Maximum tile dimension.  Keeping tiles at most 64 prevents shared-memory
# overflow on consumer GPUs (e.g. RTX 4090 with 48-100 KB per SM).
# With 64×64 tiles the kernel uses ≈48 KB of shmem (3 tiles × 64×64×4B).
_MAX_TILE = 64


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
        # Ensure inputs are contiguous to avoid reading incorrect memory.
        input = input.contiguous()
        weight_stack = weight_stack.contiguous()

        batch, in_features = input.shape
        out_features = num_blocks * block_out

        output = torch.empty(batch, out_features, device=input.device, dtype=input.dtype)

        # Tile sizes — use next power-of-2 of block dims (Triton requires pow2
        # for tl.dot), capped at _MAX_TILE to stay within shared-memory limits.
        BLOCK_K = min(_MAX_TILE, _next_power_of_2(block_in))
        BLOCK_N = min(_MAX_TILE, _next_power_of_2(block_out))
        BLOCK_BATCH = min(_MAX_TILE, _next_power_of_2(batch))

        grid = (
            num_blocks,
            math.ceil(batch / BLOCK_BATCH),
            math.ceil(block_out / BLOCK_N),
        )

        _monarch_forward_kernel[grid](
            input, output, weight_stack,
            perm_in, perm_out,
            bias if bias is not None else input,  # dummy ptr when no bias
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
            BLOCK_BATCH=BLOCK_BATCH,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
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

        # The Triton kernels address memory via raw pointer arithmetic
        # (e.g. ptr + row * stride0 + col) and assume contiguous layout.
        # PyTorch's autograd may pass non-contiguous grad_output (e.g. an
        # expanded scalar with stride (0,0) from y.sum().backward()).
        # Force contiguity so the kernels read the correct values.
        grad_output = grad_output.contiguous()
        input = input.contiguous()

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight_stack)
        grad_bias = None

        # --- grad_input kernel ---
        BLOCK_K_gi = min(_MAX_TILE, _next_power_of_2(block_out))
        BLOCK_N_gi = min(_MAX_TILE, _next_power_of_2(block_in))
        BLOCK_BATCH_gi = min(_MAX_TILE, _next_power_of_2(batch))

        grid_gi = (
            num_blocks,
            math.ceil(batch / BLOCK_BATCH_gi),
            math.ceil(block_in / BLOCK_N_gi),
        )

        _monarch_backward_input_kernel[grid_gi](
            grad_output, grad_input, weight_stack,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
            BLOCK_BATCH=BLOCK_BATCH_gi,
            BLOCK_K=BLOCK_K_gi,
            BLOCK_N=BLOCK_N_gi,
        )

        # --- grad_weight kernel ---
        BLOCK_BATCH_gw = min(_MAX_TILE, _next_power_of_2(batch))
        BLOCK_M_gw = min(_MAX_TILE, _next_power_of_2(block_out))
        BLOCK_N_gw = min(_MAX_TILE, _next_power_of_2(block_in))

        grid_gw = (
            num_blocks,
            math.ceil(block_out / BLOCK_M_gw),
            math.ceil(block_in / BLOCK_N_gw),
        )

        _monarch_backward_weight_kernel[grid_gw](
            grad_output, input, grad_weight,
            perm_in, perm_out,
            batch, in_features, out_features,
            block_in=block_in,
            block_out=block_out,
            num_blocks=num_blocks,
            BLOCK_BATCH=BLOCK_BATCH_gw,
            BLOCK_M=BLOCK_M_gw,
            BLOCK_N=BLOCK_N_gw,
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
