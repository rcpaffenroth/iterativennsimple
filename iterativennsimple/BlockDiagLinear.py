"""Block-diagonal linear layer — no permutations, pure BMM.

This is the speed-optimized variant of MonarchLinear that drops the
permutation matrices P₁ and P₂ entirely. Instead of S = P₁ M P₂,
the weight matrix is just M (block-diagonal).

The forward pass is a single ``torch.bmm`` call via reshape, which
goes through cuBLAS and is extremely fast — comparable to a dense
``nn.Linear`` for medium-sized layers.

The trade-off: without permutations, each block only connects a
contiguous slice of inputs to a contiguous slice of outputs.  When
paired with a low-rank component (in LSLinear), the low-rank term
provides the global cross-block mixing that the permutations would
have provided.

Two input-routing strategies:

1. **Contiguous** (default): block i gets inputs [i*bi : (i+1)*bi].
   Free — just a reshape/view.

2. **Stride**: block i gets inputs [i, i+K, i+2K, ...] where K = num_blocks.
   This interleaves features across blocks so every block sees features
   from across the full input space.  Costs one extra gather operation
   but the gather is on contiguous strides, so it's much cheaper than
   random permutation scatter.
"""

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BlockDiagLinear(nn.Module):
    """Block-diagonal linear layer with no permutations.

    Weight matrix is block_diag(W_0, W_1, ..., W_{K-1}) where each
    W_i has shape (block_out, block_in).  All blocks must be the same size.

    Forward: y = x @ W^T + bias, computed via reshape + torch.bmm.

    Args:
        in_features:  Total input dimension. Must be divisible by num_blocks.
        out_features: Total output dimension. Must be divisible by num_blocks.
        num_blocks:   Number of diagonal blocks.
        bias:         If True, add a learnable bias.
        stride_perm:  If True, use stride-interleave input routing instead
                      of contiguous slicing. See module docstring.
        device:       Target device.
        dtype:        Target dtype.
    """

    __constants__ = ["in_features", "out_features", "num_blocks", "block_in", "block_out"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = True,
        stride_perm: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if in_features % num_blocks != 0:
            raise ValueError(f"in_features={in_features} not divisible by num_blocks={num_blocks}")
        if out_features % num_blocks != 0:
            raise ValueError(f"out_features={out_features} not divisible by num_blocks={num_blocks}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_in = in_features // num_blocks
        self.block_out = out_features // num_blocks
        self.stride_perm = stride_perm

        # Single contiguous weight tensor: (num_blocks, block_out, block_in)
        self.weight_stack = nn.Parameter(
            torch.empty(num_blocks, self.block_out, self.block_in, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Pre-compute stride permutation indices if needed
        if stride_perm:
            # Stride permutation: feature i -> block (i % K), position (i // K)
            # Equivalent to x.reshape(B, block_in, num_blocks).permute(0,2,1).reshape(B, in_features)
            # But we store explicit indices for the general case
            idx = torch.arange(in_features).reshape(self.block_in, num_blocks).T.reshape(-1)
            self.register_buffer("_stride_idx", idx)
            # Output: same stride pattern
            out_idx = torch.arange(out_features).reshape(self.block_out, num_blocks).T.reshape(-1)
            self.register_buffer("_stride_out_idx", out_idx)
            # Inverse for output scatter
            inv_out = torch.empty_like(out_idx)
            inv_out[out_idx] = torch.arange(out_features)
            self.register_buffer("_stride_out_inv", inv_out)
        else:
            self.register_buffer("_stride_idx", None)
            self.register_buffer("_stride_out_idx", None)
            self.register_buffer("_stride_out_inv", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_blocks):
            nn.init.kaiming_uniform_(self.weight_stack[i], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ block_diag(W)^T + bias.

        Uses einsum('koi,bki->bko') which avoids the expensive
        permute(1,0,2) memory copy that torch.bmm requires.

        Args:
            input: (batch, in_features) or (in_features,).

        Returns:
            (batch, out_features) or (out_features,).
        """
        unbatched = input.dim() == 1
        if unbatched:
            input = input.unsqueeze(0)

        batch = input.shape[0]

        if self.stride_perm:
            # Stride gather: interleave inputs across blocks
            # x.reshape(B, block_in, K).permute(0,2,1) -> (B, K, block_in)
            x = input.reshape(batch, self.block_in, self.num_blocks).permute(0, 2, 1)
        else:
            # Contiguous: just reshape (free — no copy)
            x = input.reshape(batch, self.num_blocks, self.block_in)

        # einsum: weight_stack is (K, bo, bi), x is (B, K, bi) -> (B, K, bo)
        # This avoids the permute(1,0,2) copy needed by torch.bmm.
        y = torch.einsum('koi,bki->bko', self.weight_stack, x)

        if self.stride_perm:
            y = y.permute(0, 2, 1).reshape(batch, self.out_features)
        else:
            y = y.reshape(batch, self.out_features)

        if self.bias is not None:
            y = y + self.bias

        if unbatched:
            y = y.squeeze(0)

        return y

    def to_dense(self) -> torch.Tensor:
        """Return the full dense weight matrix (out_features, in_features)."""
        if self.stride_perm:
            M = torch.block_diag(*[self.weight_stack[i] for i in range(self.num_blocks)])
            # Apply stride permutations
            S = torch.zeros_like(M)
            S[self._stride_out_idx.long()] = M
            result = torch.zeros_like(S)
            result[:, self._stride_idx.long()] = S
            return result
        else:
            return torch.block_diag(*[self.weight_stack[i] for i in range(self.num_blocks)])

    def number_of_trainable_parameters(self) -> int:
        total = self.weight_stack.numel()
        if self.bias is not None:
            total += self.out_features
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_blocks={self.num_blocks}, block_in={self.block_in}, "
            f"block_out={self.block_out}, stride_perm={self.stride_perm}, "
            f"bias={self.bias is not None}"
        )

    @staticmethod
    def from_config(
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = False,
        stride_perm: bool = False,
        device=None,
        dtype=None,
    ) -> "BlockDiagLinear":
        return BlockDiagLinear(
            in_features, out_features, num_blocks,
            bias=bias, stride_perm=stride_perm,
            device=device, dtype=dtype,
        )
