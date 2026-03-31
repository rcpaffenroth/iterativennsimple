"""Block-diagonal linear layer — no permutations, pure BMM.

This is the speed-optimized variant of MonarchLinear that drops the
permutation matrices P₁ and P₂ entirely. Instead of S = P₁ M P₂,
the weight matrix is just M (block-diagonal).

The forward pass is a single ``torch.bmm`` call via reshape, which
goes through cuBLAS and is extremely fast. 

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

Factored mode (optional):
   Instead of storing num_blocks independent weight matrices, store
   k = ceil(num_blocks^(1/chain_length)) factor matrices and build
   each block as a chain product of factors.  With chain_length=m,
   this gives k^m >= num_blocks combinations from only k factor matrices.
"""

import itertools
import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BlockDiagLinear(nn.Module):
    """Block-diagonal linear layer with no permutations.

    Weight matrix is block_diag(W_0, W_1, ..., W_{K-1}) where each
    W_i has shape (block_out, block_in).  All blocks must be the same size.

    Forward: y = x @ W^T + bias, computed via reshape + torch.einsum.

    Args:
        in_features:  Total input dimension. Must be divisible by num_blocks.
        out_features: Total output dimension. Must be divisible by num_blocks.
        num_blocks:   Number of diagonal blocks.
        bias:         If True, add a learnable bias.
        stride_perm:  If True, use stride-interleave input routing instead
                      of contiguous slicing. See module docstring.
        factored:     If True, use factored block construction.
        chain_length: Number of factors multiplied per block (default 2).
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
        factored: bool = False,
        chain_length: int = 2,
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
        self._factored = factored

        if factored:
            s = min(self.block_in, self.block_out)
            self._factor_size = s
            self.chain_length = chain_length
            self.num_factors = math.ceil(num_blocks ** (1.0 / chain_length))

            self.factor_stack = nn.Parameter(
                torch.empty(self.num_factors, s, s, **factory_kwargs),
                requires_grad=True,
            )

            # Adapter for non-square blocks
            if self.block_out != self.block_in:
                self._adapter_position = "end" if self.block_out <= self.block_in else "start"
                self.adapter = nn.Parameter(
                    torch.empty(self.block_out, self.block_in, **factory_kwargs),
                    requires_grad=True,
                )
            else:
                self._adapter_position = None
                self.register_parameter("adapter", None)

            # Block recipe: all permutations-with-repetition, truncated to num_blocks
            recipe = list(itertools.islice(
                itertools.product(range(self.num_factors), repeat=chain_length),
                num_blocks,
            ))
            recipe_tensor = torch.tensor(recipe, dtype=torch.int32).flatten()
            self.register_buffer("block_recipe", recipe_tensor)

            # Pre-compute recipe indices for fast materialization
            recipe_2d = recipe_tensor.view(-1, chain_length)
            for j in range(chain_length):
                self.register_buffer(f"_recipe_idx_{j}", recipe_2d[:, j].long())

            # No weight_stack parameter in factored mode
            self.register_parameter("weight_stack", None)
        else:
            self._factor_size = None
            self.chain_length = 0
            self.num_factors = 0
            self._adapter_position = None
            self.register_parameter("adapter", None)
            self.register_buffer("block_recipe", None)

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
            idx = torch.arange(in_features).reshape(self.block_in, num_blocks).T.reshape(-1)
            self.register_buffer("_stride_idx", idx)
            out_idx = torch.arange(out_features).reshape(self.block_out, num_blocks).T.reshape(-1)
            self.register_buffer("_stride_out_idx", out_idx)
            inv_out = torch.empty_like(out_idx)
            inv_out[out_idx] = torch.arange(out_features)
            self.register_buffer("_stride_out_inv", inv_out)
        else:
            self.register_buffer("_stride_idx", None)
            self.register_buffer("_stride_out_idx", None)
            self.register_buffer("_stride_out_inv", None)

        self.reset_parameters()

    def _materialize_weight_stack(self) -> torch.Tensor:
        """Materialize (num_blocks, block_out, block_in) from factor chain + adapter.

        Fully differentiable — autograd traces through the bmm chain.
        """
        # Use pre-computed index buffers for fast indexing
        result = self.factor_stack[self._recipe_idx_0]  # (K, s, s)
        for j in range(1, self.chain_length):
            idx = getattr(self, f"_recipe_idx_{j}")
            result = torch.bmm(result, self.factor_stack[idx])
        if self._adapter_position == "end":
            adapter_exp = self.adapter.unsqueeze(0).expand(self.num_blocks, -1, -1)
            result = torch.bmm(result, adapter_exp)
        elif self._adapter_position == "start":
            adapter_exp = self.adapter.unsqueeze(0).expand(self.num_blocks, -1, -1)
            result = torch.bmm(adapter_exp, result)
        return result

    def reset_parameters(self) -> None:
        if self._factored:
            for i in range(self.num_factors):
                nn.init.kaiming_uniform_(self.factor_stack[i], a=math.sqrt(5))
            if self.adapter is not None:
                nn.init.kaiming_uniform_(self.adapter, a=math.sqrt(5))
        else:
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
            x = input.reshape(batch, self.block_in, self.num_blocks).permute(0, 2, 1)
        else:
            x = input.reshape(batch, self.num_blocks, self.block_in)

        # Get weight stack (materialized or stored)
        W = self._materialize_weight_stack() if self._factored else self.weight_stack

        # einsum: W is (K, bo, bi), x is (B, K, bi) -> (B, K, bo)
        y = torch.einsum('koi,bki->bko', W, x)

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
        W = self._materialize_weight_stack() if self._factored else self.weight_stack
        if self.stride_perm:
            M = torch.block_diag(*[W[i] for i in range(self.num_blocks)])
            S = torch.zeros_like(M)
            S[self._stride_out_idx.long()] = M
            result = torch.zeros_like(S)
            result[:, self._stride_idx.long()] = S
            return result
        else:
            return torch.block_diag(*[W[i] for i in range(self.num_blocks)])

    def number_of_trainable_parameters(self) -> int:
        if self._factored:
            total = self.factor_stack.numel()
            if self.adapter is not None:
                total += self.adapter.numel()
        else:
            total = self.weight_stack.numel()
        if self.bias is not None:
            total += self.out_features
        return total

    def extra_repr(self) -> str:
        base = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_blocks={self.num_blocks}, block_in={self.block_in}, "
            f"block_out={self.block_out}, stride_perm={self.stride_perm}, "
            f"bias={self.bias is not None}"
        )
        if self._factored:
            base += (
                f", factored=True, num_factors={self.num_factors}, "
                f"chain_length={self.chain_length}"
            )
        return base

    @staticmethod
    def from_config(
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = False,
        stride_perm: bool = False,
        factored: bool = False,
        chain_length: int = 2,
        device=None,
        dtype=None,
    ) -> "BlockDiagLinear":
        return BlockDiagLinear(
            in_features, out_features, num_blocks,
            bias=bias, stride_perm=stride_perm,
            factored=factored, chain_length=chain_length,
            device=device, dtype=dtype,
        )
