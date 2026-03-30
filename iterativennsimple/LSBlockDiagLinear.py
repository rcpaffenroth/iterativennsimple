"""Low-rank + Block-diagonal linear layer (no permutations).

Like LSLinear but uses BlockDiagLinear (pure BMM, no scatter/gather)
instead of MonarchLinear (permuted block-diagonal with scattered access).

    y = block_diag(W) @ x + A @ B @ x + bias

The block-diagonal handles local structure via fast BMM; the low-rank
AB handles global cross-block mixing — replacing what the Monarch
permutations P₁, P₂ would have provided, but learnable and adaptive.
"""

import math

import torch
import torch.nn as nn

from iterativennsimple.BlockDiagLinear import BlockDiagLinear


class LSBlockDiagLinear(nn.Module):
    """y = S(x) + x @ B^T @ A^T + bias, where S is a BlockDiagLinear.

    Args:
        sparse:  A pre-built BlockDiagLinear with bias=False.
        rank:    Rank of the low-rank component L = AB.
        bias:    If True, add a learnable bias.
        device:  Target device.
        dtype:   Target dtype.
    """

    __constants__ = ["in_features", "out_features", "rank"]

    def __init__(
        self,
        sparse: BlockDiagLinear,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if sparse.bias is not None:
            raise ValueError("BlockDiagLinear passed to LSBlockDiagLinear must have bias=False.")

        self.sparse = sparse
        self.in_features = sparse.in_features
        self.out_features = sparse.out_features
        self.rank = rank

        self.A = nn.Parameter(torch.empty(self.out_features, rank, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(rank, self.in_features, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.A)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute y = S(x) + x @ B^T @ A^T + bias."""
        y = self.sparse(input)

        unbatched = input.dim() == 1
        inp = input.unsqueeze(0) if unbatched else input

        low_rank = inp @ self.B.T  # (batch, rank)

        y_2d = y.unsqueeze(0) if unbatched else y
        y_2d = torch.addmm(y_2d, low_rank, self.A.T, beta=1.0, alpha=1.0)

        if self.bias is not None:
            y_2d = y_2d + self.bias

        return y_2d.squeeze(0) if unbatched else y_2d

    def to_dense(self) -> torch.Tensor:
        return self.sparse.to_dense() + self.A @ self.B

    def number_of_trainable_parameters(self) -> int:
        total = self.sparse.number_of_trainable_parameters()
        total += self.A.numel() + self.B.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, num_blocks={self.sparse.num_blocks}, "
            f"stride_perm={self.sparse.stride_perm}, bias={self.bias is not None}"
        )

    @staticmethod
    def from_uniform_blocks(
        in_features: int,
        out_features: int,
        num_blocks: int,
        rank: int,
        bias: bool = True,
        stride_perm: bool = False,
        device=None,
        dtype=None,
    ) -> "LSBlockDiagLinear":
        sparse = BlockDiagLinear(
            in_features, out_features, num_blocks,
            bias=False, stride_perm=stride_perm,
            device=device, dtype=dtype,
        )
        return LSBlockDiagLinear(sparse, rank=rank, bias=bias, device=device, dtype=dtype)
