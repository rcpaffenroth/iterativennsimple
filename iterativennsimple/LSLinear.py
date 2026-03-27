import math

import torch
import torch.nn as nn

from iterativennsimple.MonarchLinear import MonarchLinear


class LSLinear(nn.Module):
    """Linear layer computing y = x(L + S)^T + b  (Robust PCA decomposition: M = L + S).

    Decomposes the weight matrix into two complementary components:

    - **L = AB** (low-rank): A is (out_features x rank), B is (rank x in_features).
      Captures global, low-rank structure spanning the full input/output space.
    - **S** (sparse MonarchLinear): permuted block-diagonal.
      Captures local, structured sparse interactions within and between blocks.

    The forward pass distributes x for efficiency:

        y = S(x) + x @ B^T @ A^T + bias

    This avoids forming the full (out x in) weight matrix.  MonarchLinear uses
    its own optimised kernels for xS^T; the low-rank path is two small matmuls.

    Initialization:  A is zeroed, B is Kaiming-uniform, so the network starts
    as pure sparse (S only) and the low-rank component L grows in organically
    during training.

    Args:
        sparse:   A pre-built :class:`MonarchLinear` with ``bias=False``.
                  The outer layer owns the bias; a monarch with ``bias=True``
                  is rejected with ``ValueError``.
        rank:     Rank of the low-rank component L (r).
        bias:     If True, add a learnable bias of shape (out_features,).
        device:   Target device (applied to A, B, bias only; the sparse
                  component is assumed to already be on the correct device).
        dtype:    Target dtype.

    Shape:
        - Input:  (*, in_features)
        - Output: (*, out_features)

    Attributes:
        sparse (MonarchLinear): The S component — sparse structured layer (no bias).
        A (Parameter): Shape (out_features, rank) — left factor of L.
        B (Parameter): Shape (rank, in_features) — right factor of L.
        bias (Parameter or None): Shape (out_features,).

    Example::
        >>> layer = LSLinear.from_uniform_blocks(64, 64, num_blocks=4, rank=8)
        >>> x = torch.randn(32, 64)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([32, 64])
    """

    __constants__ = ["in_features", "out_features", "rank"]
    in_features: int
    out_features: int
    rank: int

    def __init__(
        self,
        sparse: MonarchLinear,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if sparse.bias is not None:
            raise ValueError(
                "The MonarchLinear passed to LSLinear must have bias=False. "
                "The combined layer owns the bias.  Pass bias=False when building the "
                "inner monarch (e.g. MonarchLinear.from_uniform_blocks(..., bias=False))."
            )

        self.sparse = sparse  # S component — registered as a submodule automatically
        self.in_features = sparse.in_features
        self.out_features = sparse.out_features
        self.rank = rank

        # L component — low-rank parameters: A (out x r), B (r x in)
        self.A = nn.Parameter(torch.empty(self.out_features, rank, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(rank, self.in_features, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise A, B, and bias.

        - A -> zeros  (so L = AB = 0 at init; network starts as pure sparse S)
        - B -> Kaiming uniform  (ensures x @ B^T has meaningful scale so that
          A receives well-conditioned gradients from the first step)
        - bias -> uniform(-1/sqrt(in), 1/sqrt(in))
        """
        nn.init.zeros_(self.A)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor, **sparse_kwargs) -> torch.Tensor:
        """Compute y = S(x) + x @ B^T @ A^T + bias.

        Args:
            input: Tensor of shape (batch, in_features) or (in_features,).
            **sparse_kwargs: Keyword arguments forwarded to the inner
                MonarchLinear S component (e.g. ``use_fused``, ``use_views``).

        Returns:
            Tensor of shape (batch, out_features) or (out_features,).
        """
        # S path — sparse structured (uses optimised Triton kernels when available)
        # MonarchLinear handles 1-D (unbatched) inputs internally.
        y = self.sparse(input, **sparse_kwargs)

        # L path — low-rank: two small matmuls
        #   (*, in) @ (in, r) -> (*, r) @ (r, out) -> (*, out)
        low_rank = input @ self.B.T   # (*, rank)
        low_rank = low_rank @ self.A.T  # (*, out)

        y = y + low_rank

        if self.bias is not None:
            y = y + self.bias

        return y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dense(self) -> torch.Tensor:
        """Return the full (out_features, in_features) combined weight matrix W.

        Satisfies: layer(x) = x @ W.T + bias

        W = S.to_dense() + A @ B   (i.e. L + S in Robust PCA terms)

        Useful for testing; not efficient for large layers.
        """
        return self.sparse.to_dense() + self.A @ self.B

    def number_of_trainable_parameters(self) -> int:
        """Return the total number of trainable scalar parameters.

        Counts S (sparse Monarch blocks) + L (A + B) + bias (if present).
        """
        total = self.sparse.number_of_trainable_parameters()  # S blocks only (no bias)
        total += self.A.numel() + self.B.numel()  # L component
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"num_blocks={self.sparse.num_blocks}, "
            f"bias={self.bias is not None}"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_uniform_blocks(
        in_features: int,
        out_features: int,
        num_blocks: int,
        rank: int,
        sparse_init: str | int = "kaiming",
        bias: bool = True,
        force_loop_matmul: bool = False,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "LSLinear":
        """Create an LSLinear with uniform-sized Monarch blocks for the S component.

        Args:
            in_features:   Total input dimension. Must be divisible by num_blocks.
            out_features:  Total output dimension. Must be divisible by num_blocks.
            num_blocks:    Number of diagonal blocks in the sparse S component.
            rank:          Rank of the low-rank L = AB component.
            sparse_init:   Initialisation type for S (Monarch) blocks.
                           See :meth:`MonarchLinear._initialize_block` for options.
            bias:          Whether to add a learnable bias.
            force_loop_matmul: Force loop-based block matmul in MonarchLinear
                               (useful for debugging).
            seed:          Optional seed for reproducible Monarch permutations.
            device:        Target device.
            dtype:         Target dtype.

        Returns:
            An LSLinear instance.
        """
        sparse = MonarchLinear.from_uniform_blocks(
            in_features=in_features,
            out_features=out_features,
            num_blocks=num_blocks,
            initialization_type=sparse_init,
            bias=False,  # outer layer owns the bias
            force_loop_matmul=force_loop_matmul,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        return LSLinear(sparse, rank=rank, bias=bias, device=device, dtype=dtype)

    @staticmethod
    def from_block_config(
        in_features: int,
        out_features: int,
        block_in_features: list[int],
        block_out_features: list[int],
        rank: int,
        sparse_init: str | int = "kaiming",
        bias: bool = True,
        force_loop_matmul: bool = False,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "LSLinear":
        """Create an LSLinear with explicit (possibly non-uniform) block sizes for S.

        Args:
            in_features:        Total input dimension. Must equal sum(block_in_features).
            out_features:       Total output dimension. Must equal sum(block_out_features).
            block_in_features:  Column size of each diagonal block in S.
            block_out_features: Row size of each diagonal block in S.
            rank:               Rank of the low-rank L = AB component.
            sparse_init:        Initialisation type for S blocks.
            bias:               Whether to add a learnable bias.
            force_loop_matmul:  Force loop-based block matmul in MonarchLinear.
            seed:               Optional seed for reproducible Monarch permutations.
            device:             Target device.
            dtype:              Target dtype.

        Returns:
            An LSLinear instance.
        """
        sparse = MonarchLinear.from_block_config(
            in_features=in_features,
            out_features=out_features,
            block_in_features=block_in_features,
            block_out_features=block_out_features,
            initialization_type=sparse_init,
            bias=False,
            force_loop_matmul=force_loop_matmul,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        return LSLinear(sparse, rank=rank, bias=bias, device=device, dtype=dtype)

    @staticmethod
    def from_sparsity_target(
        in_features: int,
        out_features: int,
        target_sparsity: float,
        rank: int,
        sparse_init: str | int = "kaiming",
        bias: bool = True,
        force_loop_matmul: bool = False,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "LSLinear":
        """Create an LSLinear whose S component hits a sparsity target.

        The ``target_sparsity`` applies only to the sparse S component.  The
        low-rank L = AB component adds ``rank * (in_features + out_features)``
        dense parameters on top.

        Args:
            in_features:      Total input dimension.
            out_features:     Total output dimension.
            target_sparsity:  Desired fraction of zeros in the S weight
                              matrix.  Must be in [0, 1).
            rank:             Rank of the low-rank L = AB component.
            sparse_init:      Initialisation type for S blocks.
            bias:             Whether to add a learnable bias.
            force_loop_matmul: Force loop-based block matmul.
            seed:             Optional seed for S permutations.
            device:           Target device.
            dtype:            Target dtype.

        Returns:
            An LSLinear instance.
        """
        sparse = MonarchLinear.from_sparsity_target(
            in_features=in_features,
            out_features=out_features,
            target_sparsity=target_sparsity,
            initialization_type=sparse_init,
            bias=False,
            force_loop_matmul=force_loop_matmul,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        return LSLinear(sparse, rank=rank, bias=bias, device=device, dtype=dtype)
