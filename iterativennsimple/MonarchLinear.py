import math
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MonarchLinear(nn.Module):
    """Sparse linear layer using the Monarch matrix format: S = P1 @ M @ P2.

    M is a block-diagonal matrix with trainable dense blocks.
    P1 and P2 are random permutations stored as index arrays (not matrices).

    The forward pass computes y = x S^T + b as:
        1. Permute input columns:    x_tilde = x[:, perm_in]
        2. Block-diagonal matmul:    y_tilde = x_tilde @ M^T   (via split/loop or torch.bmm)
        3. Inverse-permute outputs:  result   = y_tilde[:, perm_out]
        4. Add bias.

    This avoids constructing S explicitly and leverages dense BLAS kernels
    on GPU, making it far more efficient than unstructured COO sparse layers.

    Args:
        in_features:       Total input dimension.
        out_features:      Total output dimension.
        block_in_features: Column size of each diagonal block. Must sum to in_features.
        block_out_features: Row size of each diagonal block. Must sum to out_features.
        perm_in:           LongTensor of shape (in_features,) — input permutation index array.
        perm_out:          LongTensor of shape (out_features,) — output permutation index array.
        bias:              If True, add a learnable bias of shape (out_features,).
        device:            Target device.
        dtype:             Target dtype.

    Shape:
        - Input:  (*, in_features)
        - Output: (*, out_features)

    Attributes:
        blocks (nn.ParameterList): Trainable block weight tensors.
        perm_in  (Tensor): Input permutation buffer (not trainable).
        perm_out (Tensor): Output permutation buffer (not trainable).
        bias (Parameter or None): Learnable bias vector.

    Example::
        >>> layer = MonarchLinear.from_uniform_blocks(64, 64, num_blocks=4)
        >>> x = torch.randn(32, 64)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([32, 64])
    """

    __constants__ = ["in_features", "out_features", "num_blocks"]
    in_features: int
    out_features: int
    num_blocks: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_in_features: list[int],
        block_out_features: list[int],
        perm_in: torch.Tensor,
        perm_out: torch.Tensor,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # --- validation ---
        if sum(block_in_features) != in_features:
            raise ValueError(
                f"sum(block_in_features)={sum(block_in_features)} != in_features={in_features}"
            )
        if sum(block_out_features) != out_features:
            raise ValueError(
                f"sum(block_out_features)={sum(block_out_features)} != out_features={out_features}"
            )
        if len(block_in_features) != len(block_out_features):
            raise ValueError(
                f"len(block_in_features)={len(block_in_features)} != "
                f"len(block_out_features)={len(block_out_features)}"
            )
        if perm_in.shape != (in_features,):
            raise ValueError(f"perm_in must have shape ({in_features},), got {perm_in.shape}")
        if perm_out.shape != (out_features,):
            raise ValueError(f"perm_out must have shape ({out_features},), got {perm_out.shape}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = len(block_in_features)
        # Store as plain lists (not Parameters), so they are not serialized as tensors but
        # are accessible. block sizes are small metadata; kept as Python lists.
        self.block_in_features: list[int] = list(block_in_features)
        self.block_out_features: list[int] = list(block_out_features)

        # Trainable block weight matrices (each shape: (block_out_i, block_in_i))
        self.blocks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(block_out_features[i], block_in_features[i], **factory_kwargs),
                    requires_grad=True,
                )
                for i in range(self.num_blocks)
            ]
        )

        # Permutation index arrays — registered as buffers so they travel with
        # the module (.to(device), state_dict) but receive no gradients.
        self.register_buffer("perm_in", perm_in.long())
        self.register_buffer("perm_out", perm_out.long())

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise block weights (Kaiming uniform) and bias."""
        for block in self.blocks:
            nn.init.kaiming_uniform_(block, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute y = x S^T + b via permute → block matmul → inverse-permute.

        Args:
            input: Tensor of shape (batch, in_features) or (in_features,).

        Returns:
            Tensor of shape (batch, out_features) or (out_features,).
        """
        # Support both batched (2D) and unbatched (1D) inputs.
        unbatched = input.dim() == 1
        if unbatched:
            input = input.unsqueeze(0)

        # 1. Permute input columns
        x = input[:, self.perm_in]  # (batch, in_features)

        # 2. Block-diagonal matmul
        #    Use torch.bmm fast path when all blocks are the same size (common case),
        #    otherwise fall back to a loop over heterogeneous blocks.
        y = self._block_matmul(x)  # (batch, out_features)

        # 3. Permute output columns (scatter: result[:, perm_out[i]] = y[:, i]):
        result = torch.empty_like(y)
        result[:, self.perm_out] = y

        # 4. Bias
        if self.bias is not None:
            result = result + self.bias

        if unbatched:
            result = result.squeeze(0)

        return result

    def _block_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block-diagonal M^T to x (already permuted).

        Uses torch.bmm when blocks are uniform-sized, otherwise a Python loop.
        Both paths are fully differentiable.

        Args:
            x: Tensor of shape (batch, in_features).

        Returns:
            Tensor of shape (batch, out_features).
        """
        batch = x.shape[0]

        # Fast path: all blocks have the same shape → single torch.bmm call.
        if len(set(self.block_in_features)) == 1 and len(set(self.block_out_features)) == 1:
            block_in = self.block_in_features[0]
            block_out = self.block_out_features[0]
            # x:  (batch, num_blocks * block_in)
            # → (batch, num_blocks, block_in) → (num_blocks, batch, block_in)
            x_3d = x.reshape(batch, self.num_blocks, block_in).permute(1, 0, 2)
            # blocks: list of (block_out, block_in)  →  (num_blocks, block_out, block_in)
            W_3d = torch.stack(list(self.blocks))
            # bmm: (num_blocks, batch, block_in) × (num_blocks, block_in, block_out)
            #    = (num_blocks, batch, block_out)
            y_3d = torch.bmm(x_3d, W_3d.transpose(1, 2))
            # → (batch, num_blocks, block_out) → (batch, num_blocks * block_out)
            return y_3d.permute(1, 0, 2).reshape(batch, self.out_features)

        # General path: non-uniform block sizes — loop over blocks.
        parts = torch.split(x, self.block_in_features, dim=1)
        out_parts = [part @ self.blocks[i].T for i, part in enumerate(parts)]
        return torch.cat(out_parts, dim=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dense(self) -> torch.Tensor:
        """Construct and return the full dense weight matrix S of shape (out_features, in_features).

        S satisfies: layer(x) = x @ S.T + bias

        Useful for testing and debugging. Not efficient for large layers.

        Returns:
            Tensor of shape (out_features, in_features).
        """
        # Build block-diagonal M from the trainable blocks.
        M = torch.block_diag(*list(self.blocks))  # (out_features, in_features)

        # S[perm_out[a], perm_in[b]] = M[a, b]  for all a, b.
        # Equivalently:
        #   step 1: permute rows    — S_temp[perm_out, :] = M
        #   step 2: permute columns — S[:, perm_in] = S_temp
        S_temp = torch.zeros(
            self.out_features, self.in_features, device=M.device, dtype=M.dtype
        )
        S_temp[self.perm_out] = M  # row a of M goes to row perm_out[a]

        S = torch.zeros_like(S_temp)
        S[:, self.perm_in] = S_temp  # col b of S_temp goes to col perm_in[b]

        return S

    def to_dense_slow(self) -> torch.Tensor:
        # This is a straightforward but inefficient implementation of to_dense() that directly
        # computes the permutation matrices P1 and P2 and does the full matmul. Useful for testing against to_dense().
        # NOTE: Do not use in production code due to inefficiency.
        M = torch.block_diag(*list(self.blocks))
        P1 = torch.zeros(self.out_features, self.out_features, device=self.perm_out.device, dtype=M.dtype)
        P1[self.perm_out, torch.arange(self.out_features)] = 1.0
        P2 = torch.zeros(self.in_features, self.in_features, device=self.perm_in.device, dtype=M.dtype)
        P2[torch.arange(self.in_features), self.perm_in] = 1.0
        return P1 @ M @ P2

    def number_of_trainable_parameters(self) -> int:
        """Return the number of trainable scalar parameters (blocks + bias).

        This matches the interface expected by Sequential2D.number_of_trainable_parameters().
        """
        total = sum(b.numel() for b in self.blocks)
        if self.bias is not None:
            total += self.out_features
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_blocks={self.num_blocks}, "
            f"bias={self.bias is not None}"
        )

    # ------------------------------------------------------------------
    # Factory functions
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_permutation(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate a random permutation of [0, n) as a LongTensor."""
        return torch.randperm(n, generator=generator)

    @staticmethod
    def _make_generator(seed: int | None) -> torch.Generator | None:
        """Create a seeded Generator if seed is not None, else return None."""
        if seed is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen

    @staticmethod
    def _initialize_block(block: nn.Parameter, initialization_type: str | int | float) -> None:
        """Initialize a block parameter in place according to initialization_type.

        Supported values (mirroring the conventions in MaskedLinear):
            "kaiming"      — Kaiming uniform (default for nn.Linear)
            "zeros"        — All zeros
            1  or  "C=val" — Constant value
            "G"            — Gaussian mu=0, sigma=1
            "G=mu,sigma"   — Gaussian with specified mu and sigma
            "U"            — Uniform [-1, 1]
            "U=lo,hi"      — Uniform [lo, hi]
        """
        with torch.no_grad():
            if initialization_type == "kaiming":
                nn.init.kaiming_uniform_(block, a=math.sqrt(5))
            elif initialization_type == "zeros" or initialization_type == 0:
                nn.init.zeros_(block)
            elif initialization_type == 1:
                nn.init.ones_(block)
            elif isinstance(initialization_type, str) and initialization_type.startswith("C="):
                val = float(initialization_type[2:])
                nn.init.constant_(block, val)
            elif isinstance(initialization_type, str) and initialization_type.startswith("G"):
                if len(initialization_type) == 1:
                    mu, sigma = 0.0, 1.0
                else:
                    parts = initialization_type[2:].split(",")
                    mu, sigma = float(parts[0]), float(parts[1])
                nn.init.normal_(block, mean=mu, std=sigma)
            elif isinstance(initialization_type, str) and initialization_type.startswith("U"):
                if len(initialization_type) == 1:
                    lo, hi = -1.0, 1.0
                else:
                    parts = initialization_type[2:].split(",")
                    lo, hi = float(parts[0]), float(parts[1])
                nn.init.uniform_(block, lo, hi)
            else:
                raise ValueError(f"Unknown initialization_type: {initialization_type!r}")

    @staticmethod
    def from_block_config(
        in_features: int,
        out_features: int,
        block_in_features: list[int],
        block_out_features: list[int],
        initialization_type: str | int = "kaiming",
        bias: bool = True,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "MonarchLinear":
        """Create a MonarchLinear with explicit (potentially non-uniform) block sizes.

        Args:
            in_features:        Total input dimension. Must equal sum(block_in_features).
            out_features:       Total output dimension. Must equal sum(block_out_features).
            block_in_features:  Column size of each diagonal block.
            block_out_features: Row size of each diagonal block.
            initialization_type: How to initialize block weights. Options:
                "kaiming" (default), "zeros", 1, "C=val", "G", "G=mu,sigma",
                "U", "U=lo,hi".
            bias:  Whether to include a learnable bias.
            seed:  Optional integer seed for reproducible permutation generation.
            device: Target device.
            dtype:  Target dtype.

        Returns:
            A MonarchLinear instance with randomly generated permutations.
        """
        gen = MonarchLinear._make_generator(seed)
        perm_in = MonarchLinear._generate_permutation(in_features, gen)
        perm_out = MonarchLinear._generate_permutation(out_features, gen)

        layer = MonarchLinear(
            in_features=in_features,
            out_features=out_features,
            block_in_features=block_in_features,
            block_out_features=block_out_features,
            perm_in=perm_in,
            perm_out=perm_out,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if initialization_type != "kaiming":
            for block in layer.blocks:
                MonarchLinear._initialize_block(block, initialization_type)

        return layer

    @staticmethod
    def from_uniform_blocks(
        in_features: int,
        out_features: int,
        num_blocks: int,
        initialization_type: str | int = "kaiming",
        bias: bool = True,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "MonarchLinear":
        """Create a MonarchLinear where all diagonal blocks have the same size.

        Args:
            in_features:  Total input dimension. Must be divisible by num_blocks.
            out_features: Total output dimension. Must be divisible by num_blocks.
            num_blocks:   Number of blocks along the diagonal.
            initialization_type: See from_block_config. Default "kaiming".
            bias:   Whether to include a learnable bias.
            seed:   Optional seed for reproducible permutations.
            device: Target device.
            dtype:  Target dtype.

        Returns:
            A MonarchLinear instance.

        Raises:
            ValueError: If in_features or out_features is not divisible by num_blocks.
        """
        if in_features % num_blocks != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by num_blocks={num_blocks}"
            )
        if out_features % num_blocks != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by num_blocks={num_blocks}"
            )
        block_in = in_features // num_blocks
        block_out = out_features // num_blocks
        return MonarchLinear.from_block_config(
            in_features=in_features,
            out_features=out_features,
            block_in_features=[block_in] * num_blocks,
            block_out_features=[block_out] * num_blocks,
            initialization_type=initialization_type,
            bias=bias,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def from_sparsity_target(
        in_features: int,
        out_features: int,
        target_sparsity: float,
        initialization_type: str | int = "kaiming",
        bias: bool = True,
        seed: int | None = None,
        device=None,
        dtype=None,
    ) -> "MonarchLinear":
        """Create a MonarchLinear with approximately the desired sparsity.

        For uniform blocks of size (out_features/k) × (in_features/k):
            density  = nnz / total = (k * out_features/k * in_features/k) / (out_features * in_features)
                     = 1 / k
            sparsity = 1 - 1/k   →   k = round(1 / (1 - target_sparsity))

        The smallest feasible k that divides both in_features and out_features
        is selected; a warning is logged if the achieved sparsity differs from
        the target.

        Args:
            in_features:     Total input dimension.
            out_features:    Total output dimension.
            target_sparsity: Desired fraction of zeros. Must be in [0, 1).
            initialization_type: See from_block_config. Default "kaiming".
            bias:    Whether to include a learnable bias.
            seed:    Optional seed for reproducible permutations.
            device:  Target device.
            dtype:   Target dtype.

        Returns:
            A MonarchLinear instance.

        Raises:
            ValueError: If target_sparsity is not in [0, 1).
        """
        if not (0.0 <= target_sparsity < 1.0):
            raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")

        # Ideal k (density = 1/k → sparsity = 1 - 1/k)
        ideal_k = 1.0 / (1.0 - target_sparsity) if target_sparsity > 0.0 else 1.0
        ideal_k_int = max(1, round(ideal_k))

        # Find the nearest k that cleanly divides both dimensions.
        def is_valid(k: int) -> bool:
            return in_features % k == 0 and out_features % k == 0

        # Search outward from ideal_k_int for a valid divisor.
        num_blocks = None
        for delta in range(max(in_features, out_features)):
            for candidate in [ideal_k_int + delta, ideal_k_int - delta]:
                if candidate >= 1 and is_valid(candidate):
                    num_blocks = candidate
                    break
            if num_blocks is not None:
                break

        if num_blocks is None:
            # Fallback: k=1 is always valid (dense layer)
            num_blocks = 1

        achieved_sparsity = 1.0 - 1.0 / num_blocks
        if abs(achieved_sparsity - target_sparsity) > 0.05:
            logger.warning(
                f"target_sparsity={target_sparsity:.3f} not exactly achievable; "
                f"using num_blocks={num_blocks} for achieved_sparsity={achieved_sparsity:.3f}"
            )

        return MonarchLinear.from_uniform_blocks(
            in_features=in_features,
            out_features=out_features,
            num_blocks=num_blocks,
            initialization_type=initialization_type,
            bias=bias,
            seed=seed,
            device=device,
            dtype=dtype,
        )
