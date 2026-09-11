"""Reimplementation A of MaskedLinear:  two dense matrices, written into by slices.

The layer is an ordinary linear layer whose weight is constrained to an affine
family,

    A = W_0 + U .* Omega            (.* is entrywise)
    y = x A^T + b

with W_0 (the initial weight) and Omega (the mask) *fixed* and U the only
learnable parameter.  So Omega selects which entries of the weight are allowed
to move, and W_0 says where they start.  Omega = 1 recovers a dense
torch.nn.Linear; Omega = 0 freezes the layer at W_0.

Everything past the ~40 lines of module is a *constructor*, and every
constructor has the same job: decide the two fixed matrices W_0 and Omega.
This file does that the most concrete way available -- it allocates the layer
and then assigns into slices of A.weight_0 and A.mask, so the shape and
position of every block is on the page next to the mathematics that fills it.

Deliberately unchanged from the original implementation (see MaskedLinear.py):
  * weight_0 and mask are Parameters with requires_grad=False rather than
    buffers, because tests/test_Sequential2D.py counts len(model.parameters()).
  * 'non-zero' trainability keys off the *realised values*, not the support.
  * from_optimal_linear solves the normal equations rather than using lstsq.
Changed: the random block types draw from torch, not numpy, so torch.manual_seed
reproduces them; and the per-entry numpy.vectorize loops are gone.
"""
import math
from itertools import accumulate
from typing import Any

import torch


def _support(block_type, out_features: int, in_features: int) -> torch.Tensor:
    """The 0/1 pattern of structurally non-zero entries of a block, shape (out_features, in_features).

    block_type is the sparsity pattern of the block, and nothing else:
        0        no entries
        "W"      all entries
        "D"      the diagonal
        "R=0.5"  each entry independently with probability 0.5
        "Row=3"  3 draws per row, with replacement, so *at most* 3 per row
        "S=5"    5 draws in the whole block, with replacement, so at most 5
    The "with replacement" is why the original docstring warns you may get fewer
    entries than you asked for; drawing without replacement would need argsort
    or randperm per row, which costs more than the guarantee is worth here.
    """
    if block_type == 0:
        return torch.zeros(out_features, in_features)
    if block_type == "W":
        return torch.ones(out_features, in_features)
    if block_type == "D":
        # Rectangular "diagonal": 1 where i == j, which torch.eye gives directly.
        return torch.eye(out_features, in_features)
    if block_type.startswith("R="):
        p = float(block_type[2:])
        return (torch.rand(out_features, in_features) < p).to(torch.float32)
    if block_type.startswith("Row="):
        n = int(block_type[4:])
        cols = torch.randint(in_features, (out_features, n))         # (out_features, n) column index per draw
        support = torch.zeros(out_features, in_features)
        support.scatter_(1, cols, 1.0)                               # duplicates collapse, hence "at most n"
        return support
    if block_type.startswith("S="):
        n = int(block_type[2:])
        rows = torch.randint(out_features, (n,))                     # (n,)
        cols = torch.randint(in_features, (n,))                      # (n,)
        support = torch.zeros(out_features, in_features)
        support[rows, cols] = 1.0
        return support
    assert False, f"unknown block type {block_type}"


def _values(initialization_type, out_features: int, in_features: int) -> torch.Tensor:
    """The value each entry of a block would take if it were in the support, shape (out_features, in_features).

    initialization_type says *what* the entries are, independently of *which*
    entries there are:
        0, 1, "C=0.3"   that constant everywhere
        "G", "G=mu,sigma"   Gaussian draws, default mu=0, sigma=1
        "U", "U=lo,hi"      uniform draws, default lo=-1, hi=1
        a tensor            those values, verbatim
    Drawing a full dense block and throwing away the entries outside the support
    (see _support) wastes draws on very sparse blocks, but keeps "which entries"
    and "what values" separable, which is what makes the loop below short.
    """
    if torch.is_tensor(initialization_type):
        return initialization_type
    if initialization_type in (0, 1):
        return torch.full((out_features, in_features), float(initialization_type))
    if initialization_type.startswith("C="):
        return torch.full((out_features, in_features), float(initialization_type[2:]))
    if initialization_type == "G":
        return torch.randn(out_features, in_features)
    if initialization_type.startswith("G="):
        mu, sigma = (float(v) for v in initialization_type[2:].split(","))
        return torch.randn(out_features, in_features) * sigma + mu
    if initialization_type == "U":
        return torch.rand(out_features, in_features) * 2.0 - 1.0
    if initialization_type.startswith("U="):
        lo, hi = (float(v) for v in initialization_type[2:].split(","))
        return torch.rand(out_features, in_features) * (hi - lo) + lo
    assert False, f"unknown initialization type {initialization_type}"


class MaskedLinear(torch.nn.Module):
    """A linear layer with weight A = W_0 + U .* Omega, U learnable, W_0 and Omega fixed.

    Args:
        in_features:  size of each input sample
        out_features: size of each output sample
        bias:         include a bias term (itself always trainable)
    Shape:
        input  (*, in_features)  ->  output (*, out_features)

    Example:
        >>> m = MaskedLinear(20, 30)
        >>> m(torch.randn(128, 20)).shape
        torch.Size([128, 30])

    This is the flexible-but-slow implementation: the weight is dense and the
    mask is applied by multiplication, so it costs a dense matmul no matter how
    sparse Omega is.  Its value is as a reference -- SparseLinear and
    MonarchLinear must agree with it -- and that every stock torch optimizer
    works on it unchanged.
    """
    in_features: int
    out_features: int
    weight_0: torch.Tensor
    mask: torch.Tensor
    U: torch.Tensor

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # All three matrices are (out_features, in_features): torch stores weights
        # transposed relative to y = xA^T, so rows are outputs.  weight_0 and mask
        # are Parameters only so that they land in state_dict(); requires_grad=False
        # is what makes them fixed.
        self.weight_0 = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs),
                                           requires_grad=False)
        self.mask = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs),
                                       requires_grad=False)
        self.U = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """W_0 as torch.nn.Linear would initialise it, U = 0, Omega = 1 (fully trainable).

        U = 0 means a freshly built layer *is* W_0, so it matches a
        torch.nn.Linear built under the same seed.  a=sqrt(5) in kaiming_uniform
        is torch's own historical choice and equals uniform(-1/sqrt(fan_in),
        1/sqrt(fan_in)); see https://github.com/pytorch/pytorch/issues/57109.
        The order of these four calls fixes the random stream, so changing it
        breaks agreement with torch.nn.Linear.
        """
        torch.nn.init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        torch.nn.init.constant_(self.U, 0.)
        torch.nn.init.constant_(self.mask, 1.)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight_0 + self.U * self.mask           # A = W_0 + U .* Omega
        return torch.nn.functional.linear(input, weight, self.bias)   # y = x A^T + b

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def number_of_trainable_parameters(self) -> int:
        """Entries of U that Omega lets move, plus the bias.  Not len(parameters()),
        which would count all of W_0, Omega and U."""
        total_params = torch.count_nonzero(self.mask)
        if self.bias is not None:
            total_params += self.out_features
        return total_params

    # ---------------------------------------------------------------- constructors

    @staticmethod
    def from_description(out_features_sizes, in_features_sizes,
                         block_types, initialization_types, trainable,
                         bias: bool = True, device=None, dtype=None) -> Any:
        """Build a block matrix, one entry of each 2D argument per block.

        out_features_sizes and in_features_sizes give the block row and column
        heights/widths, so the layer is sum(out_features_sizes) by
        sum(in_features_sizes).  The other three arguments are 2D arrays of the
        same block shape, and for each block

            W_0[block] = values(initialization_types) .* support(block_types)
            Omega[block] = 1, 0, or the non-zero pattern of W_0[block]

        for trainable = True, False, 'non-zero' respectively.  See _support and
        _values for the two mini-languages.  Passing ints for the two sizes
        builds a single block, which is how Sequential2D calls this.
        """
        if (type(out_features_sizes) is int) and (type(in_features_sizes) is int):
            out_features_sizes = [out_features_sizes]
            in_features_sizes = [in_features_sizes]
            block_types = [[block_types]]
            initialization_types = [[initialization_types]]
            trainable = [[trainable]]

        # Where each block starts and stops in the assembled matrix:
        # sizes [5, 7, 9] -> bounds [0, 5, 12, 21], so block i occupies bounds[i]:bounds[i+1].
        row_bounds = [0, *accumulate(out_features_sizes)]
        col_bounds = [0, *accumulate(in_features_sizes)]

        A = MaskedLinear(in_features=col_bounds[-1], out_features=row_bounds[-1],
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            for i, out_features in enumerate(out_features_sizes):
                for j, in_features in enumerate(in_features_sizes):
                    rows = slice(row_bounds[i], row_bounds[i + 1])
                    cols = slice(col_bounds[j], col_bounds[j + 1])
                    train = trainable[i][j]

                    if block_types[i][j] == 0:
                        assert not train, "0 block should not be trainable"
                        assert initialization_types[i][j] == 0, "0 block should be initialized to 0"

                    values = _values(initialization_types[i][j], out_features, in_features)
                    support = _support(block_types[i][j], out_features, in_features)
                    A.weight_0[rows, cols] = values * support      # (out_features, in_features)

                    if train == 'non-zero':
                        # Keyed off the realised values, not the support, so an entry
                        # initialised to exactly 0.0 is frozen.  This is the original
                        # rule and the one SparseLinear.from_MaskedLinearExact assumes;
                        # it differs from `support` only for zero-valued initialisers.
                        A.mask[rows, cols] = (A.weight_0[rows, cols] != 0.0).to(A.mask.dtype)
                    else:
                        assert train in (True, False), f"unknown train type {train}"
                        A.mask[rows, cols] = float(train)
        return A

    @staticmethod
    def from_config(cfg):
        """from_description with the arguments in a dict."""
        return MaskedLinear.from_description(out_features_sizes=cfg['out_features_sizes'],
                                             in_features_sizes=cfg['in_features_sizes'],
                                             block_types=cfg['block_types'],
                                             initialization_types=cfg['initialization_types'],
                                             trainable=cfg['trainable'],
                                             bias=cfg['bias'])

    @staticmethod
    def from_coo(coo, check_mask: bool = False, bias: bool = False, device=None, dtype=None) -> Any:
        """W_0 is the COO matrix densified; the stored entries are the trainable ones."""
        A = MaskedLinear(in_features=coo.shape[1], out_features=coo.shape[0],
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            A.weight_0[:, :] = coo.to_dense()
            A.mask[:, :] = (A.weight_0 != 0)
            if check_mask:
                # A stored entry whose value happens to be 0.0 is *not* caught by the
                # line above and so is not trainable.  Slow, so it is opt-in.
                for i, j in coo.coalesce().indices().T:   # .indices() raises on an uncoalesced tensor
                    assert A.mask[i, j] != 0, f"mask is not correct at {i},{j}"
        return A

    @staticmethod
    def from_MLP(sizes, bias: bool = True, device=None, dtype=None) -> Any:
        """A feed-forward net of widths `sizes` written as one square matrix.

        With sizes = [n_0, ..., n_L] the layer is (sum n) by (sum n), the state is
        the concatenation z = (z_0, ..., z_L) of all layers' activations, and the
        MLP's k-th weight matrix sits in the block just below the diagonal, block
        row k, block column k-1:

            [ 0                  ]
            [ W_1  0             ]
            [      W_2  0        ]
            [           ...   0  ]

        so one application of A advances every layer by one step.  Everything
        off that subdiagonal is zero and frozen.
        """
        bounds = [0, *accumulate(sizes)]
        A = MaskedLinear(in_features=bounds[-1], out_features=bounds[-1],
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            A.weight_0.zero_()
            A.mask.zero_()
            for k in range(1, len(sizes)):
                rows = slice(bounds[k], bounds[k + 1])         # layer k outputs, n_k rows
                cols = slice(bounds[k - 1], bounds[k])         # layer k-1 inputs, n_{k-1} columns
                torch.nn.init.kaiming_uniform_(A.weight_0[rows, cols], a=math.sqrt(5))
                A.mask[rows, cols] = 1.0
        return A

    @staticmethod
    def from_optimal_linear(X, Y, bias: bool = False, device=None, dtype=None) -> Any:
        """Initialise at the least-squares map from X to Y, in the (x, y) state layout.

        With X of shape (N, D) and Y of shape (N, K) the state is z = (x, y), so
        the layer is (D+K) by (D+K) and

            W_0 = [ I_D       0 ]        z = (x, y)  |->  (x, W_ls x)
                  [ W_ls      0 ]

        copies x through and overwrites y with the regression prediction.  W_ls
        solves min || X W^T - Y ||_F, i.e. the normal equations

            X^T X W^T = X^T Y,   W_ls = (Y^T X)(X^T X)^{-1}.

        Omega = 0 everywhere: nothing here is trainable, this is a starting point.
        Forming and inverting X^T X squares the condition number, so
        torch.linalg.lstsq(X, Y) is the form to use if X is ever ill-conditioned.
        It is not used here only so that this reimplementation reproduces the
        original's numbers exactly; at cond(X) = 2.2 in the current test the two
        differ by 4e-6 vs 6e-6 against the true map, so switching is safe.
        """
        X_size = X.size()[1]
        Y_size = Y.size()[1]
        A = MaskedLinear(in_features=X_size + Y_size, out_features=X_size + Y_size,
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            W_ls = (torch.inverse(X.T @ X) @ X.T @ Y).T           # (Y_size, X_size)
            A.weight_0.zero_()
            A.mask.zero_()
            A.weight_0[:X_size, :X_size] = torch.eye(X_size, device=device)
            A.weight_0[X_size:, :X_size] = W_ls
        return A
