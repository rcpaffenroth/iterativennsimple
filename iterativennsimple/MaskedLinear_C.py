"""Reimplementation C of MaskedLinear:  B's (values, mask) pair, A's flat dispatch.

The layer is an ordinary linear layer whose weight is constrained to an affine
family,

    A = W_0 + U .* Omega            (.* is entrywise)
    y = x A^T + b

with W_0 (the initial weight) and Omega (the mask) *fixed* and U the only
learnable parameter.  So Omega selects which entries of the weight are allowed
to move, and W_0 says where they start.  Omega = 1 recovers a dense
torch.nn.Linear; Omega = 0 freezes the layer at W_0.

The layer is therefore determined by the pair (W_0, Omega) of equal-shaped
matrices, and that pair is the one named thing in this file: `Block` is such a
pair, block matrices of `Block`s `stack` into one `Block` (because stacking
acts on the two matrices independently), and every constructor below assembles
`Block`s and hands the result to `from_block`, which is the one place that
writes into the module's tensors.  A block-matrix constructor then reads like
the block matrix it builds.

The string mini-languages ('R=0.5' for which entries exist, 'G=0.2,0.7' for
what they are) are read where they are used, by one flat function each --
`support` and `values` -- rather than parsed into a vocabulary of types first.
There are two of these languages and each is used in exactly one place, so the
if/elif chain *is* the documentation of the language.

Deliberately unchanged from the original implementation (see MaskedLinear.py):
  * weight_0 and mask are Parameters with requires_grad=False rather than
    buffers, because tests/test_Sequential2D.py counts len(model.parameters()).
  * 'non-zero' trainability keys off the *realised values*, not the support.
  * from_optimal_linear solves the normal equations rather than using lstsq.
Changed: the random block types draw from torch, not numpy, so torch.manual_seed
reproduces them; and the per-entry numpy.vectorize loops are gone.
"""
import math
from typing import Any, NamedTuple

import torch


# ---------------------------------------------------------------- the pair (W_0, Omega)

class Block(NamedTuple):
    """One block of the weight: what the entries start at, and which may move."""
    values: torch.Tensor      # (out_features, in_features), W_0 on this block
    mask: torch.Tensor        # (out_features, in_features), Omega on this block, 1 = trainable


def stack(blocks: list[list[Block]]) -> Block:
    """Assemble a 2D array of Blocks into the single Block of the block matrix.

    Stacking a block matrix is the same operation on W_0 and on Omega
    independently, which is the whole reason for carrying them as a pair.
    """
    values = torch.vstack([torch.hstack([b.values for b in row]) for row in blocks])
    mask = torch.vstack([torch.hstack([b.mask for b in row]) for row in blocks])
    return Block(values, mask)


# The three values of `trainable`, as the three ways to turn values into a Block.

def frozen(values: torch.Tensor) -> Block:
    return Block(values, torch.zeros_like(values))


def trainable(values: torch.Tensor) -> Block:
    return Block(values, torch.ones_like(values))


def trainable_where_nonzero(values: torch.Tensor) -> Block:
    """Only the entries that came out non-zero may move.

    Note this keys off the *realised values*, not the sparsity pattern: an entry
    of the pattern initialised to exactly 0.0 is frozen.  That is the original
    rule, and the one SparseLinear.from_MaskedLinearExact assumes.
    """
    return Block(values, (values != 0.0).to(values.dtype))


# ---------------------------------------------------------------- the two mini-languages

def support(block_type, out_features: int, in_features: int) -> torch.Tensor:
    """The 0/1 pattern of structurally non-zero entries, shape (out_features, in_features).

    block_type is the sparsity pattern of the block, and nothing else:
        0        no entries
        "W"      all entries
        "D"      the diagonal
        "R=0.5"  each entry independently with probability 0.5
        "Row=3"  3 draws per row, with replacement, so *at most* 3 per row
        "S=5"    5 draws in the whole block, with replacement, so at most 5
    The "with replacement" is why the original docstring warns you may get fewer
    entries than you asked for; drawing without replacement would need a randperm
    per row, which costs more than the guarantee is worth here.
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
        return torch.zeros(out_features, in_features).scatter_(1, cols, 1.0)   # duplicates collapse
    if block_type.startswith("S="):
        n = int(block_type[2:])
        rows = torch.randint(out_features, (n,))                     # (n,)
        cols = torch.randint(in_features, (n,))                      # (n,)
        pattern = torch.zeros(out_features, in_features)
        pattern[rows, cols] = 1.0
        return pattern
    assert False, f"unknown block type {block_type}"


def values(initialization_type, out_features: int, in_features: int) -> torch.Tensor:
    """The value each entry would take if it were in the support, shape (out_features, in_features).

    initialization_type says *what* the entries are, independently of *which*
    entries there are:
        0, 1, "C=0.3"       that constant everywhere
        "G", "G=mu,sigma"   Gaussian draws, default mu=0, sigma=1
        "U", "U=lo,hi"      uniform draws, default lo=-1, hi=1
        a tensor            those values, verbatim
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


def make_block(block_type, initialization_type, trainability,
               out_features: int, in_features: int) -> Block:
    """One block of a description:  W_0 = values .* support,  Omega from `trainability`.

    The two languages are independent -- block_type says *which* entries exist,
    initialization_type says *what* they are -- so the block is a product.  The
    dense draw is thrown away outside the pattern, which wastes draws on very
    sparse blocks and keeps the two languages separable.
    """
    if block_type == 0:
        assert not trainability, "0 block should not be trainable"
        assert initialization_type == 0, "0 block should be initialized to 0"

    W = values(initialization_type, out_features, in_features) \
        * support(block_type, out_features, in_features)

    if trainability == 'non-zero':
        return trainable_where_nonzero(W)
    assert trainability in (True, False), f"unknown train type {trainability}"
    return trainable(W) if trainability else frozen(W)


def kaiming(out_features: int, in_features: int) -> torch.Tensor:
    """torch.nn.Linear's own weight initialisation, as a function returning a matrix.

    a=sqrt(5) equals uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)); it is torch's
    historical choice, see https://github.com/pytorch/pytorch/issues/57109.
    """
    W = torch.zeros(out_features, in_features)
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W


# ---------------------------------------------------------------- the module

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
        torch.nn.Linear built under the same seed.  The order of these four calls
        fixes the random stream, so changing it breaks that agreement.
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
    def from_block(block: Block, bias: bool = True, device=None, dtype=None) -> Any:
        """The one constructor: a (W_0, Omega) pair becomes a layer of that shape."""
        out_features, in_features = block.values.shape
        A = MaskedLinear(in_features=in_features, out_features=out_features,
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            A.weight_0[:, :] = block.values
            A.mask[:, :] = block.mask
        return A

    @staticmethod
    def from_description(out_features_sizes, in_features_sizes,
                         block_types, initialization_types, trainable,
                         bias: bool = True, device=None, dtype=None) -> Any:
        """Build a block matrix, one entry of each 2D argument per block.

        out_features_sizes and in_features_sizes give the block row heights and
        column widths, so the layer is sum(out_features_sizes) by
        sum(in_features_sizes).  The other three arguments are 2D arrays of the
        same block shape and go to `make_block` above, one cell at a time.
        Passing ints for the two sizes builds a single block, which is how
        Sequential2D calls this.
        """
        if (type(out_features_sizes) is int) and (type(in_features_sizes) is int):
            out_features_sizes = [out_features_sizes]
            in_features_sizes = [in_features_sizes]
            block_types = [[block_types]]
            initialization_types = [[initialization_types]]
            trainable = [[trainable]]

        blocks = [[make_block(block_types[i][j], initialization_types[i][j], trainable[i][j],
                              out_features, in_features)
                   for j, in_features in enumerate(in_features_sizes)]
                  for i, out_features in enumerate(out_features_sizes)]
        return MaskedLinear.from_block(stack(blocks), bias=bias, device=device, dtype=dtype)

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
        A = MaskedLinear.from_block(trainable_where_nonzero(coo.to_dense()),
                                    bias=bias, device=device, dtype=dtype)
        if check_mask:
            # A stored entry whose value happens to be 0.0 is *not* caught by
            # trainable_where_nonzero and so is not trainable.  Slow, so it is opt-in.
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
        blocks = [[trainable(kaiming(n_out, n_in)) if i == j + 1 else frozen(torch.zeros(n_out, n_in))
                   for j, n_in in enumerate(sizes)]
                  for i, n_out in enumerate(sizes)]
        return MaskedLinear.from_block(stack(blocks), bias=bias, device=device, dtype=dtype)

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

        Nothing here is trainable -- every block is `frozen` -- this is a
        starting point.  Forming and inverting X^T X squares the condition
        number, so torch.linalg.lstsq(X, Y) is the form to use if X is ever
        ill-conditioned.  It is not used here only so that this reimplementation
        reproduces the original's numbers exactly; at cond(X) = 2.2 in the
        current test the two differ by 4e-6 vs 6e-6, so switching is safe.
        """
        D = X.size()[1]
        K = Y.size()[1]
        with torch.no_grad():
            W_ls = (torch.inverse(X.T @ X) @ X.T @ Y).T           # (K, D)
            blocks = [[frozen(torch.eye(D, device=device)), frozen(torch.zeros(D, K, device=device))],
                      [frozen(W_ls),                        frozen(torch.zeros(K, K, device=device))]]
        return MaskedLinear.from_block(stack(blocks), bias=bias, device=device, dtype=dtype)
