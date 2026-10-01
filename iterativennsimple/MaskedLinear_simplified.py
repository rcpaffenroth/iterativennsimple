"""MaskedLinear, simplified:  the same mathematics with the string interpreter removed.

`MaskedLinear.py` is the reference implementation and this file is **not a
drop-in replacement** for it: functionality was deliberately sacrificed for
readability.  It keeps the mathematics and drops the machinery that exists to
serve YAML configuration files.  What was cut, and what it costs, is listed at
the bottom of this docstring; the experiment it came out of, with the
measurements, is `tasks/STYLE_EXPERIMENT_MaskedLinear_simplified.md`.

The layer is an ordinary linear layer whose weight is constrained to an affine
family,

    A = W_0 + U .* Omega            (.* is entrywise)
    y = x A^T + b

with W_0 (the initial weight) and Omega (the mask) *fixed* and U the only
learnable parameter.  So Omega selects which entries of the weight are allowed
to move, and W_0 says where they start.  Omega = 1 recovers a dense
torch.nn.Linear; Omega = 0 freezes the layer at W_0.

The layer is therefore *determined by* the pair (W_0, Omega) of equal-shaped
matrices, and here that is taken literally: `Block` is such a pair, the pair is
what the constructor takes, and `block_matrix` assembles a 2D array of pairs into
one pair.  A block matrix is then written the way it is written on paper.  The original's

    MaskedLinear.from_description(out_features_sizes=[5, 7], in_features_sizes=[6, 8],
                                 block_types=[[0,       'W'    ],
                                              ['D',     'R=0.5']],
                                 initialization_types=[[0,       'G=0.2,0.7'],
                                                       ['C=0.3', 'U'        ]],
                                 trainable=[[False,      True      ],
                                            ['non-zero', 'non-zero']])

becomes

    MaskedLinear(block_matrix([
        [frozen(torch.zeros(5, 6)),                 trainable(torch.randn(5, 8) * 0.7 + 0.2)],
        [trainable_where_nonzero(0.3 * torch.eye(7, 6)),
         trainable_where_nonzero(torch.rand(7, 8).mul(2).sub(1) * bernoulli(7, 8, p=0.5))],
    ]))

which is longer on the page and shorter to read: the three parallel 2D arrays
that had to stay aligned are one 2D array, the shapes are visible instead of
implied by position, any cell may be any tensor you can produce, and there is
no interpreter between what you wrote and what you get.

WHAT WAS CUT, AND WHAT IT COSTS

  * `from_description` and the two string mini-languages ('W', 'D', 'R=0.5',
    'S=15', 'Row=15', 'C=0.3', 'G=0.2,0.7', 'U=-0.5,0.5').  Values are now
    written with torch directly -- `torch.randn(5, 8) * 0.7 + 0.2` is 'G=0.2,0.7'
    -- and only the three sparsity patterns torch does not provide are kept as
    functions below.  Cost: a dict or YAML file cannot name a block type, so
    a configuration cannot reach this layer directly.  That is restored by
    `iterativennsimple/masked_linear_simplified_config.py`, which is a separate module
    holding a small closed vocabulary and a YAML loader -- it imports this file
    and this file knows nothing about it.  A configuration language exists to
    serve configuration files, so it lives with them.
  * `from_config`.  It forwarded six dict keys to identically-named arguments
    and is called from nowhere in the repository [det, grep].  Pure deletion.
  * `from_coo`, whose body is now one line the caller can write:
    `MaskedLinear(trainable_where_nonzero(coo.to_dense()))`.  Its `check_mask`
    flag never worked -- `range(coo.indices())` -- so nothing is lost there.
  * `MaskedLinear(in_features, out_features)`.  The constructor takes the pair
    it is defined by; `MaskedLinear.dense(in_features, out_features)` is the old
    behaviour.  This is the break that matters: every existing call site,
    including `Sequential2D`, passes two ints.  In exchange there is one
    construction path instead of two, no `torch.empty` followed by
    `reset_parameters`, and no allocate-then-overwrite -- so nothing is drawn
    from the generator and immediately discarded, and a recorded seed still
    reproduces.
  * `reset_parameters`.  Inlined, and it is the cut that costs nothing at all:
    each matrix's shape and initial value now appear on one line together
    instead of an empty allocation in one method and a fill in another.
  * `device` and `dtype` arguments, from every signature -- ten parameters and
    the `factory_kwargs` idiom.  Use `.to(device=..., dtype=...)` on the result.
    Cost: the initial draw happens in float32 and is then cast, so a float64
    layer's W_0 has float32 resolution.  That matters for a bit-exact float64
    regression test and for nothing else here.

  Considered and kept: `bias` (a genuinely different affine family, and the
  agreement with torch.nn.Linear rests on its fan_in-dependent initialisation);
  `U` as a separate parameter rather than masking gradients (it *is* the idea --
  what is learned is the update); `from_MLP` and `from_optimal_linear` (both
  encode mathematics of this project rather than plumbing).
"""
import math
from typing import Any, NamedTuple

import torch


# ---------------------------------------------------------------- the pair (W_0, Omega)

class Block(NamedTuple):
    """One block of the weight: what the entries start at, and which may move."""
    values: torch.Tensor      # (out_features, in_features), W_0 on this block
    mask: torch.Tensor        # (out_features, in_features), Omega on this block, 1 = trainable


def block_matrix(blocks: list[list[Block]]) -> Block:
    """Assemble a 2D array of Blocks into the single Block of the block matrix.

    This is a 2D *unzip* followed by two ordinary assemblies.  A Block is a pair,
    so a 2D array of Blocks is a 2D array of pairs -- and what is wanted is a pair
    of 2D arrays.  Unzip it, assemble each half with hstack and vstack, and pair
    the two results up again:

        [[ (V00, M00), (V01, M01) ],  unzip   [[V00, V01],       [[M00, M01],
         [ (V10, M10), (V11, M11) ]]   --->    [V10, V11]]  and   [M10, M11]]
                                                   |                   |
                                           hstack, then vstack   hstack, then vstack
                                                   v                   v
                                               W_0 (6, 7)          Omega (6, 7)

    with, say, V00 (2, 3), V01 (2, 4), V10 (4, 3), V11 (4, 4).  That assembly is
    the same operation on W_0 and on Omega and never mixes them, which is the
    whole reason the two are carried together as one object.

    Not torch.stack, which adds an axis -- this does not.  A row whose block
    heights disagree crashes in torch.hstack with both shapes in the message.
    """
    values = torch.vstack([torch.hstack([b.values for b in row]) for row in blocks])
    mask = torch.vstack([torch.hstack([b.mask for b in row]) for row in blocks])
    return Block(values, mask)


# The three ways a block can be trainable.

def frozen(values: torch.Tensor) -> Block:
    return Block(values, torch.zeros_like(values))


def trainable(values: torch.Tensor) -> Block:
    return Block(values, torch.ones_like(values))


def trainable_where_nonzero(values: torch.Tensor) -> Block:
    """Only the entries that came out non-zero may move -- the usual choice for a
    sparse block, where the zeros are structural and are not meant to fill in.

    Note this keys off the realised values: an entry of the pattern that is
    initialised to exactly 0.0 is frozen.  SparseLinear.from_MaskedLinearExact
    assumes this rule.
    """
    return Block(values, (values != 0.0).to(values.dtype))


# ---------------------------------------------------------------- what torch does not provide

# Values are written with torch: torch.ones, torch.full, torch.eye, torch.randn,
# torch.rand.  Only these four have no one-call equivalent.  Each returns a
# matrix, and a sparse block is the entrywise product of a value matrix and one
# of the three 0/1 patterns.

def bernoulli(out_features: int, in_features: int, p: float) -> torch.Tensor:
    """Each entry kept independently with probability p.  ('R=0.5' in the original.)"""
    return (torch.rand(out_features, in_features) < p).to(torch.float32)


def per_row(out_features: int, in_features: int, n: int) -> torch.Tensor:
    """n draws per row, with replacement, so at most n entries per row.  ('Row=15'.)

    Drawing without replacement would need a randperm per row; the original
    documents the shortfall rather than paying for the guarantee, and so does this.
    """
    cols = torch.randint(in_features, (out_features, n))        # (out_features, n) column index per draw
    return torch.zeros(out_features, in_features).scatter_(1, cols, 1.0)   # duplicates collapse


def scattered(out_features: int, in_features: int, n: int) -> torch.Tensor:
    """n draws in the whole block, with replacement, so at most n entries.  ('S=15'.)"""
    rows = torch.randint(out_features, (n,))                    # (n,)
    cols = torch.randint(in_features, (n,))                     # (n,)
    pattern = torch.zeros(out_features, in_features)
    pattern[rows, cols] = 1.0
    return pattern


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
        block: the pair (W_0, Omega), both (out_features, in_features)
        bias:  include a bias term (itself always trainable)
    Shape:
        input  (*, in_features)  ->  output (*, out_features)

    Example:
        >>> m = MaskedLinear.dense(20, 30)
        >>> m(torch.randn(128, 20)).shape
        torch.Size([128, 30])

    The weight is dense and the mask is applied by multiplication, so this costs
    a dense matmul no matter how sparse Omega is.  Its value is as a reference --
    SparseLinear and MonarchLinear must agree with it -- and that every stock
    torch optimizer works on it unchanged.
    """

    def __init__(self, block: Block, bias: bool = True) -> None:
        super().__init__()
        self.out_features, self.in_features = block.values.shape

        # W_0 and Omega are Parameters only so that they land in state_dict();
        # requires_grad=False is what makes them fixed.  U starts at 0, so a
        # freshly built layer *is* W_0.
        self.weight_0 = torch.nn.Parameter(block.values.clone(), requires_grad=False)
        self.mask = torch.nn.Parameter(block.mask.clone(), requires_grad=False)
        self.U = torch.nn.Parameter(torch.zeros_like(block.values))

        if bias:
            # torch.nn.Linear's bias initialisation: uniform(-1/sqrt(fan_in), +).
            bound = 1 / math.sqrt(self.in_features)
            self.bias = torch.nn.Parameter(torch.empty(self.out_features).uniform_(-bound, bound))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def dense(in_features: int, out_features: int, bias: bool = True) -> Any:
        """Omega = 1 and W_0 drawn as torch.nn.Linear draws it: the same layer, relearnable.

        Same draws in the same order as torch.nn.Linear(in_features, out_features),
        so under one seed the two agree entrywise.
        """
        return MaskedLinear(trainable(kaiming(out_features, in_features)), bias=bias)

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

    @staticmethod
    def from_MLP(sizes, bias: bool = True) -> Any:
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
        return MaskedLinear(block_matrix(blocks), bias=bias)

    @staticmethod
    def from_optimal_linear(X, Y, bias: bool = False) -> Any:
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
        number, so use torch.linalg.lstsq(X, Y) if X is ever ill-conditioned;
        the normal equations are kept only to reproduce the original's numbers.
        """
        D = X.size()[1]
        K = Y.size()[1]
        with torch.no_grad():
            W_ls = (torch.inverse(X.T @ X) @ X.T @ Y).T           # (K, D)
        return MaskedLinear(block_matrix([[frozen(torch.eye(D)), frozen(torch.zeros(D, K))],
                                   [frozen(W_ls),         frozen(torch.zeros(K, K))]]), bias=bias)
