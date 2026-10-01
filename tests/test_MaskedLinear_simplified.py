"""Tests for MaskedLinear_simplified, which has two jobs here.

Section 1 checks the mathematics against the reference implementation in
`MaskedLinear.py`.  The simplified version dropped functionality, not
mathematics, and that is a claim to be checked rather than asserted: wherever a
construction exists in both, the two must agree entrywise.

Section 2 is worked usage.  Each test is a small complete example of building,
inspecting or training a layer whose weight has a prescribed structure, and is
meant to be read as documentation of what the file is for.  The design record is
`tasks/STYLE_EXPERIMENT_MaskedLinear_simplified.md`.
"""
import torch

from iterativennsimple.MaskedLinear import MaskedLinear as Reference
from iterativennsimple.MaskedLinear_simplified import (
    MaskedLinear, Block, block_matrix, frozen, trainable, trainable_where_nonzero,
    bernoulli, per_row, scattered, kaiming,
)

IN_FEATURES, OUT_FEATURES, BATCH = 20, 30, 128

# A block description whose every block is deterministic, so the reference
# implementation's from_description can be reproduced by hand and compared.
DETERMINISTIC_DESCRIPTION = dict(
    out_features_sizes=[5, 7],
    in_features_sizes=[6, 8],
    block_types=[[0, 'W'],
                 ['D', 'W']],
    initialization_types=[[0, torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)],
                          ['C=0.3', 1]],
    trainable=[[False, True],
               ['non-zero', False]],
)


def weight(m: MaskedLinear) -> torch.Tensor:
    """The effective weight A = W_0 + U .* Omega, which is what forward() uses."""
    return m.weight_0 + m.U * m.mask


def train_briefly(m: MaskedLinear, x: torch.Tensor, target: torch.Tensor, steps: int = 5) -> None:
    """Plain SGD on the whole module.  Nothing here knows about the mask; the
    mask does its work through the gradient, dL/dU = dL/dA .* Omega."""
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        torch.nn.MSELoss()(m(x), target).backward()
        optimizer.step()


# ================================================================= 1. agreement with the reference

def test_dense_matches_torch_linear():
    """The constructor takes the (W_0, Omega) pair, so the familiar two-int call is
    MaskedLinear.dense(in, out).  It must still draw exactly as Linear does, in the
    same order, so that a layer and its dense counterpart agree under one seed."""
    x = torch.randn(BATCH, IN_FEATURES)
    torch.manual_seed(0)
    m = MaskedLinear.dense(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
    assert torch.equal(m.weight_0, linear.weight)
    assert torch.equal(m.bias, linear.bias)
    assert torch.allclose(m(x), linear(x))


def test_gradient_goes_to_U():
    """dL/dU = dL/dA .* Omega, and with Omega = 1 that is the dense layer's dL/dA.
    W_0 and Omega must receive no gradient at all."""
    x = torch.randn(BATCH, IN_FEATURES)
    target = torch.randn(BATCH, OUT_FEATURES)
    torch.manual_seed(0)
    m = MaskedLinear.dense(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)

    torch.nn.MSELoss()(m(x), target).backward()
    torch.nn.MSELoss()(linear(x), target).backward()
    assert m.weight_0.grad is None and m.mask.grad is None
    assert torch.allclose(m.U.grad, linear.weight.grad)


def test_state_dict_round_trip(tmp_path):
    """W_0 and Omega are fixed but must still be saved: they are half the layer."""
    m = MaskedLinear.dense(IN_FEATURES, OUT_FEATURES)
    x = torch.randn(BATCH, IN_FEATURES)
    torch.save(m.state_dict(), tmp_path / 'm.pt')

    reloaded = MaskedLinear.dense(IN_FEATURES, OUT_FEATURES)
    reloaded.load_state_dict(torch.load(tmp_path / 'm.pt'))
    assert torch.equal(m.weight_0, reloaded.weight_0)
    assert torch.equal(m.mask, reloaded.mask)
    assert torch.allclose(m(x), reloaded(x))


def test_reproduces_a_description_it_can_no_longer_parse():
    """The cut that matters: there is no from_description and no string
    mini-language.  Writing the same block matrix by hand must give the same two
    matrices, entry for entry."""
    reference = Reference.from_description(**DETERMINISTIC_DESCRIPTION)
    T = DETERMINISTIC_DESCRIPTION['initialization_types'][0][1]        # the given (5, 8) tensor
    m = MaskedLinear(block_matrix([
        [frozen(torch.zeros(5, 6)),                      trainable(T)],
        [trainable_where_nonzero(0.3 * torch.eye(7, 6)), frozen(torch.ones(7, 8))],
    ]))
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


def test_agrees_on_from_MLP_structure():
    """Same subdiagonal layout: same mask, same pattern of non-zeros.  The values
    differ because they are independent draws from the same distribution."""
    reference = Reference.from_MLP(sizes=[10, 5, 1])
    m = MaskedLinear.from_MLP(sizes=[10, 5, 1])
    assert torch.equal(reference.mask, m.mask)
    assert torch.equal(reference.weight_0 != 0, m.weight_0 != 0)


def test_agrees_on_from_optimal_linear():
    torch.manual_seed(0)
    X = torch.randn(BATCH, 4)
    Y = X @ torch.randn(4, 3)
    reference = Reference.from_optimal_linear(X, Y)
    m = MaskedLinear.from_optimal_linear(X, Y)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


def test_from_coo_replacement_is_one_line():
    """from_coo was cut because this is its whole body."""
    coo = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                                  values=torch.tensor([1.0, 2.0, 3.0]),
                                  size=(3, 3))
    m = MaskedLinear(trainable_where_nonzero(coo.to_dense()), bias=False)
    x = torch.randn(13, 3)
    assert torch.allclose(m(x), (coo @ x.T).T)


def test_the_three_patterns():
    """The three sparsity patterns torch does not provide.  All three draw with
    replacement, so their counts are upper bounds."""
    assert torch.all(torch.count_nonzero(per_row(7, 10, n=3), dim=1) <= 3)
    assert torch.count_nonzero(scattered(7, 10, n=5)) <= 5
    density = torch.count_nonzero(bernoulli(100, 100, p=0.5)) / 10000
    assert abs(density - 0.5) < 0.025      # 5 standard deviations is 0.025 at N = 10000


# ================================================================= 2. worked usage

def test_usage_only_the_masked_entries_ever_move():
    """The promise of the layer, and the reason to prefer it to zeroing gradients
    by hand: train a full 10x10 layer with Omega = I and the off-diagonal of the
    effective weight A is *bitwise* unchanged, not merely small."""
    torch.manual_seed(0)
    n = 10
    m = MaskedLinear(Block(values=kaiming(n, n), mask=torch.eye(n)), bias=False)
    before = weight(m).clone()

    train_briefly(m, torch.randn(64, n), torch.randn(64, n))
    after = weight(m)

    off_diagonal = ~torch.eye(n, dtype=torch.bool)
    assert torch.equal(before[off_diagonal], after[off_diagonal])        # frozen exactly
    assert not torch.allclose(before.diag(), after.diag())               # and the rest learned


def test_usage_a_frozen_layer_is_a_constant_function():
    """Omega = 0 is a fixed linear map that still composes and still backpropagates
    to whatever is upstream of it -- useful as a fixed random projection."""
    torch.manual_seed(0)
    m = MaskedLinear(frozen(kaiming(5, 8)), bias=False)
    x = torch.randn(16, 8)
    before = m(x).clone()

    train_briefly(m, x, torch.randn(16, 5))
    assert torch.equal(before, m(x))
    assert m.number_of_trainable_parameters() == 0


def test_usage_a_block_matrix_written_on_the_page():
    """The replacement for from_description: the state is z = (x, h) with x of
    width 4 and h of width 3, and the layer

        A = [ I_4   0 ]     (x is copied through, frozen)
            [ W     0 ]     (h is overwritten by a trained readout of x)

    is written as the block matrix it is.  Shapes are visible, not implied.
    """
    torch.manual_seed(0)
    m = MaskedLinear(block_matrix([
        [frozen(torch.eye(4)),        frozen(torch.zeros(4, 3))],
        [trainable(kaiming(3, 4)),    frozen(torch.zeros(3, 3))],
    ]), bias=False)

    assert m.in_features == 7 and m.out_features == 7
    assert m.number_of_trainable_parameters() == 3 * 4      # only the readout block

    z = torch.cat([torch.randn(16, 4), torch.randn(16, 3)], dim=1)
    out = m(z)
    assert torch.equal(out[:, :4], z[:, :4])                # x passed through unchanged
    assert torch.allclose(out[:, 4:], z[:, :4] @ weight(m)[4:, :4].T)


def test_usage_a_sparse_block_stays_sparse_while_it_trains():
    """A 10%-dense block: `trainable_where_nonzero` makes exactly the stored
    entries trainable, so training moves those and cannot fill in the rest.  This
    is what "sparsity is structural, not incidental" means operationally."""
    torch.manual_seed(0)
    values = torch.randn(12, 12) * bernoulli(12, 12, p=0.1)
    m = MaskedLinear(trainable_where_nonzero(values), bias=False)
    support = values != 0.0

    assert m.number_of_trainable_parameters() == int(support.sum())

    train_briefly(m, torch.randn(64, 12), torch.randn(64, 12))
    assert torch.equal(weight(m) == 0.0, ~support)          # same zeros, no new ones, none filled


def test_usage_any_pattern_you_can_write_down():
    """The point of dropping the string mini-language: `torch.tril` was never one
    of its block types and needs no new code here.  A lower-triangular weight --
    output i sees only inputs j <= i -- is one call.
    """
    torch.manual_seed(0)
    n = 8
    m = MaskedLinear(trainable_where_nonzero(kaiming(n, n) * torch.tril(torch.ones(n, n))),
                     bias=False)
    train_briefly(m, torch.randn(64, n), torch.randn(64, n))
    assert torch.equal(torch.triu(weight(m), diagonal=1), torch.zeros(n, n))


def test_usage_from_MLP_is_an_iterated_map():
    """Why the MLP is written as one square matrix: the network becomes a
    dynamical system, and one application of A advances every layer by one step.

    With sizes = [3, 4, 2] the state is z = (z_0, z_1, z_2) and A has W_1 and W_2
    on the block subdiagonal, so from z = (x, 0, 0),

        A z   = (0, W_1 x, 0)
        A^2 z = (0, 0, W_2 W_1 x)

    i.e. after L applications the last slot holds the network's output.  (No
    activation here: the nonlinearity lives in Sequential2D, not in the matrix.)
    """
    torch.manual_seed(0)
    sizes = [3, 4, 2]
    m = MaskedLinear.from_MLP(sizes, bias=False)
    W_1 = m.weight_0[3:7, 0:3]           # block (1, 0), shape (4, 3)
    W_2 = m.weight_0[7:9, 3:7]           # block (2, 1), shape (2, 4)

    x = torch.randn(3)
    z = torch.cat([x, torch.zeros(4), torch.zeros(2)])
    z = m(m(z))                          # two applications for a two-layer net
    assert torch.allclose(z[7:9], W_2 @ (W_1 @ x), atol=1e-6)


def test_usage_device_and_dtype_come_from_to():
    """There is no dtype= argument anywhere; `.to()` does the job.  The cost is
    that the initial draw happened in float32 and was then cast, so W_0 carries
    float32 resolution in a float64 layer."""
    m = MaskedLinear.dense(4, 3).to(torch.float64)
    x = torch.randn(8, 4, dtype=torch.float64)
    assert m(x).dtype == torch.float64
    assert m.weight_0.dtype == torch.float64


def test_usage_counting_parameters_two_different_ways():
    """`len(list(parameters()))` counts W_0, Omega, U and b as four tensors, and
    `sum(p.numel() ...)` over them triple-counts the weight.  The number that
    means something is count_nonzero(Omega) plus the bias."""
    torch.manual_seed(0)
    m = MaskedLinear(trainable_where_nonzero(torch.randn(6, 6) * bernoulli(6, 6, p=0.25)))

    assert len(list(m.parameters())) == 4
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 6 * 6 + 6   # U and b, dense
    assert m.number_of_trainable_parameters() == int(torch.count_nonzero(m.mask)) + 6
