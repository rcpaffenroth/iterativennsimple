"""Cross-checks for the reimplementations of MaskedLinear: A, B, C and D.

A, B and C are a style experiment: same public API, same mathematics, different
organisation.  D is a different experiment -- it drops functionality for
readability, so it is not a drop-in replacement and cannot be parametrised over
with the others.  Its tests are in the last section, and their job is to show
that the *mathematics* survived the cuts even though the API did not.  This file is the evidence that they really are the same layer.

Where construction is deterministic, exact agreement with the original
implementation is asserted.  Where a block type is random the three draw from
different generators (the original from numpy, A and B from torch) and at
different positions in the stream, so only the *structure* can be compared --
shapes, and how many entries the sparsity pattern contains.  Those are the
tests at the bottom.
"""
import pytest
import torch

from iterativennsimple.MaskedLinear import MaskedLinear as Original
from iterativennsimple.MaskedLinear_A import MaskedLinear as A
from iterativennsimple.MaskedLinear_B import MaskedLinear as B
from iterativennsimple.MaskedLinear_C import MaskedLinear as C
from iterativennsimple import MaskedLinear_D
from iterativennsimple.MaskedLinear_D import MaskedLinear as D

IMPLEMENTATIONS = [Original, A, B, C]
NAMES = ['original', 'A', 'B', 'C']
REIMPLEMENTATIONS = [A, B, C]

IN_FEATURES, OUT_FEATURES, BATCH = 20, 30, 128

# A description whose every block is deterministic, so all three must agree entrywise.
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


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_matches_torch_linear(Impl):
    """U = 0 at initialisation, so the layer *is* W_0, and W_0 is initialised
    exactly as torch.nn.Linear does -- same draws, same order."""
    x = torch.randn(BATCH, IN_FEATURES)
    torch.manual_seed(0)
    m = Impl(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
    assert torch.allclose(m(x), linear(x))


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_gradient_goes_to_U(Impl):
    """dL/dU = dL/dA .* Omega, and with Omega = 1 that is the dense layer's dL/dA."""
    x = torch.randn(BATCH, IN_FEATURES)
    target = torch.randn(BATCH, OUT_FEATURES)
    torch.manual_seed(0)
    m = Impl(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)

    torch.nn.MSELoss()(m(x), target).backward()
    torch.nn.MSELoss()(linear(x), target).backward()
    assert m.weight_0.grad is None and m.mask.grad is None
    assert torch.allclose(m.U.grad, linear.weight.grad)


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_state_dict_round_trip(Impl, tmp_path):
    """W_0 and Omega are fixed but must still be saved: they are half the layer."""
    m = Impl.from_description(**DETERMINISTIC_DESCRIPTION)
    x = torch.randn(BATCH, 6 + 8)
    torch.save(m.state_dict(), tmp_path / 'm.pt')

    reloaded = Impl(in_features=6 + 8, out_features=5 + 7)
    reloaded.load_state_dict(torch.load(tmp_path / 'm.pt'))
    assert torch.equal(m.weight_0, reloaded.weight_0)
    assert torch.equal(m.mask, reloaded.mask)
    assert torch.allclose(m(x), reloaded(x))


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_trainable_parameter_count(Impl):
    """count_nonzero(Omega) + bias, which for a fresh dense layer is everything."""
    m = Impl(IN_FEATURES, OUT_FEATURES)
    assert m.number_of_trainable_parameters() == IN_FEATURES * OUT_FEATURES + OUT_FEATURES


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_from_coo_is_the_coo_matrix(Impl):
    coo = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                                  values=torch.tensor([1.0, 2.0, 3.0]),
                                  size=(3, 3))
    m = Impl.from_coo(coo, bias=False)
    x = torch.randn(13, 3)
    assert torch.allclose(m(x), (coo @ x.T).T)


# ------------------------------------------------------------------ agreement with the original

@pytest.mark.parametrize('Impl', REIMPLEMENTATIONS, ids=['A', 'B', 'C'])
def test_agrees_on_initialisation(Impl):
    torch.manual_seed(0)
    reference = Original(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    m = Impl(IN_FEATURES, OUT_FEATURES)
    for name in ['weight_0', 'U', 'mask', 'bias']:
        assert torch.equal(getattr(reference, name), getattr(m, name)), name


@pytest.mark.parametrize('Impl', REIMPLEMENTATIONS, ids=['A', 'B', 'C'])
def test_agrees_on_from_MLP_structure(Impl):
    """The subdiagonal blocks are trainable and everything else is zero and frozen."""
    reference = Original.from_MLP(sizes=[10, 5, 1])
    m = Impl.from_MLP(sizes=[10, 5, 1])
    assert torch.equal(reference.mask, m.mask)
    assert torch.equal(reference.weight_0 != 0, m.weight_0 != 0)


def test_only_A_is_seed_compatible_with_the_original():
    """A reproduces the original's random draws for from_MLP; B and C do not.

    A allocates the layer and then draws into slices of it, which is the
    original's order, so a recorded seed gives the same weights.  B and C draw
    the blocks first and build the layer from them afterwards, so __init__'s own
    kaiming draw for weight_0 -- immediately overwritten, but still consumed --
    lands at a different point in the stream.  Same distribution, different
    numbers.  This is the price of "assemble the pair, then construct once", and
    it is recorded rather than hidden.  B and C pay it identically.
    """
    weights = {}
    for name, Impl in [('original', Original), ('A', A), ('B', B), ('C', C)]:
        torch.manual_seed(0)
        weights[name] = Impl.from_MLP(sizes=[10, 5, 1]).weight_0
    assert torch.equal(weights['original'], weights['A'])
    assert not torch.equal(weights['original'], weights['B'])
    assert torch.equal(weights['B'], weights['C'])


@pytest.mark.parametrize('Impl', REIMPLEMENTATIONS, ids=['A', 'B', 'C'])
def test_agrees_on_from_optimal_linear(Impl):
    torch.manual_seed(0)
    X = torch.randn(BATCH, 4)
    Y = X @ torch.randn(4, 3)
    reference = Original.from_optimal_linear(X, Y)
    m = Impl.from_optimal_linear(X, Y)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


@pytest.mark.parametrize('Impl', REIMPLEMENTATIONS, ids=['A', 'B', 'C'])
def test_agrees_on_deterministic_description(Impl):
    reference = Original.from_description(**DETERMINISTIC_DESCRIPTION)
    m = Impl.from_description(**DETERMINISTIC_DESCRIPTION)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


@pytest.mark.parametrize('Impl', REIMPLEMENTATIONS, ids=['A', 'B', 'C'])
def test_agrees_on_single_block_description(Impl):
    """Sequential2D passes plain ints for the sizes; that path must stay."""
    reference = Original.from_description(4, 3, 'W', 'C=2.0', True, bias=False)
    m = Impl.from_description(4, 3, 'W', 'C=2.0', True, bias=False)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


# ------------------------------------------------------------------ structure of the random patterns

@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_row_pattern_has_at_most_n_per_row(Impl):
    """"Row=3" draws 3 columns per row with replacement, so at most 3 land."""
    m = Impl.from_description(7, 10, 'Row=3', 'G', 'non-zero', bias=False)
    per_row = torch.count_nonzero(m.mask, dim=1)
    assert torch.all(per_row <= 3) and torch.all(per_row >= 1)


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_scattered_pattern_has_at_most_n_entries(Impl):
    """"S=5" draws 5 positions in the block with replacement, so at most 5 land."""
    m = Impl.from_description(7, 10, 'S=5', 'G', 'non-zero', bias=False)
    assert 1 <= torch.count_nonzero(m.mask) <= 5


@pytest.mark.parametrize('Impl', IMPLEMENTATIONS, ids=NAMES)
def test_bernoulli_pattern_density(Impl):
    """"R=0.5" keeps each entry independently: nnz/N is binomial(N, 0.5)/N, and
    N = 10000 puts 5 standard deviations at 0.025."""
    m = Impl.from_description(100, 100, 'R=0.5', 'G', 'non-zero', bias=False)
    density = torch.count_nonzero(m.mask) / m.mask.numel()
    assert abs(density - 0.5) < 0.025


# ------------------------------------------------------------------ D, which is not a drop-in

def test_D_dense_matches_torch_linear():
    """D's constructor takes the (W_0, Omega) pair, so the familiar two-int call
    is MaskedLinear.dense(in, out).  It must still draw exactly as Linear does."""
    x = torch.randn(BATCH, IN_FEATURES)
    torch.manual_seed(0)
    m = D.dense(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
    assert torch.allclose(m(x), linear(x))
    assert torch.equal(m.weight_0, linear.weight)
    assert torch.equal(m.bias, linear.bias)


def test_D_gradient_goes_to_U():
    x = torch.randn(BATCH, IN_FEATURES)
    target = torch.randn(BATCH, OUT_FEATURES)
    torch.manual_seed(0)
    m = D.dense(IN_FEATURES, OUT_FEATURES)
    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
    torch.nn.MSELoss()(m(x), target).backward()
    torch.nn.MSELoss()(linear(x), target).backward()
    assert m.weight_0.grad is None and m.mask.grad is None
    assert torch.allclose(m.U.grad, linear.weight.grad)


def test_D_state_dict_round_trip(tmp_path):
    m = D.dense(IN_FEATURES, OUT_FEATURES)
    x = torch.randn(BATCH, IN_FEATURES)
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    reloaded = D.dense(IN_FEATURES, OUT_FEATURES)
    reloaded.load_state_dict(torch.load(tmp_path / 'm.pt'))
    assert torch.allclose(m(x), reloaded(x))


def test_D_reproduces_a_description_it_can_no_longer_parse():
    """The cut that matters: D has no from_description.  Writing the same block
    matrix by hand must give the same two matrices, entry for entry."""
    reference = Original.from_description(**DETERMINISTIC_DESCRIPTION)
    T = DETERMINISTIC_DESCRIPTION['initialization_types'][0][1]        # the given (5, 8) tensor
    block = MaskedLinear_D.stack([
        [MaskedLinear_D.frozen(torch.zeros(5, 6)),
         MaskedLinear_D.trainable(T)],
        [MaskedLinear_D.trainable_where_nonzero(0.3 * torch.eye(7, 6)),
         MaskedLinear_D.frozen(torch.ones(7, 8))],
    ])
    m = D(block)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


def test_D_agrees_on_from_MLP_structure():
    reference = Original.from_MLP(sizes=[10, 5, 1])
    m = D.from_MLP(sizes=[10, 5, 1])
    assert torch.equal(reference.mask, m.mask)
    assert torch.equal(reference.weight_0 != 0, m.weight_0 != 0)


def test_D_agrees_on_from_optimal_linear():
    torch.manual_seed(0)
    X = torch.randn(BATCH, 4)
    Y = X @ torch.randn(4, 3)
    reference = Original.from_optimal_linear(X, Y)
    m = D.from_optimal_linear(X, Y)
    assert torch.equal(reference.weight_0, m.weight_0)
    assert torch.equal(reference.mask, m.mask)


def test_D_from_coo_replacement_is_one_line():
    """from_coo was cut because this is its whole body."""
    coo = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                                  values=torch.tensor([1.0, 2.0, 3.0]),
                                  size=(3, 3))
    m = D(MaskedLinear_D.trainable_where_nonzero(coo.to_dense()), bias=False)
    x = torch.randn(13, 3)
    assert torch.allclose(m(x), (coo @ x.T).T)


def test_D_patterns():
    """The three patterns torch does not provide, kept as functions."""
    assert torch.all(torch.count_nonzero(MaskedLinear_D.per_row(7, 10, n=3), dim=1) <= 3)
    assert torch.count_nonzero(MaskedLinear_D.scattered(7, 10, n=5)) <= 5
    density = torch.count_nonzero(MaskedLinear_D.bernoulli(100, 100, p=0.5)) / 10000
    assert abs(density - 0.5) < 0.025      # 5 standard deviations is 0.025 at N = 10000
