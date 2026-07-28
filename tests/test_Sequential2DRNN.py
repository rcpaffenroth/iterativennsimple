"""Tests for Sequential2DRNN.

The headline test -- that a torch.nn.RNN can be copied into a Sequential2D block
map and give the same answer -- lives in the tutorial notebook
`notebooks/7-rcp-RNN-as-Sequential2D.ipynb`, which is symlinked into this
directory and run by nbmake.  It is a better test *and* a better explanation
there, because the point is as much "here is why this works" as "this works".

What is left here is the fine print: shapes, dtypes, gradient flow, the
invariants that fail silently, and the block types that should drop in.  None of
it is instructive, all of it will eventually break.
"""

import pytest
import torch

from iterativennsimple.Sequential2DRNN import Sequential2DRNN
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D


def linear(i, o):
    """A bias-free Linear: bias belongs to the slot, not the block (Sec. 8.2)."""
    return torch.nn.Linear(i, o, bias=False)


# ---------------------------------------------------------------------------
# torch.nn.RNN equivalence, across every knob torch.nn.RNN exposes
# ---------------------------------------------------------------------------

def test_matches_torch_rnn_across_options():
    torch.manual_seed(0)
    for batch_first in [False, True]:
        for nonlinearity in ['tanh', 'relu']:
            rnn = torch.nn.RNN(4, 6, num_layers=1, nonlinearity=nonlinearity,
                               batch_first=batch_first)
            ours = Sequential2DRNN.from_rnn(rnn)

            x = torch.randn(3, 7, 4) if batch_first else torch.randn(7, 3, 4)
            h_0 = torch.randn(1, 3, 6)

            out_ref, h_ref = rnn(x, h_0)
            out, h_n = ours(x, h_0)

            assert torch.allclose(out, out_ref, atol=1e-6), \
                f'output mismatch for {nonlinearity}, batch_first={batch_first}'
            assert torch.allclose(h_n, h_ref, atol=1e-6), \
                f'h_n mismatch for {nonlinearity}, batch_first={batch_first}'


def test_matches_torch_rnn_with_default_h0():
    """h_0=None must mean zeros, as it does for torch.nn.RNN."""
    torch.manual_seed(1)
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    ours = Sequential2DRNN.from_rnn(rnn)
    x = torch.randn(2, 4, 3)

    out_ref, h_ref = rnn(x)
    out, h_n = ours(x)
    assert torch.allclose(out, out_ref, atol=1e-6)
    assert torch.allclose(h_n, h_ref, atol=1e-6)


def test_sequence_length_is_a_runtime_property():
    """The forced formulation (Sec. 8.1) must not bake the sequence length in."""
    torch.manual_seed(2)
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    ours = Sequential2DRNN.from_rnn(rnn)
    for seq_len in [1, 2, 13]:
        x = torch.randn(2, seq_len, 3)
        out_ref, h_ref = rnn(x)
        out, h_n = ours(x)
        assert out.shape == (2, seq_len, 5)
        assert torch.allclose(out, out_ref, atol=1e-6)
        assert torch.allclose(h_n, h_ref, atol=1e-6)


def test_from_rnn_rejects_unsupported_rnns():
    """Stacking is out of scope by design (Sec. 8.5), not merely unimplemented."""
    for kwargs in [{'num_layers': 2}, {'bidirectional': True}]:
        rnn = torch.nn.RNN(3, 5, **kwargs)
        with pytest.raises(AssertionError):
            Sequential2DRNN.from_rnn(rnn)


# ---------------------------------------------------------------------------
# Internal iterations
# ---------------------------------------------------------------------------

def test_K_equals_one_is_the_rnn_and_K_greater_is_not():
    """K is the dial; K=1 must be the degenerate case and K>1 must differ."""
    torch.manual_seed(3)
    rnn = torch.nn.RNN(4, 6, batch_first=True)
    x = torch.randn(2, 5, 4)

    out_ref, _ = rnn(x)
    out_K1, _ = Sequential2DRNN.from_rnn(rnn, K=1)(x)
    out_K3, _ = Sequential2DRNN.from_rnn(rnn, K=3)(x)

    assert torch.allclose(out_K1, out_ref, atol=1e-6)
    assert not torch.allclose(out_K3, out_ref, atol=1e-3), \
        'K=3 produced the same answer as K=1, so the internal loop is not running'


def test_input_persists_across_internal_iterations():
    """The x-slot must be an exact identity wire across all K steps (Sec. 8.6).

    Violating this is invisible at K=1 -- Inject overwrites the slot before it is
    read again -- so it has to be checked by driving the map directly.
    """
    torch.manual_seed(4)
    model = Sequential2DRNN.from_3x3(4, 2, 8, W_xh=linear(4, 8), W_hh=linear(8, 8),
                                     W_hy=linear(8, 2), batch_first=True)
    x_t = torch.randn(5, 4)
    z = [x_t, None, torch.zeros(5, 8)]
    for _ in range(6):
        z = model.internal_step(z)
        assert torch.equal(z[0], x_t), 'input slot drifted during internal iteration'


def test_input_persistence_check_fires():
    """A non-identity input slot must be caught at construction, not at K=10."""
    bad_maps = [
        # M_xx is a learnable Linear rather than Identity.
        dict(blocks=[[linear(4, 4), linear(4, 8)], [None, linear(8, 8)]],
             bias=[None, torch.zeros(8)],
             activation=[torch.nn.Identity(), torch.nn.Tanh()]),
        # b_x is nonzero, so the input drifts as x + k*b_x.
        dict(blocks=[[Identity(in_features=4, out_features=4), linear(4, 8)],
                     [None, linear(8, 8)]],
             bias=[torch.zeros(4), torch.zeros(8)],
             activation=[torch.nn.Identity(), torch.nn.Tanh()]),
        # A_x squashes the held input.
        dict(blocks=[[Identity(in_features=4, out_features=4), linear(4, 8)],
                     [None, linear(8, 8)]],
             bias=[None, torch.zeros(8)],
             activation=[torch.nn.Tanh(), torch.nn.Tanh()]),
    ]
    for kwargs in bad_maps:
        with pytest.raises(AssertionError):
            Sequential2DRNN(features_list=[4, 8], **kwargs)

    # ...and the check must be escapable, since these are experiments to run
    # deliberately, not mistakes to prevent (Sec. 8.6).
    Sequential2DRNN(features_list=[4, 8], check_input_persistence=False, **bad_maps[0])


# ---------------------------------------------------------------------------
# The general three-slot map
# ---------------------------------------------------------------------------

def test_from_3x3_shapes_and_gradients():
    torch.manual_seed(5)
    model = Sequential2DRNN.from_3x3(input_size=4, output_size=3, hidden_size=8,
                                     W_xh=linear(4, 8), W_hh=linear(8, 8),
                                     W_hy=linear(8, 3), W_yh=linear(3, 8),
                                     K=2, batch_first=True)
    x = torch.randn(5, 6, 4)
    out, h_n = model(x)
    assert out.shape == (5, 6, 3), f'output slot has the wrong width: {out.shape}'
    assert h_n.shape == (1, 5, 8)

    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'no gradient reached {name}'
        assert torch.isfinite(p.grad).all(), f'non-finite gradient in {name}'


def test_from_3x3_rejects_blocks_carrying_bias():
    """A block with its own bias double-counts against the slot bias (Sec. 8.2)."""
    with pytest.raises(AssertionError):
        Sequential2DRNN.from_3x3(4, 3, 8, W_xh=torch.nn.Linear(4, 8, bias=True),
                                 W_hh=linear(8, 8))


def test_y_slot_lags_h_by_one_internal_step():
    """Sequential2D is Jacobi, so a readout slot reads the *old* hidden (Sec. 8.3).

    With W_hy = I the y-slot is a pure copy of h, which makes the lag directly
    observable: after one internal step, y holds the h from before the step.
    """
    torch.manual_seed(6)
    model = Sequential2DRNN.from_3x3(
        4, 8, 8, W_xh=linear(4, 8), W_hh=linear(8, 8),
        W_hy=Identity(in_features=8, out_features=8),
        A_y=torch.nn.Identity(), bias=False, batch_first=True)

    h_before = torch.randn(5, 8)
    z = [torch.randn(5, 4), None, h_before]
    z = model.internal_step(z)
    assert torch.equal(z[1], h_before), 'y-slot is not one internal step behind h'


# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------

def test_structured_block_types_drop_in():
    """Any module with in_features/out_features is a legal block.

    This only checks that shapes and gradients survive -- not equivalence to
    anything, since a Monarch W_hh is a genuinely different model.
    """
    torch.manual_seed(7)
    for W_hh in [MonarchLinear.from_uniform_blocks(8, 8, num_blocks=2, bias=False, seed=0),
                 MaskedLinear(8, 8, bias=False)]:
        model = Sequential2DRNN.from_3x3(4, 3, 8, W_xh=linear(4, 8), W_hh=W_hh,
                                         W_hy=linear(8, 3), batch_first=True)
        out, _ = model(torch.randn(5, 6, 4))
        assert out.shape == (5, 6, 3), f'{type(W_hh).__name__} changed the output shape'
        out.sum().backward()


def test_nested_block_does_not_receive_the_outer_activation():
    """A nested block sits *inside* a slot's sum, so the outer A never sees it.

    Easy to get wrong in the other direction and silently end up with a linear
    model where a nonlinear one was intended, so it is pinned down here.
    """
    torch.manual_seed(8)
    inner = Sequential2D([8], [8], [[linear(8, 8)]])   # linear, no activation

    model = Sequential2DRNN.from_3x3(4, 3, 8, W_xh=linear(4, 8), W_hh=inner,
                                     W_hy=linear(8, 3), bias=False,
                                     A_h=torch.nn.Identity(), batch_first=True)

    # With every activation the identity and no bias, the whole system is linear,
    # which is only true if the outer A did not sneak a nonlinearity into `inner`.
    x = torch.randn(5, 6, 4)
    out_1, _ = model(x)
    out_2, _ = model(2.0 * x)
    assert torch.allclose(out_2, 2.0 * out_1, atol=1e-5), \
        'the system is not linear, so an activation is being applied somewhere'


def test_nested_block_may_carry_its_own_activation():
    """The documented way to put a nonlinearity inside a block: Sequential1D."""
    torch.manual_seed(9)
    inner = Sequential1D(torch.nn.Sequential(linear(8, 8), torch.nn.Tanh()),
                         in_features=8, out_features=8)
    model = Sequential2DRNN.from_3x3(4, 3, 8, W_xh=linear(4, 8), W_hh=inner,
                                     W_hy=linear(8, 3), batch_first=True)
    out, _ = model(torch.randn(5, 6, 4))
    assert out.shape == (5, 6, 3)


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

def test_dtype_is_preserved():
    """float64 in, float64 out.

    Worth pinning: Sequential2D.forward_vector builds its accumulator with
    torch.zeros(..., device=...) and no dtype, so it silently promotes.  This
    module uses forward_list, which never allocates an accumulator, and this
    test is what keeps it that way.
    """
    torch.manual_seed(10)
    rnn = torch.nn.RNN(4, 6, batch_first=True).double()
    ours = Sequential2DRNN.from_rnn(rnn)
    out, h_n = ours(torch.randn(2, 5, 4, dtype=torch.float64))
    assert out.dtype == torch.float64, f'dtype was promoted to {out.dtype}'
    assert h_n.dtype == torch.float64


# ---------------------------------------------------------------------------
# Block validation in from_3x3
# ---------------------------------------------------------------------------
# The source-first/target-first naming was got wrong once during design, so the
# factory checks its blocks rather than trusting the caller.

def test_blocks_are_keyword_only():
    """Positional blocks would let a reversed name land in the wrong cell."""
    with pytest.raises(TypeError):
        Sequential2DRNN.from_3x3(4, 3, 8, linear(4, 8), linear(8, 8))


def test_transposed_block_is_rejected():
    """W_xh maps x -> h, so Linear(hidden, input) is the transpose of what is wanted."""
    with pytest.raises(AssertionError, match='source-first'):
        Sequential2DRNN.from_3x3(4, 3, 8, W_xh=linear(8, 4), W_hh=linear(8, 8))


def test_transposed_weight_tensor_is_rejected():
    """Declared sizes right, weight actually transposed.

    Only detectable on a non-square block: PyTorch stores weights as
    (out_features, in_features), so overwriting `weight` with its own transpose
    changes the shape and is caught.
    """
    block = linear(4, 8)
    block.weight = torch.nn.Parameter(block.weight.data.T.clone())   # now (4, 8)
    with pytest.raises(AssertionError, match='transpose'):
        Sequential2DRNN.from_3x3(4, 3, 8, W_xh=block, W_hh=linear(8, 8))


def test_block_types_without_a_weight_attribute_are_accepted():
    """MonarchLinear keeps factors and Sequential2D keeps sub-blocks, not `.weight`.

    The shape check must skip those rather than rejecting them.
    """
    for W_hh in [MonarchLinear.from_uniform_blocks(8, 8, num_blocks=2, bias=False, seed=0),
                 Sequential2D([8], [8], [[linear(8, 8)]])]:
        Sequential2DRNN.from_3x3(4, 3, 8, W_xh=linear(4, 8), W_hh=W_hh,
                                 W_hy=linear(8, 3))


# ---------------------------------------------------------------------------
# The observation map
# ---------------------------------------------------------------------------
# y_t = g(z_t) is the second half of the state-space form and is deliberately not
# part of the dynamics.  The point of having it is that a readout carried in a
# *slot* is one internal iteration behind, whereas g is not.

def test_observation_map_has_no_lag():
    """readout(h_t) via `g` must equal applying the readout outside the model."""
    torch.manual_seed(11)
    rnn = torch.nn.RNN(4, 6, batch_first=True)
    readout = torch.nn.Linear(6, 3)                  # bias is fine: g is not a block

    hidden_only = Sequential2DRNN.from_rnn(rnn)
    with_readout = Sequential2DRNN.from_rnn(rnn, readout=readout)

    x = torch.randn(2, 5, 4)
    h_sequence, _ = hidden_only(x)                   # (batch, seq, hidden)
    y_sequence, _ = with_readout(x)                  # (batch, seq, 3)

    assert torch.allclose(y_sequence, readout(h_sequence), atol=1e-6), \
        'the observation map is not reading the current hidden state'


def test_slot_readout_does_lag_and_observation_does_not():
    """The two mechanisms must actually differ, or there is no reason for both.

    With W_hy = I the y-slot is a pure copy of h, so the slot route returns h
    shifted by one token while the observation route returns h itself.
    """
    torch.manual_seed(12)
    common = dict(W_xh=linear(4, 8), W_hh=linear(8, 8), bias=False,
                  A_y=torch.nn.Identity(), batch_first=True)
    identity_8 = Identity(in_features=8, out_features=8)

    via_slot = Sequential2DRNN.from_3x3(4, 8, 8, W_hy=identity_8, **common)
    via_g = Sequential2DRNN.from_3x3(4, 8, 8, readout=torch.nn.Identity(), **common)

    x = torch.randn(1, 5, 4)
    slot_output, _ = via_slot(x)
    g_output, _ = via_g(x)

    # The slot route is the observation route delayed by exactly one token.
    assert torch.allclose(slot_output[:, 1:, :], g_output[:, :-1, :], atol=1e-6), \
        'the y-slot is not exactly one token behind the observation map'
    assert not torch.allclose(slot_output, g_output, atol=1e-3), \
        'the two readout mechanisms agree, so the lag is not being exercised'


def test_observation_parameters_are_registered_and_trained():
    torch.manual_seed(13)
    rnn = torch.nn.RNN(4, 6, batch_first=True)
    readout = torch.nn.Linear(6, 3)
    model = Sequential2DRNN.from_rnn(rnn, readout=readout)

    names = dict(model.named_parameters())
    assert any('observation' in n for n in names), \
        f'the readout is not a registered submodule: {sorted(names)}'

    out, _ = model(torch.randn(2, 5, 4))
    out.sum().backward()
    assert readout.weight.grad is not None and readout.bias.grad is not None


def test_from_rnn_without_readout_is_still_bit_comparable():
    """Adding the observation hook must not disturb the equivalence."""
    torch.manual_seed(14)
    rnn = torch.nn.RNN(4, 6, batch_first=True)
    x, h_0 = torch.randn(3, 7, 4), torch.randn(1, 3, 6)
    out_ref, h_ref = rnn(x, h_0)
    out, h_n = Sequential2DRNN.from_rnn(rnn)(x, h_0)
    assert torch.allclose(out, out_ref, atol=1e-6)
    assert torch.allclose(h_n, h_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# PackedSequence
# ---------------------------------------------------------------------------
# Packing changes only how many rows of the batch are alive at each step.  The
# dynamics are untouched, so the tests below are all "does it agree with the
# thing it should agree with" rather than anything about M, b, A or Inject.

from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence)


@pytest.mark.parametrize('enforce_sorted', [True, False])
@pytest.mark.parametrize('h_0_given', [True, False])
def test_packed_matches_torch_rnn(enforce_sorted, h_0_given):
    """The reference implementation, including the sorted/unsorted permutations."""
    torch.manual_seed(20)
    lengths = torch.tensor([7, 7, 4, 1] if enforce_sorted else [4, 7, 1, 7])
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    ours = Sequential2DRNN.from_rnn(rnn)

    x = torch.randn(4, 7, 3)
    h_0 = torch.randn(1, 4, 5) if h_0_given else None
    packed = pack_padded_sequence(x, lengths, batch_first=True,
                                  enforce_sorted=enforce_sorted)

    out_ref, h_ref = rnn(packed, h_0)
    out, h_n = ours(packed, h_0)

    assert isinstance(out, PackedSequence), f'expected a PackedSequence, got {type(out)}'
    assert torch.equal(out.batch_sizes, out_ref.batch_sizes)
    assert torch.allclose(out.data, out_ref.data, atol=1e-6), 'packed data mismatch'
    assert torch.allclose(h_n, h_ref, atol=1e-6), 'h_n mismatch'

    # And once padded back out, which is how it is normally consumed.
    padded_ref, _ = pad_packed_sequence(out_ref, batch_first=True)
    padded, _ = pad_packed_sequence(out, batch_first=True)
    assert torch.allclose(padded, padded_ref, atol=1e-6)


def test_packed_h_n_is_each_sequence_own_last_step():
    """The whole point of packing: a short sequence must not keep being stepped.

    Checked against running each sequence on its own, which is the definition.
    """
    torch.manual_seed(21)
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    model = Sequential2DRNN.from_rnn(rnn)

    lengths = [6, 3, 1]
    x = torch.randn(3, 6, 3)
    h_0 = torch.randn(1, 3, 5)

    packed = pack_padded_sequence(x, torch.tensor(lengths), batch_first=True)
    _, h_packed = model(packed, h_0)

    for i, length in enumerate(lengths):
        _, h_alone = model(x[i:i + 1, :length, :], h_0[:, i:i + 1, :])
        assert torch.allclose(h_packed[:, i, :], h_alone[:, 0, :], atol=1e-6), \
            f'sequence {i} (length {length}) disagrees with running it alone'


def test_packed_three_slot_state_truncates_every_slot():
    """The truncation must carry the y-slot along, not just the hidden one.

    from_3x3 has a live y-slot, so this is the case where dropping a finished
    sequence has to shrink more than one tensor.  Compared against per-sequence
    runs, which cannot get the bookkeeping wrong.
    """
    torch.manual_seed(22)
    model = Sequential2DRNN.from_3x3(
        3, 2, 5, W_xh=linear(3, 5), W_hh=linear(5, 5),
        W_hy=linear(5, 2), W_yh=linear(2, 5),        # y feeds back, so y matters
        K=2, batch_first=True)

    lengths = [6, 4, 2]
    x = torch.randn(3, 6, 3)
    packed = pack_padded_sequence(x, torch.tensor(lengths), batch_first=True)
    out_packed, h_packed = model(packed)
    padded, _ = pad_packed_sequence(out_packed, batch_first=True)

    for i, length in enumerate(lengths):
        out_alone, h_alone = model(x[i:i + 1, :length, :])
        assert torch.allclose(padded[i, :length, :], out_alone[0], atol=1e-6), \
            f'sequence {i} output disagrees with running it alone'
        assert torch.allclose(h_packed[:, i, :], h_alone[:, 0, :], atol=1e-6), \
            f'sequence {i} h_n disagrees with running it alone'


def test_packed_with_observation_map_and_gradients():
    torch.manual_seed(23)
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    readout = torch.nn.Linear(5, 4)
    model = Sequential2DRNN.from_rnn(rnn, K=2, readout=readout)

    x = torch.randn(3, 6, 3)
    packed = pack_padded_sequence(x, torch.tensor([6, 4, 2]), batch_first=True)
    out, h_n = model(packed)

    assert out.data.shape == (6 + 4 + 2, 4), f'unexpected packed shape {out.data.shape}'
    out.data.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'no gradient reached {name}'


def test_packed_all_equal_lengths_matches_dense_path():
    """With no padding at all, the two code paths must give the same answer."""
    torch.manual_seed(24)
    rnn = torch.nn.RNN(3, 5, batch_first=True)
    model = Sequential2DRNN.from_rnn(rnn)

    x = torch.randn(4, 6, 3)
    dense_out, dense_h = model(x)
    packed = pack_padded_sequence(x, torch.tensor([6, 6, 6, 6]), batch_first=True)
    padded, _ = pad_packed_sequence(model(packed)[0], batch_first=True)

    assert torch.allclose(padded, dense_out, atol=1e-6)
    assert torch.allclose(model(packed)[1], dense_h, atol=1e-6)


def test_unwritten_output_slot_gives_a_useful_error():
    """An output slot with no incoming block stays None; say so clearly."""
    model = Sequential2DRNN.from_3x3(3, 2, 5, W_xh=linear(3, 5), W_hh=linear(5, 5))
    with pytest.raises(AssertionError, match='output slot'):
        model(torch.randn(2, 4, 3))
