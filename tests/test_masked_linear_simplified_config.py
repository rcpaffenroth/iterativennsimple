"""Tests for masked_linear_simplified_config: the YAML loader and the two-format converter.

The loader accepts one format, the edge list.  The grid is a view of the same
information, produced for a human to read, and the conversion has to be exact in
both directions or the view is not a check on anything -- so the round trips are
asserted here, in both directions, including through the text that gets printed.
"""
from pathlib import Path

import pytest
import torch
import yaml

from iterativennsimple.MaskedLinear_simplified import (
    block_matrix, frozen, trainable, trainable_where_nonzero,
)
from iterativennsimple.masked_linear_simplified_config import (
    from_edges, to_grid, to_edges, dump_grid, render_grid, cell_label,
)

# The same 9x16 map written both ways: three input slots, two output slots.
#
#                x_in(4)    h1_in(6)   h2_in(6)
#    h_out(6) [  1.0*I      0.5        0.25*I  ]
#    y_out(3) [  0          0          0.1     ]

EDGES_YAML = """
in_slots:  {x_in: 4, h1_in: 6, h2_in: 6}
out_slots: {h_out: 6, y_out: 3}
blocks:
  - {from: x_in,  to: h_out, values: {const: 1.0},  pattern: diagonal, trainable: none}
  - {from: h1_in, to: h_out, values: {const: 0.5},  trainable: all}
  - {from: h2_in, to: h_out, values: {const: 0.25}, pattern: diagonal, trainable: nonzero}
  - {from: h2_in, to: y_out, values: {const: 0.1},  trainable: none}
"""

GRID_YAML = """
out_slots: {h_out: 6, y_out: 3}
in_slots:  {x_in: 4, h1_in: 6, h2_in: 6}
blocks:
  - [{values: {const: 1.0}, pattern: diagonal, trainable: none},
     {values: {const: 0.5}, trainable: all},
     {values: {const: 0.25}, pattern: diagonal, trainable: nonzero}]
  - [null, null, {values: {const: 0.1}, trainable: none}]
"""

EDGES = yaml.safe_load(EDGES_YAML)
GRID = yaml.safe_load(GRID_YAML)


def canonical(edges_cfg) -> dict:
    """The same edge-list config with its blocks in a fixed order.

    The edge list is a *set* of blocks: nothing downstream depends on the order
    they are written in, and a round trip through the grid returns them in the
    grid's row-major order rather than the author's.  So the round trip is exact
    up to this reordering, and the tests say so rather than relying on an example
    that happens to be written row-major already.
    """
    return dict(edges_cfg, blocks=sorted(edges_cfg['blocks'], key=lambda e: (e['to'], e['from'])))


def test_the_config_builds_the_matrix_it_describes():
    """Against the block matrix written directly in the layer's own vocabulary."""
    expected = block_matrix([
        [frozen(torch.eye(6, 4)),
         trainable(torch.full((6, 6), 0.5)),
         trainable_where_nonzero(0.25 * torch.eye(6))],
        [frozen(torch.zeros(3, 4)),
         frozen(torch.zeros(3, 6)),
         frozen(torch.full((3, 6), 0.1))],
    ])
    m = from_edges(EDGES)
    assert torch.equal(m.weight_0, expected.values)
    assert torch.equal(m.mask, expected.mask)
    assert tuple(m.weight_0.shape) == (9, 16)
    assert m.number_of_trainable_parameters() == 6 * 6 + 6      # dense block plus one diagonal


def test_both_formats_describe_the_same_layer():
    """Only the edge list is loaded, so a grid is loaded by converting it first."""
    from_grid = from_edges(to_edges(GRID))
    reference = from_edges(EDGES)
    assert torch.equal(from_grid.weight_0, reference.weight_0)
    assert torch.equal(from_grid.mask, reference.mask)


def test_round_trip_edges_to_grid_to_edges():
    assert canonical(to_edges(to_grid(EDGES))) == canonical(EDGES)


def test_round_trip_grid_to_edges_to_grid():
    """Exact, with no reordering to allow for: a grid cell's position is its identity."""
    assert to_grid(to_edges(GRID)) == GRID


def test_the_printed_grid_parses_back():
    """dump_grid writes one line per block row rather than letting yaml spread a
    row over many.  What it prints still has to be the same config."""
    assert yaml.safe_load(dump_grid(to_grid(EDGES))) == to_grid(EDGES)


def test_a_zero_block_is_not_a_zero_valued_block():
    """`null` is frozen; a cell whose values are zero is not, and the converters
    must not confuse them -- the two have different masks."""
    explicit_zero = dict(EDGES, blocks=EDGES['blocks'] + [
        {'from': 'x_in', 'to': 'y_out', 'values': {'const': 0.0}, 'trainable': 'all'}])
    m = from_edges(explicit_zero)
    assert torch.equal(m.weight_0[6:, :4], torch.zeros(3, 4))
    assert torch.equal(m.mask[6:, :4], torch.ones(3, 4))        # trainable, unlike a null cell
    assert canonical(to_edges(to_grid(explicit_zero))) == canonical(explicit_zero)


def test_a_repeated_block_is_an_error():
    """Two edges with the same endpoints would silently override one another."""
    repeated = dict(EDGES, blocks=EDGES['blocks'] + [
        {'from': 'h1_in', 'to': 'h_out', 'values': 'ones', 'trainable': 'all'}])
    with pytest.raises(AssertionError, match='two blocks given for h1_in -> h_out'):
        from_edges(repeated)


def test_a_misspelled_slot_is_an_error():
    misspelled = dict(EDGES, blocks=[dict(EDGES['blocks'][0], **{'from': 'x_inn'})])
    with pytest.raises(AssertionError, match='x_inn -> h_out names a slot'):
        from_edges(misspelled)


def test_a_grid_of_the_wrong_shape_is_an_error():
    """The failure the grid format has and the edge list does not."""
    with pytest.raises(AssertionError, match='3 grid rows for 2 output slots'):
        to_edges(dict(GRID, blocks=GRID['blocks'] + [[None, None, None]]))
    with pytest.raises(AssertionError, match='grid row 1 has 2 cells for 3 input slots'):
        to_edges(dict(GRID, blocks=[GRID['blocks'][0], [None, None]]))


def test_the_proofreading_table_shows_the_shape_of_the_matrix():
    """render_grid is the view that makes a missing block visible."""
    table = render_grid(to_grid(EDGES)).splitlines()
    assert table[0].split() == ['x_in(4)', 'h1_in(6)', 'h2_in(6)']
    assert table[1].split() == ['h_out(6)', '1*I/none', '0.5/all', '0.25*I/nz']
    assert table[2].split() == ['y_out(3)', '.', '.', '0.1/none']


def test_cell_labels_for_the_random_specs():
    assert cell_label({'values': {'gaussian': {'mu': 0.2, 'sigma': 0.7}},
                       'pattern': {'bernoulli': 0.5}, 'trainable': 'nonzero'}) == 'G(0.2,0.7)*B(0.5)/nz'
    assert cell_label({'values': 'kaiming', 'pattern': {'per_row': 3}}) == 'K*R(3)/all'
    assert cell_label(None) == '.'


EXAMPLES = sorted((Path(__file__).parent.parent /
                   'examples' / 'masked_linear_simplified_configs').glob('*.yaml'))


def test_there_are_examples_to_check():
    """A glob that silently matches nothing would make the test below vacuous."""
    assert len(EXAMPLES) >= 6


@pytest.mark.parametrize('path', EXAMPLES, ids=lambda p: p.stem)
def test_every_example_config_builds_what_it_says(path):
    """Every file in examples/ is built here, so a broken example fails the suite
    rather than waiting to be found by whoever copies it.  The shape is checked
    against the slot widths, which is the one thing the file states twice."""
    cfg = yaml.safe_load(path.read_text())
    m = from_edges(cfg)
    assert tuple(m.weight_0.shape) == (sum(cfg['out_slots'].values()),
                                       sum(cfg['in_slots'].values()))
    assert canonical(to_edges(to_grid(cfg))) == canonical(cfg)
    assert yaml.safe_load(dump_grid(to_grid(cfg))) == to_grid(cfg)


def test_the_documented_example_is_the_documented_matrix():
    """recurrent_map.yaml is quoted in the module docstring and in the README, so
    its numbers are pinned rather than merely self-consistent."""
    cfg = yaml.safe_load((Path(__file__).parent.parent / 'examples' /
                          'masked_linear_simplified_configs' / 'recurrent_map.yaml').read_text())
    m = from_edges(cfg)
    assert tuple(m.weight_0.shape) == (9, 16)
    assert int(torch.count_nonzero(m.mask)) == 42


def test_a_random_block_has_the_density_it_asks_for():
    """The one thing a config can say that a hand-written matrix cannot check itself."""
    torch.manual_seed(0)
    cfg = {'in_slots': {'a': 100}, 'out_slots': {'b': 100},
           'blocks': [{'from': 'a', 'to': 'b', 'values': 'kaiming',
                       'pattern': {'bernoulli': 0.5}, 'trainable': 'nonzero'}]}
    m = from_edges(cfg)
    density = int(torch.count_nonzero(m.mask)) / 10000
    assert abs(density - 0.5) < 0.025      # 5 standard deviations at N = 10000
