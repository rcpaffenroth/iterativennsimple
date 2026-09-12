"""Build a MaskedLinear from a YAML configuration, and convert between the two ways of writing one.

`MaskedLinear_simplified.py` knows nothing about configuration files -- that is
deliberate, and the reasoning is in `tasks/STYLE_EXPERIMENT_MaskedLinear_simplified.md`.
This module is the one place that does.  It imports the layer; the layer does not
import it.

THE FORMAT THE LOADER ACCEPTS (the "edge list").  Slots are named, and only the
non-zero blocks are listed.  Everything unlisted is zero and frozen.

    in_slots:  {x_in: 4, h1_in: 6, h2_in: 6}
    out_slots: {h_out: 6, y_out: 3}
    blocks:
      - {from: x_in,  to: h_out, values: {const: 1.0},  pattern: diagonal, trainable: none}
      - {from: h1_in, to: h_out, values: {const: 0.5},  trainable: all}
      - {from: h2_in, to: h_out, values: {const: 0.25}, pattern: diagonal, trainable: nonzero}
      - {from: h2_in, to: y_out, values: {const: 0.1},  trainable: none}

THE FORMAT A HUMAN READS (the "grid").  The same thing as a block matrix, one
grid row per output slot and one column per input slot, in the order the slot
dicts are written.  `null` is a zero block.

    out_slots: {h_out: 6, y_out: 3}
    in_slots:  {x_in: 4, h1_in: 6, h2_in: 6}
    blocks:
      - [{values: {const: 1.0}, pattern: diagonal, trainable: none},
         {values: {const: 0.5}, trainable: all},
         {values: {const: 0.25}, pattern: diagonal, trainable: nonzero}]
      - [null, null, {values: {const: 0.1}, trainable: none}]

Only the edge list is loaded, because only one of the two should be able to
produce a model: `from_edges(to_edges(grid_config))` loads a grid if you have
one.  The grid exists to be *looked at* -- `to_grid` puts the matrix back in
block-matrix shape, where a missing or misplaced block is visible, and the
command line below prints it next to an aligned table and the resulting shape:

    uv run python -m iterativennsimple.masked_linear_simplified_config <config.yaml>

Both formats carry the same information, so the conversion round-trips
(`tests/test_masked_linear_config.py` asserts both directions).  Grid to edges to
grid is exact.  Edges to grid to edges is exact *up to the order of the blocks
list*, because the edge list is a set of blocks and the grid hands them back in
row-major order -- nothing downstream depends on the order.  What does not
survive at all is YAML comments, which `yaml.safe_dump` drops; that is why the
command line prints to stdout and nothing here writes a config file.
"""
import sys

import torch
import yaml

from iterativennsimple.MaskedLinear_simplified import (
    MaskedLinear, Block, block_matrix, frozen, trainable, trainable_where_nonzero,
    bernoulli, per_row, scattered, kaiming,
)


# ---------------------------------------------------------------- the vocabulary
# What a configuration file can say.  The menu is deliberately closed and small: a
# structure that is not on it -- torch.tril, a Kronecker pattern, a matrix loaded
# from disk -- is written in Python against MaskedLinear_simplified directly,
# rather than by growing the vocabulary here.  Extending the *language* is what
# turned the original MaskedLinear into an interpreter.

def values_from(spec, out_features: int, in_features: int) -> torch.Tensor:
    """What the entries of a block are, before any sparsity pattern is applied.

    spec is 'identity', 'ones', 'kaiming', or a one-key dict:
    {const: 0.5}, {gaussian: {mu: 0.2, sigma: 0.7}}, {uniform: {lo: -1, hi: 1}}.
    """
    if spec == 'identity':
        return torch.eye(out_features, in_features)
    if spec == 'ones':
        return torch.ones(out_features, in_features)
    if spec == 'kaiming':
        return kaiming(out_features, in_features)
    if 'const' in spec:
        return torch.full((out_features, in_features), float(spec['const']))
    if 'gaussian' in spec:
        g = spec['gaussian']
        return torch.randn(out_features, in_features) * g.get('sigma', 1.0) + g.get('mu', 0.0)
    if 'uniform' in spec:
        u = spec['uniform']
        lo, hi = u.get('lo', -1.0), u.get('hi', 1.0)
        return torch.rand(out_features, in_features) * (hi - lo) + lo
    assert False, f'unknown values spec {spec}'


def pattern_from(spec, out_features: int, in_features: int) -> torch.Tensor:
    """Which entries of a block exist: a 0/1 matrix multiplying the values.

    spec is None (dense), 'diagonal', or a one-key dict:
    {bernoulli: 0.5}, {per_row: 3}, {scattered: 15}.

    'diagonal' on a rectangular block is torch.eye(out, in) -- ones where i == j
    and nothing else -- which is the injection idiom, not an error.
    """
    if spec is None or spec == 'dense':
        return torch.ones(out_features, in_features)
    if spec == 'diagonal':
        return torch.eye(out_features, in_features)
    if 'bernoulli' in spec:
        return bernoulli(out_features, in_features, p=spec['bernoulli'])
    if 'per_row' in spec:
        return per_row(out_features, in_features, n=spec['per_row'])
    if 'scattered' in spec:
        return scattered(out_features, in_features, n=spec['scattered'])
    assert False, f'unknown pattern spec {spec}'


def block_from_cell(cell, out_features: int, in_features: int) -> Block:
    """One cell -- {values, pattern, trainable} -- as a Block.  None is the zero block.

    Note a cell whose values happen to be zero is *not* the same as `null`: a
    `{values: {const: 0.0}, trainable: all}` cell is trainable everywhere and can
    move off zero, while `null` is frozen.  The converters below never confuse
    the two.
    """
    if cell is None:
        return frozen(torch.zeros(out_features, in_features))

    W = values_from(cell['values'], out_features, in_features) \
        * pattern_from(cell.get('pattern'), out_features, in_features)

    how = cell.get('trainable', 'all')
    if how == 'all':
        return trainable(W)
    if how == 'none':
        return frozen(W)
    if how == 'nonzero':
        return trainable_where_nonzero(W)
    assert False, f'unknown trainable {how}'


# ---------------------------------------------------------------- loading

def grid_of_cells(cfg) -> dict:
    """The edge list as a dict keyed (out_slot, in_slot), with None for zero blocks.

    The duplicate check is not decoration: two edges with the same endpoints would
    silently override one another, producing a model that differs from the file in
    a way nothing downstream can see.
    """
    grid = {(to_slot, from_slot): None
            for to_slot in cfg['out_slots'] for from_slot in cfg['in_slots']}
    for edge in cfg['blocks']:
        key = (edge['to'], edge['from'])
        assert key in grid, f"block {edge['from']} -> {edge['to']} names a slot that does not exist"
        assert grid[key] is None, f"two blocks given for {edge['from']} -> {edge['to']}"
        grid[key] = edge
    return grid


def from_edges(cfg, bias: bool = False) -> MaskedLinear:
    """Build the layer an edge-list config describes."""
    grid = grid_of_cells(cfg)
    blocks = [[block_from_cell(grid[(to_slot, from_slot)], out_features, in_features)
               for from_slot, in_features in cfg['in_slots'].items()]
              for to_slot, out_features in cfg['out_slots'].items()]
    return MaskedLinear(block_matrix(blocks), bias=bias)


# ---------------------------------------------------------------- converting between the two

CELL_KEYS = ['values', 'pattern', 'trainable']


def to_grid(cfg) -> dict:
    """Edge list -> grid.  The block matrix, in block-matrix shape, to be read."""
    grid = grid_of_cells(cfg)
    return {
        'out_slots': dict(cfg['out_slots']),
        'in_slots': dict(cfg['in_slots']),
        'blocks': [[None if grid[(to_slot, from_slot)] is None else
                    {k: v for k, v in grid[(to_slot, from_slot)].items() if k in CELL_KEYS}
                    for from_slot in cfg['in_slots']]
                   for to_slot in cfg['out_slots']],
    }


def to_edges(cfg) -> dict:
    """Grid -> edge list.  Zero blocks (`null`) drop out; everything else is named."""
    out_slots, in_slots = list(cfg['out_slots']), list(cfg['in_slots'])
    assert len(cfg['blocks']) == len(out_slots), \
        f"{len(cfg['blocks'])} grid rows for {len(out_slots)} output slots"
    for i, row in enumerate(cfg['blocks']):
        assert len(row) == len(in_slots), \
            f"grid row {i} has {len(row)} cells for {len(in_slots)} input slots"

    return {
        'in_slots': dict(cfg['in_slots']),
        'out_slots': dict(cfg['out_slots']),
        'blocks': [{'from': in_slots[j], 'to': out_slots[i], **cell}
                   for i, row in enumerate(cfg['blocks'])
                   for j, cell in enumerate(row) if cell is not None],
    }


# ---------------------------------------------------------------- the command line

def cell_label(cell) -> str:
    """One cell as `values*pattern/trainable`, short enough to line up in a table."""
    if cell is None:
        return '.'

    values = cell['values']
    if isinstance(values, str):
        text = {'identity': 'I', 'ones': '1', 'kaiming': 'K'}[values]
    elif 'const' in values:
        text = f"{values['const']:g}"
    elif 'gaussian' in values:
        g = values['gaussian']
        text = f"G({g.get('mu', 0.0):g},{g.get('sigma', 1.0):g})"
    else:
        u = values['uniform']
        text = f"U({u.get('lo', -1.0):g},{u.get('hi', 1.0):g})"

    pattern = cell.get('pattern')
    if pattern == 'diagonal':
        text += '*I'
    elif isinstance(pattern, dict):
        name, argument = next(iter(pattern.items()))
        text += f"*{ {'bernoulli': 'B', 'per_row': 'R', 'scattered': 'S'}[name] }({argument:g})"

    return text + '/' + {'all': 'all', 'none': 'none', 'nonzero': 'nz'}[cell.get('trainable', 'all')]


def render_grid(grid_cfg) -> str:
    """The grid as an aligned table.  This is the proofreading view: a block in the
    wrong place, or one you meant to write and did not, shows up as a shape."""
    out_labels = [f'{name}({size})' for name, size in grid_cfg['out_slots'].items()]
    in_labels = [f'{name}({size})' for name, size in grid_cfg['in_slots'].items()]

    header = [''] + in_labels
    rows = [[out_labels[i]] + [cell_label(cell) for cell in row]
            for i, row in enumerate(grid_cfg['blocks'])]

    widths = [max(len(row[j]) for row in [header] + rows) for j in range(len(header))]
    lines = ['  '.join(text.ljust(w) for text, w in zip(row, widths)).rstrip()
             for row in [header] + rows]
    return '\n'.join(lines)


def dump_grid(grid_cfg) -> str:
    """The grid as YAML, one line per block row.

    yaml.safe_dump would spread a row over many lines and lose the one thing the
    grid is for -- that a row of the block matrix looks like a row.  Each piece is
    still dumped by yaml, so what comes out parses back to what went in.
    """
    def flow(x):
        return yaml.safe_dump(x, default_flow_style=True, width=10 ** 6, sort_keys=False).strip()

    lines = [f"out_slots: {flow(grid_cfg['out_slots'])}",
             f"in_slots:  {flow(grid_cfg['in_slots'])}",
             'blocks:']
    lines += ['  - ' + flow(row) for row in grid_cfg['blocks']]
    return '\n'.join(lines)


if __name__ == '__main__':
    cfg = yaml.safe_load(open(sys.argv[1]))

    # Which format is this?  The edge list's blocks are dicts; the grid's are lists.
    given_as_edges = isinstance(cfg['blocks'][0], dict)
    edges = cfg if given_as_edges else to_edges(cfg)
    grid = to_grid(edges)

    print(f"read as: {'edge list' if given_as_edges else 'grid'}\n")
    print(render_grid(grid))
    print('\ncells are  values*pattern/trainable ;  "." is a zero block ;'
          '  I identity, K kaiming, B bernoulli, R per_row, S scattered\n')

    print('--- the other format ' + '-' * 50)
    print(dump_grid(grid) if given_as_edges else
          yaml.safe_dump(edges, sort_keys=False, default_flow_style=None, width=100))

    m = from_edges(edges)
    print(f'--- builds: {tuple(m.weight_0.shape)} weight, '
          f'{int(torch.count_nonzero(m.mask))} trainable entries')
