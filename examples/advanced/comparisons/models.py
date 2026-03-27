"""
Unified model zoo for academic comparison experiments.

Every model exposes the same interface:
    model = build_model(name, input_dim, hidden_dim, label_dim, seq_len, **kwargs)
    logits = model(x_seq)          # x_seq: (B, T, input_dim)  -> logits: (B, label_dim)

Supported architectures:

  INN family (iterated sparse maps over sequences):
    - monarch_inn          Monarch sparse INN (structured sparsity via permuted block-diagonal)
    - masked_inn           MaskedLinear INN (same sparsity pattern as Monarch, unstructured)
    - ls_inn               L+S INN (Robust PCA: low-rank L + sparse Monarch S)

  Baselines:
    - lstm                 Standard LSTM
    - gru                  Standard GRU
    - rnn_tanh             Vanilla RNN (tanh)
    - transformer          Transformer encoder + mean-pool + head
    - mlp_flat             Flat MLP (flattens sequence, no recurrence)
"""

import math
from typing import Callable

import torch
import torch.nn as nn

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    """Count effective trainable parameters.

    Uses number_of_trainable_parameters() when available (e.g. MaskedLinear,
    MonarchLinear, LSLinear, Sequential2D) to get the correct count — this
    matters for MaskedLinear which stores a full dense U matrix but only
    trains through the mask, so p.numel() overcounts.
    """
    if hasattr(model, "number_of_trainable_parameters"):
        n = model.number_of_trainable_parameters()
        return int(n) if hasattr(n, 'item') else int(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _find_num_blocks(dim, desired):
    """Largest divisor of dim that is <= desired."""
    nb = desired
    while nb > 1 and dim % nb != 0:
        nb -= 1
    return nb


def _find_common_num_blocks(in_f, out_f, desired):
    """Largest k <= desired that divides BOTH in_f and out_f."""
    nb = desired
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    return nb


def _monarch_block(in_f, out_f, num_blocks, activation=None):
    nb = _find_common_num_blocks(in_f, out_f, num_blocks)
    layer = MonarchLinear.from_uniform_blocks(in_f, out_f, num_blocks=nb, bias=True)
    modules = [activation, layer] if activation else [layer]
    return Sequential1D(*modules, in_features=in_f, out_features=out_f)


def _masked_block(in_f, out_f, num_blocks, activation=None):
    """MaskedLinear block with same sparsity pattern as Monarch (via to_MaskedLinear)."""
    nb = _find_common_num_blocks(in_f, out_f, num_blocks)
    monarch = MonarchLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=nb, bias=True, seed=42,
    )
    masked = monarch.to_MaskedLinear()
    modules = [activation, masked] if activation else [masked]
    return Sequential1D(*modules, in_features=in_f, out_features=out_f)


def _ls_block(in_f, out_f, num_blocks, rank, activation=None):
    """LSLinear block (L+S = low-rank + sparse Monarch)."""
    nb = _find_common_num_blocks(in_f, out_f, num_blocks)
    layer = LSLinear.from_uniform_blocks(
        in_f, out_f, num_blocks=nb, rank=rank, bias=True,
    )
    modules = [activation, layer] if activation else [layer]
    return Sequential1D(*modules, in_features=in_f, out_features=out_f)


def _build_inn_map(input_dim, hidden_sizes, label_dim, block_builder):
    sizes = [input_dim] + list(hidden_sizes) + [label_dim]
    n = len(sizes)
    blocks = [[None] * n for _ in range(n)]
    blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)
    for i in range(n - 1):
        act = nn.ReLU() if i > 0 else None
        blocks[i][i + 1] = block_builder(sizes[i], sizes[i + 1], act)
    return Sequential2D(sizes, sizes, blocks)


# ---------------------------------------------------------------------------
# INN-family models
# ---------------------------------------------------------------------------

class INNClassifier(nn.Module):
    """Iterative Neural Network classifier."""

    def __init__(self, input_dim, hidden_sizes, label_dim, iterations, block_builder):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_sizes = list(hidden_sizes)
        self.iterations = iterations
        self.map = _build_inn_map(input_dim, hidden_sizes, label_dim, block_builder)

    def number_of_trainable_parameters(self) -> int:
        """Effective trainable params (respects MaskedLinear mask counting)."""
        return self.map.number_of_trainable_parameters()

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        state_dim = self.input_dim + sum(self.hidden_sizes) + self.label_dim
        state = torch.zeros(B, state_dim, device=x_seq.device, dtype=x_seq.dtype)
        for t in range(T):
            state[:, :self.input_dim] = x_seq[:, t, :]
            for _ in range(self.iterations):
                state = self.map(state)
        return state[:, -self.label_dim:]


# ---------------------------------------------------------------------------
# RNN-family models
# ---------------------------------------------------------------------------

class RNNClassifier(nn.Module):
    """Generic RNN classifier (LSTM / GRU / vanilla RNN)."""

    def __init__(self, input_dim, hidden_dim, label_dim, num_layers=2,
                 rnn_type="lstm", dropout=0.1, bidirectional=False):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn_tanh": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, label_dim)

    def forward(self, x_seq):
        out, _ = self.rnn(x_seq)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier."""

    def __init__(self, input_dim, hidden_dim, label_dim, num_layers=2,
                 nhead=4, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, label_dim)

    def forward(self, x_seq):
        x = self.proj(x_seq)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.fc(x)


# ---------------------------------------------------------------------------
# Flat MLP (no recurrence)
# ---------------------------------------------------------------------------

class FlatMLPClassifier(nn.Module):
    """Flat MLP: flattens the whole sequence and uses dense layers."""

    def __init__(self, input_dim, seq_len, hidden_dim, label_dim, num_layers=3):
        super().__init__()
        flat_dim = input_dim * seq_len
        layers = [nn.Linear(flat_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, label_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq):
        return self.net(x_seq.reshape(x_seq.size(0), -1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_model(name: str, input_dim: int, hidden_dim: int, label_dim: int,
                seq_len: int, **kwargs) -> nn.Module:
    """Build a model by name.

    Common kwargs:
        num_blocks (int):    Monarch block count (default 4)
        rank (int):          L+S rank (default 8)
        iterations (int):    INN iterations per timestep (default 4)
        num_layers (int):    RNN/Transformer layers (default 2)
        nhead (int):         Transformer attention heads (default 4)
    """
    num_blocks = kwargs.get("num_blocks", 4)
    rank = kwargs.get("rank", 8)
    iterations = kwargs.get("iterations", 4)
    num_layers = kwargs.get("num_layers", 2)
    nhead = kwargs.get("nhead", 4)

    hidden_sizes = [hidden_dim, hidden_dim]

    if name == "monarch_inn":
        return INNClassifier(
            input_dim, hidden_sizes, label_dim, iterations,
            block_builder=lambda inf, outf, act, _k=num_blocks: _monarch_block(inf, outf, _k, act),
        )

    elif name == "masked_inn":
        return INNClassifier(
            input_dim, hidden_sizes, label_dim, iterations,
            block_builder=lambda inf, outf, act, _k=num_blocks: _masked_block(inf, outf, _k, act),
        )

    elif name == "ls_inn":
        return INNClassifier(
            input_dim, hidden_sizes, label_dim, iterations,
            block_builder=lambda inf, outf, act, _k=num_blocks, _r=rank: _ls_block(inf, outf, _k, _r, act),
        )

    elif name == "lstm":
        return RNNClassifier(input_dim, hidden_dim, label_dim, num_layers, "lstm")

    elif name == "gru":
        return RNNClassifier(input_dim, hidden_dim, label_dim, num_layers, "gru")

    elif name == "rnn_tanh":
        return RNNClassifier(input_dim, hidden_dim, label_dim, num_layers, "rnn_tanh")

    elif name == "transformer":
        nh = nhead
        while hidden_dim % nh != 0:
            nh -= 1
        return TransformerClassifier(
            input_dim, hidden_dim, label_dim, num_layers, nhead=nh,
        )

    elif name == "mlp_flat":
        return FlatMLPClassifier(input_dim, seq_len, hidden_dim, label_dim, num_layers=3)

    else:
        raise ValueError(f"Unknown model: {name!r}. "
                         f"Choose from: {', '.join(ALL_MODEL_NAMES)}")


ALL_MODEL_NAMES = [
    "monarch_inn",
    "masked_inn",
    "ls_inn",
    "lstm",
    "gru",
    "rnn_tanh",
    "transformer",
    "mlp_flat",
]
