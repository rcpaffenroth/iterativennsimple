"""GPT and INN language model definitions for LSLinear vs Transformer comparison.

Three model variants:

1. **Standard GPT** — decoder-only transformer with ``nn.Linear``.
2. **LSLinear GPT** — identical skeleton, LSLinear replaces all linear projections.
3. **INN LM** — Iterative Neural Network language model using Sequential2D + LSLinear.

All GPT variants share the same code via a ``linear_factory`` injection pattern,
ensuring the comparison is fair by construction.

Usage::

    from models import build_standard_gpt, build_ls_gpt, build_inn_lm, MODEL_CONFIGS

    cfg = MODEL_CONFIGS["small"]
    model_a = build_standard_gpt(vocab_size=50257, **cfg)
    model_b = build_ls_gpt(vocab_size=50257, num_blocks=4, rank=16, **cfg)
    model_c = build_inn_lm(vocab_size=50257, num_blocks=4, rank=16, **cfg)
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "small": dict(d_model=256, n_layers=6, n_heads=4, d_ff=1024, context_len=256),
    "medium": dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072, context_len=512),
    "large": dict(d_model=2048, n_layers=24, n_heads=16, d_ff=8192, context_len=1024),
}

LS_CONFIGS = {
    "small": dict(num_blocks=4, rank=16),
    "medium": dict(num_blocks=8, rank=32),
    "large": dict(num_blocks=16, rank=64),
}

INN_CONFIGS = {
    "small": dict(num_blocks=4, rank=16, iterations=4, n_hidden=2),
    "medium": dict(num_blocks=8, rank=32, iterations=4, n_hidden=3),
    "large": dict(num_blocks=16, rank=64, iterations=4, n_hidden=4),
}

ALL_MODEL_NAMES = ["standard", "ls", "inn"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_common_num_blocks(in_f: int, out_f: int, desired: int) -> int:
    """Largest k <= desired that divides BOTH in_f and out_f."""
    nb = desired
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    return nb


def _make_ls_factory(num_blocks: int, rank: int):
    """Return a linear_factory callable that produces LSLinear3D layers."""
    def factory(in_f: int, out_f: int) -> "LSLinear3D":
        nb = _find_common_num_blocks(in_f, out_f, num_blocks)
        ls = LSLinear.from_uniform_blocks(
            in_f, out_f, num_blocks=nb, rank=rank, bias=True,
        )
        return LSLinear3D(ls)
    return factory


def _make_linear_factory():
    """Return a linear_factory callable that produces standard nn.Linear layers."""
    def factory(in_f: int, out_f: int) -> nn.Linear:
        return nn.Linear(in_f, out_f)
    return factory


# ---------------------------------------------------------------------------
# LSLinear3D — adapter for (B, T, D) tensors
# ---------------------------------------------------------------------------

class LSLinear3D(nn.Module):
    """Wraps LSLinear to handle arbitrary leading dimensions (e.g. (B, T, D)).

    LSLinear only accepts 2D (batch, features) or 1D (features,) input.
    This adapter flattens leading dims, calls LSLinear, then restores shape.
    """

    def __init__(self, ls_linear: LSLinear):
        super().__init__()
        self.linear = ls_linear
        self.in_features = ls_linear.in_features
        self.out_features = ls_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        y_2d = self.linear(x_2d)
        return y_2d.reshape(*shape[:-1], -1)

    def number_of_trainable_parameters(self) -> int:
        return self.linear.number_of_trainable_parameters()


# ---------------------------------------------------------------------------
# GPT Components
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (for causal mask buffer).
        dropout: Dropout probability.
        linear_factory: Callable(in_f, out_f) -> nn.Module for Q/K/V/O projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        factory = linear_factory or _make_linear_factory()

        # Q, K, V projections (separate for clarity; could be fused)
        self.q_proj = factory(d_model, d_model)
        self.k_proj = factory(d_model, d_model)
        self.v_proj = factory(d_model, d_model)
        self.o_proj = factory(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.o_proj(out))


class FFN(nn.Module):
    """Feed-forward network: up-project -> GELU -> down-project.

    Args:
        d_model: Model dimension.
        d_ff: Inner (expanded) dimension.
        dropout: Dropout probability.
        linear_factory: Callable(in_f, out_f) -> nn.Module.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        factory = linear_factory or _make_linear_factory()
        self.up = factory(d_model, d_ff)
        self.down = factory(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attn -> residual, LN -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout, linear_factory,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout, linear_factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Minimal GPT (decoder-only) language model.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model/embedding dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: FFN inner dimension.
        context_len: Maximum sequence length.
        dropout: Dropout probability.
        linear_factory: Callable(in_f, out_f) -> nn.Module for all projections.
            Defaults to nn.Linear. Pass an LSLinear factory for the L+S variant.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        context_len: int = 256,
        dropout: float = 0.1,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len

        # Token + positional embeddings (always dense)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_len, dropout, linear_factory)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            logits: (B, T, vocab_size) unnormalized log-probabilities.
        """
        B, T = input_ids.shape
        assert T <= self.context_len, f"Sequence length {T} > context_len {self.context_len}"

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def number_of_trainable_parameters(self) -> int:
        """Count trainable parameters, respecting LSLinear compressed counting."""
        total = 0
        counted = set()
        for name, module in self.named_modules():
            if id(module) in counted:
                continue
            if isinstance(module, LSLinear3D):
                total += module.number_of_trainable_parameters()
                counted.add(id(module))
                for child in module.modules():
                    counted.add(id(child))
            elif isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                # Skip tied weights (lm_head shares with tok_emb)
                for p in module.parameters(recurse=False):
                    if id(p) not in counted:
                        total += p.numel()
                        counted.add(id(p))
                counted.add(id(module))
        return total


# ---------------------------------------------------------------------------
# INN Language Model
# ---------------------------------------------------------------------------

class INNLM(nn.Module):
    """Iterative Neural Network language model using Sequential2D + LSLinear.

    All token positions are processed **in parallel**: embeddings are placed
    into the input slot of per-position state vectors and the Sequential2D
    map is iterated K times across the full (B*T) batch.  This is the same
    weight-sharing/iteration paradigm as the INN classifiers in the existing
    benchmarks, adapted for language modeling.

    Architecture:
        state = [embed_slot | hidden_1 | ... | hidden_n | output_slot]
        state[:, :d_model] = embeddings.reshape(B*T, d_model)
        for k in range(iterations):
            state = map(state)            # (B*T, state_dim)
        logits = lm_head(state[:, -d_model:]).reshape(B, T, vocab)

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding and hidden slot dimension.
        n_hidden: Number of hidden slots in the state vector.
        num_blocks: MonarchLinear block count for LSLinear.
        rank: Low-rank factor for LSLinear.
        iterations: Number of map iterations (depth via weight sharing).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_hidden: int = 2,
        num_blocks: int = 4,
        rank: int = 16,
        iterations: int = 4,
        context_len: int = 256,  # accepted but not used for INN sizing
        dropout: float = 0.1,
        d_ff: int = None,  # accepted for API compat, not used
        n_layers: int = None,  # accepted for API compat, not used
        n_heads: int = None,  # accepted for API compat, not used
    ):
        super().__init__()
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.iterations = iterations
        self.vocab_size = vocab_size

        # Embeddings (always dense)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Output dimension = d_model (projected to vocab via lm_head)
        self.output_dim = d_model

        # Build Sequential2D map
        # State: [embed(d_model) | h1(d_model) | ... | hn(d_model) | out(d_model)]
        sizes = [d_model] * (n_hidden + 2)  # embed + n_hidden + output
        n_slots = len(sizes)
        blocks = [[None] * n_slots for _ in range(n_slots)]

        # Identity on embed slot (preserves embedding across iterations)
        blocks[0][0] = Identity(in_features=d_model, out_features=d_model)

        # Forward connections: slot[i] -> slot[i+1] with LSLinear
        for i in range(n_slots - 1):
            act = nn.ReLU() if i > 0 else None
            nb = _find_common_num_blocks(sizes[i], sizes[i + 1], num_blocks)
            layer = LSLinear.from_uniform_blocks(
                sizes[i], sizes[i + 1], num_blocks=nb, rank=rank, bias=True,
            )
            if act is not None:
                block = Sequential1D(
                    nn.Sequential(act, layer),
                    in_features=sizes[i], out_features=sizes[i + 1],
                )
            else:
                block = Sequential1D(
                    nn.Sequential(layer),
                    in_features=sizes[i], out_features=sizes[i + 1],
                )
            blocks[i][i + 1] = block

        self.map = Sequential2D(sizes, sizes, blocks)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Init
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass — all positions processed in parallel.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            logits: (B, T, vocab_size) unnormalized log-probabilities.
        """
        B, T = input_ids.shape

        # Embed all tokens at once: (B, T, d_model)
        embeddings = self.emb_dropout(self.tok_emb(input_ids))

        # Flatten to (B*T, d_model) for Sequential2D which expects 2D input
        flat_emb = embeddings.reshape(B * T, self.d_model)

        # Build state: [embed | zeros_h1 | ... | zeros_hn | zeros_out]
        n_slots = self.n_hidden + 2
        hidden_dim = self.d_model * (n_slots - 1)  # everything after embed slot
        zeros = torch.zeros(B * T, hidden_dim, device=input_ids.device, dtype=flat_emb.dtype)
        state = torch.cat([flat_emb, zeros], dim=1)  # (B*T, state_dim)

        # Iterate the map K times (weight sharing = depth)
        for _ in range(self.iterations):
            state = self.map(state)

        # Read logits from output slot (last d_model dims of state)
        out = state[:, -self.output_dim:]             # (B*T, d_model)
        logits = self.lm_head(out)                    # (B*T, vocab_size)
        return logits.reshape(B, T, self.vocab_size)  # (B, T, vocab_size)

    def number_of_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        total = self.map.number_of_trainable_parameters()
        # Embedding (shared with lm_head via weight tying)
        total += self.tok_emb.weight.numel()
        return total


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_standard_gpt(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: int = 1024,
    context_len: int = 256,
    dropout: float = 0.1,
    **kwargs,
) -> GPT:
    """Build a standard GPT with nn.Linear projections."""
    return GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_len=context_len,
        dropout=dropout,
        linear_factory=_make_linear_factory(),
    )


def build_ls_gpt(
    vocab_size: int,
    num_blocks: int = 4,
    rank: int = 16,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: int = 1024,
    context_len: int = 256,
    dropout: float = 0.1,
    **kwargs,
) -> GPT:
    """Build a GPT with LSLinear replacing all linear projections."""
    return GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_len=context_len,
        dropout=dropout,
        linear_factory=_make_ls_factory(num_blocks, rank),
    )


def build_inn_lm(
    vocab_size: int,
    num_blocks: int = 4,
    rank: int = 16,
    iterations: int = 4,
    d_model: int = 256,
    n_hidden: int = 2,
    context_len: int = 256,
    dropout: float = 0.1,
    **kwargs,
) -> INNLM:
    """Build an INN-based language model with LSLinear blocks."""
    return INNLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_hidden=n_hidden,
        num_blocks=num_blocks,
        rank=rank,
        iterations=iterations,
        context_len=context_len,
        dropout=dropout,
    )


def build_model(name: str, vocab_size: int, config_name: str = "small", **overrides):
    """Build a model by name using a preset config.

    Args:
        name: One of "standard", "ls", "inn".
        vocab_size: Vocabulary size.
        config_name: Preset config name ("small", "medium", "large").
        **overrides: Override any config parameter.

    Returns:
        nn.Module with a forward(input_ids) -> logits interface.
    """
    cfg = dict(MODEL_CONFIGS[config_name])
    cfg.update(overrides)

    if name == "standard":
        return build_standard_gpt(vocab_size=vocab_size, **cfg)

    elif name == "ls":
        ls_cfg = dict(LS_CONFIGS.get(config_name, LS_CONFIGS["small"]))
        ls_cfg.update({k: v for k, v in overrides.items() if k in ("num_blocks", "rank")})
        cfg.update(ls_cfg)
        return build_ls_gpt(vocab_size=vocab_size, **cfg)

    elif name == "inn":
        inn_cfg = dict(INN_CONFIGS.get(config_name, INN_CONFIGS["small"]))
        inn_cfg.update({k: v for k, v in overrides.items()
                        if k in ("num_blocks", "rank", "iterations", "n_hidden")})
        cfg.update(inn_cfg)
        return build_inn_lm(vocab_size=vocab_size, **cfg)

    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from: {ALL_MODEL_NAMES}")
