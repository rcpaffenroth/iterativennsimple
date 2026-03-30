"""Llama 3 language model definitions for LSLinear vs Standard comparison.

Two model variants:

1. **Standard Llama** — decoder-only transformer with ``nn.Linear``.
2. **LSLinear Llama** — identical skeleton, LSLinear replaces all linear projections.

All variants share the same code via a ``linear_factory`` injection pattern,
ensuring the comparison is fair by construction.

Llama 3 architecture differences from GPT-2:
  - RMSNorm instead of LayerNorm
  - Rotary Position Embeddings (RoPE) instead of learned absolute position embeddings
  - Grouped Query Attention (GQA) — fewer KV heads than Q heads
  - SwiGLU activation with 3 linear projections instead of GELU with 2
  - No bias in any linear layers

Usage::

    from llama_models import build_standard_llama, build_ls_llama, LLAMA_CONFIGS

    cfg = LLAMA_CONFIGS["small"]
    model_a = build_standard_llama(vocab_size=50257, **cfg)
    model_b = build_ls_llama(vocab_size=50257, num_blocks=4, rank=16, **cfg)
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

LLAMA_CONFIGS = {
    "small": dict(d_model=256, n_layers=6, n_heads=8, n_kv_heads=4,
                  d_ff=688, context_len=256),
    "medium": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
                   d_ff=2048, context_len=512),
    "large": dict(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                  d_ff=5504, context_len=1024),
}

LLAMA_LS_CONFIGS = {
    "small": dict(num_blocks=4, rank=16),
    "medium": dict(num_blocks=8, rank=32),
    "large": dict(num_blocks=16, rank=64),
}

LLAMA_FACTORED_CONFIGS = {
    "small": dict(num_blocks=16, chain_length=4),
    "medium": dict(num_blocks=16, chain_length=4),
    "large": dict(num_blocks=16, chain_length=4),
}

LLAMA_ALL_MODEL_NAMES = ["standard", "ls", "factored"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_common_num_blocks(in_f: int, out_f: int, desired: int) -> int:
    """Largest k <= desired that divides BOTH in_f and out_f."""
    nb = desired
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    return nb


def _make_ls_factory(num_blocks: int, rank: int, bias: bool = False):
    """Return a linear_factory callable that produces LSLinear3D layers."""
    def factory(in_f: int, out_f: int) -> "LSLinear3D":
        nb = _find_common_num_blocks(in_f, out_f, num_blocks)
        ls = LSLinear.from_uniform_blocks(
            in_f, out_f, num_blocks=nb, rank=rank, bias=bias,
        )
        return LSLinear3D(ls)
    return factory


def _make_linear_factory(bias: bool = False):
    """Return a linear_factory callable that produces standard nn.Linear layers."""
    def factory(in_f: int, out_f: int) -> nn.Linear:
        return nn.Linear(in_f, out_f, bias=bias)
    return factory


def _make_factored_factory(num_blocks: int, chain_length: int = 2, bias: bool = False):
    """Return a linear_factory callable that produces factored MonarchLinear3D layers.

    Uses chain-factored blocks: k^m blocks from k factor matrices, where
    k = ceil(num_blocks^(1/m)) and m = chain_length.  Non-square projections
    use a shared adapter matrix for the dimension mismatch.
    """
    def factory(in_f: int, out_f: int) -> nn.Module:
        nb = _find_common_num_blocks(in_f, out_f, num_blocks)
        m = MonarchLinear.from_uniform_blocks(
            in_f, out_f, num_blocks=nb, bias=bias,
            factored=True, chain_length=chain_length,
        )
        return MonarchLinear3D(m)
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


class MonarchLinear3D(nn.Module):
    """Wraps MonarchLinear to handle arbitrary leading dimensions (e.g. (B, T, D)).

    MonarchLinear only accepts 2D (batch, features) or 1D (features,) input.
    This adapter flattens leading dims, calls MonarchLinear, then restores shape.
    """

    def __init__(self, monarch_linear: MonarchLinear):
        super().__init__()
        self.linear = monarch_linear
        self.in_features = monarch_linear.in_features
        self.out_features = monarch_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        y_2d = self.linear(x_2d)
        return y_2d.reshape(*shape[:-1], -1)

    def number_of_trainable_parameters(self) -> int:
        return self.linear.number_of_trainable_parameters()


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style, no bias).

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability (important for bf16 training)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables for Rotary Position Embeddings.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency (500000 for Llama 3).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 500000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        """Return (cos, sin) sliced to seq_len, each shape (seq_len, head_dim//2)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a tensor.

    Args:
        x: (B, n_heads, T, head_dim) — query or key tensor.
        cos: (T, head_dim//2) — cosine component.
        sin: (T, head_dim//2) — sine component.

    Returns:
        Rotated tensor, same shape as x.
    """
    # Reshape cos/sin for broadcasting: (1, 1, T, head_dim//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into even and odd features
    x1 = x[..., ::2]   # (..., head_dim // 2)
    x2 = x[..., 1::2]  # (..., head_dim // 2)

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back: stack on last dim then flatten
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Llama Components
# ---------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    """Grouped Query Attention with Rotary Position Embeddings.

    Args:
        d_model: Model dimension.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (GQA). Must divide n_heads evenly.
        max_seq_len: Maximum sequence length (for RoPE and causal mask).
        dropout: Dropout probability.
        linear_factory: Callable(in_f, out_f) -> nn.Module for projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Q heads per KV head
        self.head_dim = d_model // n_heads

        factory = linear_factory or _make_linear_factory()

        # Q projects to full n_heads * head_dim
        self.q_proj = factory(d_model, n_heads * self.head_dim)
        # K, V project to smaller n_kv_heads * head_dim
        self.k_proj = factory(d_model, n_kv_heads * self.head_dim)
        self.v_proj = factory(d_model, n_kv_heads * self.head_dim)
        # Output projection
        self.o_proj = factory(n_heads * self.head_dim, d_model)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Repeat K, V heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.o_proj(out))


class LlamaFFN(nn.Module):
    """SwiGLU Feed-Forward Network (3 projections: gate, up, down).

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

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
        dropout: float = 0.0,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        factory = linear_factory or _make_linear_factory()
        self.gate_proj = factory(d_model, d_ff)
        self.up_proj = factory(d_model, d_ff)
        self.down_proj = factory(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LlamaBlock(nn.Module):
    """Pre-norm Llama block: RMSNorm -> GQA Attn -> residual, RMSNorm -> SwiGLU FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = LlamaAttention(
            d_model, n_heads, n_kv_heads, max_seq_len, dropout, linear_factory,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = LlamaFFN(d_model, d_ff, dropout, linear_factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Llama Model
# ---------------------------------------------------------------------------

class Llama(nn.Module):
    """Llama 3 style decoder-only language model.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model/embedding dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (GQA).
        d_ff: SwiGLU FFN inner dimension.
        context_len: Maximum sequence length.
        dropout: Dropout probability (0.0 by default, Llama convention).
        linear_factory: Callable(in_f, out_f) -> nn.Module for all projections.
            Defaults to nn.Linear(bias=False). Pass an LSLinear factory for L+S variant.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        d_ff: int = 688,
        context_len: int = 256,
        dropout: float = 0.0,
        linear_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len

        # Token embedding only — RoPE handles position
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            LlamaBlock(d_model, n_heads, n_kv_heads, d_ff, context_len,
                       dropout, linear_factory)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

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
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            logits: (B, T, vocab_size) unnormalized log-probabilities.
        """
        B, T = input_ids.shape
        assert T <= self.context_len, f"Sequence length {T} > context_len {self.context_len}"

        x = self.emb_dropout(self.tok_emb(input_ids))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)

    def number_of_trainable_parameters(self) -> int:
        """Count trainable parameters, respecting LSLinear compressed counting."""
        total = 0
        counted = set()
        for name, module in self.named_modules():
            if id(module) in counted:
                continue
            if isinstance(module, (LSLinear3D, MonarchLinear3D)):
                total += module.number_of_trainable_parameters()
                counted.add(id(module))
                for child in module.modules():
                    counted.add(id(child))
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                for p in module.parameters(recurse=False):
                    if id(p) not in counted:
                        total += p.numel()
                        counted.add(id(p))
                counted.add(id(module))
            elif isinstance(module, RMSNorm):
                if id(module.weight) not in counted:
                    total += module.weight.numel()
                    counted.add(id(module.weight))
                counted.add(id(module))
        return total


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_standard_llama(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    d_ff: int = 688,
    context_len: int = 256,
    dropout: float = 0.0,
    **kwargs,
) -> Llama:
    """Build a standard Llama with nn.Linear(bias=False) projections."""
    return Llama(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        context_len=context_len,
        dropout=dropout,
        linear_factory=_make_linear_factory(bias=False),
    )


def build_ls_llama(
    vocab_size: int,
    num_blocks: int = 4,
    rank: int = 16,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    d_ff: int = 688,
    context_len: int = 256,
    dropout: float = 0.0,
    **kwargs,
) -> Llama:
    """Build a Llama with LSLinear replacing all linear projections."""
    return Llama(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        context_len=context_len,
        dropout=dropout,
        linear_factory=_make_ls_factory(num_blocks, rank, bias=False),
    )


def build_factored_llama(
    vocab_size: int,
    num_blocks: int = 16,
    chain_length: int = 4,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    d_ff: int = 688,
    context_len: int = 256,
    dropout: float = 0.0,
    **kwargs,
) -> Llama:
    """Build a Llama with chain-factored MonarchLinear replacing all projections.

    Uses k^m blocks from k factor matrices (exponential parameter savings).
    Non-square projections use a shared adapter matrix.
    """
    return Llama(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        context_len=context_len,
        dropout=dropout,
        linear_factory=_make_factored_factory(num_blocks, chain_length, bias=False),
    )


def build_llama_model(name: str, vocab_size: int, config_name: str = "small", **overrides):
    """Build a Llama model by name using a preset config.

    Args:
        name: One of "standard", "ls", "factored".
        vocab_size: Vocabulary size.
        config_name: Preset config name ("small", "medium", "large").
        **overrides: Override any config parameter.

    Returns:
        nn.Module with a forward(input_ids) -> logits interface.
    """
    cfg = dict(LLAMA_CONFIGS[config_name])
    cfg.update(overrides)

    if name == "standard":
        return build_standard_llama(vocab_size=vocab_size, **cfg)

    elif name == "ls":
        ls_cfg = dict(LLAMA_LS_CONFIGS.get(config_name, LLAMA_LS_CONFIGS["small"]))
        ls_cfg.update({k: v for k, v in overrides.items() if k in ("num_blocks", "rank")})
        cfg.update(ls_cfg)
        return build_ls_llama(vocab_size=vocab_size, **cfg)

    elif name == "factored":
        f_cfg = dict(LLAMA_FACTORED_CONFIGS.get(config_name, LLAMA_FACTORED_CONFIGS["small"]))
        f_cfg.update({k: v for k, v in overrides.items() if k in ("num_blocks", "chain_length")})
        cfg.update(f_cfg)
        return build_factored_llama(vocab_size=vocab_size, **cfg)

    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from: {LLAMA_ALL_MODEL_NAMES}")
