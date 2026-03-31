"""Llama 3 language model definitions for structured-sparse comparison.

Model variants:

1. **Standard Llama** — decoder-only transformer with ``nn.Linear``.
2. **LS Llama** — LSLinear (unfactored Monarch + low-rank) with permutations.
3. **LS-Factored Llama** — LSLinear (factored Monarch + low-rank) with permutations.
4. **LS-BlockDiag Llama** — LSBlockDiagLinear (unfactored block-diag + low-rank, no perms).
5. **LS-BlockDiag-Factored Llama** — LSBlockDiagLinear (factored block-diag + low-rank, no perms).

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
    model_c = build_ls_blockdiag_llama(vocab_size=50257, num_blocks=4, rank=16, **cfg)
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.LSBlockDiagLinear import LSBlockDiagLinear
from iterativennsimple.MonarchLinear import MonarchLinear


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

LLAMA_CONFIGS = {
    # Dev/test configs
    "small": dict(d_model=256, n_layers=6, n_heads=8, n_kv_heads=4,
                  d_ff=688, context_len=256),
    "medium": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
                   d_ff=2048, context_len=512),
    # xlarge: ~350M standard params — fits on 4090 (24GB) at batch=4
    "xlarge": dict(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
                   d_ff=2816, context_len=512),
    "large": dict(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                  d_ff=5504, context_len=1024),
    # Production-scale configs (Llama 3 architecture)
    # 7B (~8B): 1× RTX PRO 6000 Blackwell (96GB) or 1× H200
    "7b": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
               d_ff=14336, context_len=4096),
    # 30B: 2× RTX PRO 6000 Blackwell or 1× H200
    "30b": dict(d_model=6656, n_layers=60, n_heads=52, n_kv_heads=8,
                d_ff=17920, context_len=4096),
    # 70B: 4× RTX PRO 6000 Blackwell or 2× H200
    "70b": dict(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8,
                d_ff=28672, context_len=8192),
    # 405B: 8× H200 or 4× RTX PRO 6000 Blackwell with FSDP
    "405b": dict(d_model=16384, n_layers=126, n_heads=128, n_kv_heads=8,
                 d_ff=53248, context_len=8192),
}

LLAMA_LS_CONFIGS = {
    "small": dict(num_blocks=4, rank=16),
    "medium": dict(num_blocks=8, rank=32),
    "xlarge": dict(num_blocks=8, rank=64),
    "large": dict(num_blocks=16, rank=64),
    "7b": dict(num_blocks=32, rank=128),
    "30b": dict(num_blocks=64, rank=256),
    "70b": dict(num_blocks=64, rank=512),
    "405b": dict(num_blocks=128, rank=1024),
}

LLAMA_FACTORED_CONFIGS = {
    "small": dict(num_blocks=16, chain_length=4),
    "medium": dict(num_blocks=16, chain_length=4),
    "xlarge": dict(num_blocks=16, chain_length=4),
    "large": dict(num_blocks=16, chain_length=4),
    "7b": dict(num_blocks=32, chain_length=2),
    "30b": dict(num_blocks=64, chain_length=2),
    "70b": dict(num_blocks=64, chain_length=2),
    "405b": dict(num_blocks=128, chain_length=2),
}

LLAMA_LSBD_CONFIGS = {
    "small": dict(num_blocks=4, rank=16),
    "medium": dict(num_blocks=8, rank=32),
    "xlarge": dict(num_blocks=8, rank=64),
    "large": dict(num_blocks=16, rank=64),
    "7b": dict(num_blocks=32, rank=128),
    "30b": dict(num_blocks=64, rank=256),
    "70b": dict(num_blocks=64, rank=512),
    "405b": dict(num_blocks=128, rank=1024),
}

LLAMA_LSBD_FACTORED_CONFIGS = {
    "small": dict(num_blocks=4, rank=16, chain_length=2),
    "medium": dict(num_blocks=8, rank=32, chain_length=2),
    "xlarge": dict(num_blocks=8, rank=64, chain_length=2),
    "large": dict(num_blocks=16, rank=64, chain_length=2),
    "7b": dict(num_blocks=32, rank=128, chain_length=2),
    "30b": dict(num_blocks=64, rank=256, chain_length=2),
    "70b": dict(num_blocks=64, rank=512, chain_length=2),
    "405b": dict(num_blocks=128, rank=1024, chain_length=2),
}

LLAMA_ALL_MODEL_NAMES = [
    "standard", "ls", "ls-factored",
    "ls-blockdiag", "ls-blockdiag-factored",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_common_num_blocks(in_f: int, out_f: int, desired: int) -> int:
    """Largest k <= desired that divides BOTH in_f and out_f."""
    nb = desired
    while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
        nb -= 1
    return nb


def _make_ls_factory(
    num_blocks: int,
    rank: int,
    bias: bool = False,
    factored: bool = False,
    chain_length: int = 2,
):
    """Return a linear_factory callable that produces LSLinear3D layers.

    Builds the MonarchLinear sparse component directly so we can control
    whether it uses factored or unfactored block weights.
    """
    def factory(in_f: int, out_f: int) -> "LSLinear3D":
        nb = _find_common_num_blocks(in_f, out_f, num_blocks)
        # Cap rank to half the smaller dimension to avoid degeneracy
        # (e.g. KV projections where out_features << rank)
        effective_rank = min(rank, min(in_f, out_f) // 2)
        sparse = MonarchLinear.from_uniform_blocks(
            in_f, out_f, num_blocks=nb, bias=False,
            factored=factored, chain_length=chain_length,
        )
        ls = LSLinear(sparse, rank=effective_rank, bias=bias)
        return LSLinear3D(ls)
    return factory


def _make_linear_factory(bias: bool = False):
    """Return a linear_factory callable that produces standard nn.Linear layers."""
    def factory(in_f: int, out_f: int) -> nn.Linear:
        return nn.Linear(in_f, out_f, bias=bias)
    return factory


def _make_lsbd_factory(
    num_blocks: int,
    rank: int,
    bias: bool = False,
    factored: bool = False,
    chain_length: int = 2,
):
    """Return a linear_factory callable that produces LSBlockDiagLinear3D layers.

    LSBlockDiagLinear = BlockDiagLinear (pure BMM, no permutations) + low-rank AB.
    The low-rank component provides learnable global cross-block mixing,
    replacing what Monarch permutations P₁, P₂ would have provided.

    When factored=True, the block-diagonal weights are chain-factored:
    k^m blocks from k factor matrices (exponential parameter savings).
    """
    def factory(in_f: int, out_f: int) -> "LSBlockDiagLinear3D":
        nb = _find_common_num_blocks(in_f, out_f, num_blocks)
        # Cap rank to half the smaller dimension to avoid degeneracy
        effective_rank = min(rank, min(in_f, out_f) // 2)
        lsbd = LSBlockDiagLinear.from_uniform_blocks(
            in_f, out_f, num_blocks=nb, rank=effective_rank, bias=bias,
            factored=factored, chain_length=chain_length,
        )
        return LSBlockDiagLinear3D(lsbd)
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


class LSBlockDiagLinear3D(nn.Module):
    """Wraps LSBlockDiagLinear to handle arbitrary leading dimensions (e.g. (B, T, D)).

    LSBlockDiagLinear only accepts 2D (batch, features) input.
    This adapter flattens leading dims, calls LSBlockDiagLinear, then restores shape.
    """

    def __init__(self, lsbd_linear: LSBlockDiagLinear):
        super().__init__()
        self.linear = lsbd_linear
        self.in_features = lsbd_linear.in_features
        self.out_features = lsbd_linear.out_features

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
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len
        self.gradient_checkpointing = gradient_checkpointing

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
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
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
            if isinstance(module, (LSLinear3D, MonarchLinear3D, LSBlockDiagLinear3D)):
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

def _build_llama(vocab_size, linear_factory, gradient_checkpointing=False, **cfg):
    """Internal helper: build Llama with given factory and config."""
    return Llama(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        d_ff=cfg["d_ff"],
        context_len=cfg["context_len"],
        dropout=cfg.get("dropout", 0.0),
        linear_factory=linear_factory,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_standard_llama(vocab_size: int, gradient_checkpointing: bool = False, **cfg) -> Llama:
    """Build a standard Llama with nn.Linear(bias=False) projections."""
    return _build_llama(vocab_size, _make_linear_factory(bias=False),
                        gradient_checkpointing, **cfg)


def build_ls_llama(vocab_size: int, num_blocks: int = 4, rank: int = 16,
                   gradient_checkpointing: bool = False, **cfg) -> Llama:
    """Build a Llama with unfactored LSLinear (Monarch + low-rank, with perms)."""
    return _build_llama(vocab_size,
                        _make_ls_factory(num_blocks, rank, bias=False, factored=False),
                        gradient_checkpointing, **cfg)


def build_ls_factored_llama(vocab_size: int, num_blocks: int = 4, rank: int = 16,
                            chain_length: int = 2,
                            gradient_checkpointing: bool = False, **cfg) -> Llama:
    """Build a Llama with factored LSLinear (factored Monarch + low-rank, with perms)."""
    return _build_llama(vocab_size,
                        _make_ls_factory(num_blocks, rank, bias=False,
                                         factored=True, chain_length=chain_length),
                        gradient_checkpointing, **cfg)


def build_ls_blockdiag_llama(vocab_size: int, num_blocks: int = 4, rank: int = 16,
                             gradient_checkpointing: bool = False, **cfg) -> Llama:
    """Build a Llama with LSBlockDiagLinear (block-diag + low-rank, no perms)."""
    return _build_llama(vocab_size,
                        _make_lsbd_factory(num_blocks, rank, bias=False),
                        gradient_checkpointing, **cfg)


def build_ls_blockdiag_factored_llama(vocab_size: int, num_blocks: int = 4, rank: int = 16,
                                      chain_length: int = 2,
                                      gradient_checkpointing: bool = False, **cfg) -> Llama:
    """Build a Llama with factored LSBlockDiagLinear (factored block-diag + low-rank, no perms)."""
    return _build_llama(vocab_size,
                        _make_lsbd_factory(num_blocks, rank, bias=False,
                                           factored=True, chain_length=chain_length),
                        gradient_checkpointing, **cfg)


def build_llama_model(name: str, vocab_size: int, config_name: str = "small",
                      gradient_checkpointing: bool = False, **overrides):
    """Build a Llama model by name using a preset config.

    Args:
        name: One of LLAMA_ALL_MODEL_NAMES.
        vocab_size: Vocabulary size.
        config_name: Preset config name (small/medium/xlarge/large/7b/30b/70b/405b).
        gradient_checkpointing: Enable gradient checkpointing to save memory
            at the cost of ~30% slower training. Essential for 7B+ models.
        **overrides: Override any config parameter.

    Returns:
        nn.Module with a forward(input_ids) -> logits interface.
    """
    cfg = dict(LLAMA_CONFIGS[config_name])
    cfg.update({k: v for k, v in overrides.items()
                if k in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                         "d_ff", "context_len", "dropout")})
    cfg["gradient_checkpointing"] = gradient_checkpointing

    if name == "standard":
        return build_standard_llama(vocab_size=vocab_size, **cfg)

    elif name == "ls":
        ls_cfg = dict(LLAMA_LS_CONFIGS.get(config_name, LLAMA_LS_CONFIGS["small"]))
        ls_cfg.update({k: v for k, v in overrides.items() if k in ("num_blocks", "rank")})
        return build_ls_llama(vocab_size=vocab_size, **ls_cfg, **cfg)

    elif name == "ls-factored":
        lsf_cfg = dict(LLAMA_LS_CONFIGS.get(config_name, LLAMA_LS_CONFIGS["small"]))
        lsf_cfg.update({k: v for k, v in overrides.items()
                        if k in ("num_blocks", "rank", "chain_length")})
        if "chain_length" not in lsf_cfg:
            f_defaults = LLAMA_FACTORED_CONFIGS.get(config_name, LLAMA_FACTORED_CONFIGS["small"])
            lsf_cfg["chain_length"] = f_defaults.get("chain_length", 2)
        return build_ls_factored_llama(vocab_size=vocab_size, **lsf_cfg, **cfg)

    elif name == "ls-blockdiag":
        bd_cfg = dict(LLAMA_LSBD_CONFIGS.get(config_name, LLAMA_LSBD_CONFIGS["small"]))
        bd_cfg.update({k: v for k, v in overrides.items() if k in ("num_blocks", "rank")})
        return build_ls_blockdiag_llama(vocab_size=vocab_size, **bd_cfg, **cfg)

    elif name == "ls-blockdiag-factored":
        bdf_cfg = dict(LLAMA_LSBD_FACTORED_CONFIGS.get(config_name, LLAMA_LSBD_FACTORED_CONFIGS["small"]))
        bdf_cfg.update({k: v for k, v in overrides.items()
                        if k in ("num_blocks", "rank", "chain_length")})
        return build_ls_blockdiag_factored_llama(vocab_size=vocab_size, **bdf_cfg, **cfg)

    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from: {LLAMA_ALL_MODEL_NAMES}")
