#!/usr/bin/env python
"""Diagnose model-level factored overhead."""

import os, sys, time, gc, math
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_this_dir, "..", "..")
_advanced_llm = os.path.join(_this_dir, "..", "advanced", "llm")
sys.path.insert(0, _root)
sys.path.insert(0, _advanced_llm)

import torch
import torch.nn as nn

from iterativennsimple.BlockDiagLinear import BlockDiagLinear
from iterativennsimple.LSBlockDiagLinear import LSBlockDiagLinear
from llama_models import Llama, LLAMA_CONFIGS, LLAMA_LS_CONFIGS

torch.set_float32_matmul_precision('high')
device = torch.device("cuda")

cfg = LLAMA_CONFIGS["medium"]
ls_cfg = LLAMA_LS_CONFIGS["medium"]
nb = ls_cfg["num_blocks"]
rk = ls_cfg["rank"]

class _Wrapper3D(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
    def forward(self, x):
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        y_2d = self.linear(x_2d)
        return y_2d.reshape(*shape[:-1], -1)
    def number_of_trainable_parameters(self):
        return self.linear.number_of_trainable_parameters()

def _make_lsbd_factory(factored=False, chain_length=2):
    def factory(in_f, out_f):
        n = nb
        while n > 1 and (in_f % n != 0 or out_f % n != 0):
            n -= 1
        lsbd = LSBlockDiagLinear.from_uniform_blocks(
            in_f, out_f, n, rank=rk, bias=False,
            factored=factored, chain_length=chain_length,
        )
        return _Wrapper3D(lsbd)
    return factory

batch_size = 8
ctx_len = cfg["context_len"]
vocab_size = 50257

# Build models
print("Building models...")
torch.manual_seed(42)
model_unf = Llama(vocab_size=vocab_size, **cfg, linear_factory=_make_lsbd_factory(factored=False)).to(device)
torch.manual_seed(42)
model_fac = Llama(vocab_size=vocab_size, **cfg, linear_factory=_make_lsbd_factory(factored=True, chain_length=2)).to(device)

# Count real params
unf_stored = sum(w.number_of_trainable_parameters() for w in model_unf.modules() if isinstance(w, _Wrapper3D))
unf_other = sum(p.numel() for n, p in model_unf.named_parameters() if not any(isinstance(m, _Wrapper3D) for m in [model_unf]))
fac_stored = sum(w.number_of_trainable_parameters() for w in model_fac.modules() if isinstance(w, _Wrapper3D))

unf_surface = sum(p.numel() for p in model_unf.parameters())
fac_surface = sum(p.numel() for p in model_fac.parameters())

print(f"Unfactored: surface={unf_surface:,}")
print(f"Factored:   surface={fac_surface:,}")
print(f"Difference: {unf_surface - fac_surface:,} fewer params in factored")

# Time just forward pass
input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len), device=device)

def time_fn(fn, warmup=5, iters=20, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    print(f"  {label:45s} {ms:8.2f} ms")
    return ms

print(f"\n--- Forward only (batch={batch_size}, ctx={ctx_len}) ---")

def fwd_unf():
    with torch.no_grad():
        return model_unf(input_ids)

def fwd_fac():
    with torch.no_grad():
        return model_fac(input_ids)

time_fn(fwd_unf, label="Unfactored forward (no grad)")
time_fn(fwd_fac, label="Factored forward (no grad)")

print(f"\n--- Forward+Backward ---")

criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, vocab_size, (batch_size, ctx_len), device=device)

def fwd_bwd_unf():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_unf(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    for p in model_unf.parameters():
        if p.grad is not None:
            p.grad = None

def fwd_bwd_fac():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_fac(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    for p in model_fac.parameters():
        if p.grad is not None:
            p.grad = None

time_fn(fwd_bwd_unf, label="Unfactored fwd+bwd (bf16)")
time_fn(fwd_bwd_fac, label="Factored fwd+bwd (bf16)")

print(f"\n--- Forward+Backward (compiled) ---")

model_unf_c = torch.compile(model_unf)
model_fac_c = torch.compile(model_fac)

def fwd_bwd_unf_c():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_unf_c(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    for p in model_unf.parameters():
        if p.grad is not None:
            p.grad = None

def fwd_bwd_fac_c():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_fac_c(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    for p in model_fac.parameters():
        if p.grad is not None:
            p.grad = None

time_fn(fwd_bwd_unf_c, label="Unfactored compiled fwd+bwd (bf16)")
time_fn(fwd_bwd_fac_c, label="Factored compiled fwd+bwd (bf16)")

print(f"\n--- Full training step (fwd+bwd+optimizer) ---")

opt_unf = torch.optim.AdamW(model_unf.parameters(), lr=3e-4)
opt_fac = torch.optim.AdamW(model_fac.parameters(), lr=3e-4)

def train_step_unf():
    opt_unf.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_unf(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    opt_unf.step()

def train_step_fac():
    opt_fac.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model_fac(input_ids)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    opt_fac.step()

time_fn(train_step_unf, label="Unfactored full train step")
time_fn(train_step_fac, label="Factored full train step")

# Compute tok/s
unf_ms = time_fn(train_step_unf, label="Unfactored (final)")
fac_ms = time_fn(train_step_fac, label="Factored (final)")

tokens_per_step = batch_size * ctx_len
print(f"\n  Unfactored: {tokens_per_step / (unf_ms/1000):,.0f} tok/s")
print(f"  Factored:   {tokens_per_step / (fac_ms/1000):,.0f} tok/s")
print(f"  Ratio:      {unf_ms / fac_ms:.3f}x")
