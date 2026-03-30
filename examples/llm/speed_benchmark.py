#!/usr/bin/env python
"""Speed benchmark: compare layer-level and model-level throughput.

Tests multiple linear layer variants at the exact dimensions used in
the Llama medium config to understand where time is spent.

Usage:
    python speed_benchmark.py                    # layer-level microbenchmark
    python speed_benchmark.py --model            # full model training speed
    python speed_benchmark.py --model --epochs 3 # quick model benchmark
"""

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn as nn

# Setup paths
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_this_dir, "..", "..")
_advanced_llm = os.path.join(_this_dir, "..", "advanced", "llm")
sys.path.insert(0, _root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _advanced_llm not in sys.path:
    sys.path.insert(0, _advanced_llm)

from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.LSLinear import LSLinear
from iterativennsimple.BlockDiagLinear import BlockDiagLinear
from iterativennsimple.LSBlockDiagLinear import LSBlockDiagLinear


def benchmark_layer(layer, x, label, warmup=10, iters=100, backward=True):
    """Time forward+backward for a layer. Returns ms per iteration."""
    layer = layer.cuda()
    x = x.cuda().requires_grad_(backward)

    # Warmup
    for _ in range(warmup):
        y = layer(x)
        if backward:
            y.sum().backward()
            if x.grad is not None:
                x.grad = None

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(iters):
        y = layer(x)
        if backward:
            y.sum().backward()
            if x.grad is not None:
                x.grad = None

    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000  # ms

    params = layer.number_of_trainable_parameters() if hasattr(layer, 'number_of_trainable_parameters') else sum(p.numel() for p in layer.parameters())
    print(f"  {label:45s}  {elapsed:7.2f} ms  params={params:>10,}")
    return elapsed


def layer_benchmark():
    """Microbenchmark individual layers at Llama medium dimensions."""
    print("\n" + "=" * 90)
    print(" LAYER-LEVEL SPEED BENCHMARK")
    print("=" * 90)

    device = torch.device("cuda")
    batch = 32 * 512  # batch_size * context_len (flattened for 2D layers)

    configs = [
        # (in_features, out_features, label_suffix)
        (768, 768, "attn_proj"),       # Q, K, V, O projections
        (768, 2048, "ffn_up"),         # FFN gate/up
        (2048, 768, "ffn_down"),       # FFN down
    ]

    num_blocks = 8
    rank = 32

    for in_f, out_f, suffix in configs:
        print(f"\n--- {suffix}: ({in_f} -> {out_f}), batch={batch} ---")

        x = torch.randn(batch, in_f)

        # 1. Standard nn.Linear
        linear = nn.Linear(in_f, out_f, bias=False)
        baseline = benchmark_layer(linear, x, f"nn.Linear")
        del linear; gc.collect(); torch.cuda.empty_cache()

        # 2. BlockDiagLinear (no permutations, contiguous)
        bd = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, stride_perm=False)
        benchmark_layer(bd, x, f"BlockDiagLinear (contiguous)")
        del bd; gc.collect(); torch.cuda.empty_cache()

        # 3. BlockDiagLinear (contiguous, compiled)
        bd = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, stride_perm=False).cuda()
        bd_compiled = torch.compile(bd)
        benchmark_layer(bd_compiled, x, f"BlockDiagLinear (contig+compile)")
        del bd, bd_compiled; gc.collect(); torch.cuda.empty_cache()

        # 4. BlockDiagLinear (factored, chain_length=2)
        bd_f = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=True, chain_length=2)
        benchmark_layer(bd_f, x, f"BlockDiagLinear (factored, m=2)")
        del bd_f; gc.collect(); torch.cuda.empty_cache()

        # 5. BlockDiagLinear (factored, chain_length=3)
        bd_f3 = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=True, chain_length=3)
        benchmark_layer(bd_f3, x, f"BlockDiagLinear (factored, m=3)")
        del bd_f3; gc.collect(); torch.cuda.empty_cache()

        # 5b. BlockDiagLinear (factored+compiled, m=2)
        bd_fc = BlockDiagLinear(in_f, out_f, num_blocks, bias=False, factored=True, chain_length=2).cuda()
        bd_fc_compiled = torch.compile(bd_fc)
        benchmark_layer(bd_fc_compiled, x, f"BlockDiagLinear (factored+compile, m=2)")
        del bd_fc, bd_fc_compiled; gc.collect(); torch.cuda.empty_cache()

        # 6. LSBlockDiagLinear (BlockDiag + low-rank, contiguous)
        lsbd = LSBlockDiagLinear.from_uniform_blocks(in_f, out_f, num_blocks, rank=rank, bias=False, stride_perm=False)
        benchmark_layer(lsbd, x, f"LSBlockDiagLinear (contig+LR)")
        del lsbd; gc.collect(); torch.cuda.empty_cache()

        # 7. LSBlockDiagLinear (contig, compiled)
        lsbd = LSBlockDiagLinear.from_uniform_blocks(in_f, out_f, num_blocks, rank=rank, bias=False, stride_perm=False).cuda()
        lsbd_compiled = torch.compile(lsbd)
        benchmark_layer(lsbd_compiled, x, f"LSBlockDiagLinear (contig+compile)")
        del lsbd, lsbd_compiled; gc.collect(); torch.cuda.empty_cache()

        # 8. LSBlockDiagLinear (factored BlockDiag + low-rank)
        lsbd_f = LSBlockDiagLinear.from_uniform_blocks(
            in_f, out_f, num_blocks, rank=rank, bias=False,
            factored=True, chain_length=2,
        )
        benchmark_layer(lsbd_f, x, f"LSBlockDiagLinear (factored+LR)")
        del lsbd_f; gc.collect(); torch.cuda.empty_cache()

        # 8b. LSBlockDiagLinear (factored+compiled)
        lsbd_fc = LSBlockDiagLinear.from_uniform_blocks(
            in_f, out_f, num_blocks, rank=rank, bias=False,
            factored=True, chain_length=2,
        ).cuda()
        lsbd_fc_compiled = torch.compile(lsbd_fc)
        benchmark_layer(lsbd_fc_compiled, x, f"LSBlockDiagLinear (factored+compile)")
        del lsbd_fc, lsbd_fc_compiled; gc.collect(); torch.cuda.empty_cache()

        # 9. nn.Linear (compiled, for fair comparison)
        linear = nn.Linear(in_f, out_f, bias=False).cuda()
        linear_compiled = torch.compile(linear)
        benchmark_layer(linear_compiled, x, f"nn.Linear (compiled)")
        del linear, linear_compiled; gc.collect(); torch.cuda.empty_cache()

        # 10. MonarchLinear (with permutations, BMM path)
        monarch = MonarchLinear.from_uniform_blocks(in_f, out_f, num_blocks, bias=False, factored=False)
        benchmark_layer(monarch, x, f"MonarchLinear (BMM, perm)")
        del monarch; gc.collect(); torch.cuda.empty_cache()

        # 11. LSLinear (MonarchLinear + low-rank, original)
        ls = LSLinear.from_uniform_blocks(in_f, out_f, num_blocks, rank=rank, bias=False)
        benchmark_layer(ls, x, f"LSLinear (monarch+LR)")
        del ls; gc.collect(); torch.cuda.empty_cache()


def model_benchmark(epochs=5, batch_size=32, config_name="medium"):
    """Full model training speed comparison."""
    print("\n" + "=" * 80)
    print(f" MODEL TRAINING SPEED BENCHMARK ({config_name} config, {epochs} epochs)")
    print("=" * 80)

    # Import model builders
    from llama_models import (
        build_standard_llama, build_ls_llama, build_factored_llama,
        Llama, LLAMA_CONFIGS, LLAMA_LS_CONFIGS,
        _make_linear_factory, _find_common_num_blocks,
    )
    from data import load_wikitext, create_dataloaders

    cfg = LLAMA_CONFIGS[config_name]
    ls_cfg = LLAMA_LS_CONFIGS[config_name]
    ctx_len = cfg["context_len"]

    # Data
    version = "2"  # wikitext-2 for speed benchmarks
    train_tokens, val_tokens, _ = load_wikitext(version, verbose=True)
    train_loader, val_loader, vocab_size = create_dataloaders(
        train_tokens, val_tokens, context_len=ctx_len, batch_size=batch_size,
    )

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()

    class _Wrapper3D(nn.Module):
        """Wraps a 2D layer to handle (B, T, D) inputs."""
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

    def _make_bd_factory(num_blocks, bias=False, stride_perm=False, factored=False, chain_length=2):
        """Factory for BlockDiagLinear3D layers."""
        def factory(in_f, out_f):
            nb = num_blocks
            while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
                nb -= 1
            bd = BlockDiagLinear(in_f, out_f, nb, bias=bias, stride_perm=stride_perm,
                                factored=factored, chain_length=chain_length)
            return _Wrapper3D(bd)
        return factory

    def _make_lsbd_factory(num_blocks, rank, bias=False, stride_perm=False, factored=False, chain_length=2):
        """Factory for LSBlockDiagLinear3D layers."""
        def factory(in_f, out_f):
            nb = num_blocks
            while nb > 1 and (in_f % nb != 0 or out_f % nb != 0):
                nb -= 1
            lsbd = LSBlockDiagLinear.from_uniform_blocks(
                in_f, out_f, nb, rank=rank, bias=bias, stride_perm=stride_perm,
                factored=factored, chain_length=chain_length,
            )
            return _Wrapper3D(lsbd)
        return factory

    nb = ls_cfg["num_blocks"]
    rk = ls_cfg["rank"]

    models_to_test = [
        ("standard", lambda: build_standard_llama(vocab_size=vocab_size, **cfg), False),
        ("standard (compiled)", lambda: build_standard_llama(vocab_size=vocab_size, **cfg), True),
        ("ls (monarch+LR)", lambda: build_ls_llama(
            vocab_size=vocab_size, **cfg, **ls_cfg), False),
        ("blockdiag (contig)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_bd_factory(nb, stride_perm=False)), False),
        ("blockdiag (compiled)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_bd_factory(nb, stride_perm=False)), True),
        ("blockdiag (factored)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_bd_factory(nb, factored=True, chain_length=2)), False),
        ("blockdiag (factored+compiled)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_bd_factory(nb, factored=True, chain_length=2)), True),
        ("ls-blockdiag (contig)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_lsbd_factory(nb, rk, stride_perm=False)), False),
        ("ls-blockdiag (compiled)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_lsbd_factory(nb, rk, stride_perm=False)), True),
        ("ls-blockdiag (factored)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_lsbd_factory(nb, rk, factored=True, chain_length=2)), False),
        ("ls-blockdiag (factored+compiled)", lambda: Llama(
            vocab_size=vocab_size, **cfg,
            linear_factory=_make_lsbd_factory(nb, rk, factored=True, chain_length=2)), True),
    ]

    results = {}

    for name, builder, use_compile in models_to_test:
        print(f"\n--- {name} ---")
        torch.manual_seed(42)
        try:
            model = builder().to(device)
            if use_compile:
                model = torch.compile(model)

            # Count params
            if hasattr(model, 'number_of_trainable_parameters'):
                stored = model.number_of_trainable_parameters()
            else:
                stored = sum(p.numel() for p in model.parameters())
            surface = sum(p.numel() for p in model.parameters())
            print(f"  Stored: {stored:>12,}  Surface: {surface:>12,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            total_steps = epochs * len(train_loader)
            warmup_steps = int(total_steps * 0.05)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            model.train()
            total_tokens = 0
            torch.cuda.synchronize()
            t0 = time.time()

            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                epoch_tokens = 0
                t_epoch = time.time()

                for input_ids, targets in train_loader:
                    input_ids = input_ids.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        logits = model(input_ids)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    batch_tokens = input_ids.numel()
                    epoch_loss += loss.item() * batch_tokens
                    epoch_tokens += batch_tokens

                total_tokens += epoch_tokens
                elapsed_epoch = time.time() - t_epoch
                avg_loss = epoch_loss / epoch_tokens
                tok_s = epoch_tokens / elapsed_epoch
                ppl = math.exp(min(avg_loss, 100))
                print(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                      f"ppl={ppl:.1f}  tok/s={tok_s:,.0f}  ({elapsed_epoch:.1f}s)")

            torch.cuda.synchronize()
            total_time = time.time() - t0
            avg_tok_s = total_tokens / total_time
            print(f"  TOTAL: {avg_tok_s:,.0f} tok/s  ({total_time:.1f}s)")

            results[name] = {
                "stored": stored, "surface": surface,
                "avg_tok_s": avg_tok_s, "total_time": total_time,
            }

            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 100}")
    print(f" SUMMARY")
    print(f"{'=' * 100}")
    std_toks = results.get("standard", {}).get("avg_tok_s", 1)
    print(f"{'Model':30s} {'Stored':>12s} {'Surface':>12s} {'Tok/s':>10s} {'Speedup':>8s} {'Time':>8s}")
    print(f"{'-' * 100}")
    for name, r in results.items():
        speedup = r['avg_tok_s'] / std_toks if std_toks > 0 else 0
        print(f"{name:30s} {r['stored']:>12,} {r['surface']:>12,} "
              f"{r['avg_tok_s']:>10,.0f} {speedup:>7.2f}x {r['total_time']:>7.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="Run full model benchmark")
    parser.add_argument("--layer", action="store_true", help="Run layer microbenchmark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", default="medium")
    args = parser.parse_args()

    if not args.model and not args.layer:
        args.layer = True  # default to layer benchmark

    torch.set_float32_matmul_precision('high')

    if args.layer:
        layer_benchmark()

    if args.model:
        model_benchmark(epochs=args.epochs, batch_size=args.batch_size, config_name=args.config)
