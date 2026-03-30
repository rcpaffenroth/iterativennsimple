# Task 1.5 — GPU Performance Validation

## Objective

Verify that `MonarchLinear` is significantly faster than `SparseLinear` (and competitive with `MaskedLinear`) on GPU for large layers, validating the motivation for the Monarch format.

---

## Approach

Create a notebook or script that benchmarks `forward` and `backward` wall-clock time for the three layer types at various sizes.

### Suggested File

`notebooks/advanced/8-rcp-monarch-performance.ipynb` (or a script in `notebooks/advanced/sparse_scripts/`).

### Benchmark Parameters

| Parameter | Values |
|-----------|--------|
| `in_features` = `out_features` | 256, 512, 1024, 2048, 4096 |
| `batch_size` | 128, 512 |
| Sparsity (for `SparseLinear` and `MonarchLinear`) | ~75% (i.e., `num_blocks = 4` for Monarch) |
| Device | `cuda` (skip if unavailable) |
| Warmup iterations | 10 |
| Timed iterations | 100 |

### What to Measure

1. **Forward time** — `layer(x)`, averaged over timed iterations.
2. **Backward time** — `loss.backward()`, averaged.
3. **Total (forward + backward)** — most relevant for training.
4. **Memory** — `torch.cuda.max_memory_allocated()` before and after.

### Expected Outcomes

| Layer | Forward | Why |
|-------|---------|-----|
| `MaskedLinear` | Fast (dense matmul) | Uses `F.linear`, fully dense GEMM on GPU |
| `SparseLinear` | Slow | Python loop over COO entries, no GPU parallelism |
| `MonarchLinear` | Fast | Indexing + loop of small dense matmuls (or `bmm`) |

For *large* layers (≥1024), `MonarchLinear` should match or approach `MaskedLinear` speed at a fraction of the parameters, and should be orders of magnitude faster than `SparseLinear`.

### Plotting

- Plot time vs. `in_features` for each layer type (log-log scale).
- Plot speedup of `MonarchLinear` over `SparseLinear`.
- Optionally plot memory usage.

---

## Correctness Sanity Check

Before benchmarking, verify that all three layer types produce the same output for a small test case (up to the sparsity pattern difference). Specifically:

1. Create a `MonarchLinear` with known seed.
2. Extract its dense matrix via `to_dense()`.
3. Create a `MaskedLinear` initialized with that dense matrix.
4. Forward the same input through both and assert `allclose`.

This does not need to be part of the benchmark loop — just a one-time check at the top of the notebook.

---

## Notes

- This task is lower priority than Tasks 1.1–1.4. It validates the design but is not required for the initial PR.
- The existing scripts in `notebooks/advanced/sparse_scripts/` (`1-rcp-gpu-dense-large-as-possible.py`, `3-rcp-gpu-monarch-large-as-possible.py`) provide a template for the benchmarking style.
- The `torch.bmm` fast path for uniform blocks should be tested separately to confirm it provides additional speedup over the loop.
