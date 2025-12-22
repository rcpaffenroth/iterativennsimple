# GPU Matrix Multiplication Benchmarks

This directory contains scripts for benchmarking large-scale matrix multiplication on GPUs, comparing different matrix formats and sparsity patterns.

## Overview

These scripts test the limits of GPU memory and computational performance by allocating and multiplying matrices approaching ~1 billion parameters (2^30). Each script targets different matrix representations to understand their memory footprint and computational efficiency.

**Why .py files instead of notebooks?** These scripts intentionally push GPU memory limits and may trigger out-of-memory errors. Running them as standalone Python scripts prevents Jupyter kernel crashes during experimentation.

## Scripts

### 1. Dense Matrix Multiplication (`1-rcp-gpu-dense-large-as-possible.py`)

Baseline benchmark using standard dense matrix multiplication.

**Key Features:**
- Tests maximum size of square dense matrices that fit in GPU memory
- Uses float16 precision to maximize matrix size
- Targets ~1B parameters: 32768 × 32768 matrices (adjusted for GPU capacity)
- Measures allocation time and matrix multiplication performance
- Provides reference performance for comparison with sparse methods

**Typical Output:**
- GPU memory usage for dense storage
- Time for allocation and `torch.matmul(A, B)`
- Useful for understanding dense baseline before exploring sparse optimizations

**Example Timing (32768×32768 on RTX 4090 24GB):**

```text
Time taken to allocate matrices: 0.1253 seconds
Time taken for matrix multiplication: 0.4065 seconds
```

### 2. Sparse CSR Matrix Multiplication (`2-rcp-gpu-csr-large-as-possible.py`)

Tests sparse matrix multiplication using Compressed Sparse Row (CSR) format.

**Key Features:**
- Creates random sparse matrices in CSR format
- Configurable sparsity density (default: 2% non-zero elements)
- Converts COO → CSR format for efficient GPU sparse operations
- Measures sparse @ dense matrix multiplication performance
- Reports memory usage for sparse storage (values + indices)

**Memory Advantages:**
- CSR stores only non-zero elements plus row/column indices
- At 2% density: ~50× memory reduction vs dense
- Enables much larger effective matrix dimensions

**Performance Caveat:**
- ⚠️ **CSR multiplication is significantly slower than dense on GPUs**
- At 10% density: ~3-5× slower than dense operations
- PyTorch's CSR implementation lacks the optimizations of dense GEMM
- Memory savings come at substantial computational cost
- Only beneficial when extreme sparsity (<<1%) enables otherwise impossible computations

**Example Timing (32768×32768 on RTX 4090 24GB):**

```text
Density: 10% - Time for sparse CSR @ dense: 2.1881 seconds
(Dense baseline: 0.4065 seconds - CSR is 5.4× SLOWER)
```

**Use Cases:**
- Sparse neural network layers
- Pruned models
- Attention mechanisms with sparsity patterns

### 3. Block-Diagonal (Monarch) Matrix Multiplication (`3-rcp-gpu-monarch-large-as-possible.py`)

Implements block-diagonal matrix structure similar to Monarch matrices for memory-efficient operations.

**Key Features:**
- Decomposes matrices into block-diagonal structure
- Configurable number of blocks (default: 4)
- Three implementations:
  - **Uncompiled**: Native PyTorch with Python loops
  - **Compiled**: Using `torch.compile()` for optimization
  - **Stacked**: Batch matrix multiply via `torch.bmm()`
- Warm-up phase to separate compilation overhead from runtime
- Comparative benchmarking across all three approaches

**Memory Advantages:**
- For N blocks: stores only (1/N^2) of full dense matrix
- With 4 blocks on 32768×32768: 16× memory reduction
- Maintains expressiveness for certain structured operations

**Performance Insights:**
- `torch.compile()` adds overhead for small computations
- Benefit emerges with multiple blocks (num_blocks > 1)
- Stacked version uses optimized batch operations, typically fastest
- In-place operations degrade compilation performance

**Example Timing (32768×32768 with 4 blocks on RTX 4090 24GB):**

```text
Time taken (uncompiled): 0.1540 seconds
Time taken (compiled): 0.1193 seconds
Speedup (compiled vs uncompiled): 1.29x
```

## Performance Summary

Comparison of multiplication times for 32768×32768 matrices (RTX 4090 24GB, float16):

| Method | Time (seconds) | Memory Reduction | Speed vs Dense |
| --- | --- | --- | --- |
| Dense (baseline) | 0.4065 | 1.0× | 1.0× |
| CSR (10% density) | 2.1881 | ~10× | 5.4× **slower** |
| Block-Diagonal (4 blocks, uncompiled) | 0.1540 | 16× | 2.6× faster |
| Block-Diagonal (4 blocks, compiled) | 0.1193 | 16× | 3.4× faster |

**Key Takeaways:**

- **CSR at 10% density is 5.4× slower than dense** - PyTorch's CSR implementation lacks GPU optimization
- Block-diagonal structures offer both memory savings (16× reduction) and speed improvements (2-3×)
- CSR only beneficial when memory constraints prevent dense operations entirely
- Structured sparsity (block-diagonal) significantly outperforms random sparsity (CSR)

## Usage

Run as standalone Python scripts:

```bash
python 1-rcp-gpu-dense-large-as-possible.py
python 2-rcp-gpu-csr-large-as-possible.py
python 3-rcp-gpu-monarch-large-as-possible.py
```

Or execute interactively in IPython/Jupyter:

```python
%run 1-rcp-gpu-dense-large-as-possible.py
```

## GPU Memory Reference

Approximate matrix sizes that fit various GPU configurations (float16, accounting for overhead):

| GPU | Memory | Max Dense Matrix | CSR (2% density) | Block-Diagonal (4 blocks) |
|-----|--------|------------------|------------------|---------------------------|
| RTX 3060 (laptop) | 6 GB | ~17k × 17k | ~39k × 39k | ~34k × 34k |
| RTX 3060 | 12 GB | ~25k × 25k | ~55k × 55k | ~50k × 50k |
| RTX 4090 | 24 GB | ~35k × 35k | ~78k × 78k | ~70k × 70k |
| A100 (40 GB) | 40 GB | ~46k × 46k | ~100k × 100k | ~92k × 92k |
| A100 (80 GB) | 80 GB | ~65k × 65k | ~145k × 145k | ~130k × 130k |

## Configuration

Key parameters to adjust in each script:

- `target_size`: Matrix dimensions (default: 32768)
- `dtype`: Precision (default: `torch.float16`)
- `csr_density`: Sparsity level for CSR (script 2)
- `num_blocks`: Number of diagonal blocks (script 3)

## Requirements

```bash
torch >= 2.0  # torch.compile() requires PyTorch 2.0+
```

For sparse CSR operations, ensure CUDA-enabled PyTorch build.

## Research Context

These benchmarks inform design decisions for:
- Sparse neural network architectures
- Memory-efficient training of large models
- Structured matrix approximations (e.g., Monarch, Butterfly)
- Trade-offs between memory usage, computational speed, and expressiveness
