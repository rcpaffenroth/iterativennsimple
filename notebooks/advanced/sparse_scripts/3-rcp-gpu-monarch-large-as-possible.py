# What is the largest matrix that fits on my GPU and how fast can I multiply it?

# NOTE: this "notebook" is actually a .py file can either be run iteractively in a Jupyter environment
# or as a standard python script. 
# I do it this way since I will purposefully running out of memory on my GPU,
# and I don't want to keep crashing my Jupyter kernel when that happens.

# In this notebook we analyze the the maximum size of a square matrix that can
# fit into the memory of a GPU, and measure the time it takes to perform matrix
# multiplication on that matrix.
# We also look at how the performance scales with matrix size and
# compare to theoretical expectations and monarch matrices.

import torch
import warnings
import time
warnings.filterwarnings("ignore")

# Some rough starting points:

# - RTX 3060(laptop): 6 GB
# - RTX 3060: 12 GB
# - RTX 4090: 24 GB
# - A100: 40 GB or 80 GB

# create a tensor on the GPU to test memory limits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")    
max_memory = torch.cuda.get_device_properties(0).total_memory
print(f"Total GPU memory: {max_memory / (1024**3):.2f} GB")
# Also print the free memory
free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
print(f"Free GPU memory: {free_memory / (1024**3):.2f} GB")
# Estimate the maximum size of a square matrix that can fit into GPU memory
# Each float16 takes 2 bytes, so we can fit max_memory / 2 bytes
max_matrix_size = int((max_memory / 2) ** 0.5)
print(f"Estimated maximum square matrix size: {max_matrix_size} x {max_matrix_size}")

# Let's focus on a billion parameters for now.  This is approximately 2^30 parameters.
# For a square matrix, this is sqrt(2^30) = 2^15 = 32768
print(f"Target square matrix size for ~1B parameters: 32768 x 32768")
# NOTE: 32768*15//8 does not quite fit onto a 24 GB GPU, so we use 32768*7//4 as our largest size 
target_size = 32768
print(f"Actual size in bytes for two (one input and one output) matrices of size {target_size} x {target_size}: {2 * target_size * target_size * 2 / (1024**3):.2f} GB")
num_blocks = 4
block_size = target_size // num_blocks
print(f"Using {num_blocks} blocks of size {block_size} x {block_size} for monarch matrix multiplication.")
print(f"Actual size in bytes for monarch matrix (blocks only) {target_size} x {target_size}: {(num_blocks * block_size * block_size) * 2 / (1024**3):.2f} GB")

def block_diagonal_matrix_multiply(A, B, P1, P2):
    """Multiply a list of square blocks of A and a dense B using block diagonal multiplication.
    A: list of square 2D tensors, each of shape (block_size_i, block_size_i)
    B: 2D tensor of shape (N, M)
    block_size_i: size of each block in A
    Returns: 2D tensor of shape (N, M)
    1. For each block in A, multiply it with the corresponding rows in B
    2. Accumulate the results into the output matrix
    3. Return the output matrix
    """
    # assert that the sum of block sizes equals the number of rows in B
    total_rows = sum(block.size(0) for block in A)
    assert total_rows == B.size(0), "Sum of block sizes in A must equal number of rows in B"
    # Create output tensor instead of modifying B in-place
    result = B[P1, :].clone()
    for i, block in enumerate(A):
        start_row = i * block.size(0)
        end_row = start_row + block.size(0)
        result[start_row:end_row, :] = block @ B[start_row:end_row, :]
    return result[P2]

block_diagonal_matrix_multiply_opt = torch.compile(block_diagonal_matrix_multiply)

# We can try to allocate a matrix of this size and see if it fits
try:
    start_time = time.time()
    print(f"Allocating list of {num_blocks} matrices of size {block_size} x {block_size} on GPU...")
    A = [torch.randn((block_size, block_size), device=device, dtype=torch.float16) for _ in range(num_blocks)]
    print(f"Allocating dense matrix of size {target_size} x {target_size} on GPU...")
    B = torch.randn((target_size, target_size), device=device, dtype=torch.float16)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    end_time = time.time()
    allocation_time = end_time - start_time
    print(f"Time taken to allocate matrices: {allocation_time:.4f} seconds")
    print(f"Successfully allocated matrices of size {target_size} x {target_size} on GPU.")
    # print the actual memory used
    actual_memory_used = A[0].element_size() * A[0].nelement() * num_blocks + B.element_size() * B.nelement()
    print(f"Actual memory used for two matrices: {actual_memory_used / (1024**3):.2f} GB")
except RuntimeError as e:   
    print(f"Failed to allocate matrices of size {target_size} x {target_size} on GPU: {e}")

P1 = torch.randperm(target_size, device=device)
P2 = torch.randperm(target_size, device=device)

# Warm up compiled function (first run includes compilation time)
print("\nWarming up compiled function...")
_ = block_diagonal_matrix_multiply_opt(A, B.clone(), P1, P2)
torch.cuda.synchronize()
print("Warmup complete.")

# Benchmark uncompiled version
print("\n--- Benchmarking uncompiled version ---")
B_copy = B.clone()
start_time = time.time()
C_uncompiled = block_diagonal_matrix_multiply(A, B_copy, P1, P2)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time_uncompiled = end_time - start_time
print(f"Time taken (uncompiled): {elapsed_time_uncompiled:.4f} seconds")

# Benchmark compiled version
print("\n--- Benchmarking compiled version ---")
B_copy = B.clone()
start_time = time.time()
C_compiled = block_diagonal_matrix_multiply_opt(A, B_copy, P1, P2)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time_compiled = end_time - start_time
print(f"Time taken (compiled): {elapsed_time_compiled:.4f} seconds")

# Compare results
print(f"\nSpeedup: {elapsed_time_uncompiled / elapsed_time_compiled:.2f}x")

