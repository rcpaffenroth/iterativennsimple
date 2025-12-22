# What is the largest CSR matrix that fits on my GPU and how fast can I multiply it?

# NOTE: this "notebook" is actually a .py file can either be run iteractively in a Jupyter environment
# or as a standard python script. 
# I do it this way since I will purposefully running out of memory on my GPU, 
# and I don't want to keep crashing my Jupyter kernel when that happens.

# In this notebook we analyze the the maximum size of a rectangular CSR matrix that can 
# fit into the memory of a GPU, and measure the time it takes to perform matrix 
# multiplication between that CSR matrix and a dense matrix.
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

rows_csr = 32768
cols_csr = rows_dense = 32768
cols_dense = 32768
csr_density = 0.02  # 99% density

csr_size = int(rows_csr * cols_csr * csr_density)
print(f"CSR matrix will have {csr_size} non-zero elements.")
input_size = rows_dense * cols_dense
print(f"Dense matrix will have {input_size} elements.")
target_size = rows_csr * cols_dense 
print(f"Target output matrix will have {target_size} elements.")
print(f"Actual size in bytes for all three (CSR matrix, dense input, and dense output) matrices: {(csr_size + input_size + target_size) * 2 / (1024**3):.2f} GB")
# We can try to allocate a matrix of this size and see if it fits

try:
    start_time = time.time()
    print(f"Allocating sparse CSR matrix of size {rows_csr} x {cols_csr} with {csr_density*100:.1f}% density...")
    # Create random sparse matrix in COO format first, then convert to CSR
    num_nonzero = csr_size
    row_indices = torch.randint(0, rows_csr, (num_nonzero,), device=device)
    col_indices = torch.randint(0, cols_csr, (num_nonzero,), device=device)
    values = torch.randn(num_nonzero, device=device, dtype=torch.float16)
    
    # Create COO sparse tensor and convert to CSR format
    indices = torch.stack([row_indices, col_indices])
    A_coo = torch.sparse_coo_tensor(indices, values, (rows_csr, cols_csr), device=device, dtype=torch.float16)
    A_coo = A_coo.coalesce()  # Remove duplicate indices
    A = A_coo.to_sparse_csr()
    
    print(f"Allocating dense matrix of size {rows_dense} x {cols_dense}...")
    B = torch.randn((rows_dense, cols_dense), device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    end_time = time.time()
    allocation_time = end_time - start_time
    print(f"Time taken to allocate matrices: {allocation_time:.4f} seconds")
    print(f"Successfully allocated sparse CSR matrix and dense matrix on GPU.")
    
    # Print actual memory used
    actual_memory_sparse = A.values().element_size() * A.values().nelement() + A.crow_indices().element_size() * A.crow_indices().nelement() + A.col_indices().element_size() * A.col_indices().nelement()
    actual_memory_dense = B.element_size() * B.nelement()
    print(f"Actual memory used for sparse CSR matrix: {actual_memory_sparse / (1024**3):.2f} GB")
    print(f"Actual memory used for dense matrix: {actual_memory_dense / (1024**3):.2f} GB")
    print(f"Total memory used: {(actual_memory_sparse + actual_memory_dense) / (1024**3):.2f} GB")
except RuntimeError as e:
    print(f"Failed to allocate matrices: {e}")

# Warm up compiled function (first run includes compilation time)
print("\nWarming up function...")
_ = torch.matmul(A, B)
torch.cuda.synchronize()
print("Warmup complete.")

# Time the sparse-dense matrix multiplication with GPU synchronization
start_time = time.time()
print(f"Performing sparse CSR @ dense matrix multiplication...")
C = torch.matmul(A, B)
torch.cuda.synchronize()  # Ensure all GPU operations are complete
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for sparse CSR @ dense multiplication: {elapsed_time:.4f} seconds")

# clean up GPU memory
del A
del B
del C