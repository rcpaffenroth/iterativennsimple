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
# NOTE: 32768*15//8 does not quite fit onto a 40 GB GPU, so we use 32768*7//4 instead
target_size = 32768*7//4
print(f"Actual size in bytes for three (two input and one output) matrices of size {target_size} x {target_size}: {3 * target_size * target_size * 2 / (1024**3):.2f} GB")
# We can try to allocate a matrix of this size and see if it fits
try:
    start_time = time.time()
    print(f"Allocating two matrices of size {target_size} x {target_size} on GPU...")
    A = torch.randn((target_size, target_size), device=device, dtype=torch.float16)
    B = torch.randn((target_size, target_size), device=device, dtype=torch.float16)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    end_time = time.time()
    allocation_time = end_time - start_time
    print(f"Time taken to allocate matrices: {allocation_time:.4f} seconds")
    print(f"Successfully allocated matrices of size {target_size} x {target_size} on GPU.")
    # print the actual memory used
    actual_memory_used = A.element_size() * A.nelement() + B.element_size() * B.nelement()
    print(f"Actual memory used for two matrices: {actual_memory_used / (1024**3):.2f} GB")
except RuntimeError as e:   
    print(f"Failed to allocate matrices of size {target_size} x {target_size} on GPU: {e}")

# Time the matrix multiplication with gpu synchronization
start_time = time.time()
print(f"Performing matrix multiplication of size {target_size} x {target_size} on GPU...")    
C = torch.matmul(A, B)
torch.cuda.synchronize()  # Ensure all GPU operations are complete
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for matrix multiplication of size {target_size} x {target_size}: {elapsed_time:.4f} seconds")

# Some sanity checks
print(f"Result matrix size: {C.size()}")
print(f"Result matrix dtype: {C.dtype}")
# Make sure that C is different from A and B
print(f"C is A: {torch.equal(C, A)}")
print(f"C is B: {torch.equal(C, B)}")

# clean up GPU memory
del A
del B