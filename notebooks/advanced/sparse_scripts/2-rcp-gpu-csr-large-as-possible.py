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
# TODO Add these imports and test MaskeLinear and SparseLinear
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.SparseLinear import SparseLinear

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
csr_density = 0.2  # 20% density

csr_size = int(rows_csr * cols_csr * csr_density)
print(f"CSR matrix will have {csr_size} non-zero elements.")
input_size = rows_dense * cols_dense
print(f"Dense matrix will have {input_size} elements.")
target_size = rows_csr * cols_dense 
print(f"Target output matrix will have {target_size} elements.")
print(f"Actual size in bytes for all three (CSR matrix, dense input, and dense output) matrices: {(csr_size + input_size + target_size) * 2 / (1024**3):.2f} GB")
# We can try to allocate a matrix of this size and see if it fits

FILL IN WITH CORRECT CSR CODE

# Some sanity checks
print(f"Result matrix size: {C.size()}")
print(f"Result matrix dtype: {C.dtype}")
# Make sure that C is different from A and B
print(f"C is A: {torch.equal(C, A)}")
print(f"C is B: {torch.equal(C, B)}")

# clean up GPU memory
del A
del B