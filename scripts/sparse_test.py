import torch
from icecream import ic
import timeit

from iterativennsimple.SparseLinear import SparseLinear
from iterativennsimple.MaskedLinear import MaskedLinear

# import the linear layer from torch.nn
from torch.nn import Linear

# We manually set the seed to ensure that the results are reproducible
torch.manual_seed(0)

# Test if cuda is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Select the number of threads used for parallelization
# This is to make sure that the results are consistent, since you may
# run this on a computer with a different number of cores
num_threads = 1
torch.set_num_threads(num_threads)

num_samples = 5
num_in_features = 2000
num_out_features = 1000


# Create a sparse linear layer
sparse_model = SparseLinear.from_singleBlock(row_size=num_out_features,
                                             col_size=num_in_features,
                                             block_type='R=0.01',
                                             initialization_type='G=0.0,1.0',
                                             optimized_implementation=True,
                                             transpose=True,
                                             bias=False,
                                             dtype=torch.float32,
                                             device=device)

# Print the number of trainable parameters and the min and max values
ic(num_out_features * num_in_features)
ic(sparse_model.number_of_trainable_parameters())
ic(sparse_model.number_of_trainable_parameters()/(num_out_features * num_in_features))

ic(sparse_model.sparse_trainable_indices.shape)
ic(torch.abs(sparse_model.sparse_trainable_values).max())
ic(torch.abs(sparse_model.sparse_trainable_values).min())

ic(sparse_model.sparse_trainable_indices[0,:].max())
ic(sparse_model.sparse_trainable_indices[0,:].min())
ic(sparse_model.sparse_trainable_indices[1,:].max())
ic(sparse_model.sparse_trainable_indices[1,:].min())

sparse_weights = torch.sparse_coo_tensor(sparse_model.sparse_trainable_indices, 
                                         sparse_model.sparse_trainable_values, 
                                         size=(num_out_features, num_in_features),
                                         dtype=torch.float32,
                                         device=device)

ic(sparse_weights._indices().shape[1])
ic(torch.abs(sparse_weights._values()).max())
ic(torch.abs(sparse_weights._values()).min())

dense_weights = sparse_weights.to_dense().to(device)

dense_model = Linear(in_features=num_in_features, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)

x = torch.randn((num_samples, num_in_features),
                dtype=torch.float32,
                device=device)

# Create a linear layer with the same weights as the sparse model
ic(dense_model.weight.shape)
with torch.no_grad():
    dense_model.weight.data = dense_weights
ic(dense_model.weight.shape)

# Run the forward pass for both models

raw_dense_output = x @ (dense_weights.T)
raw_sparse_output = x @ (sparse_weights.T)
dense_output = dense_model(x)
sparse_output = sparse_model(x)

ic(raw_dense_output.shape)
ic(raw_sparse_output.shape)
ic(dense_output.shape)
ic(sparse_output.shape)

ic(raw_dense_output[:2,:2])
ic(raw_sparse_output[:2,:2])
ic(dense_output[:2,:2])
ic(sparse_output[:2,:2])

raw_dense_time = timeit.timeit('x @ (dense_weights.T)', globals=globals(), number=10)
raw_sparse_time = timeit.timeit('x @ (sparse_weights.T)', globals=globals(), number=10)
dense_time = timeit.timeit('dense_model(x)', globals=globals(), number=10)
sparse_time = timeit.timeit('sparse_model(x)', globals=globals(), number=10)

ic(raw_dense_time)
ic(raw_sparse_time)
ic(dense_time)
ic(sparse_time)