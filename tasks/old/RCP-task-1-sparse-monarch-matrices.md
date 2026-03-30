I want a plan to implement a new sparse linear layer based on the "Monarch matrix" format, replacing the previous implementation that used torch-sparse and torch-scatter. The new implementation should be efficient on GPUs and should not cause installation issues.

The idea of a Monarch matrix is to implement a sparse matrix using a combination of a block-diagonal matrix M and two random permutations P1 and P2.  In particular, I want a sparse linear layer, similar to MaskedLinear, but instead of using a binary mask to determine which weights are active, I want to use the Monarch matrix format.  This will allow us to have a sparse linear layer that is more efficient on GPUs and does not require any additional dependencies.

In particular:

1) The sparse matrix S is computed as S = P1 * M * P2, where M is a block-diagonal matrix with blocks of size (block_size x block_size), and P1 and P2 are random permutations.
2) DO NOT IMPLEMENT THE PERMUTATIONS AS ACTUAL PERMUTATION MATRICES.  Instead, implement them as index arrays that can be used to permute the input and output of the linear layer.
3) As as example, M could be of the form:
   M = [[B1, 0, 0],
        [0, B2, 0],
        [0, 0, B3]]
   where B1, B2, and B3 are dense blocks of size (block_size_i x block_size_j).
4) Of course, the number of blocks and their sizes can be determined by the user.
5) There should a factory function that can generate the Monarch matrix given the desired input and output sizes, block sizes, and sparsity level, similar in flavor to the factory function for MaskedLinear, but specialized for the Monarch matrix format.
6) The forward pass of the sparse linear layer should efficiently compute the output using the Monarch matrix format, without explicitly constructing the full sparse matrix S.  Instead, it should leverage the block structure and the permutations to compute the output efficiently.