from typing import Any

import numpy as np
import torch
try:
    import torch_sparse
except ImportError:
    torch_sparse = None

import logging
logger = logging.getLogger(__name__)
import time

from iterativennsimple.MaskedLinear import MaskedLinear

class SparseLinear(torch.nn.Module):
    """
    This implements a module with two sparse matrices. One is trainable
    and one is not.  This leads to a very general setup where each entry
    can be either:

    0 and not trainable
    0 and trainable
    arbitrary and not trainable
    arbitrary and trainable

    Only the later three cost computation, while the first is "free".

    The weight is defined as

    :math:`A = S_T + S_N`

    Where S_T is sparse and trainable, and S_N is sparse and not
    trainable.

    Once that is done this is identical to a linear layer, and it applies a 
    linear transformation to the incoming data: 

    :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
        dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
        are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
            >>> mask = torch.ones(20, 30)        
            >>> m = MaskedLinear(20, 30, mask)
            >>> s = SparseLinear.from_MaskedLinearExact(m)
            >>> input = torch.randn(128, 20)
            >>> output = s(input)
            >>> print(output.size())
            torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight_0: torch.Tensor
    mask: torch.Tensor
    U: torch.Tensor

    def __init__(self, 
                 sparse_trainable: torch.Tensor, 
                 sparse_not_trainable: torch.tensor = None, 
                 optimized_implementation: bool = True,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()
        # Check everything is as expected
        assert sparse_trainable.is_sparse, "sparse_trainable must be sparse"
        assert sparse_not_trainable is None or sparse_not_trainable.is_sparse, "sparse_not_trainable must be sparse"
        if not sparse_not_trainable is None:    
            assert sparse_trainable.shape == sparse_not_trainable.shape, "shape mismatch"

        # Check to see if we have torch_sparse
        self.optimized_implementation = optimized_implementation
        if torch_sparse is None and self.optimized_implementation:
            logger.warning("torch_sparse is not installed, falling back to slow implementation")
            self.optimized_implementation = False
            
        # Unpack the standard pytorch sparse matrix
        sparse_trainable = sparse_trainable.coalesce()
        self.sparse_trainable_values = torch.nn.parameter.Parameter(torch.tensor(sparse_trainable.values(), **factory_kwargs), 
                                                                    requires_grad=True)
        self.sparse_trainable_indices = sparse_trainable.indices()

        # Note, we don't need to unpack this, since we don't care about the gradients
        if sparse_not_trainable is None:    
            self.sparse_not_trainable = None
        else:
            sparse_not_trainable = sparse_not_trainable.coalesce()
            self.sparse_not_trainable = sparse_not_trainable

        # Save the shape both for the forward pass and for Sequential2D
        self.in_features = sparse_trainable.shape[1]
        self.out_features = sparse_trainable.shape[0]

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self.out_features, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def _getInitializer(initialization_type):
        # We need an initialization function I can call below to initialize
        # the data.
        if initialization_type == 1:
            def initialize(i, j):
                return 1.0
        elif initialization_type[0] == "C":
            val = float(initialization_type[2:])

            def initialize(i, j, val=val):
                return val
        elif initialization_type[0] == "G":
            if len(initialization_type) == 1:
                mu = 0.0
                sigma = 1.0
            else:
                mu = float(initialization_type[2:].split(',')[0])
                sigma = float(initialization_type[2:].split(',')[1])

            def initialize(i, j, mu=mu, sigma=sigma):
                return np.random.randn()*sigma + mu
        elif initialization_type[0] == "U":
            if len(initialization_type) == 1:
                min = -1.0
                max = 1.0
            else:
                min = float(initialization_type[2:].split(',')[0])
                max = float(initialization_type[2:].split(',')[1])

            def initialize(i, j, min=min, max=max):
                return np.random.rand()*(max-min) + min
        else:
            assert False, "Unknown initialization type"
        return initialize

    @staticmethod
    def _getBlock(initialize, block_type, row_size, col_size):
        u = []
        v = []
        vals = []
        if block_type[0] == "D":
            n = min(row_size, col_size)
            for k in range(n):
                u.append(k)
                v.append(k)
                vals.append(initialize(k, k))
        elif block_type[0:3] == "Row":
            n = int(block_type[4:])
            for k in range(row_size):
                for l in range(n):
                    u.append(k)
                    v.append(np.random.randint(0, int(col_size)))
                    vals.append(initialize(u[-1], v[-1]))
        elif block_type[0] == "R":
            n = int(float(block_type[2:])*row_size*col_size)
            for k in range(n):
                u.append(np.random.randint(0, int(row_size)))
                v.append(np.random.randint(0, int(col_size)))
                vals.append(initialize(u[-1], v[-1]))
        elif block_type[0] == "S":
            n = int(block_type[2:])
            for k in range(n):
                u.append(np.random.randint(0, int(row_size)))
                v.append(np.random.randint(0, int(col_size)))
                vals.append(initialize(u[-1], v[-1]))
        else:
            assert False, "Unknown block type"
        return torch.sparse_coo_tensor(torch.stack([torch.tensor(u), torch.tensor(v)]), torch.tensor(vals), (row_size, col_size))

    def to_coo(self):
        """Returns a COO tensor representation of the sparse matrix.

        Returns:
            Tensor: the COO tensor representation of the sparse matrix.
        """
        assert self.sparse_not_trainable is None, "Not implemented when sparse_not_trainable is not None"
        return torch.sparse_coo_tensor(self.sparse_trainable_indices, self.sparse_trainable_values, (self.out_features, self.in_features))

    @staticmethod
    def from_coo(coo,
                 optimized_implementation: bool = True,
                 bias: bool = False, device=None, dtype=None) -> Any:
        """
        Create a sparse matrix from a COO tensor.
        Args:
            coo: A COO tensor
        """
        A = SparseLinear(coo, None, optimized_implementation=optimized_implementation, 
                         bias=bias, device=device, dtype=dtype)
        return A

    @staticmethod
    def from_singleBlock(row_size, col_size, block_type, initialization_type,
                         optimized_implementation: bool = True,
                         bias: bool = True, device=None, dtype=None) -> Any:
        """
        Create a sparse matrix from a single block description.
        Args:
            row_sizes: The number of rows in the matrix
            col_sizes: The number of columns in the matrix
            block_type: A string describing the block type. 
                "D": A block with all off-diagonal entries 0 and only the diagonal 
                        entries initialized. Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
                "R=0.5": A sparse block with randomly placed 0s.
                        The probability an entry is non-zero is 0.5 in this case. However, you might get less 
                        than the requested number of non-zero entries if you happen to draw the same indices twice
                "S=15": A sparse block with randomly placed 0s.
                        This draws, for example, 15 random indices in the block and makes those entries trainable.
                        However, you might get less than 15 since if you happen to draw the same indices twice
                        then they will get coalesced into a single entry
                        (see https://pytorch.org/docs/stable/sparse.html for details)
                        Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
                "Row=15": A sparse block with randomly placed 0s.
                        This draws, for example, 15 random indices *per row* in the block and makes those entries trainable.
                        However, you might get less that 15 entries in each row
                        since if you happen to draw the same indices twice then they will get
                        coalesced into a single entry (see https://pytorch.org/docs/stable/sparse.html for details)
                        Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
            initialization_types: A string description of the initialization for each non-zero, trainable entry
            --------------------------------------------------------
                1 : Initialization each entry with 1
                "C=0.3": Initialize each entry with the value 0.3
                "G": Initialize each entry with draw from Gaussian mu=0.0,sigma=1.0
                "G=0.2,0.7": Initialize each entry with draw from Gaussian mu=0.2,sigma=0.7
                "U": Initialize each entry with draw from Uniform min=-1.0,max=1.0
                "U=-0.5,0.5": Initialize each entry with draw from Uniform min=-0.5,max=0.5                        
        """
        initializer = SparseLinear._getInitializer(initialization_type)
        # Note the strange order of row_size and col_size.  This is because the block is transposed for the left multiplication.
        # block = SparseLinear._getBlock(initializer, block_type, col_size, row_size)
        block = SparseLinear._getBlock(initializer, block_type, row_size, col_size)
        A = SparseLinear(block, None, optimized_implementation=optimized_implementation,
                         bias=bias, device=device, dtype=dtype)
        return A
        
    @staticmethod
    def from_MaskedLinear(M: MaskedLinear, 
                          optimized_implementation: bool = True,
                          device=None, dtype=None) -> Any:
        """
        Create a sparse matrix from a masked matrix.  This is a simpler and faster version of from_MaskedLinearExact.
        It only keeps non-zero entries that have a mask=1 and they are all trainable.
        Args:
            M: the MaskedLinear matrix to copy
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
        """ 
        with torch.no_grad():
            M_matrix = M.weight_0 + M.U * M.mask
            sparse_trainable = (M_matrix*M.mask).to_sparse_coo()
            if not M.bias is None:
                A = SparseLinear(sparse_trainable, None, 
                                 optimized_implementation=optimized_implementation, 
                                 bias=True, device=device, dtype=dtype)
                A.bias[:] = M.bias[:] 
            else:
                A = SparseLinear(sparse_trainable, None, 
                                 optimized_implementation=optimized_implementation, 
                                 bias=False, device=device, dtype=dtype)
        return A


    @staticmethod
    def from_MaskedLinearExact(M: MaskedLinear, 
                               keep_trainable_zeros: bool = True, 
                               optimized_implementation: bool = True, 
                               device=None, dtype=None) -> Any:
        """
        Create a sparse matrix from a masked matrix.  This should be, mathematically, the same linear operator
        as M, including what entries and trainable and which are not.
        Args:
            M: the MaskedLinear matrix to copy
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            keep_trainable_zeros: If set to ``True``, entries in the MaskedLinear whose values are 0, but
                which are marked as trainable will be kept as trainable.  If set to ``False``, all values which
                are 0 in the MaskedLinear will be untrainable in the SparseLinear, even if they were trainable
                in the original MaskedLinear.
        """
        with torch.no_grad():
            M_matrix = M.weight_0 + M.U * M.mask
            # There are 4 cases
            # First the 2 cases for non-trainable entries
            # If M_matrix[i,j]==0 *and* M_matrix[i,j]==0, This is the one case we can discard the entry, but hopefully there are many of these
            # If M_matrix[i,j]!=0 *and* M_matrix[i,j]==0, keep the entry and have it be not trainable
            sparse_not_trainable = (M_matrix*(1.0-M.mask)).to_sparse_coo().coalesce()

            # Second the two cases for trainable entries
            # If M_matrix[i,j]!=0 *and* M_matrix[i,j]==1, keep the non-zero entry and have it be trainable
            # If M_matrix[i,j]==0 *and* M_matrix[i,j]==1, keep the *zero* entry and have it be trainable 
            
            # RCP There is one corner case here.  If M_matrix[i,j] == 0 then the entry is will not
            # be trainable even if M.mask[i,j] == 1 unless we do something fancy.  I.e., 0 is a special value
            # that is used to designate entries that are not kept in the sparse matrix.  However,
            # sometimes we want to keep the 0 entries since they may be trained to be non-zero.
            # This makes things tricky, and the code below tries to get around that.

            sparse_trainable = (M_matrix*M.mask).to_sparse_coo()
            if keep_trainable_zeros:
                # This is a bit of a hack, but the least hacky of the other fixes I thought of.
                # M.mask != 0 everywhere that M is trainable, so sparse_mask is exactly the 
                # the trainable entries in the original MaskedLinear
                sparse_mask = M.mask.to_sparse_coo().coalesce()
                # The next line makes sure that that sparse trainable has an entry for 
                # every trainable entry in M.  This is because sparse_mask is non-zero
                # for all of those entries and therefore has entries for them.
                # So, the sum below will create entries in sparse trainable for all
                # the non-zero entries in sparse_mask, but their values will be 0.
                sparse_trainable += 0.0*sparse_mask
                # It seems strange to do this!  However, the addition creates entries, but
                # does not change the values of the entries already in sparse_trainable.
                # So, we end up with trainable entries whose value happens to be 0!
            sparse_trainable = sparse_trainable.coalesce()

            logger.debug(f'len(sparse_trainable.values())={len(sparse_trainable.values())}')
            logger.debug(f'len(sparse_not_trainable.values())={len(sparse_not_trainable.values())}')
            
            if not M.bias is None:
                A = SparseLinear(sparse_trainable, sparse_not_trainable, 
                                 optimized_implementation=optimized_implementation, 
                                 bias=True, device=device, dtype=dtype)
                A.bias[:] = M.bias[:] 
            else:
                A = SparseLinear(sparse_trainable, sparse_not_trainable, 
                                 optimized_implementation=optimized_implementation,
                                 bias=False, device=device, dtype=dtype)

        return A

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()
        if self.optimized_implementation:
            # FIXME: This is a hack to get around the fact that the indices are not on the same device as the input
            # I am not sure this is the fastest way to do this
            tmp_indices = self.sparse_trainable_indices.to(input.device)
            # NOTE: torch.sparse mm does provide gradients for sparse
            # This is the fast way to do it, but the library is a little funky.  I.e., I am not sure
            # how well it is maintained, though it seems fine as of 12/2024.
            y_before_T = torch_sparse.spmm(tmp_indices,
                                           self.sparse_trainable_values,
                                           self.out_features,
                                           self.in_features,
                                           # NOTE: The operation we want is y = xA^T, to be consistent with
                                           # https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
                                           # However, the library does y = Ax, so we need to transpose the input
                                           input.T)
        else:
            # This is the slow way to do it
            # NOTE: the strange order of the arguments
            # to zip is because of the transpose.
            y_before_T = torch.zeros(self.out_features, input.shape[0], device=input.device, dtype=input.dtype)
            for (i,j), v in zip(self.sparse_trainable_indices.T,
                                self.sparse_trainable_values):
                # NOTE: The operation we want is y = xA^T, to be consistent with
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
                # However, the library does y = Ax, so we need to transpose the input
                y_before_T[i,:] += v*(input.T)[j,:]
        logger.debug(f'sparse mm {time.perf_counter()-start_time:e}')

        start_time = time.perf_counter()
        if not self.sparse_not_trainable is None:
            # FIXME: This is a hack to get around the fact that self.sparse_not_trainable is
            # not on the same device as the input. I am not sure this is the fastest way to do this
            tmp_sparse_not_trainable = self.sparse_not_trainable.to(input.device)
            # NOTE: torch.mm does *not* provide gradients for sparse
            # NOTE: The operation we want is y = xA^T, to be consistent with
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
            # However, the library does y = Ax, so we need to transpose the input
            y_before_T += torch.mm(tmp_sparse_not_trainable, input.T)
        logger.debug(f'sparse not trainable {time.perf_counter()-start_time:e}')

        # NOTE: The operation we want is y = xA^T, to be consistent with
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
        # but what we have is y_before_T = Ax^T, so we need to transpose it to get
        # y = xA^T = (Ax^T)^T = y_before_T.T
        y = y_before_T.T

        start_time = time.perf_counter()
        # Note, this does :math:`y = xA^T + b`.  Beware the transpose.
        if not self.bias is None:
            y = y + self.bias
        logger.debug(f'bias {time.perf_counter()-start_time:e}')
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def number_of_trainable_parameters(self) -> int:
        return len(self.sparse_trainable_values)

    def number_of_not_trainable_parameters(self) -> int:
        tmp = self.sparse_not_trainable.coalesce()
        return len(tmp)
