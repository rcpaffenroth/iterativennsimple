import math
import time
import logging
from typing import Any

import numpy as np
import torch

from iterativennsimple.bmatrix import bmatrix

logger = logging.getLogger(__name__)

class MaskedLinear(torch.nn.Module):
    """
    This implements a module with a masked weight.   
    The weight is defined as

    :math:`A = U \cdot \Omega + W_0`

    Where W_0 is the initial weight, \Omega is the mask, 
    and U is the *trainable* update.  In particular, :math:`U`
    is the only learnable parameter, and :math:`\Omega` and :math:`W_0` are
    fixed.

    Once that is done this is identical to a linear layer, and it applies a 
    linear transformation to the incoming data: 

    :math:`y = xA^T + b`

    Note: This is not particularily efficient, but it is a good way to get 
    started and can be used as a regression test for the fancier implementations.
    I.e., it is the most flexible implementation and the faster implementations
    should give the same answer as this one.  Also, all of the normal pytorch
    optimizers can be used with this implementation.

    Args:
        in_features (int): size of each input sample
        out_features(int): size of each output sample
        bias (bool, optional): Include a bias term. Defaults to True.
        device (optional): The device to use (e.g. CPU vs GPU). Defaults to None.
        dtype (optional): The data type to use (e.g. float32 vs float64). Defaults to None.
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
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight_0: torch.Tensor
    mask: torch.Tensor
    U: torch.Tensor

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The initial weights do not require gradients.
        self.weight_0 = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), requires_grad=False, **factory_kwargs), requires_grad=False)

        # The update does require gradients and we make that explicit here.
        self.U = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), requires_grad=True, **factory_kwargs), requires_grad=True)

        # The mask
        self.mask = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), requires_grad=False, **factory_kwargs), requires_grad=False)

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, requires_grad=True, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        # Note, self.mask gets set to 1 and self.U get set to 0 in here.
        self.reset_parameters()

    @staticmethod
    def _getBlock(initialize, block_type, out_features, in_features):
        if block_type == 0:
            block = torch.zeros((out_features, in_features))
        elif block_type == "W":
            block = np.fromfunction(np.vectorize(initialize), 
                                    (out_features, in_features))
            block = torch.tensor(block)
        elif block_type[0] == "D":
            block = torch.zeros((out_features, in_features))
            n = torch.min(out_features, in_features)
            for k in range(n):
                block[k, k] = initialize(k, k)
        elif block_type[0:3] == "Row":
            n = int(block_type[4:])
            block = torch.zeros((out_features, in_features))
            for k in range(out_features):
                for l in range(n):
                    v = np.random.randint(0, int(in_features))
                    block[k, v] = initialize(k, v)
        elif block_type[0] == "R":
            def func(k, l, p=float(block_type[2:])):
                if np.random.rand() < p:
                    return initialize(k, l)
                else:
                    return 0.0
            block = np.fromfunction(np.vectorize(func), 
                                    (out_features, in_features))
            block = torch.tensor(block)
        elif block_type[0] == "S":
            n = int(block_type[2:])
            block = torch.zeros((out_features, in_features))
            for k in range(n):
                u = np.random.randint(0, int(out_features))
                v = np.random.randint(0, int(in_features))
                block[u, v] = initialize(u, v)
        else:
            assert False, "Unknown block type"
        return block

    @staticmethod
    def _getInitializer(initialization_type):
        # We need an initialization function I can call below to initialize
        # the data.
        if torch.is_tensor(initialization_type):
            def initialize(i, j, W=initialization_type):
                return W[int(i), int(j)]
        elif initialization_type == 0:
            def initialize(i, j):
                return 0.0
        elif initialization_type == 1:
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
            assert False, f"Unknown initialization type {initialization_type}"
        return initialize

    @staticmethod
    def from_coo(coo, check_mask: bool = False, bias: bool = False, device=None, dtype=None) -> Any:
        """
        Create a MaskedLinear from a sparse COO matrix.

        Args:
            coo (dict): the COO matrix
            bias (bool, optional): Include a bias term. Defaults to True.
            device (optional): The device to use (e.g. CPU vs GPU). Defaults to None.
            dtype (optional): The data type to use (e.g. float32 vs float64). Defaults to None.

        Returns:
            MaskedLinear: a MaskedLinear object from the configuration
        """
        A = MaskedLinear(in_features=coo.shape[1], 
                         out_features=coo.shape[0],
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            # The weights are just the COO matrix in dense form
            A.weight_0[:, :] = coo.to_dense()
            # The mask is 1 where the COO matrix is non-zero
            A.mask[:, :] = (A.weight_0[:, :] != 0)
            if check_mask:
                # In some cases we want to check that the mask is correct in that there might
                # be a zero value in the COO that is not in the mask and therefore not trainable.  
                # This is a sanity check which is slow, so we don't want to do it all the time.
                for idx in range(coo.indices()):
                    i = idx[0]
                    j = idx[1]
                    assert A.mask[i, j] != 0, f"mask is not correct at {i},{j}"
        return A

    @staticmethod
    def from_config(cfg):
        """
        A wrapper for the from_description method.

        Args:
            cfg (dict): a configuration dictionary

        Returns:
            MaskedLinear: a MaskedLinear object from the configuration
        """
        return MaskedLinear.from_description(out_features_sizes=cfg['out_features_sizes'],
                                            in_features_sizes=cfg['in_features_sizes'],
                                            block_types=cfg['block_types'],
                                            initialization_types=cfg['initialization_types'],
                                            trainable=cfg['trainable'],
                                            bias=cfg['bias'])

    @staticmethod
    def from_description(out_features_sizes, in_features_sizes,
                         block_types,
                         initialization_types,
                         trainable,
                         bias: bool = True, device=None, dtype=None) -> Any:
        """
        Create a block sparse matrix.
        Args:
            rows_sizes (list of int): The number of rows in each block
            cols_sizes (list of int): The number of columns in each block

            block_types (2D-array of number or string): A 2D-array that describe each block.
                Each dict has a key named "type" and perhaps other keys to describe the block
            --------------------------------------------------------
            0  : A block of zeros.
            "W": A dense block with all entries initialized.
            "D": A block with all off-diagonal entries 0 and only the diagonal 
                    entries initialized. Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
            "R=0.5": A sparse block with randomly placed 0s.
                    The probability an entry is non-zero is 0.5 in this case.
                    Note, this can be slow since a random number is drawn for each entry in the block. Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
            "S=15": A sparse block with randomly placed 0s.
                    This draws, for example, 15 random indices in the block and makes those entries trainable.
                    However, you might get less that 15 since if you happen to draw the same indices twice
                    then they will get coalesced into a single entry
                    (see https://pytorch.org/docs/stable/sparse.html for details)
                    Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.
            "Row=15": A sparse block with randomly placed 0s.
                    This draws, for example, 15 random indices *per row* in the block and makes those entries trainable.
                    However, you might get less that 15 entries in each row
                    since if you happen to draw the same indices twice then they will get
                    coalesced into a single entry (see https://pytorch.org/docs/stable/sparse.html for details)
                    Note either all entries are trainable or all entries are not trainable.  (i.e., 0 entries are trainable if all entries are trainable).  This is a limitation of the current implementation.

            initialization_types (2D-array of number of string): A 2D-array of dicts that describes
                the initialization of the trainable parameters in each block.
                the initialization of the trainable parameters in each block.
                Each dict has a key named "type" and perhaps other keys to describe the block.
            --------------------------------------------------------
            0 : Initialization each entry with 0
            1 : Initialization each entry with 1
            "C=0.3": Initialize each entry with the value 0.3
            "G": Initialize each entry with draw from Gaussian mu=0.0,sigma=1.0
            "G=0.2,0.7": Initialize each entry with draw from Gaussian mu=0.2,sigma=0.7
            "U": Initialize each entry with draw from Uniform min=-1.0,max=1.0
            "U=-0.5,0.5": Initialize each entry with draw from Uniform min=-0.5,max=0.5
            T : Initialize with the given tensor

            trainable (2D-array of boolean):  A 2D-array of bools that defines if a block is
                trainable.
            --------------------------------------------------------
            True:  block is trainable
            False:  block is not trainable
            'non-zero': the non-zero entries are trainable
        """
        # Take care of the case where I want just a single block
        if (type(out_features_sizes) is int) and (type(in_features_sizes) is int):
            out_features_sizes = [out_features_sizes]
            in_features_sizes = [in_features_sizes]
            block_types = [[block_types]]
            initialization_types = [[initialization_types]]
            trainable = [[trainable]]

        out_features_sizes = torch.tensor(out_features_sizes)
        in_features_sizes = torch.tensor(in_features_sizes)

        A = MaskedLinear(in_features=int(torch.sum(in_features_sizes)), 
                         out_features=int(torch.sum(out_features_sizes)),
                         bias=bias, device=device, dtype=dtype)

        weight_0 = [[None]*len(in_features_sizes) for i in range(len(out_features_sizes))]
        mask = [[None]*len(in_features_sizes) for i in range(len(out_features_sizes))]

        with torch.no_grad():
            for current_row, out_features in enumerate(out_features_sizes):
                for current_col, in_features in enumerate(in_features_sizes):
                    block_type = block_types[current_row][current_col]
                    train = trainable[current_row][current_col]
                    initialization_type = initialization_types[current_row][current_col]
                    if block_type == 0:
                        assert not train, "0 block should not be trainable"
                        assert initialization_type == 0, "0 block should be initialized to 0"

                    initialize = MaskedLinear._getInitializer(initialization_type)

                    # The implementations of the various block types
                    block = MaskedLinear._getBlock(initialize, block_type,
                                                   out_features, in_features)
                    weight_0[current_row][current_col] = block
                    if train==True:
                        mask[current_row][current_col] = torch.ones(out_features, in_features)
                    elif train==False:
                        mask[current_row][current_col] = torch.zeros(out_features, in_features)
                    elif train=='non-zero':
                        mask[current_row][current_col] = torch.zeros(out_features, in_features)
                        mask[current_row][current_col][block != 0.0] = 1
                    else:
                        assert False, f"unknow train type {train}"
            A.weight_0[:, :] = bmatrix(weight_0)
            A.mask[:, :] = bmatrix(mask) 
    
            return A

    @staticmethod
    def from_MLP(sizes, bias: bool = True, device=None, dtype=None) -> Any:
        """
        Create a MaskedLinear from a MLP.  This is useful for testing and comparing.

        Args:
            sizes: A list of sizes for the layers of the MLP
            bias (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Each entry in sizes give rise to a kxk block on the diagonal of the matrix, the the total size
        # is the sum of the entries in sizes.
        out_features_sizes = sizes
        in_features_sizes = sizes

        A = MaskedLinear(in_features=int(torch.sum(torch.Tensor(in_features_sizes))),
                         out_features=int(torch.sum(torch.Tensor(out_features_sizes))),
                         bias=bias, device=device, dtype=dtype)

        weight_0 = [[None]*len(in_features_sizes) for i in range(len(out_features_sizes))]
        mask = [[None]*len(in_features_sizes) for i in range(len(out_features_sizes))]
        with torch.no_grad():
            for i,rows in enumerate(out_features_sizes):
                for j,cols in enumerate(in_features_sizes):
                    # The MLP is right below the diagonal, so we need to offset the index
                    if i == j+1:
                        # Only these blocks are initialized and trained
                        weight_0[i][j] = torch.zeros(rows, cols)
                        torch.nn.init.kaiming_uniform_(weight_0[i][j], a=math.sqrt(5))
                        mask[i][j] = torch.ones(rows, cols)
                    ###########################
                    else:
                        weight_0[i][j] = torch.zeros(rows, cols)
                        mask[i][j] = torch.zeros(rows, cols)

            # The requires_grad=False is important, otherwise the gradient will be computed when we don't want it to.
            A.weight_0[:, :] = bmatrix(weight_0)
            A.mask[:, :] = bmatrix(mask) 
    
            return A
    
    @staticmethod
    def from_optimal_linear(X, Y, bias: bool = False, device=None, dtype=None) -> Any:
        """
        This function initializes the initial weights "weight_0" and the update "U" 
        to be the optimal least squares solution to the linear regression problem
        which maps the input X to the output Y.  This is done by computing
    
        :math:`
            XW=Y\\
            X^T X W = X^T Y \\
            (X^T X)^{-1} X^T X W = (X^T X)^{-1} X^T Y \\
            W = (X^T X)^{-1} X^T Y 
        `

        and setting

        :math:`
            W = 
            \begin{bmatrix}
            I_3 & 0 \\
            W_{true} & 0 \\
            \end{bmatrix}
        `
        Args:
            X: A tensor of shape (N, D) where N is the number of samples and D is the number of features.
            Y: A tensor of shape (N, K) where N is the number of samples and K is the number of outputs.
        """
        # This idea for initialization is 
        # from https://stackoverflow.com/questions/22375612/python-multiple-ways-to-initialize-a-class
        X_size = X.size()[1]
        Y_size = Y.size()[1]
        A = MaskedLinear(in_features=X_size+Y_size, 
                         out_features=X_size+Y_size,
                         bias=bias, device=device, dtype=dtype)
        with torch.no_grad():
            W_init = (torch.inverse(X.T @ X)@X.T@Y).T
            # The requires_grad=False is important, otherwise the gradient will be computed when we don't want it to.
            A.weight_0[:, :] = bmatrix([[torch.eye(X_size, device=device), torch.zeros(X_size, Y_size, device=device)],
                                        [W_init,                           torch.zeros(Y_size, Y_size, device=device)]])
            A.mask[:, :] = bmatrix([[torch.zeros(X_size, X_size, device=device), torch.zeros(X_size, Y_size, device=device)],
                                    [torch.zeros(Y_size, X_size, device=device), torch.zeros(Y_size, Y_size, device=device)]])
        return A

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        torch.nn.init.constant_(self.U, 0.)
        torch.nn.init.constant_(self.mask, 1.)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()
        weight = self.weight_0 + self.U * self.mask
        # NOTE: this does :math:`y = xA^T + b`.  Beware the transpose.
        result = torch.nn.functional.linear(input, weight, self.bias)
        logger.debug(f'forward time {time.perf_counter()-start_time:e}')
        return result

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def number_of_trainable_parameters(self) -> int:
        total_params = torch.count_nonzero(self.mask)
        if self.bias is not None:
            total_params += self.out_features
        return total_params

