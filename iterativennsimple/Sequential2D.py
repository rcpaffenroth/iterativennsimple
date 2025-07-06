import torch

from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.SparseLinear import SparseLinear

class Identity(torch.nn.Identity):
    """
    A torch.nn.Identity module that also has in_features and out_features attributes.
    """
    def __init__(self, *args, in_features=None, out_features=None, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)
        assert in_features is not None, "in_features must be specified"
        assert out_features is not None, "out_features must be specified"
        self.in_features = in_features
        self.out_features = out_features

class Sequential2D(torch.nn.Module):
    """
    A 2D version of the torch.nn.Sequential module.

    The idea is to have a simple class that takes a 2D list (or list like) of modules and then applies them in sequence.  This just does a few things:

    1)  Makes sure that the sizes match up.
    2)  Implements the "+" combiner (which is what it means for the sizes to match up).
    3)  Actually calls the modules and returns the result.
    4)  It is efficient, since if a block is None, then it just acts as if the block returns a 0-vector of the correct size.

    *Note, it requires that the modules have a "forward" method and that they have a "in_features" and "out_features" attribute.*

    Also, it assumes that things like what parameters are in trainable in the model and how they are initialized are handled by the modules themselves.

    There are some nice side effects:

    1) Linear layers can be used as blocks.
    2) MaskedLinear layers can be used as blocks.
    3) SparseLinear layers can be used as blocks.
    4) As can another Sequential2D!  This allows the blocks to be nested quite arbitrarily.

    Args:
        in_features_list (list): A list of integers, where each integer is the number of input features for each block.
        out_features_list (list): A list of integers, where each integer is the number of output features for each block.
        blocks (list): A list of lists of torch.nn.Module objects. The blocks[i][j] is the block that takes in_features_list[i] features and 
                       outputs out_features_list[j] features.  If blocks[i][j] is None, then we assume that the output is 
                       just a 0-vector of the correct size.
    """
    def __init__(self, in_features_list, out_features_list, blocks):
        super(Sequential2D, self).__init__()
        # Note, it is redundant to require the in_features_list and out_features_list, since we also require the blocks to have their own in_features and out_features.  
        # However, it is convenient to have them here, since we can use them to check the dimensions of the input and output. 
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.in_features = sum(in_features_list)
        self.out_features = sum(out_features_list)

        with torch.no_grad():
            # This makes sure that each of the modules parameters are registered as parameters of the Sequential2D module.
            self.blocks = torch.nn.ModuleDict()
            # Check that the dimensions are correct
            for i in range(len(self.in_features_list)):
                for j in range(len(self.out_features_list)):
                    # Note, not all blocks need to be defined, so we need to check if they are None
                    # if a block is None, then we assume that its output is just a 0-vector of the correct size
                    if blocks[i][j] is not None:
                        assert blocks[i][j].in_features == self.in_features_list[i], f"{blocks[i][j]}: {blocks[i][j].in_features}, {self.in_features_list[i]}"
                        assert blocks[i][j].out_features == self.out_features_list[j], f"{blocks[i][j]}: {blocks[i][j].out_features}, {self.out_features_list[j]}"
                        self.blocks[str((i, j))] = blocks[i][j]
                    
    def forward(self, X_in):
        """
        A forward method that takes an input tensor and applies the blocks in sequence.
        
        Args:
            X_in (torch.Tensor): A tensor of shape (batch_size, in_features).
        
        Returns:
            X_out (torch.Tensor): A tensor of shape (batch_size, out_features).
        """
        if isinstance(X_in, list):
            return self.forward_list(X_in)
        else:
            return self.forward_vector(X_in)
        
    def forward_list(self, X_in):
        """
        A forward method that takes a list of inputs and applies the blocks in sequence.
        
        Args:
            X_in (list): A list of tensors, where each tensor has shape (batch_size, in_features_list[i]).
        
        Returns:
            X_out (list): A list of tensors, where each tensor has shape (batch_size, out_features_list[j]).
        """
        assert False, "The forward_list method is not implemented. Use forward_vector instead."
        assert len(X_in) == len(self.in_features_list), f'The input has the wrong number of features. {len(X_in), self.in_features_list }'

        X_out = [torch.zeros((X_in[0].shape[0], self.out_features_list[j]), device=X_in[0].device) for j in range(len(self.out_features_list))]

        for i in range(len(self.in_features_list)):
            for j in range(len(self.out_features_list)):
                if str((i, j)) in self.blocks.keys() and X_in[i] is not None
                    tmp_block = self.blocks[str((i, j))].forward(X_in[i])
                    X_out[j] += tmp_block
        return X_out

    def forward_vector(self, X_in):
        assert X_in.shape[1] == self.in_features, f'The input has the wrong number of features. {X_in.shape, self.in_features }'
        X_out = torch.zeros((X_in.shape[0], self.out_features), device=X_in.device)

        for i in range(len(self.in_features_list)):
            for j in range(len(self.out_features_list)):
                # Note, this conditional has the same effect having a 0-vector as the output of the block
                # If a block is None, then we assume that its output is just a 0-vector of the correct size
                if str((i, j)) in self.blocks.keys():
                    # When i=0 or j=0 we have [:0] is empty
                    # similarily [:i+1] and [:j+1] are the whole thing
                    in_start = sum(self.in_features_list[:i])
                    in_end = sum(self.in_features_list[:i+1])
                    out_start = sum(self.out_features_list[:j])
                    out_end = sum(self.out_features_list[:j+1])
                    tmp_block = self.blocks[str((i, j))].forward(X_in[:, in_start:in_end])
                    X_out[:, out_start:out_end] += tmp_block
        return X_out

    def number_of_trainable_parameters(self):
        # FIXME: I am not sure this is the best way of doing this, but
        # knowing the total number of trainable parameters is important
        # for our work. This is needed because the number of trainable
        # for MaskedLinear is strange.

        trainable_parameters = 0
        for i in range(len(self.in_features_list)):
            for j in range(len(self.out_features_list)):
                if str((i, j)) in self.blocks.keys():
                    model = self.blocks[str((i,j))]
                    if hasattr(model, "number_of_trainable_parameters"):
                        trainable_parameters += model.number_of_trainable_parameters()
                    elif issubclass(model.__class__, torch.nn.Module):
                        trainable_parameters += sum(p.numel() for p in model.parameters() if p.requires_grad)
                    else:
                        trainable_parameters += 0
        return trainable_parameters

    @staticmethod
    def from_config(cfg):
        """Constructs a Sequential2D object from a configuration dictionary.

        Args:
            cfg (dictionary): A dictionary with the following keys

        Returns:
            Sequential2D: The constructed Sequential2D object.
        """
        in_features_list = cfg['in_features_list']
        out_features_list = cfg['out_features_list']
        block_types = cfg['block_types']
        blocks = []
        for i, row in enumerate(block_types):
            blocks_row = []
            for j, block_type in enumerate(row):
                if block_type is None:
                    blocks_row.append(None)
                elif block_type == 'None':
                    blocks_row.append(None)
                elif block_type == 'Identity':
                    blocks_row.append(Identity(in_features=in_features_list[i],
                                               out_features=out_features_list[j]))
                elif block_type == 'Linear':
                    blocks_row.append(torch.nn.Linear(in_features_list[i], out_features_list[j]))
                elif block_type == 'MaskedLinear':
                    blocks_row.append(MaskedLinear(in_features_list[i], out_features_list[j]))
                elif block_type == 'MaskedLinear.from_description':
                    # Note, the odd ordering of the first two arguments is on purpose.  The first argument is the output dimension(rows), and the second is the input dimension (columns) 
                    blocks_row.append(MaskedLinear.from_description(
                                        out_features_list[j], in_features_list[i],  
                                        cfg['block_kwargs'][i][j]['block_type'],
                                        cfg['block_kwargs'][i][j]['initialization_type'],
                                        cfg['block_kwargs'][i][j]['trainable'],
                                        bias = cfg['block_kwargs'][i][j]['bias'],
                                    ))
                elif block_type == 'SparseLinear.from_description':
                    # Note, the odd ordering of the first two arguments is on purpose.  The first argument is the output dimension(rows), and the second is the input dimension (columns) 
                    # FIXME:  This is inefficient, because we are creating a MaskedLinear and 
                    # then converting it to a SparseLinear.  We should just create a 
                    # SparseLinear directly.  This will only really be an issue
                    # for large networks.
                    tmp_block = MaskedLinear.from_description(
                                    out_features_list[j], in_features_list[i],  
                                    cfg['block_kwargs'][i][j]['block_type'],
                                    cfg['block_kwargs'][i][j]['initialization_type'],
                                    cfg['block_kwargs'][i][j]['trainable'],
                                    bias = cfg['block_kwargs'][i][j]['bias'],
                                )
                    blocks_row.append(SparseLinear.from_MaskedLinearExact(tmp_block))
                elif block_type == 'SparseLinear.from_singleBlock':
                    # Note, the odd ordering of the first two arguments is on purpose.  The first argument is the output dimension(rows), and the second is the input dimension (columns) 
                    # This is efficient, in that it only created the needed entries.
                    block = SparseLinear.from_singleBlock(
                                    out_features_list[j], in_features_list[i],  
                                    cfg['block_kwargs'][i][j]['block_type'],
                                    cfg['block_kwargs'][i][j]['initialization_type'],
                                    bias = cfg['block_kwargs'][i][j]['bias'],
                                )
                    blocks_row.append(block)
                else:
                    raise ValueError(f"Unknown block type {block_type}")
            blocks.append(blocks_row)

        model = Sequential2D(in_features_list, out_features_list, blocks)
        return model