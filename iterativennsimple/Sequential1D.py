import torch

class Sequential1D(torch.nn.Sequential):
    """
    This is a simple extension of torch.nn.Sequential just provides in_features and out_features
    so that it can be used in a Sequential2D.

    Args:
        in_features (int): The number of input features for the first module.
        out_features (int): The number of output features for the last module.

    """
    def __init__(self, *args, in_features=None, out_features=None, **kwargs):
        super(Sequential1D, self).__init__(*args, **kwargs) 
        assert in_features is not None, "in_features must be specified"
        assert out_features is not None, "out_features must be specified"
        self.in_features = in_features
        self.out_features = out_features

    def number_of_trainable_parameters(self):
        # FIXME: I am not sure this is the best way of doing this, but
        # knowing the total number of trainable parameters is important
        # for our work. This is needed because the number of trainable
        # for MaskedLinear is strange.
        
        trainable_parameters = 0
        for model in self:
            if hasattr(model, "number_of_trainable_parameters"):
                    trainable_parameters += model.number_of_trainable_parameters()
            elif issubclass(model.__class__, torch.nn.Module):
                trainable_parameters += sum(p.numel() for p in model.parameters() if p.requires_grad)
            else:
                trainable_parameters += 0
        return trainable_parameters