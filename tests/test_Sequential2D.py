import torch

from iterativennsimple.Sequential2D import Sequential2D
from iterativennsimple.MaskedLinear import MaskedLinear

def test_Sequential2D_Linear():
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    X_in = torch.randn(10, 5)
    X_out = model.forward(X_in)

def test_Sequential2D_MaskedLinear():
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[MaskedLinear(2, 4), MaskedLinear(2, 5)],
              [MaskedLinear(3, 4), MaskedLinear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    X_in = torch.randn(10, 5)
    X_out = model.forward(X_in)

def test_Sequential2D_Composite():
    in_features_list = [2, 3]
    out_features_list = [3, 2]
    blocks = [[MaskedLinear(2, 3), MaskedLinear(2, 2)],
              [MaskedLinear(3, 3), MaskedLinear(3, 2)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    in_features_list = [5, 6]
    out_features_list = [5, 7]
    blocks = [[model, MaskedLinear(5, 7)],
              [MaskedLinear(6, 5), MaskedLinear(6, 7)]]
    model2 = Sequential2D(in_features_list, out_features_list, blocks)

    X_in = torch.randn(10, 11)
    X_out = model2.forward(X_in)

def test_sequential2D_factory1():
    cfg = {
        "sequential2D": {
            "in_features_list": [784, 200, 10], # 784 + 200 + 10 = 994
            "out_features_list": [784, 140, 50, 20], # = # 784 + 140 + 50 + 20 = 994
            "block_types": [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
            ],
            "block_kwargs": [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
            ]
        }
    }

    model = Sequential2D.from_config(cfg['sequential2D'])
    X_in = torch.randn(sum(cfg['sequential2D']['in_features_list']), sum(cfg['sequential2D']['out_features_list']))
    X_out = model.forward(X_in)
    assert tuple(X_out.shape)==(sum(cfg['sequential2D']['in_features_list']), sum(cfg['sequential2D']['out_features_list'])), "incorrect shape"
    assert torch.all(X_out == 0.0), "incorrect values" 
    assert list(model.parameters()) == [], "incorrect parameters"

def test_sequential2D_factory2():
    cfg = {
        "sequential2D": {
            "in_features_list": [784, 200, 10], # 784 + 200 + 10 = 994
            "out_features_list": [784, 140, 50, 20], # = # 784 + 140 + 50 + 20 = 994
            "block_types": [
                ['MaskedLinear', None, None, None],
                ['MaskedLinear', 'Linear', 'Linear', None],
                ['Linear', 'MaskedLinear.from_description', None, 'Linear'],
            ],
            "block_kwargs": [
                [None, None, None, None],
                [None, None, None, None],
                [None, {'block_type':'S=15', 'initialization_type':'G=0.2,0.7', 'trainable':True, 'bias':True}, None, None],
            ]
        }
    }

    model = Sequential2D.from_config(cfg['sequential2D'])
    X_in = torch.randn(sum(cfg['sequential2D']['in_features_list']), sum(cfg['sequential2D']['out_features_list']))
    X_out = model.forward(X_in)
    assert tuple(X_out.shape)==(sum(cfg['sequential2D']['in_features_list']), sum(cfg['sequential2D']['out_features_list'])), "incorrect shape" 
    assert len(list(model.parameters())) == 20, "incorrect parameters"

