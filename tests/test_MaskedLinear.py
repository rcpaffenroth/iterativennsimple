import os

import torch

from iterativennsimple.MaskedLinear import MaskedLinear


class GetMaskedLinear:
    def __init__(self):
        self.in_features = 20
        self.out_features = 30
        self.input_batch_size = 128
        self.input = torch.randn(self.input_batch_size, self.in_features)
        torch.manual_seed(0)
        self.m = MaskedLinear(self.in_features, self.out_features)
        torch.manual_seed(0)
        self.l = torch.nn.Linear(self.in_features, self.out_features)


def test_MaskedLinearInit():
    tml = GetMaskedLinear()
    output = tml.m(tml.input)
    assert output.size() == (tml.input_batch_size, tml.out_features)


def test_MaskedLinearSameAsLinear():
    tml = GetMaskedLinear()
    torch.manual_seed(0)
    output_m = tml.m(tml.input)
    output_l = tml.l(tml.input)
    diff_is_small = torch.isclose(output_m, output_l)
    assert torch.all(diff_is_small)


def test_MaskedLinearGrad():
    tml = GetMaskedLinear()
    torch.manual_seed(0)
    model_m = torch.nn.Sequential(tml.m)
    torch.manual_seed(0)
    model_l = torch.nn.Sequential(tml.l)
    target = torch.randn(tml.input_batch_size, tml.out_features)

    output_m = model_m(tml.input)
    output_l = model_l(tml.input)
    loss = torch.nn.MSELoss()
    loss_m = loss(target, output_m)
    loss_l = loss(target, output_l)

    model_m.zero_grad()
    loss_m.backward()
    model_l.zero_grad()
    loss_l.backward()
    assert model_l[0].weight.grad.size() == model_m[0].U.grad.size()
    diff_is_small = torch.isclose(model_l[0].weight.grad, model_m[0].U.grad)
    assert torch.all(diff_is_small)


def test_MaskedLinearCheckSaving():
    tml = GetMaskedLinear()
    torch.manual_seed(0)
    model_m = torch.nn.Sequential(tml.m)
    output_m = model_m(tml.input)
    torch.save(model_m.state_dict(), "tmp")
    model_m.state_dict()
    model = torch.nn.Sequential(
        MaskedLinear(tml.in_features, tml.out_features))
    model.load_state_dict(torch.load("tmp"))
    output = model(tml.input)
    os.remove("tmp")
    # Save and recover module
    diff_is_small = torch.isclose(output_m, output)
    assert torch.all(diff_is_small)


def test_MaskedLinearSetMatrices():
    tml = GetMaskedLinear()
    mask = torch.zeros(tml.out_features,
                       tml.in_features)
    weight_0 = torch.zeros_like(mask)
    bias = torch.zeros(tml.out_features)
    with torch.no_grad():
        tml.m.weight_0 = torch.nn.Parameter(weight_0)
        tml.m.bias = torch.nn.Parameter(bias)
        tml.m.mask = torch.nn.Parameter(mask)
    output = tml.m(tml.input)
    diff_is_small = torch.isclose(output, torch.zeros(tml.input_batch_size, tml.out_features))
    assert torch.all(diff_is_small)
    

def test_fromOptimalLinear():
    tml = GetMaskedLinear()
    W_true = torch.randn(tml.out_features, tml.in_features)
    # Genrate data which is actually linear
    tml.output = tml.input @ W_true.T
    # intialize with the optimal linear model
    m = tml.m.from_optimal_linear(tml.input, tml.output)
    z = torch.concat([tml.input, torch.rand(tml.input_batch_size, tml.out_features)], dim=1)
    z_true = torch.concat([tml.input, tml.output], dim=1)
    z_hat = m(z)
    # So the output should have zero error
    diff_is_small = torch.isclose(z_hat, z_true)
    assert not torch.all(diff_is_small)

# def test_from_grown_model():
#     sizes=[20, 30]
#     sizes_map=['x', 'y']

#     model = MaskedLinear.from_MLP(sizes=sizes)
#     grown_model = MaskedLinear.from_grown_model(model, added_columns=5, added_rows=5)
#     assert grown_model.weight_0.size() == (20+30 + 5, 20+30 + 5)

def test_fromMLP():
    tml = GetMaskedLinear()
    MLP_model = tml.m.from_MLP(sizes=[10, 5, 1])
    assert MLP_model.weight_0.size() == (10 + 5 + 1, 10 + 5 + 1)


def test_fromDescription():
    tml = GetMaskedLinear()
    out_features_sizes = [5, 7, 9]
    in_features_sizes = [6, 8, 10]
    block_types = [[0, 'W', 'D'],
                   ['R=0.5', 'S=5', 'Row=3'],
                   ['S=2', 'R=0.9', 'Row=1']]
    initialization_types = [[0, torch.ones((5, 8)), 'C=0.3'],
                            ['G', 'G=0.2,0.7', 'U'],
                            ['U=-0.5,0.5', 1, torch.randn(size=(9, 10))]]
    trainable = [[0, 1, 0],
                 [1, 0, 1],
                 [1, 1, 1]]
    model = tml.m.from_description(out_features_sizes=out_features_sizes,
                                   in_features_sizes=in_features_sizes,
                                   block_types=block_types,
                                   initialization_types=initialization_types,
                                   trainable=trainable)
    assert model.weight_0.size() == (5 + 7 + 9, 6 + 8 + 10)

def test_fromCOO():
    coo = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                                  values=torch.tensor([1.0, 2.0, 3.0]),
                                  size=(3, 3))                                                
    model = MaskedLinear.from_coo(coo, bias=False)

    assert model.weight_0.size() == (3, 3)

    x = torch.randn(13, 3)
    y1 = model(x)
    y2 = coo @ x.T

    diff_is_small = torch.isclose(y1, y2.T)
    assert torch.all(diff_is_small)
