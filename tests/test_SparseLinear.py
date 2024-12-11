from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.SparseLinear import SparseLinear
import torch

def test_fromDescription():
    row_sizes = [5, 7, 9]
    col_sizes = [6, 8, 10]
    block_types = [[0, 'W', 'D'],
                   ['R=0.5', 'S=5', 'Row=3'],
                   ['S=2', 'R=0.9', 'Row=1']]
    initialization_types = [[0, torch.ones((5, 8)), 'C=0.3'],
                            ['G', 'G=0.2,0.7', 'U'],
                            ['U=-0.5,0.5', 1, torch.randn(size=(9, 10))]]
    trainable = [[0, 1, 0],
                 [1, 0, 1],
                 [1, 1, 1]]
    model = MaskedLinear.from_description(row_sizes=row_sizes,
                                          col_sizes=col_sizes,
                                          block_types=block_types,
                                          initialization_types=initialization_types,
                                          trainable=trainable)
    assert model.weight_0.size() == (5 + 7 + 9, 6 + 8 + 10)
    model = SparseLinear.from_MaskedLinearExact(model)


def test_fromDescription_gradient():
    # More tests can be found in:
    #
    #  iterativenn/notebooks/06-rcp-sparse-comparison.ipynb
    #
    row_sizes = [5, 7, 9]
    col_sizes = [6, 8, 10]
    block_types = [[0, 'W', 'D'],
                   ['R=0.5', 'D', 'Row=3'],
                   ['S=2', 'R=0.9', 'Row=1']]
    initialization_types = [[0, torch.ones((5, 8)), 'C=0.3'],
                            ['G', 'G=0.2,0.7', 'U'],
                            ['U=-0.5,0.5', 1, torch.randn(size=(9, 10))]]
    trainable = [[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]]
    model = MaskedLinear.from_description(row_sizes=row_sizes,
                                          col_sizes=col_sizes,
                                          block_types=block_types,
                                          initialization_types=initialization_types,
                                          trainable=trainable)


    sparse_model = SparseLinear.from_MaskedLinearExact(model)

    input = torch.randn(128, model.in_features)
    output = torch.randn(128, model.out_features)

    output_model = model(input)
    output_sparse_model = sparse_model(input)

    diff_is_small = torch.isclose(output_model, output_sparse_model, atol=1e-6)
    assert torch.all(diff_is_small)

    # dense version
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    output_model = model(input)
    loss = torch.norm(output-output_model)

    loss.backward()
    optimizer.step()
    output_model_after = model(input)

    # sparse version
    optimizer = torch.optim.SGD(sparse_model.parameters(), lr=0.01)
    output_sparse_model = sparse_model(input)
    loss = torch.norm(output-output_sparse_model)

    loss.backward()
    optimizer.step()
    output_sparse_model_after = sparse_model(input)

    diff_is_small = torch.isclose(output_model_after, output_sparse_model_after, atol=1e-6)
    assert torch.all(diff_is_small)

def test_singleBlock():
    S = SparseLinear.from_singleBlock(3, 4, 'R=0.5', 'C=0.2')
    S = SparseLinear.from_singleBlock(3, 4, 'S=6', 'G')
    S = SparseLinear.from_singleBlock(3, 4, 'Row=2', 'U=-0.5,0.5')
    S = SparseLinear.from_singleBlock(3, 4, 'D', 1)
    S = SparseLinear.from_singleBlock(6, 3, 'D', 1)
    S = SparseLinear.from_singleBlock(6, 3, 'R=0.5', 'C=0.2')
    S = SparseLinear.from_singleBlock(6, 3, 'S=6', 'G=0.2,0.7')
    S = SparseLinear.from_singleBlock(6, 3, 'Row=2', 'U')

def test_fromDescription_exact():
    row_sizes = [5, 7, 9]
    col_sizes = [6, 8, 10]
    block_types = [[0, 'W', 'D'],
                   ['R=0.5', 'S=5', 'Row=3'],
                   ['S=2', 'R=0.9', 'Row=1']]
    initialization_types = [[0, torch.ones((5, 8)), 'C=0.3'],
                            ['G', 'G=0.2,0.7', 'U'],
                            ['U=-0.5,0.5', 1, torch.randn(size=(9, 10))]]
    trainable = [[0, 'non-zero', 'non-zero'],
                 ['non-zero', 'non-zero', 'non-zero'],
                 ['non-zero', 'non-zero', 'non-zero']]
    model = MaskedLinear.from_description(row_sizes=row_sizes,
                                          col_sizes=col_sizes,
                                          block_types=block_types,
                                          initialization_types=initialization_types,
                                          trainable=trainable)
    sparse_model_exact = SparseLinear.from_MaskedLinearExact(model)
    sparse_model = SparseLinear.from_MaskedLinear(model)

    input = torch.randn(128, sparse_model_exact.in_features)

    output_sparse_model_exact = sparse_model_exact(input)
    output_sparse_model = sparse_model(input)
    diff_is_small = torch.isclose(output_sparse_model, output_sparse_model_exact, atol=1e-6)
    assert torch.all(diff_is_small)

def test_from_coo():
    coo = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                                  values=torch.tensor([1.0, 2.0, 3.0]),
                                  size=(3, 3))
    model = SparseLinear.from_coo(coo, bias=False)

    x = torch.randn(13, 3)
    y1 = model(x)
    y2 = coo @ x.T

    diff_is_small = torch.isclose(y1, y2.T)
    assert torch.all(diff_is_small)

def test_to_coo():
    row_sizes = [5, 7, 9]
    col_sizes = [6, 8, 10]
    block_types = [[0, 'W', 'D'],
                   ['R=0.5', 'S=5', 'Row=3'],
                   ['S=2', 'R=0.9', 'Row=1']]
    initialization_types = [[0, torch.ones((5, 8)), 'C=0.3'],
                            ['G', 'G=0.2,0.7', 'U'],
                            ['U=-0.5,0.5', 1, torch.randn(size=(9, 10))]]
    trainable = [[0, 'non-zero', 'non-zero'],
                 ['non-zero', 'non-zero', 'non-zero'],
                 ['non-zero', 'non-zero', 'non-zero']]
    model = MaskedLinear.from_description(row_sizes=row_sizes,
                                          col_sizes=col_sizes,
                                          block_types=block_types,
                                          initialization_types=initialization_types,
                                          trainable=trainable,
                                          bias=False)
    sparse_model = SparseLinear.from_MaskedLinear(model)

    coo = sparse_model.to_coo()
    model2 = SparseLinear.from_coo(coo)

    x = torch.randn(13, model.in_features)
    y1 = model(x)
    y2 = model2(x)
    y3 = x @ coo.T

    print(y1-y2)

    diff_is_small = torch.isclose(y1, y2, atol=1e-4, rtol=1e-4)
    assert torch.all(diff_is_small)

    diff_is_small = torch.isclose(y1, y3, atol=1e-4, rtol=1e-4)
    assert torch.all(diff_is_small)

    diff_is_small = torch.isclose(y2, y3, atol=1e-4, rtol=1e-4)
    assert torch.all(diff_is_small)
