import torch

from iterativennsimple.Sequential2D import Sequential2D
from iterativennsimple.MaskedLinear import MaskedLinear
from iterativennsimple.MonarchLinear import MonarchLinear

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

def test_forward_list_basic():
    """Test forward_list with basic tensor inputs"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Create list of input tensors
    X_in_list = [torch.randn(10, 2), torch.randn(10, 3)]
    X_out_list = model.forward_list(X_in_list)
    
    assert len(X_out_list) == 2, "Output list should have 2 elements"
    assert X_out_list[0].shape == (10, 4), f"First output shape should be (10, 4), got {X_out_list[0].shape}"
    assert X_out_list[1].shape == (10, 5), f"Second output shape should be (10, 5), got {X_out_list[1].shape}"

def test_forward_list_with_none_inputs():
    """Test forward_list with None inputs to verify optimization"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Test with None in first position
    X_in_list = [None, torch.randn(10, 3)]
    X_out_list = model.forward_list(X_in_list)
    
    assert len(X_out_list) == 2, "Output list should have 2 elements"
    # Only blocks from second input should contribute
    expected_out_0 = blocks[1][0](X_in_list[1])
    expected_out_1 = blocks[1][1](X_in_list[1])
    
    assert torch.allclose(X_out_list[0], expected_out_0), "First output should match expected"
    assert torch.allclose(X_out_list[1], expected_out_1), "Second output should match expected"

def test_forward_list_with_partial_none():
    """Test forward_list with some None inputs"""
    in_features_list = [2, 3, 4]
    out_features_list = [5, 6]
    blocks = [[torch.nn.Linear(2, 5), torch.nn.Linear(2, 6)],
              [None, torch.nn.Linear(3, 6)],  # First block is None
              [torch.nn.Linear(4, 5), None]]  # Second block is None
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    X_in_list = [torch.randn(8, 2), torch.randn(8, 3), torch.randn(8, 4)]
    X_out_list = model.forward_list(X_in_list)
    
    assert len(X_out_list) == 2, "Output list should have 2 elements"
    
    # Manually compute expected outputs
    expected_out_0 = blocks[0][0](X_in_list[0]) + blocks[2][0](X_in_list[2])
    expected_out_1 = blocks[0][1](X_in_list[0]) + blocks[1][1](X_in_list[1])
    
    assert torch.allclose(X_out_list[0], expected_out_0), "First output should match expected"
    assert torch.allclose(X_out_list[1], expected_out_1), "Second output should match expected"

def test_forward_vector_vs_forward_list_consistency():
    """Test that forward_vector and forward_list give consistent results"""
    in_features_list = [3, 4]
    out_features_list = [5, 6]
    blocks = [[torch.nn.Linear(3, 5), torch.nn.Linear(3, 6)],
              [torch.nn.Linear(4, 5), torch.nn.Linear(4, 6)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Create inputs
    batch_size = 10
    X_in_vector = torch.randn(batch_size, 7)  # 3 + 4 = 7
    X_in_list = [X_in_vector[:, :3], X_in_vector[:, 3:]]
    
    # Get outputs from both methods
    X_out_vector = model.forward_vector(X_in_vector)
    X_out_list = model.forward_list(X_in_list)
    
    # Convert list output to vector for comparison
    X_out_from_list = torch.cat(X_out_list, dim=1)
    
    assert torch.allclose(X_out_vector, X_out_from_list, atol=1e-6), \
        "Vector and list forward methods should give identical results"

def test_forward_dispatch():
    """Test that forward method correctly dispatches to forward_vector or forward_list"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Test with tensor input (should use forward_vector)
    X_in_tensor = torch.randn(10, 5)  # 2 + 3 = 5
    X_out_tensor = model.forward(X_in_tensor)
    assert isinstance(X_out_tensor, torch.Tensor), "Should return tensor for tensor input"
    assert X_out_tensor.shape == (10, 9), f"Output shape should be (10, 9), got {X_out_tensor.shape}"
    
    # Test with list input (should use forward_list)
    X_in_list = [torch.randn(10, 2), torch.randn(10, 3)]
    X_out_list = model.forward(X_in_list)
    assert isinstance(X_out_list, list), "Should return list for list input"
    assert len(X_out_list) == 2, "Output list should have 2 elements"

def test_forward_list_dimension_mismatch():
    """Test that forward_list raises appropriate errors for dimension mismatches"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Test with wrong number of inputs
    X_in_wrong_length = [torch.randn(10, 2)]  # Should have 2 elements
    try:
        model.forward_list(X_in_wrong_length)
        assert False, "Should raise assertion error for wrong input length"
    except AssertionError:
        pass  # Expected behavior

def test_forward_vector_dimension_mismatch():
    """Test that forward_vector raises appropriate errors for dimension mismatches"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Test with wrong input dimensions
    X_in_wrong_dim = torch.randn(10, 7)  # Should be 5 (2+3)
    try:
        model.forward_vector(X_in_wrong_dim)
        assert False, "Should raise assertion error for wrong input dimensions"
    except AssertionError:
        pass  # Expected behavior

def test_forward_list_with_masked_linear():
    """Test forward_list with MaskedLinear blocks"""
    in_features_list = [3, 4]
    out_features_list = [5, 6]
    blocks = [[MaskedLinear(3, 5), MaskedLinear(3, 6)],
              [MaskedLinear(4, 5), MaskedLinear(4, 6)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    X_in_list = [torch.randn(8, 3), torch.randn(8, 4)]
    X_out_list = model.forward_list(X_in_list)
    
    assert len(X_out_list) == 2, "Output list should have 2 elements"
    assert X_out_list[0].shape == (8, 5), f"First output shape should be (8, 5), got {X_out_list[0].shape}"
    assert X_out_list[1].shape == (8, 6), f"Second output shape should be (8, 6), got {X_out_list[1].shape}"

def test_number_of_trainable_parameters():
    """Test the number_of_trainable_parameters method"""
    in_features_list = [2, 3]
    out_features_list = [4, 5]
    blocks = [[torch.nn.Linear(2, 4), torch.nn.Linear(2, 5)],
              [torch.nn.Linear(3, 4), torch.nn.Linear(3, 5)]]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    
    # Calculate expected parameters
    # Linear(2,4): 2*4 + 4 = 12, Linear(2,5): 2*5 + 5 = 15
    # Linear(3,4): 3*4 + 4 = 16, Linear(3,5): 3*5 + 5 = 20
    expected_params = 12 + 15 + 16 + 20  # = 63
    
    actual_params = model.number_of_trainable_parameters()
    assert actual_params == expected_params, f"Expected {expected_params} parameters, got {actual_params}"

def test_identity_block():
    """Test Sequential2D with Identity blocks"""
    cfg = {
        "sequential2D": {
            "in_features_list": [5, 10],
            "out_features_list": [5, 10],
            "block_types": [
                ['Identity', None],
                [None, 'Identity'],
            ],
            "block_kwargs": [
                [None, None],
                [None, None],
            ]
        }
    }
    
    model = Sequential2D.from_config(cfg['sequential2D'])
    X_in = torch.randn(8, 15)  # 5 + 10 = 15
    X_out = model.forward(X_in)
    
    # With Identity blocks in diagonal positions, input should equal output
    assert torch.allclose(X_in, X_out), "Identity blocks should preserve input"


# ---------------------------------------------------------------------------
# MonarchLinear tests
# ---------------------------------------------------------------------------

def test_Sequential2D_MonarchLinear():
    """Test Sequential2D constructed directly with MonarchLinear blocks."""
    in_features_list = [8, 16]
    out_features_list = [8, 16]
    blocks = [
        [MonarchLinear.from_uniform_blocks(8, 8, num_blocks=2, seed=0),
         MonarchLinear.from_uniform_blocks(8, 16, num_blocks=2, seed=1)],
        [MonarchLinear.from_uniform_blocks(16, 8, num_blocks=2, seed=2),
         MonarchLinear.from_uniform_blocks(16, 16, num_blocks=4, seed=3)],
    ]
    model = Sequential2D(in_features_list, out_features_list, blocks)
    X_in = torch.randn(10, 24)  # 8 + 16 = 24
    X_out = model.forward(X_in)
    assert X_out.shape == (10, 24), f"Expected shape (10, 24), got {X_out.shape}"


def test_sequential2D_factory_monarch_uniform_blocks():
    """Test Sequential2D.from_config with MonarchLinear.from_uniform_blocks."""
    cfg = {
        "in_features_list": [64, 64],
        "out_features_list": [64, 64],
        "block_types": [
            ["MonarchLinear.from_uniform_blocks", "MonarchLinear.from_uniform_blocks"],
            ["MonarchLinear.from_uniform_blocks", "MonarchLinear.from_uniform_blocks"],
        ],
        "block_kwargs": [
            [{"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 0},
             {"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 1}],
            [{"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 2},
             {"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 3}],
        ],
    }
    model = Sequential2D.from_config(cfg)
    X_in = torch.randn(8, 128)  # 64 + 64 = 128
    X_out = model.forward(X_in)
    assert X_out.shape == (8, 128), f"Expected shape (8, 128), got {X_out.shape}"


def test_sequential2D_factory_monarch_block_config():
    """Test Sequential2D.from_config with MonarchLinear.from_block_config."""
    cfg = {
        "in_features_list": [12],
        "out_features_list": [12],
        "block_types": [
            ["MonarchLinear.from_block_config"],
        ],
        "block_kwargs": [
            [{"block_in_features": [4, 4, 4], "block_out_features": [4, 4, 4],
              "bias": True, "initialization_type": "kaiming", "seed": 42}],
        ],
    }
    model = Sequential2D.from_config(cfg)
    X_in = torch.randn(5, 12)
    X_out = model.forward(X_in)
    assert X_out.shape == (5, 12), f"Expected shape (5, 12), got {X_out.shape}"


def test_sequential2D_factory_monarch_sparsity_target():
    """Test Sequential2D.from_config with MonarchLinear.from_sparsity_target."""
    cfg = {
        "in_features_list": [64],
        "out_features_list": [64],
        "block_types": [
            ["MonarchLinear.from_sparsity_target"],
        ],
        "block_kwargs": [
            [{"target_sparsity": 0.75, "bias": True, "initialization_type": "kaiming", "seed": 7}],
        ],
    }
    model = Sequential2D.from_config(cfg)
    X_in = torch.randn(6, 64)
    X_out = model.forward(X_in)
    assert X_out.shape == (6, 64), f"Expected shape (6, 64), got {X_out.shape}"


def test_sequential2D_monarch_gradient_flow():
    """Verify gradients flow through MonarchLinear blocks in Sequential2D."""
    cfg = {
        "in_features_list": [16],
        "out_features_list": [16],
        "block_types": [
            ["MonarchLinear.from_uniform_blocks"],
        ],
        "block_kwargs": [
            [{"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 0}],
        ],
    }
    model = Sequential2D.from_config(cfg)
    X_in = torch.randn(4, 16)
    X_out = model.forward(X_in)
    loss = X_out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"


def test_sequential2D_monarch_parameter_count():
    """Verify trainable parameter count for a 1×1 MonarchLinear grid."""
    num_blocks = 4
    in_out = 64
    # Each block: (in_out/num_blocks) * (in_out/num_blocks) weights + in_out biases
    block_size = in_out // num_blocks
    expected = num_blocks * block_size * block_size + in_out  # weights + bias

    cfg = {
        "in_features_list": [in_out],
        "out_features_list": [in_out],
        "block_types": [["MonarchLinear.from_uniform_blocks"]],
        "block_kwargs": [[{"num_blocks": num_blocks, "bias": True,
                           "initialization_type": "kaiming", "seed": 0}]],
    }
    model = Sequential2D.from_config(cfg)
    assert model.number_of_trainable_parameters() == expected, (
        f"Expected {expected} parameters, got {model.number_of_trainable_parameters()}"
    )


def test_sequential2D_monarch_mixed_blocks():
    """Test mixing MonarchLinear with Linear and None blocks in Sequential2D."""
    cfg = {
        "in_features_list": [32, 32],
        "out_features_list": [32, 32],
        "block_types": [
            ["MonarchLinear.from_uniform_blocks", "Linear"],
            [None, "MonarchLinear.from_uniform_blocks"],
        ],
        "block_kwargs": [
            [{"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 0}, None],
            [None, {"num_blocks": 4, "bias": True, "initialization_type": "kaiming", "seed": 1}],
        ],
    }
    model = Sequential2D.from_config(cfg)
    X_in = torch.randn(10, 64)  # 32 + 32 = 64
    X_out = model.forward(X_in)
    assert X_out.shape == (10, 64), f"Expected shape (10, 64), got {X_out.shape}"

