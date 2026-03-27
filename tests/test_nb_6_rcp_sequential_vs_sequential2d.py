"""
Tests for notebook 6-rcp-Sequential-vs-Sequential2D.ipynb

Verifies functional equivalence, training equivalence, and complex
connectivity patterns between Sequential and Sequential2D.
"""

import torch
import torch.nn as nn
from iterativennsimple.Sequential2D import Sequential2D
from iterativennsimple.Sequential1D import Sequential1D


def _create_equivalent_networks():
    """
    Create functionally equivalent Sequential and Sequential2D networks.
    Returns (sequential_net, sequential2d_net).
    """
    input_size = 784
    hidden_size = 128
    output_size = 10

    f1 = nn.Linear(input_size, hidden_size)
    f2 = nn.Linear(hidden_size, hidden_size)
    f3 = nn.Linear(hidden_size, output_size)

    sequential_net = nn.Sequential(f1, nn.ReLU(), f2, nn.ReLU(), f3)

    in_features_list = [input_size, hidden_size, hidden_size, output_size]
    out_features_list = [input_size, hidden_size, hidden_size, output_size]

    F1 = Sequential1D(nn.Sequential(f1), in_features=input_size, out_features=hidden_size)
    F2 = Sequential1D(nn.Sequential(nn.ReLU(), f2), in_features=hidden_size, out_features=hidden_size)
    F3 = Sequential1D(nn.Sequential(nn.ReLU(), f3), in_features=hidden_size, out_features=output_size)

    blocks = [
        [None, F1,   None, None],
        [None, None, F2,   None],
        [None, None, None, F3],
        [None, None, None, None],
    ]

    sequential2d_net = Sequential2D(in_features_list, out_features_list, blocks)
    return sequential_net, sequential2d_net


def test_functional_equivalence():
    torch.manual_seed(42)
    seq_net, seq2d_net = _create_equivalent_networks()

    batch_size = 32
    test_input = torch.randn(batch_size, 784)

    with torch.no_grad():
        seq_output = seq_net(test_input)

        seq2d_output = [test_input, None, None, None]
        for _ in range(3):
            seq2d_output = seq2d_net(seq2d_output)

    max_diff = torch.max(torch.abs(seq_output - seq2d_output[3])).item()
    assert max_diff < 1e-6, f"Outputs differ by {max_diff:.2e}, expected < 1e-6"


def test_parameter_count_equivalence():
    torch.manual_seed(42)
    seq_net, seq2d_net = _create_equivalent_networks()

    seq_params = sum(p.numel() for p in seq_net.parameters())
    seq2d_params = sum(p.numel() for p in seq2d_net.parameters())

    assert seq_params == 118282, f"Sequential params: {seq_params}, expected 118282"
    assert seq2d_params == 118282, f"Sequential2D params: {seq2d_params}, expected 118282"


def test_training_equivalence():
    torch.manual_seed(42)

    input_size = 784
    hidden_size = 128
    output_size = 10

    # Create base weights
    f1_base = nn.Linear(input_size, hidden_size)
    f2_base = nn.Linear(hidden_size, hidden_size)
    f3_base = nn.Linear(hidden_size, output_size)

    # Build Sequential with copied weights
    f1_seq = nn.Linear(input_size, hidden_size)
    f2_seq = nn.Linear(hidden_size, hidden_size)
    f3_seq = nn.Linear(hidden_size, output_size)
    f1_seq.weight.data.copy_(f1_base.weight.data)
    f1_seq.bias.data.copy_(f1_base.bias.data)
    f2_seq.weight.data.copy_(f2_base.weight.data)
    f2_seq.bias.data.copy_(f2_base.bias.data)
    f3_seq.weight.data.copy_(f3_base.weight.data)
    f3_seq.bias.data.copy_(f3_base.bias.data)
    seq_net = nn.Sequential(f1_seq, nn.ReLU(), f2_seq, nn.ReLU(), f3_seq)

    # Build Sequential2D with copied weights (independent copy)
    f1_seq2d = nn.Linear(input_size, hidden_size)
    f2_seq2d = nn.Linear(hidden_size, hidden_size)
    f3_seq2d = nn.Linear(hidden_size, output_size)
    f1_seq2d.weight.data.copy_(f1_base.weight.data)
    f1_seq2d.bias.data.copy_(f1_base.bias.data)
    f2_seq2d.weight.data.copy_(f2_base.weight.data)
    f2_seq2d.bias.data.copy_(f2_base.bias.data)
    f3_seq2d.weight.data.copy_(f3_base.weight.data)
    f3_seq2d.bias.data.copy_(f3_base.bias.data)

    in_features_list = [input_size, hidden_size, hidden_size, output_size]
    out_features_list = [input_size, hidden_size, hidden_size, output_size]

    F1 = Sequential1D(nn.Sequential(f1_seq2d), in_features=input_size, out_features=hidden_size)
    F2 = Sequential1D(nn.Sequential(nn.ReLU(), f2_seq2d), in_features=hidden_size, out_features=hidden_size)
    F3 = Sequential1D(nn.Sequential(nn.ReLU(), f3_seq2d), in_features=hidden_size, out_features=output_size)

    blocks = [
        [None, F1,   None, None],
        [None, None, F2,   None],
        [None, None, None, F3],
        [None, None, None, None],
    ]
    seq2d_net = Sequential2D(in_features_list, out_features_list, blocks)

    # Verify identical outputs before training
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, output_size)

    with torch.no_grad():
        out_seq = seq_net(x)
        seq2d_input = [x, None, None, None]
        for _ in range(3):
            seq2d_input = seq2d_net(seq2d_input)
        out_seq2d = seq2d_input[3]
        pre_diff = torch.max(torch.abs(out_seq - out_seq2d)).item()
    assert pre_diff < 1e-6, f"Pre-training diff {pre_diff:.2e}"

    # Train Sequential
    criterion = nn.MSELoss()
    optimizer_seq = torch.optim.SGD(seq_net.parameters(), lr=0.01)
    optimizer_seq.zero_grad()
    loss_seq = criterion(seq_net(x), target)
    loss_seq.backward()
    optimizer_seq.step()

    # Train Sequential2D
    optimizer_seq2d = torch.optim.SGD(seq2d_net.parameters(), lr=0.01)
    optimizer_seq2d.zero_grad()
    seq2d_input = [x, None, None, None]
    for _ in range(3):
        seq2d_input = seq2d_net(seq2d_input)
    loss_seq2d = criterion(seq2d_input[3], target)
    loss_seq2d.backward()
    optimizer_seq2d.step()

    # Verify losses match
    loss_diff = abs(loss_seq.item() - loss_seq2d.item())
    assert loss_diff < 1e-6, f"Loss diff {loss_diff:.2e}"

    # Verify outputs match after training
    with torch.no_grad():
        test_x = torch.randn(16, input_size)
        out_seq_after = seq_net(test_x)
        seq2d_test = [test_x, None, None, None]
        for _ in range(3):
            seq2d_test = seq2d_net(seq2d_test)
        out_seq2d_after = seq2d_test[3]
        post_diff = torch.max(torch.abs(out_seq_after - out_seq2d_after)).item()
    assert post_diff < 1e-6, f"Post-training diff {post_diff:.2e}"


def test_complex_sequential2d_tensor_input():
    torch.manual_seed(42)

    cfg = {
        'in_features_list': [50, 100, 200, 150],
        'out_features_list': [100, 200, 150, 10],
        'block_types': [
            ['Linear', 'Linear', None, None],
            [None, 'Linear', 'Linear', 'Linear'],
            [None, None, 'Linear', 'Linear'],
            [None, None, None, 'Linear'],
        ],
    }
    complex_net = Sequential2D.from_config(cfg)

    batch_size = 16
    total_input_size = 50 + 100 + 200 + 150  # 500
    test_input = torch.randn(batch_size, total_input_size)

    output = complex_net(test_input)
    assert output.shape == (16, 460), f"Expected (16, 460), got {output.shape}"


def test_complex_sequential2d_list_input():
    torch.manual_seed(42)

    cfg = {
        'in_features_list': [50, 100, 200, 150],
        'out_features_list': [100, 200, 150, 10],
        'block_types': [
            ['Linear', 'Linear', None, None],
            [None, 'Linear', 'Linear', 'Linear'],
            [None, None, 'Linear', 'Linear'],
            [None, None, None, 'Linear'],
        ],
    }
    complex_net = Sequential2D.from_config(cfg)

    batch_size = 16
    input_list = [torch.randn(batch_size, 50), None, None, None]

    output_list = complex_net.forward_list(input_list)

    assert output_list[0] is not None, "Path 0 should have output"
    assert output_list[0].shape == (16, 100), f"Path 0: expected (16, 100), got {output_list[0].shape}"
    assert output_list[1] is not None, "Path 1 should have output"
    assert output_list[1].shape == (16, 200), f"Path 1: expected (16, 200), got {output_list[1].shape}"
    assert output_list[2] is None, "Path 2 should be None"
    assert output_list[3] is None, "Path 3 should be None"
