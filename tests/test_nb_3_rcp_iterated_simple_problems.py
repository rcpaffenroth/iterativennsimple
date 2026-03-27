"""
Tests for notebook 3-rcp-iterated-simple-problems.ipynb

Trains an iterated Sequential model (same map applied multiple times)
on simple 2D regression problems and verifies training reduces loss.
"""

import torch
from generatedata.load_data import load_data
from generatedata.df_to_tensor import df_to_tensor
from generatedata.StartTargetData import StartTargetData


def test_iterated_model_construction():
    data_dict = load_data('regression_line')
    z_size = data_dict['start'].shape[1]
    hidden_size = 15

    model = torch.nn.Sequential(
        torch.nn.Linear(z_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, z_size),
    )

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0

    # Verify output shape matches input shape (required for iteration)
    x = torch.randn(16, z_size)
    out = model(x)
    assert out.shape == x.shape, \
        f"Output shape {out.shape} must match input shape {x.shape} for iteration"


def test_iterated_forward_pass():
    torch.manual_seed(42)

    z_size = 2
    hidden_size = 15
    iterations = 10

    model = torch.nn.Sequential(
        torch.nn.Linear(z_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, z_size),
    )

    x = torch.randn(16, z_size)
    mapped = x
    for _ in range(iterations):
        mapped = model(mapped)
        assert mapped.shape == (16, z_size), \
            f"Shape changed during iteration: {mapped.shape}"


def test_iterated_model_trains():
    torch.manual_seed(42)

    data_dict = load_data('regression_line')
    z_start = data_dict['start']
    z_target = data_dict['target']
    z_size = z_start.shape[1]
    hidden_size = 15
    iterations = 10

    model = torch.nn.Sequential(
        torch.nn.Linear(z_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, z_size),
    )

    z_start_tensor = df_to_tensor(z_start)
    z_target_tensor = df_to_tensor(z_target)

    train_data = StartTargetData(z_start_tensor, z_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Record initial loss (with iterations)
    with torch.no_grad():
        mapped = z_start_tensor
        for _ in range(iterations):
            mapped = model(mapped)
        initial_loss = criterion(mapped, z_target_tensor).item()

    # Train for 20 epochs (reduced from 500)
    max_epochs = 20
    for epoch in range(max_epochs):
        for start, target in train_loader:
            optimizer.zero_grad()
            mapped = start
            loss = 0.0
            for i in range(iterations):
                mapped = model(mapped)
                loss += criterion(mapped, target)
            loss.backward()
            optimizer.step()

    # Verify loss decreased
    with torch.no_grad():
        mapped = z_start_tensor
        for _ in range(iterations):
            mapped = model(mapped)
        final_loss = criterion(mapped, z_target_tensor).item()

    assert final_loss < initial_loss, \
        f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
