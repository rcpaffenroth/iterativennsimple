"""
Tests for notebook 2-rcp-seqential-simple-problems.ipynb

Trains a Sequential model on simple 2D regression problems and verifies
that the model can be constructed and that training reduces loss.
"""

import torch
from generatedata.load_data import load_data
from generatedata.df_to_tensor import df_to_tensor
from generatedata.StartTargetData import StartTargetData


def test_data_loading():
    data_dict = load_data('regression_line')
    z_start = data_dict['start']
    z_target = data_dict['target']
    assert len(z_start) > 0
    assert len(z_target) > 0
    assert z_start.shape[1] == 2
    assert z_target.shape[1] == 2


def test_sequential_model_construction():
    data_dict = load_data('regression_line')
    z_size = data_dict['start'].shape[1]
    hidden_size = 20

    model = torch.nn.Sequential(
        torch.nn.Linear(z_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, z_size),
    )

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0

    # Check forward pass shape
    x = torch.randn(16, z_size)
    out = model(x)
    assert out.shape == (16, z_size)


def test_sequential_model_trains():
    torch.manual_seed(42)

    data_dict = load_data('regression_line')
    z_start = data_dict['start']
    z_target = data_dict['target']
    z_size = z_start.shape[1]
    hidden_size = 20

    model = torch.nn.Sequential(
        torch.nn.Linear(z_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, z_size),
    )

    z_start_tensor = df_to_tensor(z_start)
    z_target_tensor = df_to_tensor(z_target)

    train_data = StartTargetData(z_start_tensor, z_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Record initial loss
    with torch.no_grad():
        initial_loss = criterion(model(z_start_tensor), z_target_tensor).item()

    # Train for 20 epochs (reduced from 500)
    max_epochs = 20
    for epoch in range(max_epochs):
        for start, target in train_loader:
            optimizer.zero_grad()
            mapped = model(start)
            loss = criterion(mapped, target)
            loss.backward()
            optimizer.step()

    # Verify loss decreased
    with torch.no_grad():
        final_loss = criterion(model(z_start_tensor), z_target_tensor).item()

    assert final_loss < initial_loss, \
        f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
