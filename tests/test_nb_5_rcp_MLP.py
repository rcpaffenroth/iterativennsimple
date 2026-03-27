"""
Tests for notebook 5-rcp-MLP.ipynb

Builds a Sequential2D MLP with Identity and Sequential1D blocks on MNIST,
verifies mask computation, model construction, forward pass, and training.
"""

import torch
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.Sequential1D import Sequential1D
from generatedata.load_data import load_data


def _df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)


def _transpose_blocks(blocks):
    return [[blocks[j][i] for j in range(len(blocks))] for i in range(len(blocks[0]))]


def _load_mnist_data(max_num_samples=200):
    """Load MNIST data and compute masks. Returns tensors and mask info."""
    data_dict = load_data('MNIST')
    z_start = data_dict['start']
    z_target = data_dict['target']

    z_start_tensor = _df_to_tensor(z_start)
    z_target_tensor = _df_to_tensor(z_target)

    # Limit samples for speed
    num_samples = min(max_num_samples, z_start_tensor.shape[0])
    z_start_tensor = z_start_tensor[:num_samples]
    z_target_tensor = z_target_tensor[:num_samples]

    # Compute masks: columns where start == target are x (identity), rest are y (to predict)
    mask = (z_start_tensor == z_target_tensor).all(axis=0)
    x_mask = mask
    y_mask = ~mask

    return z_start_tensor, z_target_tensor, x_mask, y_mask


def _build_model(input_size, h1_size, h2_size, output_size):
    """Build the Sequential2D MLP model as in the notebook."""
    I = Identity(in_features=input_size, out_features=input_size)
    f1 = Sequential1D(
        torch.nn.Linear(in_features=input_size, out_features=h1_size),
        torch.nn.ReLU(),
        in_features=input_size, out_features=h1_size,
    )
    f2 = Sequential1D(
        torch.nn.Linear(in_features=h1_size, out_features=h2_size),
        torch.nn.ReLU(),
        in_features=h1_size, out_features=h2_size,
    )
    f3 = torch.nn.Linear(in_features=h2_size, out_features=output_size)

    in_features_list = [input_size, h1_size, h2_size, output_size]
    out_features_list = [input_size, h1_size, h2_size, output_size]
    blocks = [
        [I,    None, None, None],
        [f1,   None, None, None],
        [None, f2,   None, None],
        [None, None, f3,   None],
    ]

    model = Sequential2D(
        in_features_list=in_features_list,
        out_features_list=out_features_list,
        blocks=_transpose_blocks(blocks),
    )
    return model, in_features_list


def test_mask_computation():
    _, _, x_mask, y_mask = _load_mnist_data()

    # For MNIST: 784 image pixels are x (identity), 10 label columns are y
    assert int(x_mask.sum()) == 784, f"Expected x_mask.sum()==784, got {int(x_mask.sum())}"
    assert int(y_mask.sum()) == 10, f"Expected y_mask.sum()==10, got {int(y_mask.sum())}"


def test_model_construction():
    torch.manual_seed(42)
    input_size = 784
    h1_size = 20
    h2_size = 20
    output_size = 10

    model, _ = _build_model(input_size, h1_size, h2_size, output_size)

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0

    # f1: Linear(784, 20) = 784*20+20 = 15700
    # f2: Linear(20, 20) = 20*20+20 = 420
    # f3: Linear(20, 10) = 20*10+10 = 210
    expected = 15700 + 420 + 210
    assert num_params == expected, f"Expected {expected} params, got {num_params}"


def test_forward_pass_shape():
    torch.manual_seed(42)
    input_size = 784
    h1_size = 20
    h2_size = 20
    output_size = 10
    total_size = input_size + h1_size + h2_size + output_size
    iterations = 3

    model, _ = _build_model(input_size, h1_size, h2_size, output_size)

    batch_size = 16
    x = torch.randn(batch_size, total_size)

    mapped = x
    for _ in range(iterations):
        mapped = model(mapped)

    assert mapped.shape == (batch_size, total_size), \
        f"Expected ({batch_size}, {total_size}), got {mapped.shape}"


def test_training_reduces_loss():
    torch.manual_seed(42)
    z_start_tensor, z_target_tensor, x_mask, y_mask = _load_mnist_data(max_num_samples=200)

    input_size = int(x_mask.sum())
    h1_size = 20
    h2_size = 20
    output_size = int(y_mask.sum())
    iterations = 3

    x_idx = torch.arange(0, input_size)
    h_idx = torch.arange(input_size, input_size + h1_size + h2_size)
    y_idx = torch.arange(input_size + h1_size + h2_size,
                         input_size + h1_size + h2_size + output_size)

    model, _ = _build_model(input_size, h1_size, h2_size, output_size)

    # Build augmented tensors (with zero hidden state)
    zh_start_tensor = torch.cat((
        z_start_tensor[:, x_mask],
        torch.zeros(z_start_tensor.shape[0], len(h_idx)),
        z_start_tensor[:, y_mask],
    ), dim=1)
    zh_target_tensor = torch.cat((
        z_target_tensor[:, x_mask],
        torch.zeros(z_target_tensor.shape[0], len(h_idx)),
        z_target_tensor[:, y_mask],
    ), dim=1)

    # Simple dataset class
    class Data(torch.utils.data.Dataset):
        def __init__(self, z_start, z_target):
            self.z_start = z_start
            self.z_target = z_target

        def __len__(self):
            return len(self.z_start)

        def __getitem__(self, idx):
            return self.z_start[idx], self.z_target[idx]

    train_data = Data(zh_start_tensor, zh_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Record initial loss
    with torch.no_grad():
        mapped = zh_start_tensor
        for _ in range(iterations):
            mapped = model(mapped)
        initial_loss = criterion(mapped[:, y_idx], zh_target_tensor[:, y_idx]).item()

    # Train for 20 epochs (reduced from 500)
    for epoch in range(20):
        for start, target in train_loader:
            optimizer.zero_grad()
            mapped = start
            for _ in range(iterations):
                mapped = model(mapped)
            loss = criterion(mapped[:, y_idx], target[:, y_idx])
            loss.backward()
            optimizer.step()

    # Verify loss decreased
    with torch.no_grad():
        mapped = zh_start_tensor
        for _ in range(iterations):
            mapped = model(mapped)
        final_loss = criterion(mapped[:, y_idx], zh_target_tensor[:, y_idx]).item()

    assert final_loss < initial_loss, \
        f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
