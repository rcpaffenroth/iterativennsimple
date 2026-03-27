"""
Tests for notebook 4-rcp-pulled-apart.ipynb

Defines a custom MLP class with raw tensor parameters and skip connections,
trains on MNIST1D, and verifies model construction and training.
"""

import torch
from generatedata.load_data import load_data
from generatedata.df_to_tensor import df_to_tensor
from generatedata.StartTargetData import StartTargetData


# Inline MLP class from the notebook
class MLP(torch.nn.Module):
    def __init__(self, x_idx, y_idx, hidden_size=100):
        super(MLP, self).__init__()
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.hidden_size = hidden_size

        self.sigma = torch.nn.ReLU()
        self.reset()

    def reset(self):
        with torch.no_grad():
            self.W1_raw = torch.zeros(size=(self.hidden_size, len(self.x_idx)),
                                      requires_grad=True, dtype=torch.float32)
            self.W2_raw = torch.zeros(size=(self.hidden_size, self.hidden_size),
                                      requires_grad=True, dtype=torch.float32)
            self.W3_raw = torch.zeros(size=(len(self.y_idx), self.hidden_size),
                                      requires_grad=True, dtype=torch.float32)

            self.b1_raw = torch.zeros(size=(self.hidden_size,))
            self.b2_raw = torch.zeros(size=(self.hidden_size,))
            self.b3_raw = torch.zeros(size=(len(self.y_idx),))

            torch.nn.init.kaiming_uniform_(self.W1_raw, a=5**0.5)
            torch.nn.init.kaiming_uniform_(self.W2_raw, a=5**0.5)
            torch.nn.init.kaiming_uniform_(self.W3_raw, a=5**0.5)
            torch.nn.init.zeros_(self.b1_raw)
            torch.nn.init.zeros_(self.b2_raw)
            torch.nn.init.zeros_(self.b3_raw)

            self.W1 = torch.nn.Parameter(self.W1_raw)
            self.W2 = torch.nn.Parameter(self.W2_raw)
            self.W3 = torch.nn.Parameter(self.W3_raw)
            self.b1 = torch.nn.Parameter(self.b1_raw)
            self.b2 = torch.nn.Parameter(self.b2_raw)
            self.b3 = torch.nn.Parameter(self.b3_raw)

    def forward(self, z):
        x = z[:, self.x_idx]

        h_1 = x @ self.W1.T + self.b1
        h_1 = self.sigma(h_1)

        # Skip connection (matches MNIST1D reference implementation)
        h_2 = h_1 + h_1 @ self.W2.T + self.b2
        h_2 = self.sigma(h_2)

        y = h_2 @ self.W3.T + self.b3
        y = torch.nn.functional.log_softmax(y, dim=1)

        return torch.cat([z[:, self.x_idx], y], dim=1)


def _load_mnist1d():
    """Load MNIST1D data and return tensors, indices, and data loader."""
    name = 'MNIST1D'
    z_size = 50
    x_idx = range(40)
    y_idx = range(40, 50)

    data_dict = load_data(name)
    z_start = data_dict['start']
    x_target = data_dict['target']

    z_start_tensor = df_to_tensor(z_start)
    x_target_tensor = df_to_tensor(x_target)

    train_data = StartTargetData(z_start_tensor, x_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

    return z_start_tensor, x_target_tensor, x_idx, y_idx, train_loader


def test_data_loading():
    data_dict = load_data('MNIST1D')
    z_start = data_dict['start']
    x_target = data_dict['target']
    assert len(z_start) > 0
    assert z_start.shape[1] == 50, f"Expected 50 columns, got {z_start.shape[1]}"
    assert x_target.shape[1] == 50, f"Expected 50 columns, got {x_target.shape[1]}"


def test_mlp_construction():
    torch.manual_seed(42)
    x_idx = range(40)
    y_idx = range(40, 50)
    model = MLP(x_idx, y_idx)

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0

    # W1: (100, 40), W2: (100, 100), W3: (10, 100), b1: 100, b2: 100, b3: 10
    expected_params = 100 * 40 + 100 * 100 + 10 * 100 + 100 + 100 + 10
    assert num_params == expected_params, \
        f"Expected {expected_params} params, got {num_params}"


def test_mlp_forward_pass():
    torch.manual_seed(42)
    x_idx = range(40)
    y_idx = range(40, 50)
    model = MLP(x_idx, y_idx)

    batch_size = 16
    z = torch.randn(batch_size, 50)
    out = model(z)

    # Output should have same total size (40 x-features + 10 y-features)
    assert out.shape == (batch_size, 50), f"Expected (16, 50), got {out.shape}"

    # The x part should be preserved (identity)
    assert torch.allclose(out[:, :40], z[:, :40]), \
        "x-features should pass through unchanged"

    # The y part should be log-softmax (sum of exp should be ~1)
    log_probs = out[:, 40:]
    probs = torch.exp(log_probs)
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), \
        f"log_softmax output should sum to 1, got {prob_sums}"


def test_mlp_training():
    torch.manual_seed(42)
    z_start_tensor, x_target_tensor, x_idx, y_idx, train_loader = _load_mnist1d()

    model = MLP(x_idx, y_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()

    # Record initial loss
    with torch.no_grad():
        z_pred = model(z_start_tensor)
        z_target_labels = torch.argmax(x_target_tensor[:, y_idx], dim=1)
        initial_loss = loss_fn(z_pred[:, y_idx], z_target_labels).item()

    # Train for 20 epochs (reduced from 150)
    for epoch in range(20):
        for z_start, z_target in train_loader:
            optimizer.zero_grad()
            z_pred = model.forward(z_start)
            z_target_labels = torch.argmax(z_target[:, y_idx], dim=1)
            loss = loss_fn(z_pred[:, y_idx], z_target_labels)
            loss.backward()
            optimizer.step()

    # Verify loss decreased
    with torch.no_grad():
        z_pred = model(z_start_tensor)
        z_target_labels = torch.argmax(x_target_tensor[:, y_idx], dim=1)
        final_loss = loss_fn(z_pred[:, y_idx], z_target_labels).item()

    assert final_loss < initial_loss, \
        f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
