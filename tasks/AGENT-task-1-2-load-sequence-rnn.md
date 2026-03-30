# Task 1-2 — RNN, LSTM, and GRU Models

## Goal

Create the second section of the notebook. Define simple RNN, LSTM, and GRU sequence classifiers, a shared training loop, and train/evaluate all three.

## Prerequisites

This section assumes all variables from Task 1-1 are available:
- `train_loader`, `val_loader` — PyTorch DataLoaders
- `input_dim` (38), `label_dim` (10), `seq_len` (28)
- `hidden_size` (64), `num_epochs` (10), `learning_rate` (1e-3)
- `device`, `batch_size`

## Cells to Create

### Cell 1 — Markdown: Section header

```markdown
## 2. Baseline Models: RNN, LSTM, and GRU

We define three simple recurrent models. Each processes the sequence one timestep at a time and produces a classification from the hidden state at the **last** timestep.

All three models share the same structure:
1. A recurrent layer (`nn.RNN`, `nn.LSTM`, or `nn.GRU`) that processes the sequence
2. A linear layer that maps the final hidden state to class logits
```

### Cell 2 — Code: Model definitions

Define a single flexible class that wraps any of `nn.RNN`, `nn.LSTM`, or `nn.GRU`:

```python
class SimpleRecurrentClassifier(nn.Module):
    """A simple sequence classifier using RNN, LSTM, or GRU.

    Args:
        rnn_type: One of "RNN", "LSTM", or "GRU".
        input_dim: Number of features per timestep.
        hidden_size: Hidden state dimension.
        label_dim: Number of output classes.
    """

    def __init__(self, rnn_type, input_dim, hidden_size, label_dim):
        super().__init__()
        # Select the recurrent layer type
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, label_dim)
        self.rnn_type = rnn_type

    def forward(self, x_seq):
        """
        Args:
            x_seq: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, label_dim)
        """
        # Run the recurrent layer over the full sequence
        output, _ = self.rnn(x_seq)  # output: (batch, seq_len, hidden_size)
        # Take the hidden state at the last timestep
        last_hidden = output[:, -1, :]  # (batch, hidden_size)
        # Map to class logits
        logits = self.fc(last_hidden)  # (batch, label_dim)
        return logits
```

**Important notes for the student:**
- `batch_first=True` means the input shape is `(batch, seq_len, features)` instead of `(seq_len, batch, features)`.
- For LSTM, `self.rnn(x_seq)` returns `(output, (h_n, c_n))`, but we only use `output` so the `_` catches both hidden and cell states.
- We only classify based on the last timestep's output. This is the simplest approach.

### Cell 3 — Markdown: Training loop explanation

```markdown
### Training Loop

We define a simple training function that works for any model that takes `x_seq` as input and returns `logits`. We use `CrossEntropyLoss` and the `Adam` optimizer.
```

### Cell 4 — Code: Training function

```python
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Train a model and print train/val accuracy each epoch.

    Args:
        model: Any nn.Module with forward(x_seq) -> logits.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        device: torch device (cpu or cuda).

    Returns:
        dict with 'train_acc', 'val_acc' (lists of per-epoch accuracies).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        correct, total = 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (logits.detach().argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        train_acc = correct / total * 100

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        val_acc = correct / total * 100

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:2d}/{num_epochs}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

    return history
```

### Cell 5 — Markdown: Train the models

```markdown
### Train RNN, LSTM, and GRU

We create one model of each type and train them with the same hyperparameters.
```

### Cell 6 — Code: Train all three models

```python
results = {}

for rnn_type in ["RNN", "LSTM", "GRU"]:
    print(f"\n{'='*50}")
    print(f"Training {rnn_type}")
    n_params = sum(p.numel() for p in SimpleRecurrentClassifier(rnn_type, input_dim, hidden_size, label_dim).parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*50}")

    model = SimpleRecurrentClassifier(rnn_type, input_dim, hidden_size, label_dim)
    history = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    results[rnn_type] = history

print("\n--- Summary ---")
for name, hist in results.items():
    print(f"  {name}: best val_acc = {max(hist['val_acc']):.1f}%")
```

## Verification

- All three models should train without errors.
- Validation accuracy should be well above random (10% for MNIST).
- With `hidden_size=64` and 10 epochs, expect roughly 85–95% val accuracy for LSTM/GRU, possibly lower for vanilla RNN.
- The `results` dictionary should have keys `"RNN"`, `"LSTM"`, `"GRU"`, each with `train_acc` and `val_acc` lists.
