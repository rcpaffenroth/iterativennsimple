# Task 1-1 — Data Loading and Exploration

## Goal

Create the first section of the notebook `notebooks/advanced/11-rcp-load-sequence-example.ipynb`. This section handles imports, data loading with `load_data_as_sequence`, train/val splitting, and building PyTorch DataLoaders.

## Context

The `load_data_as_sequence` function lives in the external `generatedata` package. It loads a flat dataset and reshapes it into a sequence:

```python
from generatedata.load_data import load_data_as_sequence

X_seq, labels = load_data_as_sequence(
    name="MNIST",
    step_size=28,
    label_every_step=True,
)
# X_seq  shape: (num_points, seq_len, step_size + label_dim)
#   For MNIST with step_size=28: (60000, 28, 38)  — 28 pixels + 10 one-hot label
# labels shape: (num_points, label_dim)
#   For MNIST: (60000, 10)  — one-hot encoded
```

- `step_size=28` means each timestep is one row of the 28×28 image → `seq_len = 784 // 28 = 28`.
- `label_every_step=True` appends the one-hot label (10 dims for MNIST) to each timestep's input, so `input_dim = 28 + 10 = 38`.

## Cells to Create

### Cell 1 — Markdown: Title and introduction

```markdown
# 11. Load Sequence Example: RNN vs LSTM vs GRU vs MonarchLinear

This notebook demonstrates how to:
1. Load a dataset as a sequence using `load_data_as_sequence`
2. Train and compare RNN, LSTM, and GRU models on the sequence data
3. Build an iterative neural network using `Sequential2D` with `MonarchLinear` blocks
4. Iterate the Sequential2D model over multiple time steps

We use MNIST as our dataset, treating each row of pixels (28 values) as one timestep in a sequence of length 28.
```

### Cell 2 — Code: Imports

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from generatedata.load_data import load_data_as_sequence

from iterativennsimple.MonarchLinear import MonarchLinear
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.Sequential2D import Sequential2D, Identity

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Cell 3 — Markdown: Data loading explanation

```markdown
## 1. Load Data as a Sequence

We use `load_data_as_sequence` to load MNIST and reshape it into a sequence.

- **step_size=28**: Each timestep contains one row of the 28×28 image (28 pixel values).
- **label_every_step=True**: The ground-truth one-hot label (10 dimensions) is appended to each timestep's input. This gives the model access to the label at every step — useful for the iterative model to refine its predictions.

The result is:
- `X_seq`: shape `(N, 28, 38)` — 28 timesteps, each with 28 pixels + 10 label values
- `labels`: shape `(N, 10)` — one-hot encoded class labels
```

### Cell 4 — Code: Load and split data

```python
# Load MNIST as a sequence
X_seq, labels = load_data_as_sequence(
    name="MNIST",
    step_size=28,
    label_every_step=True,
)

# Convert to PyTorch tensors
X_seq = torch.from_numpy(X_seq.astype(np.float32))
labels = torch.from_numpy(labels.astype(np.float32))

# Integer class labels for CrossEntropyLoss
y_cls = labels.argmax(dim=1)  # (N,)

# Dimensions
N = len(X_seq)
seq_len = X_seq.shape[1]
input_dim = X_seq.shape[2]     # step_size + label_dim = 28 + 10 = 38
label_dim = labels.shape[1]    # 10 for MNIST

print(f"Dataset: MNIST")
print(f"  N={N}, seq_len={seq_len}, input_dim={input_dim}, label_dim={label_dim}")
print(f"  X_seq shape: {X_seq.shape}")
print(f"  labels shape: {labels.shape}")
```

### Cell 5 — Code: Train/val split and DataLoaders

```python
# Train/validation split
val_fraction = 1 / 7
rng = np.random.default_rng(42)
idx = rng.permutation(N)
n_val = max(1, int(N * val_fraction))
val_idx, train_idx = idx[:n_val], idx[n_val:]

X_train, y_train = X_seq[train_idx], y_cls[train_idx]
X_val, y_val = X_seq[val_idx], y_cls[val_idx]

print(f"  train={len(y_train)}, val={len(y_val)}")

# DataLoaders
batch_size = 256
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
```

### Cell 6 — Markdown: Hyperparameters

```markdown
## Hyperparameters

We use small models and few epochs so the notebook runs quickly. These are not tuned for best accuracy — the goal is to demonstrate the workflow.
```

### Cell 7 — Code: Hyperparameters

```python
# Shared hyperparameters
hidden_size = 64   # Hidden state size for RNN/LSTM/GRU and MonarchLinear
num_epochs = 10
learning_rate = 1e-3
```

## Verification

After running these cells, the following should be true:
- `X_train.shape` is approximately `(51429, 28, 38)` (6/7 of 60000)
- `X_val.shape` is approximately `(8571, 28, 38)` (1/7 of 60000)
- `y_train` and `y_val` contain integer class labels 0–9
- `train_loader` and `val_loader` are ready for use
- `input_dim == 38`, `label_dim == 10`, `seq_len == 28`
