# Task 1-3 — Sequential2D with MonarchLinear

## Goal

Create the third section of the notebook. Build a `Sequential2D` map with `Identity` and `MonarchLinear` blocks and wrap it as a sequence classifier. This section explains the mathematical structure and implements it.

## Prerequisites

All variables from Tasks 1-1 and 1-2 are available, plus the imports:
- `MonarchLinear`, `Sequential1D`, `Sequential2D`, `Identity` (imported in Task 1-1)
- `input_dim` (38), `label_dim` (10), `hidden_size` (64)
- `train_loader`, `val_loader`, `device`

## Background: The Sequential2D Map

The key idea is to represent the model's state as a vector with three parts:

$$\text{state} = \begin{bmatrix} x \\ y \\ h \end{bmatrix}$$

where:
- $x$ = input features (dim = `input_dim` = 38)
- $y$ = predicted labels (dim = `label_dim` = 10)
- $h$ = hidden state (dim = `hidden_size` = 64)

The `Sequential2D` map applies a block matrix to update the state:

$$\begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix} \begin{bmatrix} x_k \\ y_k \\ h_k \end{bmatrix} = \begin{bmatrix} x_{k+1} \\ y_{k+1} \\ h_{k+1} \end{bmatrix}$$

- **Row 0** (Identity): The input $x$ passes through unchanged — it's "frozen" during iterations.
- **Row 1** ($y$ update): The new $y$ is computed from all three parts of the state via MonarchLinear blocks $M_1, M_2, M_3$.
- **Row 2** ($h$ update): The new $h$ is computed from all three parts via MonarchLinear blocks $M_4, M_5, M_6$.

The `MonarchLinear` blocks are sparse linear layers that use the Monarch matrix format for efficiency.

## Cells to Create

### Cell 1 — Markdown: Section header and math explanation

```markdown
## 3. Sequential2D with MonarchLinear

Instead of a traditional recurrent layer, we can use a **block-structured linear map** to update the model's state. The state vector is split into three groups:

- **x** (input, dim=38): The current timestep's input features
- **y** (output, dim=10): The predicted class probabilities
- **h** (hidden, dim=64): A learned hidden representation

At each application of the map, the state is updated as:

$$\begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix} \begin{bmatrix} x_k \\ y_k \\ h_k \end{bmatrix} = \begin{bmatrix} x_{k+1} \\ y_{k+1} \\ h_{k+1} \end{bmatrix}$$

The identity block in the top-left ensures the input is preserved. The $M_i$ blocks are `MonarchLinear` layers — sparse linear maps that are efficient on GPUs.
```

### Cell 2 — Code: Helper to build MonarchLinear blocks

This helper creates a `MonarchLinear` block wrapped in a `Sequential1D`. The wrapping is needed so `Sequential2D` can query `in_features` and `out_features`. Optionally prepend a ReLU activation.

```python
def make_monarch_block(in_features, out_features, num_blocks=4, activation=None):
    """Create a MonarchLinear block wrapped in Sequential1D.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        num_blocks: Number of blocks in the Monarch matrix.
        activation: Optional activation ("ReLU", "ELU", "Tanh", or None).

    Returns:
        Sequential1D wrapping the MonarchLinear layer.
    """
    # Adjust num_blocks so it divides both dimensions evenly
    nb = num_blocks
    while nb > 1 and (in_features % nb != 0 or out_features % nb != 0):
        nb -= 1

    layer = MonarchLinear.from_uniform_blocks(
        in_features=in_features,
        out_features=out_features,
        num_blocks=nb,
        bias=True,
    )

    if activation is not None:
        act_fn = {"ReLU": nn.ReLU, "ELU": nn.ELU, "Tanh": nn.Tanh}[activation]
        seq = nn.Sequential(act_fn(), layer)
    else:
        seq = nn.Sequential(layer)

    return Sequential1D(seq, in_features=in_features, out_features=out_features)
```

### Cell 3 — Code: Build the Sequential2D map

```python
def build_seq2d_map(input_dim, label_dim, hidden_size, num_blocks=4):
    """Build a Sequential2D block matrix map.

    The state is [x | y | h] with dimensions [input_dim | label_dim | hidden_size].

    Block layout:
        [Identity,  None,       None      ]   <- x is preserved
        [M1,        M2,         M3        ]   <- y updated from all
        [M4,        M5,         M6        ]   <- h updated from all

    Args:
        input_dim: Dimension of input features (x).
        label_dim: Dimension of labels (y).
        hidden_size: Dimension of hidden state (h).
        num_blocks: Number of Monarch blocks per layer.

    Returns:
        Sequential2D module.
    """
    sizes = [input_dim, label_dim, hidden_size]

    # Create the 3x3 block grid (None = zero block)
    blocks = [[None, None, None],
              [None, None, None],
              [None, None, None]]

    # Row 0: Identity for x
    blocks[0][0] = Identity(in_features=input_dim, out_features=input_dim)

    # Row 1: y update — M1(x->y), M2(y->y), M3(h->y)
    blocks[1][0] = make_monarch_block(input_dim, label_dim, num_blocks)
    blocks[1][1] = make_monarch_block(label_dim, label_dim, num_blocks)
    blocks[1][2] = make_monarch_block(hidden_size, label_dim, num_blocks)

    # Row 2: h update — M4(x->h), M5(y->h), M6(h->h)
    blocks[2][0] = make_monarch_block(input_dim, hidden_size, num_blocks, activation="ReLU")
    blocks[2][1] = make_monarch_block(label_dim, hidden_size, num_blocks, activation="ReLU")
    blocks[2][2] = make_monarch_block(hidden_size, hidden_size, num_blocks, activation="ReLU")

    return Sequential2D(sizes, sizes, blocks)
```

### Cell 4 — Markdown: Explain the sequence classifier wrapper

```markdown
### Wrapping as a Sequence Classifier

To process a sequence, we:
1. Initialize the state with zeros (and a uniform guess for y).
2. At each timestep, overwrite the x-slot with the current input.
3. Apply the Sequential2D map once (we'll explore multiple iterations in the next section).
4. After the last timestep, read the y-slot as our prediction.
```

### Cell 5 — Code: MonarchSequenceClassifier (single iteration)

```python
class MonarchSequenceClassifier(nn.Module):
    """Sequence classifier using a Sequential2D block map.

    At each timestep:
      1. Overwrite the x-slot with the current input
      2. Apply the Sequential2D map (once per timestep for now)
      3. Read the y-slot as the prediction at the final timestep

    Args:
        input_dim: Features per timestep.
        label_dim: Number of output classes.
        hidden_size: Hidden state dimension.
        num_blocks: Monarch blocks per layer.
    """

    def __init__(self, input_dim, label_dim, hidden_size, num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.hidden_size = hidden_size
        self.seq2d = build_seq2d_map(input_dim, label_dim, hidden_size, num_blocks)

    def forward(self, x_seq, iterations=1):
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
            iterations: Number of times to apply the map per timestep.

        Returns:
            logits: (batch, label_dim) — prediction from the final timestep.
        """
        B, T, _ = x_seq.shape
        total_dim = self.input_dim + self.label_dim + self.hidden_size

        # Initialize state: [x | y | h] with uniform y, zero h
        state = torch.zeros(B, total_dim, device=x_seq.device)
        # Initialize y-slot with uniform distribution
        state[:, self.input_dim:self.input_dim + self.label_dim] = 1.0 / self.label_dim

        for t in range(T):
            # Overwrite x-slot with current timestep's input
            state[:, :self.input_dim] = x_seq[:, t, :]

            # Apply the map `iterations` times
            for _ in range(iterations):
                state = self.seq2d(state)

        # Read y-slot as logits
        logits = state[:, self.input_dim:self.input_dim + self.label_dim]
        return logits
```

### Cell 6 — Code: Train the Monarch model (1 iteration)

```python
print(f"\n{'='*50}")
print("Training MonarchLinear Sequential2D (1 iteration per timestep)")
monarch_model = MonarchSequenceClassifier(input_dim, label_dim, hidden_size, num_blocks=4)
n_params = sum(p.numel() for p in monarch_model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")
print(f"{'='*50}")

history = train_model(monarch_model, train_loader, val_loader, num_epochs, learning_rate, device)
results["Monarch (1 iter)"] = history

print(f"\n  Best val_acc = {max(history['val_acc']):.1f}%")
```

## Important Notes

- The `Sequential2D` constructor takes `(in_features_list, out_features_list, blocks)` where `blocks` is a list-of-lists of modules (or None for zero blocks).
- `Sequential1D` is a thin wrapper around `nn.Sequential` that adds `in_features` and `out_features` attributes, which `Sequential2D` needs.
- The `train_model` function from Task 1-2 works here because `MonarchSequenceClassifier.forward` returns logits just like the RNN models. Note: when calling with `iterations=1` (default), the model applies the map once per timestep.

## Verification

- The model should train without errors.
- Validation accuracy should be above random (10%).
- The model should have significantly fewer parameters than the LSTM (due to Monarch sparsity).
