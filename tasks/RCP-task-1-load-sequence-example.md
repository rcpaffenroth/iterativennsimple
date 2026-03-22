In generateddata.load_data.load_data_as_sequence function, the data is loaded as a sequence of tokens. Each token represents a specific element in the data, such as a word, character, or symbol. The function processes the input data and converts it into a format that can be used for training machine learning models or performing other analyses. The sequence of tokens allows for efficient handling of the data and enables various natural language processing tasks to be performed effectively.

The code looks like

```python
def load_data_as_sequence(
    name: str,
    step_size: int,
    local: bool = False,
    data_dir: Path | str | None = None,
    label_every_step: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load any dataset with x_y_index and reshape it into a sequence.

    The sequence length is computed as x_y_index // step_size.  This allows
    any flat dataset to be treated as a time-series without storing sequence
    metadata in info.json.

    Args:
        name: Dataset name.
        step_size: Number of feature values per timestep.
        local: If True, load from local processed data directory.
        data_dir: Override the default data directory.
        label_every_step: If True, broadcast labels across all timesteps
            and concatenate with pixel sequence; if False, return pixels only.

    Returns:
        (X_seq, labels) where X_seq has shape
        (num_points, seq_len, step_size [+ label_dim]) and
        labels has shape (num_points, label_dim).

    Raises:
        ValueError: If x_y_index is missing or not divisible by step_size.
    """
    data = load_data(name, local=local, data_dir=data_dir)
    info = data["info"]

    if "x_y_index" not in info:
        raise ValueError(
            f"Dataset '{name}' has no x_y_index metadata. Cannot reshape as sequence."
        )

    x_y_index = info["x_y_index"]

    if x_y_index % step_size != 0:
        raise ValueError(
            f"x_y_index ({x_y_index}) is not evenly divisible by step_size ({step_size})."
        )

    seq_len = x_y_index // step_size
    target_df = data["target"]
    num_points = len(target_df)

    pixels = target_df.iloc[:, :x_y_index].to_numpy()   # (num_points, x_y_index)
    labels = target_df.iloc[:, x_y_index:].to_numpy()   # (num_points, label_dim)
    label_dim = labels.shape[1]

    X_seq = pixels.reshape(num_points, seq_len, step_size)

    if label_every_step:
        labels_broadcast = np.broadcast_to(
            labels[:, np.newaxis, :], (num_points, seq_len, label_dim)
        ).copy()
        X_seq = np.concatenate([X_seq, labels_broadcast], axis=2)

    return X_seq, labels
```

I want a *very simple* example of using this function to load a dataset and compare an RNN, LSTM, and GRU model on the loaded sequence data. Below is a simple example using synthetic data to demonstrate how to use the `load_data_as_sequence` function and compare the three types of recurrent neural networks.

Now, with that example I also want to ability to address the same dataset with a MonarchLinear models living in a Sequential2D.  For example, something like

$$
\begin{bmatrix}
I & 0 & 0 \\
M_1 & M_2 & M_3 \\
M_4 & M_5 & M_6 \\
\end{bmatrix}
\begin{bmatrix}
x_k \\
y_k \\
h_k \\
\end{bmatrix}
=
\begin{bmatrix}
x_{k+1} \\
y_{k+1} \\
h_{k+1} \\
\end{bmatrix}
$$

for a single iteration of the Sequential2D model, where \(I\) is the identity matrix, \(M_i\) are the learnable parameters of the MonarchLinear model, and \(x_k\), \(y_k\), and \(h_k\) represent the input features, labels, and hidden state at time step \(k\) respectively.

Now, finally, I want to be able to iterate the Sequential2D model over multiple time steps to see how the hidden state evolves and how the predictions change over the iterations at every timesetp.  For example, something like the below for 3 iterations of the Sequential2D model:

$$
\begin{bmatrix}
I & 0 & 0 \\
M_1 & M_2 & M_3 \\
M_4 & M_5 & M_6 \\
\end{bmatrix}
\begin{bmatrix}
I & 0 & 0 \\
M_1 & M_2 & M_3 \\
M_4 & M_5 & M_6 \\
\end{bmatrix}
\begin{bmatrix}
I & 0 & 0 \\
M_1 & M_2 & M_3 \\
M_4 & M_5 & M_6 \\
\end{bmatrix}
\begin{bmatrix}
x_k \\
y_k \\
h_k \\
\end{bmatrix}
=
\begin{bmatrix}
x_{k+1} \\
y_{k+1} \\
h_{k+1} \\
\end{bmatrix}
$$

Put the simple example into a jupyter notebook format and include explanations for each step. The notebook should be called:

notebooks/advanced/11-rcp-load-sequence-example.ipynb

Divide this into tasks and create markdown files for each task in the tasks directory. Each markdown file should contain a sufficient explanation of the task to be handed to a simpler model to complete the task.

Remember, keep it simple!  This notebook is intended to be an example of how to use the `load_data_as_sequence` function and compare different models, so the focus should be on clarity and simplicity rather than complex data or models. I want to be able to give this to a graduate student who is not an expert in pytorch.
