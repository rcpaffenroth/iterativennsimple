# Task 1-0 — Load Sequence Example Notebook: Overview

## Summary

Create a Jupyter notebook at `notebooks/advanced/11-rcp-load-sequence-example.ipynb` that demonstrates how to use the `load_data_as_sequence` function from the `generatedata` package and compares four sequence model architectures on a real dataset:

1. **RNN** (vanilla `nn.RNN`)
2. **LSTM** (`nn.LSTM`)
3. **GRU** (`nn.GRU`)
4. **Sequential2D with MonarchLinear** — an iterative neural network approach

The notebook is aimed at graduate students who are **not** PyTorch experts, so clarity and simplicity are paramount.

## Notebook Outline

| Section | Subtask File | Description |
|---------|-------------|-------------|
| 1. Setup & Data Loading | `AGENT-task-1-1-load-sequence-data.md` | Imports, load data with `load_data_as_sequence`, train/val split, DataLoaders |
| 2. RNN / LSTM / GRU Models | `AGENT-task-1-2-load-sequence-rnn.md` | Define simple RNN, LSTM, GRU classifiers; shared training loop; train and evaluate |
| 3. Sequential2D + MonarchLinear | `AGENT-task-1-3-load-sequence-monarch.md` | Build a Sequential2D map with Identity + MonarchLinear blocks; wrap as a sequence classifier |
| 4. Iterated Sequential2D & Comparison | `AGENT-task-1-4-load-sequence-iterated.md` | Show iteration of the Sequential2D map at each timestep; train; compare all models |
| 5. Final Assembly | `AGENT-task-1-5-load-sequence-assemble.md` | Combine all sections into the final `.ipynb` file with markdown explanations |

## Implementation Order

Complete the subtasks in order (1-1 through 1-5). Each subtask produces code and markdown cells that will be assembled into the final notebook.

## Key Conventions

- Use the `"quick"` preset style: small hidden sizes, few epochs, so the notebook runs fast.
- Dataset: use `"MNIST"` with `step_size=28` (one row of pixels per timestep → 28 timesteps).
- `label_every_step=True` so labels are appended to input at each step.
- Supervise at the last timestep only for simplicity.
- Use `CrossEntropyLoss` and `Adam` optimizer throughout.
- Print train/val accuracy each epoch.
- Keep all model definitions in notebook cells (no external files needed).

## Dependencies

All dependencies are already in `pyproject.toml`:
- `torch`, `numpy`, `matplotlib`
- `generatedata` (external package with `load_data_as_sequence`)
- `iterativennsimple` (this package: `MonarchLinear`, `Sequential2D`, `Identity`, `Sequential1D`)

## Reference

See `examples/monarch_mnist.py` for a production-quality version of this workflow. The notebook should be a simplified, educational version of that script.
