# Running Jupyter Notebooks on Turing

This directory contains scripts for running Jupyter notebooks on the Turing compute cluster, including examples for GPU-accelerated workloads.

## Environment Setup

Before submitting notebook jobs to Turing, set up the Python environment locally using uv:

```bash
# From the project root
uv sync --group dev

# Activate the environment
source .venv/bin/activate
```

## Non-interactive Batch Execution

Jupyter notebooks can be run non-interactively on Turing using Papermill, which enables:
- Running notebooks as batch jobs with GPU access
- Parameterizing notebooks for parameter sweeps
- Running multiple notebooks in parallel

### Example: Run Sequential Comparison Notebook

```bash
sbatch Sequential-vs-Sequential2D.sh
```

This submits a batch job that executes `Sequential-vs-Sequential2D.ipynb` on a GPU compute node.

## Parameterizing Notebooks with Papermill

Papermill allows you to pass parameters to notebooks before execution, which is useful for:
- Testing different hyperparameters
- Running parameter sweeps across multiple jobs
- Automating experiment variations

For more information, see the [Papermill documentation](https://papermill.readthedocs.io/en/latest/usage-parameterize.html).

### Example Parameterization

In your notebook, add a cell tagged with `parameters`:

```python
# Parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 100
```

Then run with custom parameters:

```bash
papermill input_notebook.ipynb output_notebook.ipynb -p learning_rate 0.01 -p batch_size 64
```
