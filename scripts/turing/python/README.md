# Running Python Scripts on Turing

This directory contains example scripts for running Python on the Turing compute cluster.

## Environment Setup

Before submitting jobs to Turing, set up the Python environment locally using uv:

```bash
# From the project root
uv sync --group dev

# Activate the environment
source .venv/bin/activate
```

For sparse computing workloads:
```bash
uv sync --group dev --group sparse
source .venv/bin/activate
```

## Running Examples

### Scikit-learn Example

This script runs a small scikit-learn example on a compute node. Results appear in a file called `slurm-XXXXXX.out` where XXXXX is the SLURM job ID.

```bash
sbatch plot_digits_classification_exercise_sbatch.sh
```

### PyTorch with GPU

This script runs PyTorch and verifies GPU access on a remote compute node. Results appear in `slurm-XXXXXX.out`.

```bash
sbatch pytorch_using_GPU_sbatch.sh
```

## Submitting Your Own Jobs

When submitting custom Python scripts to Turing, the job scripts should:
1. Activate the uv-created virtual environment
2. Run your Python script from the activated environment
3. Redirect output to capture results

Example sbatch script pattern:
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

source /path/to/.venv/bin/activate
python your_script.py
```
