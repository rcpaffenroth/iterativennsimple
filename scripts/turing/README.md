# Turing Cluster Guide

This directory contains scripts and documentation for running experiments on the Turing compute cluster.

## Quick Start

Before using Turing, set up the Python environment locally using uv:

```bash
# From the project root
uv sync --group dev

# Activate the environment
source .venv/bin/activate
```

For GPU-accelerated sparse computing:
```bash
uv sync --group dev --group sparse
```

## Overview

This repository contains several directories with scripts and docs:

- **`jupyter_notebooks/`**: Scripts for running Jupyter notebooks non-interactively with GPU support
- **`python/`**: Examples for running Python scripts on compute nodes  
- **`scripts/`**: General utilities for working with Turing (interactive nodes, job monitoring, etc.)

# Getting Started with Turing

Turing is WPI's key computational resource for large-scale machine learning experiments. While powerful, it can be daunting at first. This guide makes it easier to get started.

## What is Turing and Getting Started with SLURM

Turing is a collection of compute nodes accessible from a head node. When you log onto Turing:

```bash
ssh turing.wpi.edu
```

You are on the **head node** - do NOT run intensive computations here. Instead, request compute nodes using **SLURM** (Simple Linux Utility for Resource Management).

### Basic SLURM Workflow

```bash
# Submit a job for scheduling
sbatch myjob.sh

# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job_id>
```

SLURM handles fair resource allocation across users and nodes. For complete documentation:  
https://arc.wpi.edu/cluster-documentation/build/html/index.html

## PyTorch and GPU Access

To run PyTorch efficiently, request a GPU-enabled compute node in your SLURM script:

```bash
#SBATCH --gres=gpu:1
```

### Current GPU Hardware (as of 2026)

Turing has several GPU generations available. Hardware evolves over time, but current options include:

| GPU | Use Case |
|-----|----------|
| **V100** | Learning, small workloads |
| **L40S** | Standard ML workloads, good balance |
| **A100** | High-memory, compute-intensive work |
| **H100** | State-of-the-art performance |
| **H200** | Newest, maximum memory |

Specify GPU type in SLURM scripts or accept the default allocation.

# Advanced Topics for Power Users

The following tools can dramatically improve your workflow on Turing:

## VS Code Remote Development

VS Code (https://code.visualstudio.com/) is an excellent editor for Turing development. You can:
- Run VS Code on the Turing head node with the GUI on your local machine
- Edit files and notebooks with full IDE support
- Run code directly from your "desktop" (actually Turing)

This provides a seamless remote development experience.

## Papermill for Notebook Automation

Papermill (https://papermill.readthedocs.io/en/latest/) enables non-interactive notebook execution with powerful parameter management:

- Run multiple notebooks in parallel across Turing compute nodes
- Parameterize notebooks for hyperparameter sweeps
- Automate architecture and activation function studies
- Generate hundreds of experimental variations

This is especially powerful for parameter studies and batch experimentation. See `jupyter_notebooks/README.md` for examples.

## MLOps Tools for Experiment Tracking

Managing large-scale ML experiments requires organization. Popular options include:

- **Weights & Biases** (https://wandb.ai/): Comprehensive tracking, visualization, and collaboration
- **Comet.ml** (https://www.comet.ml/): Educational accounts with generous limits
- **TensorBoard** (https://www.tensorflow.org/tensorboard): TensorFlow-native tracking

These tools help track, compare, and visualize hundreds of experiments across Turing jobs. Highly recommended for serious research work.
