# Introduction

Here are some examples of running python scripts on Turing

## Scikit learn

This is an example that will run a small scikit learn example. The result will appear in a file called "slurm-XXXXXX.out"
where "XXXXX" is the ID number of your slurm run.

sbatch plot_digits_classification_exercise_sbatch.sh

## Basic pytorch

This is an example that will run pytorch on a remote node and just make sure it
has access to a GPU. The result will appear in a file called "slurm-XXXXXX.out"
where "XXXXX" is the ID number of your slurm run.

sbatch pytorch_using_GPU_sbatch.sh
