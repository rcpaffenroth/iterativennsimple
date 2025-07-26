# Introduction

Here are some examples of running jupyter notebooks on Turing

## Non-interactive batch run of a jupyter notebook

This is an example that will run a "batch" Jupyter notebook using a GPU. I.e., a Jupyter notebook that will run non-interactively. This idea can be used
to several long running Jupyter notebooks in parallel. This uses a program called "papermill" (https://papermill.readthedocs.io/en/latest/) that runs Jupyter notebooks without user input

sbatch Sequential-vs-Sequential2D.sh

Note, this is far more effective if you know a bit about parameterizing Juypter notebooks using papermill (https://papermill.readthedocs.io/en/latest/usage-parameterize.html).
