# Introduction

Here are some examples of running jupyter notebooks on Turing

## Interactive notebook

This will allow you to access an interactive Jupyter notebook running on a Turing node with a GPU! Note, please read the instructions
carefully that the script outputs, as it will tell you how to setup your local machine to access the remote Jupyter server.

sbatch jupyter_interactive_sbatch.sh

## Non-interactive batch run of a jupyter notebook

This is an example that will run a "batch" Jupyter notebook using a GPU. I.e., a Jupyter notebook that will run non-interactively. This idea can be used
to several long running Jupyter notebooks in parallel. This uses a program called "papermill" (https://papermill.readthedocs.io/en/latest/) that runings Jupyter notebooks without user input

sbatch pytorch_lightning_cifar10_sbatch.sh

Note, this is far more effective if you know a bit about parameterizing Juypter notebooks using papermill (https://papermill.readthedocs.io/en/latest/usage-parameterize.html).
