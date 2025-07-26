#!/bin/bash
# One node, one task, 16GB memory, one GPU (L40S)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S"

# Of course, you need to change this to be the script you want to run.,
VENVPATH=/home/rcpaffenroth/projects/2_research/iterativennsimple/.venv

source $VENVPATH/bin/activate
# Run the script with papermill
papermill Sequential-vs-Sequential2D.ipynb tmp_output.ipynb


