#!/bin/bash

# One node, one task, 16GB memory, one GPU (L40S)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb

# Of course, you need to change this to be the script you want to run.,
VENVPATH=/home/rcpaffenroth/projects/2_research/iterativennsimple/.venv
source $VENVPATH/bin/activate

# Of course, you need to change this to be the script you want to run.,
python plot_digits_classification_exercise.py


