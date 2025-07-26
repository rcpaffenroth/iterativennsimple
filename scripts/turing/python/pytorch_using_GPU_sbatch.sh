#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|A100|H100|H200"

# Of course, you need to change this to be directory of your virtual environment.
VENVPATH=/home/rcpaffenroth/projects/2_research/iterativennsimple/.venv

source $VENVPATH/bin/activate

# Of course, you need to change this to be the script you want to run.
python pytorch_using_GPU.py


