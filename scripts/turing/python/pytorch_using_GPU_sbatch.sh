#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="K80|V100|P100|T4"

# Of course, you need to change this to be the script you want to run.,
/work/rcpaffenroth/opt/anaconda3/bin/python pytorch_using_GPU.py


