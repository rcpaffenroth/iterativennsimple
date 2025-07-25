#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="K80|V100|P100|T4"

# The jupyter notebook needs this to have write permissions
# to its temp directory
export XDG_RUNTIME_DIR=""

# Choose a random port.  This might fail if you get unlucky and choose
# a port already in use.  In that case just try again.
PORT=$(shuf -i 10000-11000 -n 1)

echo -e "\nStarting Jupyter on port ${PORT} on the $(hostname) server."
echo -e "\nOn your *local* machine you need to run the SSH tunnel command below and"
echo -e "\n*then* point your *local* web browswer at the given URI."
echo -e "\n"
echo -e "\nSSH tunnel command: ssh -NL 6600:$(hostname):${PORT} ${USER}@turing.wpi.edu"
echo -e "\nLocal URI: http://localhost:6600"
echo -e "\n"
echo -e "\nIf you look at the output from the jupyter notebook then you will see the"
echo -e "\ntoken you need to actually access the notebook."

# Setup a port forward to rcp.wpi.edu so that I can access this from outside
ssh -f -N -R 6600:localhost:${PORT} turing.wpi.edu

NOTEBOOKDIR="/home/rcpaffenroth/projects"
# Obviously change this line to match your user notebook-dir
echo -e "You are using notebook-dir ${NOTEBOOKDIR}"
echo -e "If this is not correc then you need to change it in the script!"
/work/rcpaffenroth/opt/anaconda3/bin/jupyter lab --notebook-dir=${NOTEBOOKDIR} --no-browser --port=${PORT} --ip=0.0.0.0 
