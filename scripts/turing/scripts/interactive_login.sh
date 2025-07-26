#! /bin/bash
# Other common command line options
# To get larger memory
# --mem=16gb

# Get a node with just a CPU
#srun --pty bash    
# Get a node with a "random" GPU
#srun --pty --gres=gpu --constraint="V100|L40S|A100|H100|H200" bash
# Get a node with a "small" GPU
#srun --pty --gres=gpu --constraint="L40S" bash
# Get a node with a "new" GPU
#srun --pty --gres=gpu --constraint="A100" bash
# Get a node with a "huge" GPU
#srun --pty --gres=gpu --constraint="H200" bash

# Get a specific node
# srun --pty --nodelist=compute-0-25 bash

# Get a specific queue.  The options are 
# short: Jobs lasting less than 24 hours
# long: Jobs lasting less that a week
# srun --pty -p short bash

srun --pty --gres=gpu --constraint="V100|L40S|A100|H100|H200" --mem=16gb bash

