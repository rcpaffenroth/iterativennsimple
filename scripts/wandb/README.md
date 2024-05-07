# Weights and Biases demo

This is a simple example of how to use Weights and Biases to log metrics and visualize results.  It also includes a simple example of how to use the `wandb` command line to run a sweep.

## run_demo.py

This is the most useful script in this directory.  It is a simple example of how to use the `wandb` library to log configuration, metrics, and artifacts. 

The key idea is that this uses command line arguments to set the configuration of the run.  This is a good practice because it makes it easy to run the same script with different configurations.  It also makes it easy to run the script with different configurations in a sweep.

For example, you can run the script with the following command:

```bash
python run_demo.py --model_type=square --data_name=MNIST --gpu
```

You can also use the `wandb` command line to run a sweep.  For example, you can run the following command:

```bash
$ wandb sweep run_sweep.yml
wandb: Creating sweep from: run_sweep.yml
wandb: Creating sweep with ID: 97e790bg
wandb: View sweep at: https://wandb.ai/rcpaffenroth-wpi/test-sweep/sweeps/97e790bg
wandb: Run sweep agent with: wandb agent rcpaffenroth-wpi/test-sweep/97e790bg
$ wandb agent rcpaffenroth-wpi/test-sweep/97e790bg
wandb: Starting wandb agent 🕵️
2024-05-07 03:40:39,664 - wandb.wandb_agent - INFO - Running runs: []
2024-05-07 03:40:39,844 - wandb.wandb_agent - INFO - Agent received command: run
2024-05-07 03:40:39,844 - wandb.wandb_agent - INFO - Agent starting run with config:
-------LOTS MORE OUTPUT :-)---------------
```

for details see https://docs.wandb.ai/guides/sweeps.  The cool part is that once the sweep
is created, you can run the command "wandb agent rcpaffenroth-wpi/test-sweep/97e790bg" on as many
remote machines as you like and it will automatically run the script with all the configurations
in the sweep.

## plot_demo.py
It also demonstrates how to use the `wandb` library to log images and plots.  This is somewhat kludgy at the moment, but it works.  The key idea is that you can log images and plots as artifacts.  This is useful because it makes it easy to view the images and plots in the Weights and Biases dashboard.  The script just demonstrates how to do this.  Note, one thing that makes the plotting much easier is to use the plotly library.  This is a very powerful library for creating plots and it is easy to use with Weights and Biases (and Google Colab for that matter).