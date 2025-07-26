# TL;DR

This repository contains several scripts to help you get started running Python on Turing. I have attempted to document the scripts so please look at the scripts themselves for more information.


## jupyter_notebooks

Some scripts for running Jupyter notebooks on Turing. There are examples for Pytorch and Scilkit-learn

## python

Some scripts for running Python on Turing. There are examples for Pytorch and Scilkit-learn

## scripts

Some scripts for using Turing more generally. For example, for getting an interactive node.

# Introduction to Turing (as of 7/26/2025)

## Getting started

Turing is the key computational resource at WPI and is quite useful for doing large scale machine learning experiments, far larger than one could do on one's laptop. However, Turing can be a bit daunting to use at first, so I have attempted to make things a bit easier with this material.

## What is Turing and getting started with "slurm"

Turing is a collection of "compute" nodes all accessible from a "head" node. When you log onto turing, say by running:

ssh turing.wpi.edu

you will find yourself on the turing head node. Normally you do _not_ run long or computational intensive jobs but instead ask for a "compute" node to be assigned to you. How you do you that? By using a system called "slurm". Slurm takes care of assigning users compute nodes in a fair manner, and more information can be found here:

https://arc.wpi.edu/cluster-documentation/build/html/index.html

All of my scripts above make use of slurm for accessing compute nodes on turing.

## A few notes on pytorch

To run pytorch efficiently you require access to a turing compute node with a GPU. Things are moving fast with GPU technology, and GPU hardware can reach its end-of-life suprisingly quickly. At this moment these are the GPUs called:

V100:  Older GPU, but still works for many things.
L40S:  A newer GPU, but not as much memory as fancier GPUs. For many    uses this is the best GPU to use.
A100:  A newer GPU with a lot of memory.  This is the baseline high-performance GPU on turing.
H100:  A newer GPU with a lot of memory.  This is the high-performance GPU on turing better than the A100.
H200:  The latest and greatest GPU on turing.  

Of course, when you read this things may have changed, but I have attempted to provide a set of scripts that at least work on these compute nodes.

# A few note for advanced users

There are some additional tools that I have use all the time, but are perhaps only of interest to mroe advanced users

## vscode is awesome

I use vscode (https://code.visualstudio.com/) as my editor of choice. It is awesome and makes working with turing (and python an general) a joy. You can run vscode on turing, but have the window running on your local machine. Also, the python support has advanced to the point where you can run and edit jupyter notebooks just like any other file. I.e., as far as you are concerned a turing compute node is on your desktop :-)

## parameterized papermill is awesome

papermill (https://papermill.readthedocs.io/en/latest/) not only lets you run many jupyter notebooks in parallel on turing, it also let's you parameterize them is a very slick way (https://papermill.readthedocs.io/en/latest/usage-parameterize.html). This is gret for doing parameter students (e.g., looking at different architectures, activation functions, etc.). I have generated hundreds of notebooks this way for papers.

## comet.ml is awesome

But, how do you keep track of hundreds of notebooks? MLOps is an emerging field (https://en.wikipedia.org/wiki/MLOps) and there are many groups working on organizing larger scale ML experiments. The classic software in this domain is tensorboard (https://www.tensorflow.org/tensorboard), but here are now many competitors. My favorite in this domain is Comet.ml (https://www.comet.ml/site/). They have a very generous educational account option, and I can't tell you how many hours this has saved me!
