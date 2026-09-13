Start by reading PRINCIPLES.md and RESEARCH_LOG.md.  Once you have done that, then 
take a look at examples/lra_benchmark.py to get a fell for what has gone before, but
*this is intended to be a different set of experiment* so there is not need to follow
the conventions for the previous lra experiments, other than I hope that we will use
the same datasets here.

For this sequence of experiments I want to explore some ideas in dynamical systems,
such as convergence, along side ideas such as homotopies and parameter continuation
methods.  Now, again, don't put too much weight on the precise definition of these
ideas in the dynamical systems literature, but rather let's be inspired by them.

So, I propose the following sort of algorithm.

Conceptuall, let's start with vectors that look like:

$$
z_k=
\begin{bmatrix}
x_k \\
y_k \\ 
h_k 
\end{bmatrix}
$$

Where $x_k$ is the input to some lra type problem (e.g., for mnist it is some set of
pixel values), $y_k$ is some current guess to the eventual target (e.g., a vector
of probabilities for a classification), and $h_k$ is some internal state of the
dynamical system to be defined later.

I want to take "baby steps" and add complexity slowly so that we can see what each
addition buys us, if anything.

Initially, one could write something like

Experiment 1
$$
x_{k+1} = read next data from lrs dataset
y_{k+1} = f_{x to y}(x_k)
$$

I would like to start with something simple for $f_{x to y}$ such as linear 
least squares (or perhaps adding in a Ridge normalization if we like).

Next, let's consider

Experiment 2 
$$
x_{k+1} = read next data from lrs dataset
y_{k+1} = f_{x to y}(x_k) + f_{y to y}(y_k)
$$

The idea is that previous values of $y_k$ may help with future predictions.  Now,
two thing immediately arise.  First, how to initialize $z_k$? $x_k$ is just from the
dataset so is clear, but perhaps start out with $y_0=0$.  Second, I acknowledge that
$y$ is little strange here, since it is both a prediction and a "state", but let's
do this anyway just to see what happens.

Now, there are two directions I want to go.  

First, we can let,

Experiment 3
$$
x_{k+1} = read next data from lrs dataset
y_{k+1} = f_{x and y to y}(x_k, y_k)
$$

I.e., $f_{x and y to y}$ is some non-linear mixing.

Second, and perhaps more tricky from the implementation perspective

Experiment 4 
$$
x_{k+1} = *somtimes* read next data from lrs dataset
y_{k+1} = f_{x and y to y}(x_k, y_k)
$$

I.e., sometimes $x_k = x_{k+1}$.  The idea here is to uncouple the update rates
of $x_k$ and $y_k$, so that $y_k$ has a chance to "converge" before a new value of
$x_k$ is introduced.  Now, there are many things one can do here that will be 
interesting to try!  I would recommend that we start off simple (e.g., $x_k$ stays
constant for some proscribed number of iterations) but leave the ability to change
this later (e.g., $x_k$ stays constant until the change in $y_k$ is sufficiently small).  Note, this will likely need us to think about loss functions (e.g., is the convergence of $y_k$ for fixed $x_k$ a loss function itself), and I would like you to question me about this.

Experiment 5 is then just combining Experiment 3 and 4.

Now, things get interesting!

Experiment 6 
$$
x_{k+1} = *somtimes* read next data from lrs dataset
y_{k+1} = f_{x to y}(x_k) + f_{y to y}(y_k) + f_{h to y}(h_k)
h_{k+1} = g_(x and y and h to h)(x_k, y_k, h_k)
$$

Note I have split $f$ again on purpose.  This is where homotopy/continuation methods/boosting come in.
I am interesting in training $f_{y to y}$ in the presence of an *already trained*
$f_{x to y}$, and training $f_{h to y}$ (and g) in the presence of the other functions
also being *already trained*.



