#!/usr/bin/env python
# coding: utf-8

# # Load modules

# In[64]:


# This modles that we will use for this notebook.
import json

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

import requests
import pickle

import torch
from torchvision import datasets, transforms
torch.manual_seed(0)
import iterativennsimple


# ## Utility functions

# In[65]:


# The directory in which the notebook is located.
base_dir = pathlib.Path(iterativennsimple.__path__[0])
# The directory where the data is stored.
data_dir = base_dir / '../data/processed'


# In[66]:


def save_data(name, start_data, target_data, x_y_index=None, data_dir=data_dir):
    """ Save data to parquet files and update info.json.  

    The start_data is the start of a trajectory and the target_data is the end of the trajectory.  
    Classically the target_data lay on the manifold of interest and is the "clean" data.
    Simlarly the start_data is the "noisy" data, for example an MNIST image with the classification label ramdomly chosen.

    Args: name (str): Name of the dataset.
            start_data (list): List of start data.
            target_data (list): List of target data.    

    Returns: None
    """
    target_df = pd.DataFrame(target_data)
    start_df = pd.DataFrame(start_data)
    assert start_df.shape[1]==target_df.shape[1], 'shape mismatch'
    size = start_df.shape[1]

    data_info = {
        'num_points': start_df.shape[0],
        'size': size,       
    }

    if x_y_index is not None:
        data_info['x_y_index'] = x_y_index
        data_info['x_size'] = x_y_index
        data_info['y_size'] = size-x_y_index

    with open(data_dir / f'info.json', 'r') as f:
        info = json.load(f)

    if name in info:
        info[name].update(data_info)
    else:
        info[name] = data_info

    with open(data_dir / f'info.json', 'w') as f:
        json.dump(info, f)

    start_df.to_parquet(data_dir / f'{name}_start.parquet')
    target_df.to_parquet(data_dir / f'{name}_target.parquet')


# # The Data
# 
# 

# We want a simple yet non-trivial problem for demonstrating the methods, so we do a manifold problem.  I.e., there is a space that contains a manifold of some given form and dimension, and each training example is a point on the manifold.  The idea is to train the model to be a dynamical system where the manifold is an attractor.
# 
# For example, we can use a simple manifold, such as a circle in 2D.  We can use a circle because it is easy to visualize and easy to generate.  
# However, other manifolds are also possible.
# 
# Another example would be a 2D Swiss roll, which is a classic 2D manifold in 3D space.  This is a bit more complicated to generate, but it is still easy to visualize.
# 
# A final example could be MNIST, where the manifold is the set of all possible images of a given digit.  This is a much more complicated manifold.  Note, we can, and should, condition the problem where each image is labeled with its true digit.
# 
# Note, we will proceed in two steps.  First we will generat the appropriate training data as a Pandas frame, then we will read in the data to train the model.  This notebook focuses on the first step.

# In[67]:


# Create an empty json file at ../data/processed/info.json
info = {}
with open(data_dir / f'info.json', 'w+') as f:
    json.dump(info, f)


# The data consists of pairs of vectors, where the first vector is a starting vector and the second vector is the target vector.  The input vector is a point not on the manifold, and the target vector some point on the manifold we want the input vector to converge to.  The idea is to train the model to be a dynamical system where the manifold is an attractor where the starting vector is attracted to the target vector.

# # Regression line

# In[68]:


num_points = 1000
name='regression_line'
# The poits on the line
x_on = np.random.uniform(0, 1, num_points)
y_on = 0.73*x_on
# The points off the line
x_off = x_on
y_off = y_on + np.random.normal(0, 0.1, num_points)


# In[69]:


_, ax = plt.subplots()
ax.plot(x_on, y_on, 'o')
ax.plot(x_off, y_off, '+')
ax.set_aspect('equal')


# In[70]:


idx = np.random.choice(range(num_points), 10)
_, ax = plt.subplots()
ax.plot(x_on[idx], y_on[idx], 'o')
ax.plot(x_off[idx], y_off[idx], '+')
ax.quiver(x_off[idx], y_off[idx], x_on[idx]-x_off[idx], y_on[idx]-y_off[idx],
          scale=1, scale_units='xy', angles='xy', color='r', width=0.005)
ax.set_aspect('equal')


# In[71]:


start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(name, start_data, target_data, x_y_index=1)


# # PCA line

# In[72]:


num_points = 1000
name='pca_line'
# The poits on the line
rho = np.random.uniform(0, 1, num_points)
theta = 0.54
x_on = rho*np.cos(theta)
y_on = rho*np.sin(theta)
# The points off the line
gamma = np.random.normal(0, 0.1, num_points)
x_off = x_on + gamma*np.cos(theta+np.pi/2.0)
y_off = y_on + gamma*np.sin(theta+np.pi/2.0)


# In[73]:


_, ax = plt.subplots()
ax.plot(x_on, y_on, 'o')
ax.plot(x_off, y_off, '+')
ax.set_aspect('equal')


# In[74]:


idx = np.random.choice(range(num_points), 10)
_, ax = plt.subplots()
ax.plot(x_on[idx], y_on[idx], 'o')
ax.plot(x_off[idx], y_off[idx], '+')
ax.quiver(x_off[idx], y_off[idx], x_on[idx]-x_off[idx], y_on[idx]-y_off[idx],
          scale=1, scale_units='xy', angles='xy', color='r', width=0.005)
ax.set_aspect('equal')


# In[75]:


start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(name, start_data, target_data)


# # Circle

# In[76]:


num_points = 1000
name='circle'
thetas = np.random.uniform(0, 2*np.pi, num_points)
# The poits on the unit circle
x_on = np.cos(thetas)
y_on = np.sin(thetas)
# The points off the unit circle
r_off = np.random.uniform(0.8, 1.2, num_points)
x_off = r_off*np.cos(thetas)
y_off = r_off*np.sin(thetas)


# In[77]:


_, ax = plt.subplots()
ax.plot(x_on, y_on, 'o')
ax.plot(x_off, y_off, '+')
ax.set_aspect('equal')


# In[78]:


idx = np.random.choice(range(num_points), 100)
_, ax = plt.subplots()
ax.plot(x_on[idx], y_on[idx], 'o')
ax.plot(x_off[idx], y_off[idx], '+')
ax.quiver(x_off[idx], y_off[idx], x_on[idx]-x_off[idx], y_on[idx]-y_off[idx],
          scale=1, scale_units='xy', angles='xy', color='r', width=0.005)
ax.set_aspect('equal')


# In[79]:


start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(name, start_data, target_data)


# # Regression circle

# In[80]:


num_points = 1000
name='regression_circle'
thetas = np.random.uniform(0, 2*np.pi, num_points)
# The poits on the unit circle
x_on = np.cos(thetas)
y_on = np.sin(thetas)
# The points off the unit circle
y_noise = np.random.normal(0, 0.2, num_points)
x_off = np.cos(thetas)
y_off = np.sin(thetas)+y_noise


# In[81]:


_, ax = plt.subplots()
ax.plot(x_on, y_on, 'o')
ax.plot(x_off, y_off, '+')
ax.set_aspect('equal')


# In[82]:


idx = np.random.choice(range(num_points), 100)
_, ax = plt.subplots()
ax.plot(x_on[idx], y_on[idx], 'o')
ax.plot(x_off[idx], y_off[idx], '+')
ax.quiver(x_off[idx], y_off[idx], x_on[idx]-x_off[idx], y_on[idx]-y_off[idx],
          scale=1, scale_units='xy', angles='xy', color='r', width=0.005)
ax.set_aspect('equal')


# In[83]:


start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(name, start_data, target_data, x_y_index=1)


# # Manifold

# In[84]:


num_points = 1000
name='manifold'

# Define the function that generates the Swiss roll points
def swiss_roll(n_samples=1000):
    # Generate random values for u and v
    u = np.random.rand(n_samples) 
    v = np.random.rand(n_samples)

    # Define the function that maps u and v to x, y, and z coordinates
    x = np.cos(u * 2 * np.pi) * (1-0.5*u)
    y = np.sin(u * 2 * np.pi) * (1-0.5*u)
    z = v

    # Return the points as a numpy array
    return np.column_stack((x, y, z))

x_on = swiss_roll(num_points)

# The points off the manifold
x_off = x_on + np.random.normal(0, 0.1, (num_points, 3))


# In[85]:


# a plot of the manifold
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_on[:, 0], x_on[:, 1], x_on[:, 2], 'o')
ax.scatter(x_off[:, 0], x_off[:, 1], x_off[:, 2], '+')


# In[86]:


start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(name, start_data, target_data)


#  # MNIST 1D
# 
# 

# In[87]:


name='MNIST1D'

# A simpler version of the MNIST dataset based on the work of Greydanus et al. (2018)
# https://arxiv.org/abs/2011.14439

url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
r = requests.get(url, allow_redirects=True)

mnist1d_dataset = pickle.loads(r.content)


# In[88]:


num_digits = 4000
vector_dim = 40

# Create a list of PyTorch tensors containing the MNIST digits
digit_tensors = []
for i in range(num_digits):
    digit_tensor, label = torch.tensor(mnist1d_dataset['x'][i]), mnist1d_dataset['y'][i]

    # Append the classification label as a one-hot encoding to the end of the tensor
    one_hot = torch.zeros(10)
    one_hot[label] = 1
    # template = torch.tensor(mnist1d_dataset['templates']['x'][label,:])
    # digit_tensor = torch.cat((digit_tensor, template, one_hot))
    digit_tensor = torch.cat((digit_tensor, one_hot))

    # Append the tensor to the list of digit tensors
    digit_tensors.append(digit_tensor)

# x_on contains the MNIST digits with the correct classification
x_on = torch.stack(digit_tensors)
# x_ff contains the MNIST digits with a random classification
x_off = torch.stack(digit_tensors)
# x_off[:, -10:] = torch.rand(size=(num_digits,10))
x_off[:, -10:] = 0.1


# In[89]:


plot_index = 0
fig, ax = plt.subplots()
ax.plot(x_on[plot_index].numpy())
print(f'The start classification label is {x_off[plot_index,-10:]}')
print(f'The target classification label is {x_on[plot_index,-10:]}')


# In[90]:


start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(name, start_data, target_data, x_y_index=vector_dim)


#  # MNIST
# 
# 

# In[91]:


name='MNIST'

# get a list of pytorch vectors, one for each MNIST digit, with the last entry being the classification

# Define the transformation to apply to the MNIST images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='../data/external/', train=True, download=True, transform=transform)

# Define the number of digits and the dimensionality of each vector
num_digits = 60000
vector_dim = 28 * 28

# Create a list of PyTorch tensors containing the MNIST digits
digit_tensors = []
for i in range(num_digits):
    random_image, label = mnist_dataset[np.random.randint(len(mnist_dataset))]

    # Flatten the image into a PyTorch tensor
    digit_tensor = random_image.view(-1)

    # Append the classification label as a one-hot encoding to the end of the tensor
    one_hot = torch.zeros(10)
    one_hot[label] = 1
    digit_tensor = torch.cat((digit_tensor, one_hot))

    # Append the tensor to the list of digit tensors
    digit_tensors.append(digit_tensor)

# x_on contains the MNIST digits with the correct classification
x_on = torch.stack(digit_tensors)
# x_ff contains the MNIST digits with a random classification
x_off = torch.stack(digit_tensors)
# x_off[:, -10:] = torch.rand(size=(num_digits,10))
x_off[:, -10:] = 0.1


# In[92]:


plot_index = 0
fig, ax = plt.subplots()
ax.imshow(x_on[plot_index, :-10].view(28, 28).numpy())
print(f'The start classification label is {x_off[plot_index,-10:]}')
print(f'The target classification label is {x_on[plot_index,-10:]}')


# In[93]:


start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(name, start_data, target_data, x_y_index=vector_dim)


# # Electric Field Range Localization
# 

# In[94]:


name = 'EMlocalization'
X = torch.load('../data/raw/EM_X_train.pt')
Y = torch.load('../data/raw/EM_Y_train.pt')
print(X.shape)
x_on = torch.cat((X, Y), dim=1)
print(x_on.shape)


# In[95]:


# Now set the last entry of x_off to be a random range in the same range as the target
x_off = x_on.clone()
#x_off[:, -1].uniform_(float(x_on[:,-1].min()), float(x_on[:,-1].max()))
x_off[:, -1] += torch.randn(x_off[:, -1].shape)*5000.0
print(x_on[:3, -1])
print(x_off[:3, -1])


# In[96]:


print(x_on[0,-5:])
print(x_off[0,-5:])


# In[97]:


start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(name, start_data, target_data, x_y_index=160)


# # Lunar lander

# In[98]:


name = 'LunarLander'
# Load the LunarLander dataset from a parquet file
lunarlander_df = pd.read_parquet('../data/raw/lander_all_data.parquet')
x_off = []
x_on = []
for i,state in enumerate(['random', 'trained', 'good', 'better']):
    X = lunarlander_df.loc[(state, slice(None)), (slice(None),('x','y', 'vx', 'vy'))]
    Y_off = torch.zeros((X.shape[0], 4))
    Y_on = torch.zeros((X.shape[0], 4)) 
    Y_on[:, i] = 1

    x_off.append(torch.cat((torch.tensor(X.values), Y_off), dim=1))
    x_on.append(torch.cat((torch.tensor(X.values), Y_on), dim=1))

x_off = torch.cat(x_off, dim=0)
x_on = torch.cat(x_on, dim=0)

x_on.shape


# In[99]:


print(x_on[0,-6:])
print(x_off[0,-6:])


# In[100]:


start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(name, start_data, target_data, x_y_index=404)

