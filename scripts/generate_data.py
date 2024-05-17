import json

import pandas as pd
import numpy as np
import pathlib

import requests
import pickle

import torch
from torchvision import datasets, transforms
torch.manual_seed(0)
import iterativennsimple
from iterativennsimple.utils.save_data import save_data

# The directory in which the notebook is located.
base_dir = pathlib.Path(iterativennsimple.__path__[0])
# The directory where the data is stored.
data_dir = base_dir / '../data/processed'

# Create an empty json file at ../data/processed/info.json
info = {}
with open(data_dir / f'info.json', 'w+') as f:
    json.dump(info, f)

#########################
# Regression line
#########################

num_points = 1000
name='regression_line'
# The poits on the line
x_on = np.random.uniform(0, 1, num_points)
y_on = 0.73*x_on
# The points off the line
x_off = x_on
y_off = y_on + np.random.normal(0, 0.1, num_points)

start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(data_dir, name, start_data, target_data, x_y_index=1)


#########################
# PCA line
#########################

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


start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(data_dir, name, start_data, target_data)

#########################
# Circle
#########################

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

start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(data_dir, name, start_data, target_data)

#########################
# Regression circle
#########################

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

start_data = {f'x{i}': columns for i, columns in enumerate([x_off, y_off])}
target_data = {f'x{i}': columns for i, columns in enumerate([x_on, y_on])}
save_data(data_dir, name, start_data, target_data, x_y_index=1)


#########################
# Manifold
#########################

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

start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(data_dir, name, start_data, target_data)

#########################
# MNIST 1D
#########################

name='MNIST1D'

# A simpler version of the MNIST dataset based on the work of Greydanus et al. (2018)
# https://arxiv.org/abs/2011.14439

url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
r = requests.get(url, allow_redirects=True)

mnist1d_dataset = pickle.loads(r.content)

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

start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(data_dir, name, start_data, target_data, x_y_index=vector_dim)


#########################
# MNIST
#########################
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
# num_digits = 60000
num_digits = 1000
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

start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(data_dir, name, start_data, target_data, x_y_index=vector_dim)


#########################
# Electric Field Range Localization
#########################
name = 'EMlocalization'
X = torch.load('../data/raw/EM_X_train.pt')
Y = torch.load('../data/raw/EM_Y_train.pt')
x_on = torch.cat((X, Y), dim=1)

y_max = x_on[:, -1].max()
y_min = x_on[:, -1].min()
x_on[:, -1] = (x_on[:, -1] - y_min) / (y_max - y_min)

# Now set the last entry of x_off to be a random range in the same range as the target
x_off = x_on.clone()
x_off[:, -1] = 0

start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(data_dir, name, start_data, target_data, x_y_index=160)


#########################
# Lunar lander
#########################
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

start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
save_data(data_dir, name, start_data, target_data, x_y_index=404)

#########################
# Mass spectrometry
#########################
name = 'MassSpec'
# Load the mass spec dataset from a parquet file
mass_spec_df = pd.read_parquet('../data/raw/mass_spec.parquet')

# Reorder the columns so that the chemception features are at the end
# This means moving columns 915-1426 to the end
new_order = list(mass_spec_df.columns[:915]) + list(mass_spec_df.columns[1427:-1]) + list(mass_spec_df.columns[915:1427])
start_data = mass_spec_df[new_order]
target_data = mass_spec_df[new_order]

pd.options.mode.copy_on_write = True
start_data.iloc[:, -512:] = 0.0

save_data(data_dir, name, start_data, target_data, x_y_index=1433-512)
