import torch
import pandas as pd

import iterativennsimple
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.Sequential1D import Sequential1D

from iterativennsimple.SparseLinear import SparseLinear
from generatedata.load_data import load_data

import os
import pathlib

def test_for_Neil(long_test=False):
    # This is based on notebooks/4-rcp-MLP.ipynb

    # Turn a pandas dataframe into a pytorch tensor
    def df_to_tensor(df):
        return torch.tensor(df.values, dtype=torch.float32)

    z = load_data('MNIST')

    # Data preprocessing

    z_start_tensor = df_to_tensor(z['start'])
    z_target_tensor = df_to_tensor(z['target'])

    # Only use the given number of samples
    if long_test:
        max_num_samples = 500
    else:
        max_num_samples = 10

    num_samples = min(max_num_samples, z_start_tensor.shape[0])
    z_start_tensor = z_start_tensor[:num_samples]
    z_target_tensor = z_target_tensor[:num_samples]

    mask = (z_start_tensor == z_target_tensor).all(axis=0)
    x_mask = mask
    y_mask = ~mask

    input_size = int(x_mask.sum())
    h1_size = 20
    h2_size = 20
    output_size = int(y_mask.sum())

    h_idx = torch.arange(input_size, input_size+h1_size+h2_size)
    y_idx = torch.arange(input_size+h1_size+h2_size, input_size+h1_size+h2_size+output_size)

    iterations = 3
    sparsity = 'R=0.4'

    I = Identity(in_features=input_size, out_features=input_size)

    l1 = SparseLinear.from_singleBlock(in_features=input_size, out_features=h1_size,  
                                       block_type=sparsity, initialization_type='G=0.0,0.001', 
                                       optimized_implementation=True)
    f1 = Sequential1D(l1, 
                    torch.nn.ReLU(), 
                    in_features=input_size, out_features=h1_size)


    l2 = SparseLinear.from_singleBlock(in_features=h1_size, out_features=h2_size,  
                                    block_type=sparsity, initialization_type='G=0.0,0.001', 
                                    optimized_implementation=True)
    f2 = Sequential1D(l2, 
                    torch.nn.ReLU(), 
                    in_features=h1_size,    out_features=h2_size)

    f3 = SparseLinear.from_singleBlock(in_features=h2_size, out_features=output_size,  
                                    block_type=sparsity, initialization_type='G=0.0,0.001', 
                                    optimized_implementation=True)

    in_features_list  = [input_size, h1_size, h2_size, output_size]
    out_features_list = [input_size, h1_size, h2_size, output_size]
    blocks = [[I,    None, None, None],
            [f1,   None, None, None],
            [None, f2,   None, None],
            [None, None, f3,   None]]

    def transpose_blocks(blocks):
        return [[blocks[j][i] for j in range(len(blocks))] for i in range(len(blocks[0]))]

    map = Sequential2D(
        in_features_list=in_features_list,
        out_features_list=out_features_list,
        blocks=transpose_blocks(blocks)
    )   

    zh_start_tensor = torch.cat((z_start_tensor[:, x_mask],
                                torch.zeros(z_start_tensor.shape[0], len(h_idx)), 
                                z_start_tensor[:, y_mask]), dim=1)
    zh_target_tensor = torch.cat((z_target_tensor[:, x_mask], 
                                torch.zeros(z_target_tensor.shape[0], len(h_idx)), 
                                z_target_tensor[:, y_mask]), dim=1)


    # a dataloader which returns a batch of start and target data
    class Data(torch.utils.data.Dataset):
        def __init__(self, z_start, z_target):
            self.z_start = z_start
            self.z_target = z_target
        def __len__(self):
            return len(self.z_start)
        def __getitem__(self, idx):
            return self.z_start[idx], self.z_target[idx]
        
    train_data = Data(zh_start_tensor, zh_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(map.parameters(), lr=0.001)

    if long_test:
        max_epochs = 50
    else:
        max_epochs = 2

    last_loss = 10**9
    # Train the model
    for epoch in range(max_epochs):
        for batch_idx, (start, target) in enumerate(train_loader):
            optimizer.zero_grad()

            mapped = start

            loss = 0.0
            for i in range(iterations):
                mapped = map(mapped)

            loss = criterion(mapped[:, y_idx], target[:, y_idx])
            loss.backward()

            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
            assert loss.item() < last_loss
            last_loss = loss.item()

if __name__ == "__main__":
    test_for_Neil()
