
# %%
# Note, the above line allows this code to be run iteractively
# simialr to a jupyter notebook using an appropriate IDE (such as VSCode)

# Import necessary libraries
import pathlib
import torch
import wandb
import pandas as pd
import click
import iterativennsimple
from iterativennsimple.utils.df_to_tensor import df_to_tensor
from iterativennsimple.utils.StartTargetData import StartTargetData

def load_data(name: str) -> dict:
    """Load in the dataset with the given name

    Args:
        name (str): the name of the dataset

    Returns:
        dict: the start and target points of the dataset
    """
    # The directory in which the notebook is located.
    base_dir = pathlib.Path(iterativennsimple.__path__[0])
    # The directory where the data is stored.
    data_dir = base_dir / '../data/processed'

    # Read the start data
    z_start = pd.read_parquet(data_dir / f'{name}_start.parquet')
    # Read the target data
    z_target = pd.read_parquet(data_dir / f'{name}_target.parquet')
    return {'start': z_start, 'target': z_target}

def get_model(model_type: str, size: int) -> torch.nn.Module:
    """Generate a model of the given type

    Args:
        model_type (str): The type of model to generate
        size (int): The dimension of the input and output

    Raises:
        ValueError: If the model type is not recognized

    Returns:
        torch.nn.Module: The model
    """
    if model_type == 'non-square':    
        model = torch.nn.Sequential(
            torch.nn.Linear(size, size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(size//2, size)
        )
    elif model_type == 'square':    
        model = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            torch.nn.ReLU(),
            torch.nn.Linear(size, size)
        )      
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    return model

# Configure can happen in two ways.  First, we can just
# set the values in the config dictionary directly.
config = {
    # Replace this with your wandb entity and project
    'wandb_entity': 'rcpaffenroth-wpi',
    'wandb_project': 'test'
}

# Alternatively, we can set them as command line arguments
@click.command()
# The following are set above in the config dictionary, but here is an example of 
# how to set them as command line arguments
# @click.option("--wandb_entity", default='rcpaffenroth-wpi', help="The wandb entity to use.")
# @click.option("--wandb_project", default='test', help="The wandb project to use.")
@click.option("--data_name", default='regression_line', 
              help="The name of the dataset to use.")
@click.option("--model_type", default='non-square', 
              help="The name of the model to use.")
@click.option("--gpu", is_flag=True,
              help="Whether to use a GPU if available.")
@click.option("--threads", default=1, 
              help="The number of threads to use.")
def main(**kwargs):
    """Run the training of the model

    Args:
        dataname (str): the name of the dataset to use
    """
    # Update the config dictionary with the command line arguments
    config.update(kwargs)
    # Initialize wandb
    wandb.init(entity=config['wandb_entity'], 
               project=config['wandb_project'],
               config=config)

    # Set the number of threads
    torch.set_num_threads(config['threads'])

    # Find out is there is a GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not config['gpu']:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load the data
    z_data = load_data(config['data_name'])
    config['data_size'] = z_data['start'].shape[1]

    # Get the model, we call it a map to emphasize the relationship 
    # to the dynamical system
    map = get_model(config['model_type'], config['data_size'])
    map.to(device)

    # Make two pytorch tensor datasets from the start and target data
    z_start_tensor = df_to_tensor(z_data['start'])
    z_target_tensor = df_to_tensor(z_data['start'])

    z_start_tensor = z_start_tensor.to(device)
    z_target_tensor = z_target_tensor.to(device)
        
    train_data = StartTargetData(z_start_tensor, z_target_tensor)  
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, 
                                               shuffle=True)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(map.parameters(), lr=0.001)

    max_epochs = 500
    iterations = 3
    # Train the model
    for epoch in range(max_epochs):
        for batch_idx, (start, target) in enumerate(train_loader):
            optimizer.zero_grad()
            mapped = start
            loss = 0.0
            for i in range(iterations):
                mapped = map(mapped)
                loss += criterion(mapped, target)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            wandb.log({'loss': loss.item()})
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

if __name__ == '__main__':
    main()

