
# %%
# Note, the above line allows this code to be run iteractively
# simialr to a jupyter notebook using an appropriate IDE (such as VSCode)

# Import necessary libraries

#  The main torch library
import torch
# Weights and biases
import wandb
# Click is a library for creating command line interfaces and is used here for command line arguments
import click

# The following are all of the functions that we have created in the iterativennsimple package
# Turn a dataframe into a tensor
from iterativennsimple.utils.df_to_tensor import df_to_tensor
# A class to help with returning pairs of start and target data
from iterativennsimple.utils.StartTargetData import StartTargetData
# Load the datasets that come with iterativennsimple
from iterativennsimple.utils.load_data import load_data

# Setup logging.  This is better than using print statements.
import logging
logging.basicConfig(level=logging.INFO)

def get_model(model_type: str, size: int) -> torch.nn.Module:
    """Generate a model of the given type

    These are simple examples of models, and this is a place where you can
    experiment with different architectures.

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
    elif model_type == 'affine':    
        model = torch.nn.Sequential(
            torch.nn.Linear(size, size),
        )      
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    return model

# Configure can happen in two ways.  First, we can just
# set the values in this config dictionary directly.
config = {
    # NOTE: Replace this with your wandb entity and project,
    # otherwise you will be writing to Prof. Paffenroth's account :-)
    'wandb_entity': 'rcpaffenroth-wpi',
    'wandb_project': 'test'
}

# Second, we can use the click library to set the values from the command line
@click.command()
@click.option("--data_name", default='MNIST1D',
              help="The name of the dataset to use.")
@click.option("--model_type", default='square',
              help="The name of the model to use.")
@click.option("--gpu", is_flag=True,
              help="Whether to use a GPU if available.")
@click.option("--threads", default=1,
              help="The number of threads to use.")
def main(**kwargs):
    """Run the training of the model

    This is a simple example of all of the pieces that you need to train a model, do logging, and
    save the results to wandb.

    Args:
        dataname (str): the name of the dataset to use
    """
    # Update the config dictionary with the command line arguments
    # if they are present in both places, the command line arguments
    # will take precedence
    config.update(kwargs)

    # Initialize wandb
    wandb.init(entity=config['wandb_entity'], 
               project=config['wandb_project'],
               config=config)

    # Set the number of threads
    # By default pytorch uses all of the threads on the machine, and this
    # can cause problems when running multiple experiments at once. It is ok to
    # use all the threads when running a single experiment, and it will make your experiment
    # run faster (sometimes :-), but you want to be sure to do this on purpose and not by accident.
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
    mymap = get_model(config['model_type'], config['data_size'])
    # Move the model to the device.  This copies the model to the GPU if we are using the GPU.
    mymap.to(device)

    # optional: log the model
    # This will upload many details about the model to wandb
    wandb.watch(mymap)

    # Make two pytorch tensor datasets from the start and target data
    z_start_tensor = df_to_tensor(z_data['start'])
    z_target_tensor = df_to_tensor(z_data['target'])

    # Move the data to the device.
    # Note if the data and the model are on different devices, this will cause an error!
    z_start_tensor = z_start_tensor.to(device)
    z_target_tensor = z_target_tensor.to(device)
        
    # Create a dataloader.  This does several things for us:
    # 1. It shuffles the data
    # 2. It groups the data into batches
    train_data = StartTargetData(z_start_tensor, z_target_tensor)  
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100,
                                               shuffle=True)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    # We use the Adam optimizer, which is a good general purpose optimizer.
    # As an exercise, you might want to try different optimizers and learning rates.
    # These can be made command line arguments!  And even better, you could use wandb
    # sweeps to try different hyperparameters.
    optimizer = torch.optim.Adam(mymap.parameters(), lr=0.001)

    # Again, these are all hyperparameters that could be set from the command line
    max_epochs = 50
    # and this is the first place we stray from the normal use of pytorch.  We are going to
    # iterate the map several times and one might do with a dynamical system!
    iterations = 3
    # Train the model
    for epoch in range(max_epochs):
        for batch_idx, (start, target) in enumerate(train_loader):
            # Zero the parameter gradients.  This is important to do before each batch!
            optimizer.zero_grad()

            # We record the per iteration loss and the total loss
            iteration_loss = []

            # This is the forward pass
            # We initialize mapped to be the start data
            mapped = start
            for i in range(iterations):
                # Each iteration we apply the map
                mapped = mymap(mapped)
                # And calculate the loss, i.e. how far we are from the target data/attracting fixed point
                iteration_loss.append(criterion(mapped, target))

            # Given the iteration losses, we want to minimize the total loss, and there are many ways to do this
            # Here we are just summing the losses, but you could try other things like a weighted sum.
            total_loss = 0
            for loss in iteration_loss:
                total_loss += loss

            # That is the end of the forward pass.  Now we do some logging to wandb.
            # Log the total loss
            wandb.log({'total loss': total_loss.item()})
            # and log the loss of each iteration
            for i, loss in enumerate(iteration_loss):
                wandb.log({f'iterations loss {i}': loss.item()})
                # Now, if the the iteration losses are all the *exactly* the same, then we likely have
                # a bug in our code.  So, we program defensively and check for this occurance.
                # This is a good example of where logging can help you find bugs in your code.
                if i > 0 and loss == iteration_loss[i-1]:
                    logging.warning(f'Iteration losses are the same!')
                    logging.warning(f'epoch {epoch} batch_idx {batch_idx} iteration {i-1}={iteration_loss[i-1]} iteration {i}={loss}')

            # This is the backward pass where we calculate the gradients
            total_loss.backward()
            # This is the step where we update the weights
            optimizer.step()
        if epoch % 10 == 0:
            # An example of logging the loss every n epochs
            logging.info('Epoch %s, Loss %s', epoch, total_loss.item())

if __name__ == '__main__':
    main()

