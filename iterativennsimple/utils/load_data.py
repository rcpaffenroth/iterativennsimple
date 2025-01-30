import pandas as pd
import pathlib
import iterativennsimple
import json
import requests

DATA_URL = 'http://users.wpi.edu/~rcpaffenroth/data/iterativennsimple/1-30-2025/'

def data_names(local=False) -> list:
    """List the names of the datasets that are available to load.

    Returns:
        list: the names of the datasets
    """
    if local:
        # The directory in which the notebook is located.
        base_dir = pathlib.Path(iterativennsimple.__path__[0])
        # The directory where the data is stored.
        data_dir = base_dir / '../data/processed'
        data_info = json.load(open(data_dir / 'info.json', 'r'))
    else:
        # Read the info json file from the URL http://users.wpi.edu/~rcpaffenroth/data/iterativennsimple/1-30-2025/info.json
        response = requests.get(DATA_URL+'/info.json')    
        data_info = response.json()
    return list(data_info.keys())

def load_data(name: str, local=False) -> dict:
    """Load in the dataset with the given name.  This functions loads in a variety of datasets created by the
    `generate-data` notebook.

    Args:
        name (str): the name of the dataset

    Returns:
        dict: the start and target points of the dataset
    """
    if local:
        # The directory in which the notebook is located.
        base_dir = pathlib.Path(iterativennsimple.__path__[0])
        # The directory where the data is stored.
        data_dir = base_dir / '../data/processed'

        # load in the info for the datasets
        with open(data_dir / f'info.json', 'r') as f:
            data_info = json.load(f)[name]

        # Read the start data
        z_start = pd.read_parquet(data_dir / f'{name}_start.parquet')
        # Read the target data
        z_target = pd.read_parquet(data_dir / f'{name}_target.parquet')
    else:
        # Read in the info for the datasets
        response = requests.get(DATA_URL+'/info.json')    
        data_info = response.json()
        # Read the start data
        z_start = pd.read_parquet(DATA_URL+f'/{name}_start.parquet')
        # Read the target data
        z_target = pd.read_parquet(DATA_URL+f'/{name}_target.parquet')
        
    return {'info': data_info, 'start': z_start, 'target': z_target}