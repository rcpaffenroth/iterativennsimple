import pandas as pd
import pathlib
import iterativennsimple
import json

def data_names() -> list:
    """List the names of the datasets that are available to load.

    Returns:
        list: the names of the datasets
    """
    # The directory in which the notebook is located.
    base_dir = pathlib.Path(iterativennsimple.__path__[0])
    # The directory where the data is stored.
    data_dir = base_dir / '../data/processed'
    # List the files in the directory
    data_files = data_dir.glob('*.parquet')
    # Extract the names of the files
    data_names = [f.stem.split('_')[0] for f in data_files]
    return data_names

def load_data(name: str) -> dict:
    """Load in the dataset with the given name.  This functions loads in a variety of datasets created by the
    `generate-data` notebook.

    Args:
        name (str): the name of the dataset

    Returns:
        dict: the start and target points of the dataset
    """
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
    return {'info': data_info, 'start': z_start, 'target': z_target}