import pandas as pd
import pathlib
import iterativennsimple

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

    # Read the start data
    z_start = pd.read_parquet(data_dir / f'{name}_start.parquet')
    # Read the target data
    z_target = pd.read_parquet(data_dir / f'{name}_target.parquet')
    return {'start': z_start, 'target': z_target}