"""
Tests for notebook 1-rcp-visualize-data.ipynb

Verifies that all datasets load correctly via generatedata.load_data
with proper structure, shapes, and non-empty data.
"""

import numpy as np
from generatedata.load_data import load_data


def _check_2d_dataset(name):
    """Helper: load a 2D dataset and verify basic structure."""
    data = load_data(name)
    assert 'target' in data, f"{name}: missing 'target' key"
    assert 'start' in data, f"{name}: missing 'start' key"
    assert 'x0' in data['target'].columns, f"{name}: target missing 'x0' column"
    assert 'x1' in data['target'].columns, f"{name}: target missing 'x1' column"
    assert 'x0' in data['start'].columns, f"{name}: start missing 'x0' column"
    assert 'x1' in data['start'].columns, f"{name}: start missing 'x1' column"
    assert len(data['target']) > 0, f"{name}: target is empty"
    assert len(data['start']) > 0, f"{name}: start is empty"
    return data


def test_load_regression_line():
    _check_2d_dataset('regression_line')


def test_load_pca_line():
    _check_2d_dataset('pca_line')


def test_load_circle():
    _check_2d_dataset('circle')


def test_load_regression_circle():
    _check_2d_dataset('regression_circle')


def test_load_manifold():
    data = load_data('manifold')
    assert 'target' in data
    assert 'start' in data
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert x_on.ndim == 2 and x_on.shape[1] == 3, \
        f"manifold target should be (n, 3), got {x_on.shape}"
    assert x_off.ndim == 2 and x_off.shape[1] == 3, \
        f"manifold start should be (n, 3), got {x_off.shape}"
    assert len(x_on) > 0


def test_load_MNIST1D():
    data = load_data('MNIST1D')
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert x_on.ndim == 2
    assert x_off.ndim == 2
    assert len(x_on) > 0
    # Last 10 columns are classification labels
    labels = x_on[0, -10:]
    assert len(labels) == 10


def test_load_MNIST():
    data = load_data('MNIST')
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert x_on.ndim == 2
    assert len(x_on) > 0
    # Image data (first 784 columns) reshapes to 28x28, plus 10 label columns
    assert x_on.shape[1] >= 784 + 10, \
        f"MNIST should have at least 794 columns, got {x_on.shape[1]}"
    img = x_on[0, :-10].reshape(28, 28)
    assert img.shape == (28, 28)


def test_load_EMlocalization():
    data = load_data('EMlocalization')
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert len(x_on) > 0
    assert len(x_off) > 0


def test_load_LunarLander():
    data = load_data('LunarLander')
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert len(x_on) > 0
    assert len(x_off) > 0
    # Last 6 columns are action data
    assert x_on.shape[1] >= 6


def test_load_MassSpec():
    data = load_data('MassSpec')
    x_on = np.array(data['target'])
    x_off = np.array(data['start'])
    assert len(x_on) > 0
    assert len(x_off) > 0
