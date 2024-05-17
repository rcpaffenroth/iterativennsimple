#! /bin/bash

# Install the python dependencies
poetry shell
poetry install
cd scripts; python3 generate_data.py

# Or, install the developer version
# poetry install --with dev
