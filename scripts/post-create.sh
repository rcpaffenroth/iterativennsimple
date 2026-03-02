#! /bin/bash

# Install the python dependencies
uv sync

# Or, install the developer version
# uv sync --group dev

# Or, install the developer version with the sparse dependencies
# uv sync --group dev --group sparse

