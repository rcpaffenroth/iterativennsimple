#! /bin/bash

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
CUDA=cu118

pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip install pandas matplotlib pyarrow ipykernel ipympl jupyterlab
pip install .
