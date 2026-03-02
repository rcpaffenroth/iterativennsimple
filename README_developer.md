# Notes for developers

## Defaults
See .devcontainer/devcontainer.json for the defaults.  This can 
automatically install python packages, run scripts, install extensions, etc.

## How to use

If you are using this in Codespaces you can just run the following commands in the terminal:

```bash
uv sync --group dev
source .venv/bin/activate
```

## pytorch and uv
PyTorch package index configuration is already defined in `pyproject.toml` under `[tool.uv.index]` and `[tool.uv.sources]`.

```bash
# Sync dependencies (including dev group)
uv sync --group dev
```

## vscode and uv
uv uses a local `.venv` by default, which VS Code can discover automatically.

```bash
uv sync
```

## Generating test data
To generate a local copy of the test data you can run the following command:

```bash
cd scripts
python generate_data.py
```