# Notes for developers

## Defaults
See .devcontainer/devcontainer.json for the defaults.  This can 
automatically install python packages, run scripts, install extensions, etc.

## How to use

If you are using this in Codespaces you can just run the following commands in the terminal:

```bash
uv sync --group dev --group sparse
source .venv/bin/activate
```

## pytorch and uv
PyTorch and sparse package index configuration is already defined in `pyproject.toml` under `[tool.uv.index]` and `[tool.uv.sources]`.

```bash
# Sync dependencies including sparse extras
uv sync --group dev --group sparse
```

## vscode and uv
uv uses a local `.venv` by default, which VS Code can discover automatically.

```bash
uv sync
```