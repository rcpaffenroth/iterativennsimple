# Notes for developers

## Environment Setup

This project uses **uv** for fast, reliable Python dependency management. The configuration is defined in `pyproject.toml`.

### DevContainer Defaults
See `.devcontainer/devcontainer.json` for development environment defaults, which automatically installs Python packages, runs setup scripts, and installs VS Code extensions.

## Installation

### Option 1: GitHub Codespaces (Recommended)
If using GitHub Codespaces, simply run:
```bash
uv sync --extra dev
source .venv/bin/activate
```

### Option 2: Local Development
1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync --extra dev
   source .venv/bin/activate
   ```

## Managing Dependencies with uv

### Sync Development Environment
```bash
# Install all dependencies including dev and test tools
uv sync --extra dev
```

### PyTorch Configuration
PyTorch package index configuration is defined in `pyproject.toml` under `[tool.uv.sources]`. This ensures the correct PyTorch version is installed for your system.

### VS Code Integration
uv creates a local `.venv` directory by default, which VS Code automatically discovers for Python intellisense and debugging.

## Running Tests

```bash
# Run all tests including notebook tests
pytest

# Run specific test file
pytest tests/test_Sequential2D.py
```
