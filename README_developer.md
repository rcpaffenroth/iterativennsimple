# Notes for developers

## Defaults
See .devcontainer/devcontainer.json for the defaults.  This can 
automatically install python packages, run scripts, install extensions, etc.

## How to use

If you are using this in Codespaces you can just run the following commands in the terminal:

```bash
poetry shell
poetry install --with dev
```

## pytorch and poetry
To get poetry to install the correct version of pytorch you need to so something like this

```bash
# This adds the pytorch repo to the list of sources
poetry source add -p explicit pytorch https://download.pytorch.org/whl/cpu
# This uses the pytorch repo to install pytorch instead of the default
poetry add --source pytorch torch torchvision
```

## vscode and poetry
You want the virtual environment to be local, so that vscode can find it.  The "--local" flag is the important part
and will create the poetry.toml file that we can submit to git.

```bash
poetry config virtualenvs.in-project true --local
```

## Generating test data
To generate a local copy of the test data you can run the following command:

```bash
cd scripts
python generate_data.py
```