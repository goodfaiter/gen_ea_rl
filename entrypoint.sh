#!/bin/bash

# Activate the virtual environment
source /.venv/bin/activate

# Install the package in editable mode
cd workspace
pip install -e .

# Instalal tools
cd /workspace/src/gen_ea_rl/tools/instructor
pip install -e .

# Run the command passed to the container
exec "$@"