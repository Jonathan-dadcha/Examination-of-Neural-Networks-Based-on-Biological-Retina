#!/bin/bash
# Script to run CBEM Jupyter notebook using the conda environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use the conda Python to run jupyter
"$PROJECT_ROOT/.conda/bin/python" -m jupyter notebook exampleCBEMfitting.ipynb

