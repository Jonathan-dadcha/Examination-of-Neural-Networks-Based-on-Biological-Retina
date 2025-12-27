#!/bin/bash
# Example script to run Deep-Retina models
# Replace the values with actual experiment and stimulus data you have

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Example 1: BN_CNN model with whitenoise
echo "Running BN_CNN model with whitenoise..."
"$PROJECT_ROOT/.conda/bin/python" fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN

# Uncomment and modify the examples below as needed:

# Example 2: BN_CNN model with naturalscene
# "$PROJECT_ROOT/.conda/bin/python" fit_models.py --expt 15-10-07 --stim naturalscene --model BN_CNN

# Example 3: LN model (requires --cell)
# "$PROJECT_ROOT/.conda/bin/python" fit_models.py --expt 15-10-07 --stim whitenoise --model LN_softplus --cell 0

# Example 4: Different experiment
# "$PROJECT_ROOT/.conda/bin/python" fit_models.py --expt 15-11-21a --stim whitenoise --model BN_CNN

