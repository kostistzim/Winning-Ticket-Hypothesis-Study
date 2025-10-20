#!/bin/bash
#BSUB -J "lth_exp[1-60]%4"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 10:00
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/lth_%J_%I.out
#BSUB -e outputs/lth_%J_%I.err

set -euo pipefail

echo "Starting job $LSB_JOBID, index $LSB_JOBINDEX on host $HOSTNAME"

# Load a stable Python
module load python3/3.11.9
module load cuda/11.7

# Make sure uv is already installed (e.g., installed from login node)
export PATH="$HOME/.local/bin:$PATH"

# Just to be safe:
unset PYTHONHOME
unset PYTHONPATH

# Sync environment once (optional)
uv sync

# Read config
CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" configs.txt)
echo "Running: uv run python-u main.py $CONFIG_LINE"

uv run python -u main.py $CONFIG_LINE
