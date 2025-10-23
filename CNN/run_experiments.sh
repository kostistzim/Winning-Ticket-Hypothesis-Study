#!/bin/bash
#BSUB -J "lth_exp[1-20]%4"
#BSUB -q gpuh100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 6:00
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/lth_%J_%I.out
#BSUB -e outputs/lth_%J_%I.err

set -euo pipefail

echo "Starting job $LSB_JOBID, index $LSB_JOBINDEX on host $HOSTNAME"

# Load stable modules
module load python3/3.11.9
module load cuda/12.1

export PATH="$HOME/.local/bin:$PATH"
unset PYTHONHOME
unset PYTHONPATH

# Sync environment if needed
uv sync

# Read config for this job
CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" configs.txt)
echo "Running: uv run python -u main.py $CONFIG_LINE"

# Execute
uv run python -u main.py $CONFIG_LINE
