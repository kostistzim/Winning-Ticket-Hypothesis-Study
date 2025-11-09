#!/bin/bash
#BSUB -J "fewshot_transfer[1-36]%6"
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/fewshot_%J_%I.out
#BSUB -e outputs/fewshot_%J_%I.err

set -euo pipefail

module load python3/3.11.9
module load cuda/12.1

export PATH="$HOME/.local/bin:$PATH"
unset PYTHONHOME
unset PYTHONPATH

# Sync your Python environment if you use uv (optional)
uv sync || echo "⚠️ uv sync skipped (not installed or no lockfile)"

CONFIG_FILE="fewshot_config.txt"
CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIG_FILE")

if [ -z "$CONFIG_LINE" ]; then
  echo "No config line found for job index $LSB_JOBINDEX in $CONFIG_FILE"
  exit 1
fi

echo "Running config: $CONFIG_LINE"

uv run python -u few_shot_training.py $CONFIG_LINE

echo "Job $LSB_JOBID (index $LSB_JOBINDEX) finished successfully."
