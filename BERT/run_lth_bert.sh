#!/bin/bash
#BSUB -J "bert_lth[1-6]%6"
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/bert_lth_%J_%I.out
#BSUB -e outputs/bert_lth_%J_%I.err

set -euo pipefail

module load python3/3.11.9
module load cuda/12.1

export PATH="$HOME/.local/bin:$PATH"
unset PYTHONHOME
unset PYTHONPATH

uv sync || echo "uv sync skipped (not installed or no lockfile)"

CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" configs_bert.txt)
if [ -z "$CONFIG_LINE" ]; then
  echo "No config line found for job index $LSB_JOBINDEX in configs_bert.txt"
  exit 1
fi

echo "Running config: $CONFIG_LINE"

uv run python -u lottery_ticket_bert.py $CONFIG_LINE

echo "Job $LSB_JOBID (index $LSB_JOBINDEX) finished successfully."