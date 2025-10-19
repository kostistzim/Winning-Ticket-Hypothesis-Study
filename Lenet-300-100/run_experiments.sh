#!/bin/bash
#BSUB -J "lottery[1-5]"
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -o outputs/output_%J_%I.out
#BSUB -e outputs/error_%J_%I.err

module load python3/3.11.7

# Adjust trial number so it starts from 0 in Python
trial_index=$((LSB_JOBINDEX - 1))

uv run main.py --trial ${trial_index}
