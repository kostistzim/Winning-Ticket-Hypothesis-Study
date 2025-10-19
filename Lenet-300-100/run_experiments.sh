#!/bin/bash
#BSUB -J lottery_ticket[0-4]
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o outputs/output_%J_%I.out
#BSUB -e outputs/error_%J_%I.err

module load python3/3.11.7

uv run main.py --trial ${LSB_JOBINDEX}