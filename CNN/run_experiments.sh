#!/bin/sh
# --- BSUB Directives (LSF Scheduler) ---
#BSUB -J "lth_exp[1-60]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 10:00
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -o bsub_logs/lth_%J_%I.out
#BSUB -e bsub_logs/lth_%J_%I.err

echo "---"
echo "Starting LSF Job: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Running on host: $HOSTNAME"
echo "---"

# Load required modules
module load python3/3.10.13
module load cuda/11.7

# Activate Python virtual environment
source assignment2_env/bin/activate

# Debug: Check current directory
echo "Current directory: $(pwd)"

# Get the arguments for this specific job from configs.txt
CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" configs.txt)

# Debug: Print the config line
echo "Config (Line $LSB_JOBINDEX): $CONFIG_LINE"

echo "Running command:"
echo "python main.py $CONFIG_LINE"
echo "---"

# Execute main script with selected arguments
python main.py $CONFIG_LINE

echo "---"
echo "Job $LSB_JOBID finished."