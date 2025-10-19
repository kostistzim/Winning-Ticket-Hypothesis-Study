#!/bin/sh
#
# --- BSUB Directives (LSF Scheduler) ---
#
# These lines tell the scheduler what resources to reserve for each job.

#BSUB -J "lth_exp[1-60]"        # Job name "lth_exp" and array size (1-60)
                                 # This MUST match the number of lines in configs.txt

#BSUB -q gpuv100                # MUST VERIFY: The queue to submit to.
                                 # 'gpuv100' or 'gpua100' are common. Check DTU's docs.

#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU in exclusive mode
#BSUB -n 4                      # Request 4 CPU cores for this job
#BSUB -W 10:00                  # Walltime limit (10 hours, 0 minutes).
                                 # Adjust if your Conv-6 jobs take longer.

#BSUB -R "rusage[mem=16G]"      # Memory request (16GB)
#BSUB -R "span[hosts=1]"        # Ensure all 4 CPUs are on the same machine

#BSUB -o bsub_logs/lth_%J_%I.out  # Standard output log file
                                 # %J is the Job ID, %I is the Array Index
#BSUB -e bsub_logs/lth_%J_%I.err  # Standard error log file

# --- Environment Setup ---
echo "---"
echo "Starting LSF Job: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Running on host: $HOSTNAME"
echo "---"

# Load the required modules
# These versions are examples from DTU's docs
module load python3/3.11.3
module load cuda/11.7

# Activate your Python virtual environment
# MUST VERIFY: Make sure this path is correct for your cluster setup
source (assignement2)/bin/activate

# --- Get Task Configuration ---
# Get the arguments for *this specific job* from configs.txt
# $LSB_JOBINDEX is the LSF variable for the array job number (1, 2, 3...)
CONFIG_LINE=$(sed -n "${LSB_JOBINDEX}p" configs.txt)

# --- Run the Experiment ---
echo "Running command:"
echo "python CNN/main.py $CONFIG_LINE"
echo "---"

# Execute your main script with the selected arguments
python CNN/main.py $CONFIG_LINE

echo "---"
echo "Job $LSB_JOBID finished."