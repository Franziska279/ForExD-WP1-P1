#!/bin/bash
#SBATCH --job-name=sentinel_processing
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log
#SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
#SBATCH --array=1-10%5    # Run 20 tasks in parallel (adjust as needed)
#SBATCH --ntasks=1        # One task per job
#SBATCH --cpus-per-task=4 # Adjust based on CPU requirements
#SBATCH --gres=gpu:1      # Request one GPU per job (adjust for your cluster)
#SBATCH --mem=80G         # Memory per task (adjust based on requirements)

# Define paths to the arguments
ENV_PATH="/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/environment/.env"

# Run the Python script with the SLURM_ARRAY_TASK_ID as the index
python s2_downloader.py --env "$ENV_PATH" $SLURM_ARRAY_TASK_ID 
