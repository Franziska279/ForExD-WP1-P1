#!/bin/bash
#SBATCH --job-name=S2_Download
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log
#SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
#SBATCH --ntasks=1        # One task per job
#SBATCH --gres=gpu:1      # Request one GPU per job (adjust for your cluster)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

# # Read command-line arguments
# ENV_PATH=$1
# SPATIAL_BUFFER=$2

# # Get the task ID (SLURM_ARRAY_TASK_ID is set automatically)
# TASK_ID=$SLURM_ARRAY_TASK_ID

# # Run the Python script with SLURM task ID and spatial buffer
# python s2_downloader.py --env "$ENV_PATH" --buffer "$SPATIAL_BUFFER" --index "$TASK_ID"



# Read environment variables and spatial buffer
ENV_PATH="/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/environment/.env"
BUFFER=500

# Get SLURM task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Run the Python script
python s2_downloader.py --env "$ENV_PATH" --buffer "$BUFFER" --index "$TASK_ID"