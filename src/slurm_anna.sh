#!/bin/bash
#SBATCH --job-name=era5_land
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log
#SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
#SBATCH --time=4-00:00:00   # 4 days
#SBATCH --array=2005-2016%5    # Run 20 tasks in parallel (adjust as needed)
#SBATCH --ntasks=1        # One task per job
#SBATCH --cpus-per-task=4 # Adjust based on CPU requirements
#SBATCH --mem=120G         # Memory per task (adjust based on requirements)


# Run the Python script with the SLURM_ARRAY_TASK_ID as the index
python downloader_anna.py  $SLURM_ARRAY_TASK_ID 
