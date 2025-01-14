#!/bin/bash
#SBATCH --job-name=run_main            # Job name
#SBATCH --output=logs/run_main_%j.out  # Standard output log (%j will be replaced with job ID)
#SBATCH --error=logs/run_main_%j.err   # Standard error log
#SBATCH --time=5:00:00              # Time limit hh:mm:ss
#SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
#SBATCH --ntasks=1                     # Number of tasks (1 for a single Python script)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --mem=80G                     # Memory per node
#SBATCH --mail-type=END                # Send email on job start, end, fail (optional)
#SBATCH --mail-user=franziska_rosa-maria.mueller@uni-leipzig.de # Your email (if mail-type is enabled)

# Define paths to the arguments
ENV_PATH="/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/environment/.env"
METADATA_OUTPUT="/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/results_clean/metadata_table.csv"

# Run the Python script with arguments
python main.py --env "$ENV_PATH" --metadata-output "$METADATA_OUTPUT"
