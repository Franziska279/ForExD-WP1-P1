#!/bin/bash
#SBATCH --job-name=run_main            # Job name
#SBATCH --output=logs/run_main_%j.out # Standard output log (%j will be replaced with job ID)
#SBATCH --error=logs/run_main_%j.err  # Standard error log
#SBATCH --time=04:00:00               # Time limit hh:mm:ss
#SBATCH --partition=clara           # Partition name (adjust to your system's available partitions)
#SBATCH --ntasks=1                    # Number of tasks (1 for a single Python script)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --mail-type=END               # Send email on job start, end, fail (optional)
#SBATCH --mail-user=franziska_rosa-maria.mueller@uni-leipzig.de # Your email (if mail-type is enabled)

# Load necessary modules (adjust as needed for your cluster environment)
module load python/3.9  # Example Python module (replace with your cluster's Python version)

# Set working directory (if not already in the correct directory)
cd /home/sc.uni-leipzig.de/sy58xupo/ForExD-WP1-P1/src/

# Run the Python script
python main.py

