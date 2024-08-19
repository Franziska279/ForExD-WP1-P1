#!/bin/bash
#SBATCH --array=1
#SBATCH --partition=work
#SBATCH --nodes=1              # Use only one node
#SBATCH --ntasks=4             # Use 4 tasks
#SBATCH --cpus-per-task=1      # Use 1 CPU per task
#SBATCH --mem=200G             # Total memory for the node
#SBATCH --job-name=mini_downloading
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/mini_downloading/logs-%A-%a.out
#SBATCH -e /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/mini_downloading/errors-%A-%a.err  # Redirect stderr to error file

# Print SLURM array task ID
echo "Running Task ID $SLURM_ARRAY_TASK_ID"

# Path to the Python interpreter
PYTHON=/Net/Groups/BGI/scratch/fmueller/miniconda3/envs/minicuber/bin/python

# Specify input file and output directory
OUTPUT_DIR="/Net/Groups/BGI/scratch/fmueller/Data/s2/512/"
#INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/fire_drought_minicubes.shp"
INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/cleaned_centroids.shp"


# Execute Python script with SLURM array task ID, input file, and output directory arguments
$PYTHON /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/src/05_s2_minicubes_downloading.py $SLURM_ARRAY_TASK_ID $INPUT_FILE $OUTPUT_DIR

# Exit script after completion
exit
