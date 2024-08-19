#!/bin/bash
#SBATCH --array=46
#SBATCH --partition=work
#SBATCH --nodes=1              # Use only one node
#SBATCH --ntasks=2             # Use 4 tasks
#SBATCH --cpus-per-task=1      # Use 1 CPU per task
#SBATCH --mem=80G             # Total memory for the node
#SBATCH --job-name=mini_preprocessing
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/mini_preprocessing/logs-%A-%a.out
#SBATCH -e /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/mini_preprocessing/errors-%A-%a.err  # Redirect stderr to error file

# Print SLURM array task ID
echo "Running Task ID $SLURM_ARRAY_TASK_ID"

# Path to the Python interpreter
PYTHON=/Net/Groups/BGI/scratch/fmueller/miniconda3/envs/emp/bin/python

# Specify input file and output directory
INPUT_DIR="/Net/Groups/BGI/scratch/fmueller/Data/s2_region8_nc_256px_vi/"
#INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/fire_drought_minicubes.shp"
INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/cleaned_centroids.shp"

# Execute Python script with SLURM array task ID, input file, and output directory arguments
$PYTHON /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/src/05_1_s2_minicubes_preprocessing.py $SLURM_ARRAY_TASK_ID $INPUT_FILE $INPUT_DIR 

# Exit script after completion
exit
