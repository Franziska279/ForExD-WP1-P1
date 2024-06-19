#!/bin/bash
#SBATCH --array=8-8%20
#SBATCH --partition=work
#SBATCH --nodes=2               # Increase the number of nodes to 2
#SBATCH --ntasks=2             # Increase the number of tasks to 4
#SBATCH --cpus-per-task=2       # Number of CPUs per task
#SBATCH --mem-per-cpu=120G       # Memory per CPU
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
OUTPUT_DIR="/Net/Groups/BGI/scratch/fmueller/Data/test_s2/"
INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/intersected_grid.shp"

# Execute Python script with SLURM array task ID, input file, and output directory arguments
$PYTHON /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/src/05_s2_minicubes_preprocessing.py $SLURM_ARRAY_TASK_ID $INPUT_FILE $OUTPUT_DIR

# Exit script after completion
exit
