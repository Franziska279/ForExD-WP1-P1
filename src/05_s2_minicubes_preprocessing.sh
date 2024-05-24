#!/bin/bash
#SBATCH --array=0-1%20
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=200G
#SBATCH --job-name=mini_downloading
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH -o logs/minicube_downloading_pipeline/logs-%A-%a.out

echo "Running Task ID $SLURM_ARRAY_TASK_ID"

# Path to the Python interpreter
PYTHON=/Net/Groups/BGI/scratch/fmueller/miniconda3/envs/minicuber/bin/python

OUTPUT_DIR="/Net/Groups/BGI/scratch/fmueller/Data/s2/"
INPUT_FILE="/Net/Groups/BGI/scratch/fmueller/Project:ForEXD/intersections_csv/minicube_grid/minicube_grid_2024.shp"

$PYTHON minicube_download_preprocessing_pipeline.py $SLURM_ARRAY_TASK_ID $INPUT_FILE $OUTPUT_DIR

exit