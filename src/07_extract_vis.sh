#!/bin/bash
#SBATCH --array=0-7340%50
#SBATCH --partition=work
#SBATCH --nodes=1              # Use only one node
#SBATCH --ntasks=1             # Use 4 tasks
#SBATCH --cpus-per-task=1      # Use 1 CPU per task
#SBATCH --mem=80G             # Total memory for the node
#SBATCH --job-name=vi_bb
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/vegitation_index/logs-%A-%a.out
#SBATCH -e /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/vegitation_index/errors-%A-%a.err  # Redirect stderr to error file

# Print SLURM array task ID
echo "Running Task ID $SLURM_ARRAY_TASK_ID"

PYTHON=/Net/Groups/BGI/scratch/fmueller/miniconda3/envs/emp/bin/python

# Define the paths and disturbance type
IDS_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons_mini_idx.csv"
REFDM_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping_mini_idx_dissolved.shp"
DISTURBANCE_TYPE="wind"
DIR_PATH="/Net/Groups/BGI/scratch/fmueller/Data/s2_region8_nc_256px_vi/"
OUTPUT_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/tmp_wind/"

# Run the Python script with the specified arguments
$PYTHON /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/src/07_extract_vis.py $IDS_PATH $REFDM_PATH $DISTURBANCE_TYPE $DIR_PATH $OUTPUT_PATH $SLURM_ARRAY_TASK_ID

# Exit script after completion
exit
