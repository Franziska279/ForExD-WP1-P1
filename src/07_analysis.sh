#!/bin/bash
#SBATCH --partition=work
#SBATCH --nodes=1              # Use only one node
#SBATCH --ntasks=4             # Use 4 tasks
#SBATCH --cpus-per-task=1      # Use 1 CPU per task
#SBATCH --mem=180G             # Total memory for the node
#SBATCH --job-name=vi_bb
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/vegitation_index/logs-%A-%a.out
#SBATCH -e /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/logs/vegitation_index/errors-%A-%a.err  # Redirect stderr to error file


PYTHON=/Net/Groups/BGI/scratch/fmueller/miniconda3/envs/emp/bin/python

# Define the paths and disturbance type
IDS_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons_mini_idx.csv"
REFDM_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping_mini_idx_dissolved.shp"
MINICUBES_SHAPE_PATH="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/s2_minicube_bounderies_all.shp"
DISTURBANCE_TYPE="bark_beetle"

# Run the Python script with the specified arguments
$PYTHON /Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/src/07_analysis.py $IDS_PATH $REFDM_PATH $MINICUBES_SHAPE_PATH $DISTURBANCE_TYPE

# Exit script after completion
exit

