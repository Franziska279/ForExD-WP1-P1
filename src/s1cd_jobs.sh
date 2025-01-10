#!/bin/bash
#SBATCH --job-name=s1cd_extraction
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=120G
#SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --output=logs/s1cd_extraction_%j.out  # Standard output log (%j will be replaced with job ID)
#SBATCH --error=logs/s1cd_extraction_%j.err   # Standard error log

ENV_PATH="/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/environment/.env"
INPUT_FILE=$1
OUTPUT_DIR=$2

python s1cd_process_file_script.py "$INPUT_FILE" "$OUTPUT_DIR" --env "$ENV_PATH"
