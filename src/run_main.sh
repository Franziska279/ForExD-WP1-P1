#!/bin/bash
# =============================================================================
# FILE: run_main.sh
# AUTHOR: Franziska Rosa-Maria Mueller
# DATE: 2025-04-05
# PROJECT: ForExD-WP1-P1 - Reproducible Satellite Data Processing Pipeline
# PURPOSE: Submit a SLURM job to run the analysis pipeline using configuration
#          from the .env file. Individual stages can be enabled independently.
#
# DESCRIPTION:
#   This script submits a job to the HPC cluster to run the main analysis
#   pipeline. It uses a .env file for configuration and passes parameters via CLI.
#
#   The pipeline has four stages that build on each other:
#     1. IDS processing    (--run-ids)   filter and reproject USDA IDS polygons
#     2. TCC processing    (--run-tcc)   prepare tree canopy cover mask
#     3. S1CD processing   (--run-s1cd)  match Sentinel-1 change detection to IDS
#     4. Plot generation   (--run-plotter) generate all analysis figures
#
#   Uncomment only the stages you need. If IDS, TCC, and S1CD results already
#   exist, you can run --run-plotter alone.
#
#   Output logs are saved in the 'logs/' directory.
#
# USAGE:
#   sbatch run_main.sh
#
# DEPENDENCIES:
#   - Python (with required packages in environment)
#   - SLURM scheduler (Clara partition)
#   - .env file with configuration (see environment/.env.example)
#
# LICENSE: MIT (see LICENSE)
# =============================================================================

# =============================================================================
# SLURM JOB DIRECTIVES
# =============================================================================
#SBATCH --job-name=run_main            # Job name
#SBATCH --output=logs/run_main_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=logs/run_main_%j.err   # Standard error log
#SBATCH --time=2:00:00                 # Max runtime: 2 hours
#SBATCH --partition=clara              # HPC partition (adjust if needed)
#SBATCH --ntasks=1                     # One task (single Python script)
#SBATCH --cpus-per-task=10             # 10 CPU cores per task
#SBATCH --mem=20G                      # 20 GB memory per node
#SBATCH --mail-type=END                # Send email on job completion
#SBATCH --mail-user=YOUR_EMAIL@institution.de                    # Your email

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to the .env file (must be accessible on HPC)
ENV_PATH="$(dirname "$0")/../environment/.env"
echo "$ENV_PATH"

# Check if .env file exists
if [[ ! -f "$ENV_PATH" ]]; then
    echo "❌ ERROR: .env file not found at $ENV_PATH"
    exit 1
fi

# =============================================================================
# RUN MAIN PYTHON SCRIPT
# =============================================================================
echo "🚀 Starting analysis pipeline..."
echo "📁 Using .env file: $ENV_PATH"
echo "📅 Time range: 2016 - 2020"
echo "🧩 Spatial buffer: 500 meters"
echo "🔢 Max parallel jobs: 10"

# Uncomment stages as needed — they build on each other:
#   --run-ids      Step 1: filter and reproject USDA IDS polygons
#   --run-tcc      Step 2: prepare tree canopy cover mask
#   --run-s1cd     Step 3: match Sentinel-1 change detection to IDS polygons
#   --run-plotter  Step 4: generate all analysis figures
python main.py \
    --env "$ENV_PATH" \
    --start-year 2016 \
    --end-year 2020 \
    --buffer-years 2 \
    --spatial-buffer 500 \
    --max-jobs 10 \
    --run-plotter
    # --run-tcc
    # --run-s1cd
    #--run-ids 

# =============================================================================
# FINAL MESSAGE
# =============================================================================
echo "✅ Job completed. Check logs in 'logs/' directory."