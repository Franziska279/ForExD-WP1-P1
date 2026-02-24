#!/bin/bash
# =============================================================================
# FILE: run_main.sh
# AUTHOR: Franziska Rosa-Maria Mueller
# DATE: 2025-04-05
# PROJECT: ForExD-WP1-P1 - Reproducible Satellite Data Processing Pipeline
# PURPOSE: Submit a SLURM job to run the full analysis pipeline (S1CD + plotting)
#          using configuration from .env file.
#
# DESCRIPTION:
#   This script submits a job to the HPC cluster to run the main analysis
#   pipeline. It uses a .env file for configuration and passes parameters via CLI.
#
#   The pipeline includes:
#     - S1CD processing (if --run-s1cd is enabled)
#     - Plot generation (if --run-plotter is enabled)
#
#   Output logs are saved in the 'logs/' directory.
#
# USAGE:
#   sbatch run_main.sh
#
# DEPENDENCIES:
#   - Python (with required packages in environment)
#   - SLURM scheduler (Clara partition)
#   - .env file with configuration (see .env.example)
#
# LICENSE: MIT (see LICENSE.md)
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
#SBATCH --mail-user=franziska_rosa-maria.mueller@uni-leipzig.de  # Your email

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to the .env file (must be accessible on HPC)
ENV_PATH="/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/environment/.env"

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
echo "📅 Time range: 2016 - 2021"
echo "🧩 Spatial buffer: 500 meters"
echo "🔢 Max parallel jobs: 10"

# Run the main Python script with all required arguments
python main.py \
    --env "$ENV_PATH" \
    --start-year 2016 \
    --end-year 2021 \
    --buffer-years 2 \
    --spatial-buffer 500 \
    --max-jobs 10 \
    --run-ids \
    --run-tcc \
    --run-s1cd \
    --run-plotter

# =============================================================================
# FINAL MESSAGE
# =============================================================================
echo "✅ Job completed. Check logs in 'logs/' directory."



# #!/bin/bash
# #SBATCH --job-name=run_main            # Job name
# #SBATCH --output=logs/run_main_%j.out  # Standard output log (%j will be replaced with job ID)
# #SBATCH --error=logs/run_main_%j.err   # Standard error log
# #SBATCH --time=2:00:00              # Time limit hh:mm:ss
# #SBATCH --partition=clara              # Partition name (adjust to your system's available partitions)
# #SBATCH --ntasks=1                     # Number of tasks (1 for a single Python script)
# #SBATCH --cpus-per-task=10              # Number of CPU cores per task
# #SBATCH --mem=20G                     # Memory per node
# #SBATCH --mail-type=END                # Send email on job start, end, fail (optional)
# #SBATCH --mail-user=franziska_rosa-maria.mueller@uni-leipzig.de # Your email (if mail-type is enabled)

# # Define path to the .env file
# ENV_PATH="/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/environment/.env"

# # Run Python script (region is now loaded from .env)
# python main.py --env "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env" --start-year 2016 --end-year 2021 --buffer-years 2 --spatial-buffer 500 --max-jobs 10 --run-s1cd --run-plotter