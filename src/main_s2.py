from pathlib import Path
import argparse
import logging
import os
from dotenv import load_dotenv  # Import dotenv
import subprocess
import geopandas as gpd

from equi7_grid_creator import Equi7GridCreator

def main(env_path, spatial_buffer, max_jobs, run_grid, run_download, run_calculation, run_plotter):
    
    logging.info("Loading environment variables...")

    load_dotenv(env_path)
    region = str(os.getenv('REGION', '01')).zfill(2)
    resolution=10
    pixel_size=512

    logging.info(f"Starting Sentinel-2 and S1DM pipeline for region {region}...")

    if run_grid:
        try:
            logging.info("Running Equi7GridCreator...")
            grid_creator = Equi7GridCreator(resolution=resolution, pixel_size=pixel_size, buffer=spatial_buffer, env_path=env_path)
            grid_cells = grid_creator.create_grid()
            logging.info("Equi7 Grid creation completed.")
        except Exception as e:
            logging.error(f"Error in Equi7 Grid creation: {e}")

    if run_download:
        try:
            
            base_dir = os.getenv('EQUI7_GRIDS_DIR')
            intersected_grid_filepath = os.path.join(base_dir, os.getenv('INTERSECTED_GRIDS').format(resolution=resolution, pixel_size=pixel_size, region_id=region, buffer=spatial_buffer))
            grid_cells = gpd.read_file(intersected_grid_filepath)
            num_jobs = len(grid_cells)
            print(f"Amount of Grid Cells: {num_jobs}")
            logging.info(f"Submitting SLURM job array with {num_jobs} tasks (max 20 concurrent)...")

            cmd = f"sbatch --array=0-{num_jobs-1}%20 s2_downloader.sh"
            # Submit SLURM array job for downloading
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


            # Extract SLURM job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"Download SLURM job array submitted with Job ID: {job_id}")

            # # Submit preprocessing job (runs only after all download jobs finish)
            # preprocess_command = f"sbatch --dependency=afterok:{job_id} preprocess_job.sh {env_path}"
            # subprocess.run(preprocess_command, shell=True, check=True)
            # logging.info("Preprocessing job submitted successfully.")

        except Exception as e:
            logging.error(f"Error submitting SLURM jobs: {e}")


    # if run_calculation:
    #     try:
    #         calculator = Calculator(region_id=region, env_path=env_path)
    #         calculator.calculate(aggregation_mode, reference_year, variable, dca_keys, year_range, rsc_method)
    #         logging.info("Sentinel-2 downloading and processing completed successfully.")
    #     except Exception as e:
    #         logging.error(f"Error in Sentinel-2 downloading and processing: {e}")


    # if run_plotter:
    #     try:
    #         logging.info("Running Plotter...")
    #         plotter_s2 = PlotterS2(env_path, spatial_buffer=spatial_buffer)
    #         plotter_s2.plot()
    #         logging.info("Plot generation completed.")
    #     except Exception as e:
    #         logging.error(f"Error in Plotting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IDS and related processors.")
    parser.add_argument("--env", required=True, help="Path to the .env file")
    parser.add_argument("--spatial-buffer", type=int, default=500, help="List of spatial buffer distances for processing")
    parser.add_argument("--max-jobs", type=int, default=8, help="Max parallel jobs for processing")
    
    # Flags for optional processes
    parser.add_argument("--run-grid", action="store_true", help="Run IDSProcessor")
    parser.add_argument("--run-download", action="store_true", help="Run TCCProcessor")
    parser.add_argument("--run-calculation", action="store_true", help="Run S1CDProcessor")
    parser.add_argument("--run-plotter", action="store_true", help="Run Plotter for visualizing results")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename="process_s2_s1dm.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    main(
        env_path=Path(args.env),
        spatial_buffer=args.spatial_buffer,
        max_jobs=args.max_jobs,
        run_grid=args.run_grid,
        run_download=args.run_download,  # Fixed variable name
        run_calculation=args.run_calculation,
        run_plotter=args.run_plotter
    )