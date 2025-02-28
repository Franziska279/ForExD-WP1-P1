from pathlib import Path
import argparse
import logging
import os
from dotenv import load_dotenv  # Import dotenv

from ids_processor import IDSProcessor
from tcc_processor import TCCProcessor
from s1cd_processor import S1CDProcessor
from plotter import Plotter
from equi7_grid_creator import Equi7GridCreator

# Load environment variables from .env file
def load_env_variables(env_path):
    load_dotenv(env_path)
    return {
        "region": int(os.getenv("REGION", 8)),  # Default to 0 if not set
    }

def main(env_path, start_year, end_year, excluded_dca_types, buffer_years, spatial_buffer, max_jobs, 
         run_ids, run_tcc, run_s1cd, run_plotter, run_grid_creator):
    
    logging.info("Loading environment variables...")
    env_vars = load_env_variables(env_path)
    region = env_vars["region"]

    logging.info(f"Starting IDS processing pipeline for region {region}...")

    if run_ids:
        try:
            processor = IDSProcessor(region_id=region, env_path=env_path)
            processor.loading()
            data = processor.exclude_include_overlapping_entries()
            processor.filter_data(start_year=start_year, end_year=end_year, excluded_dca_types=excluded_dca_types)
            processor.print_status()
            processor.save_and_plot()
            logging.info("IDS processing completed successfully.")
        except Exception as e:
            logging.error(f"Error in IDS processing: {e}")
    
    if run_tcc:
        try:
            logging.info(f"--------------------------------------------")
            logging.info("Running TCCProcessor...")
            logging.info(f"--------------------------------------------")
            tcc_processor = TCCProcessor(env_path)
            tcc_processor.process()
            logging.info("TCC processing completed.")
        except Exception as e:
            logging.error(f"Error in TCC processing: {e}")

    if run_s1cd:
        try:
            logging.info("Running S1CDProcessor...")
            s1cd_processor = S1CDProcessor(env_path, buffer_years=buffer_years, spatial_buffer=spatial_buffer, max_jobs=max_jobs)
            s1cd_processor.process_files()
            logging.info("S1CD processing completed.")
        except Exception as e:
            logging.error(f"Error in S1CD processing: {e}")

    if run_plotter:
        try:
            logging.info("Running Plotter...")
            plotter = Plotter(env_path, spatial_buffer=spatial_buffer)
            plotter.plot()
            logging.info("Plot generation completed.")
        except Exception as e:
            logging.error(f"Error in Plotting: {e}")

    if run_grid_creator:
        try:
            logging.info("Running Equi7GridCreator...")
            grid_creator = Equi7GridCreator(resolution=10, pixel_size=512, env_path=env_path)
            grid_creator.create_grid()
            logging.info("Equi7 Grid creation completed.")
        except Exception as e:
            logging.error(f"Error in Equi7 Grid creation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IDS and related processors.")
    parser.add_argument("--env", required=True, help="Path to the .env file")
    parser.add_argument("--start-year", type=int, default=2016, help="Start year for filtering disturbances")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for filtering disturbances")
    parser.add_argument("--excluded-dca-types", nargs='+', default=['other', 'multi_damage', 'other_abiotic', 'other_biotic'], help="List of disturbance types to exclude")
    parser.add_argument("--buffer-years", type=int, default=2, help="Temporal buffer for S1CDProcessor")
    parser.add_argument("--spatial-buffer", nargs='+', type=int, default=[100, 250, 500, 1000], help="List of spatial buffer distances for processing")
    parser.add_argument("--max-jobs", type=int, default=8, help="Max parallel jobs for processing")
    
    # Flags for optional processes
    parser.add_argument("--run-ids", action="store_true", help="Run IDSProcessor")
    parser.add_argument("--run-tcc", action="store_true", help="Run TCCProcessor")
    parser.add_argument("--run-s1cd", action="store_true", help="Run S1CDProcessor")
    parser.add_argument("--run-plotter", action="store_true", help="Run Plotter for visualizing results")
    parser.add_argument("--run-grid-creator", action="store_true", help="Run Equi7GridCreator")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename="process.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run main function with parsed arguments
    main(
        env_path=Path(args.env),
        start_year=args.start_year,
        end_year=args.end_year,
        excluded_dca_types=args.excluded_dca_types,
        buffer_years=args.buffer_years,
        spatial_buffer=args.spatial_buffer,
        max_jobs=args.max_jobs,
        run_ids=args.run_ids,
        run_tcc=args.run_tcc,
        run_s1cd=args.run_s1cd,
        run_plotter=args.run_plotter,
        run_grid_creator=args.run_grid_creator
    )
