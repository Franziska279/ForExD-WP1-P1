"""
ForExD-WP1-P1 — Main Pipeline Entry Point
==========================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD Work Package 1, Part 1
         Improving forest disturbance labels using Sentinel-1 change detection

Description
-----------
This script runs the full analysis pipeline that validates and refines USDA
Insect and Disease Survey (IDS) forest disturbance labels by cross-referencing
them with Sentinel-1 SAR change detection data.

The pipeline has four sequential stages — each builds on the output of the previous:

  1. IDSProcessor   -- Loads the USDA IDS dataset for a given region, removes
                       temporal and spatial overlaps, filters to the disturbance
                       types and year range of interest, and reprojects to the
                       target CRS. Output: filtered IDS shapefiles.

  2. TCCProcessor   -- Processes NLCD Tree Canopy Cover (TCC) rasters for each
                       year. TCC is used as a forest mask in the S1CD step to
                       exclude non-forested pixels. Output: resampled/cropped
                       TCC rasters per year.

  3. S1CDProcessor  -- Spatially and temporally matches Sentinel-1 change
                       detection tiles to the filtered IDS polygons (using
                       configurable spatial and temporal buffers), applies the
                       TCC mask, and computes overlap metrics.
                       Output: matched S1 disturbance shapefiles.

  4. Plotter        -- Generates all analysis figures from the processed outputs
                       (study area map, year-lag histograms, size/position change,
                       overlap percentages, manual validation comparisons, etc.).

Usage
-----
Run via SLURM:   sbatch run_main.sh
Run directly:    python main.py --run-plotter

Key parameters you may want to adjust (passed as CLI arguments or via .env):
  --start-year / --end-year   : disturbance year range to analyse (default 2016–2020)
  --spatial-buffer            : buffer around IDS polygons in metres (default 500)
  --buffer-years              : temporal tolerance for S1/IDS matching in years (default 2)
  --max-jobs                  : parallel workers for S1CD processing (default 8)
  --excluded-dca-types        : disturbance types to exclude from the analysis

Configuration
-------------
All file paths and settings are loaded from environment/.env (copy from
environment/.env.example and set FOREXD_DIR to your local clone path).
The log file location is set via LOG_FILE in the .env (default: process.log).
"""

from pathlib import Path
import argparse
import logging
import os
from dotenv import load_dotenv

from ids_processor import IDSProcessor
from tcc_processor import TCCProcessor
from s1cd_processor import S1CDProcessor
from plotter import Plotter

DEFAULT_ENV = Path(__file__).resolve().parent.parent / "environment" / ".env"


def main(env_path, start_year, end_year, excluded_dca_types, buffer_years, spatial_buffer, max_jobs,
         run_ids, run_tcc, run_s1cd, run_plotter):

    load_dotenv(dotenv_path=env_path)
    region = int(os.getenv("REGION", 8))

    logging.info(f"Starting pipeline for region {region}...")

    if run_ids:
        try:
            processor = IDSProcessor(region_id=region, env_path=env_path)
            processor.load()
            processor.resolve_polygon_overlaps()
            processor.filter_by_study_criteria(start_year=start_year, end_year=end_year, excluded_dca_types=excluded_dca_types)
            processor.log_summary()
            processor.save_and_plot()
            logging.info("IDS processing completed.")
        except Exception as e:
            logging.error(f"Error in IDS processing: {e}")

    if run_tcc:
        try:
            logging.info("Running TCCProcessor...")
            # TCC data is needed for the year prior to each disturbance year
            # so that the canopy cover mask reflects pre-disturbance forest state.
            # Year range: [start_year - 1, end_year - 1].
            tcc_years = range(start_year - 1, end_year)
            for year in tcc_years:
                processor = TCCProcessor(env_path, year)
                processor.process()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ForExD-WP1-P1 processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", default=str(DEFAULT_ENV),
                        help="Path to the .env configuration file (default: environment/.env)")
    parser.add_argument("--start-year", type=int, default=2016,
                        help="Start year for filtering disturbances (default: 2016)")
    parser.add_argument("--end-year", type=int, default=2020,
                        help="End year for filtering disturbances (default: 2020)")
    parser.add_argument("--excluded-dca-types", nargs='+',
                        default=['other', 'multi_damage', 'other_abiotic', 'other_biotic'],
                        help="Disturbance types to exclude from the analysis")
    parser.add_argument("--buffer-years", type=int, default=2,
                        help="Temporal tolerance (±years) for matching S1 detections to IDS labels (default: 2)")
    parser.add_argument("--spatial-buffer", nargs='+', type=int, default=[500],
                        help="Spatial buffer around IDS polygons in metres (default: 500)")
    parser.add_argument("--max-jobs", type=int, default=8,
                        help="Max parallel workers for S1CD processing (default: 8)")
    # Pipeline stage flags — run in order; each stage depends on the previous.
    parser.add_argument("--run-ids",     action="store_true", help="Step 1: filter and reproject USDA IDS polygons")
    parser.add_argument("--run-tcc",     action="store_true", help="Step 2: prepare tree canopy cover mask")
    parser.add_argument("--run-s1cd",    action="store_true", help="Step 3: match Sentinel-1 change detection to IDS polygons")
    parser.add_argument("--run-plotter", action="store_true", help="Step 4: generate analysis figures")

    args = parser.parse_args()

    # Load .env first so LOG_FILE is available before logging is configured
    load_dotenv(args.env)
    log_file = os.getenv("LOG_FILE", "process_det.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
    )
