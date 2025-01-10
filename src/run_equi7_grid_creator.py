import argparse
from pathlib import Path
from equi7_grid_creator import Equi7GridCreator  # Assuming the class is in a file named equi7_grid_creator.py
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

def main():

    # Load environment variables from the .env file
    env_path = Path('/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    # Retrieve environment variables
    s2_minicubes_folder = os.getenv('EQUI7_GRIDS')
    print(f"Equi7 grids folder: {s2_minicubes_folder}")

    # Retrieve the CRS (Coordinate Reference System) for Equi7 NA
    equi7_crs = os.getenv('EQUI7_NA_EPSG')

    # Ensure the 'REGION' environment variable is set
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

    # Format region ID as a two-digit string
    region_id = str(region).zfill(2)

    # Parameters for the grid
    resolution = 10
    pixel_size = 512

    # Define file paths for shapefiles and output locations
    usa_filepath = f"{os.getenv('REGION_SHAPE')}/S_USA.AdministrativeRegion.shp"

   
    # Create output paths dynamically based on region ID
    output_paths = {
        "grid_output_path": f"{s2_minicubes_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}.shp",
        "grid_figure_output_path": f"{os.getenv('FIGURES')}/p1_f4_grid_equi7_{resolution}_{pixel_size}_region_{region_id}.png",
        "convex_hulls_output_path": f"{os.getenv('RESULTS')}/radar_results/convex_hulls_refdm_region_{region_id}_epsg_4326.shp",
        "intersection_output_path": f"{s2_minicubes_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersetion.shp",
        "intersection_figure_output_path": f"{os.getenv('FIGURES')}/p1_f4_grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersection.png",
        "refdm_path": f"{os.getenv('RESULTS')}/radar_results/radar_enhanced_forest_disturbance_mapping_region_{region_id}.shp",
        "ids_path": f"{os.getenv('RESULTS')}/region{region_id}_dca_filtered_ids_usda_polygons_espg_27705.shp",
        "output_path_refdm": f"{os.getenv('RESULTS')}/radar_results/radar_enhanced_forest_disturbance_mapping_region_{region_id}_epsg_27705.shp",
        "batches_figure_output_path": f"{os.getenv('FIGURES')}/p1_f5_grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersecting_cells_batches.png"
    }

    # Instantiate and run the grid creator
    grid_creator = Equi7GridCreator(
        usa_filepath=usa_filepath,
        resolution=resolution,
        pixel_size=pixel_size,
        env_path=env_path,
        output_paths=output_paths
    )

    # Run the grid creation process
    grid_creator.create_grid()


if __name__ == "__main__":
    main()
