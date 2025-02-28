import geopandas as gpd
from sentle import sentle
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
import logging
import shutil
import time
import concurrent.futures


class S2Downloader:

    def __init__(self, env_path):
        """
        Initializes the Sentinel-2 Downloader class.

        Parameters:
            region_id (int or str): Region ID for the dataset.
            env_path (str or Path): Path to the .env file.
        """
        self._set_up_logging()
        self._load_env_variables(env_path)
        self.resolution = 10
        self.pixel_size = 512

    def _set_up_logging(self):
        """Set up logging to a file with timestamps for tracking the process."""
        logging.basicConfig(
            filename='log_plotter.log',
            level=logging.INFO, 
            format='%(asctime)s - %(message)s'
        )

    def _load_env_variables(self, env_path):
        """Load required environment variables from a .env file."""

        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        # Load environment variables and validate
        self.region = os.getenv('REGION')
        if not self.region:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)

        # Set target CRS (Coordinate Reference System)
        self.equi7_crs = os.getenv('EQUI7_NA_EPSG')


        # Define file paths for shapefiles and output locations
        self.usa_filepath = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.ids_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        self.tcc_downsampled = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(region_id=self.region_id))
        self.s1_tiles_boundary_path =  os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id))

    def _load_env(self):
        """Loads environment variables from the specified .env file."""
        load_dotenv(dotenv_path=self.env_path)

        # Validate the necessary environment variables
        self.s2_minicube_path = os.getenv("SENTINEL2_MINICUBES")
        self.equi7_grid_path = os.getenv("EQUI7_GRIDS")

        if not self.s2_minicube_path or not self.equi7_grid_path:
            raise ValueError("Missing required environment variables in .env file.")

        # Construct the grid shapefile path
        self.grid_path = f"{self.equi7_grid_path}/grid_equi7_{self.resolution}_{self.pixel_size}_region_{self.region_id}_intersetion.shp"

        logger.info(f"Using grid shapefile: {self.grid_path}")

    def load_sentle(self, idx):
        """
        Loads Sentinel data using Sentle for a given grid index.

        Parameters:
            idx (int): Index of the grid cell.

        Returns:
            da: Processed Sentinel dataset.
        """
        try:
            intersected_gdf_equi7 = gpd.read_file(self.grid_path)
            bounds = intersected_gdf_equi7.iloc[idx].geometry.bounds
            equi7_crs = intersected_gdf_equi7.crs

            logger.info(f"Processing grid {idx} with bounds {bounds} at resolution {self.resolution}")

            da = sentle.process(
                target_crs=equi7_crs,
                bound_left=int(bounds[0]),
                bound_bottom=int(bounds[1]),
                bound_right=int(bounds[2]),
                bound_top=int(bounds[3]),
                datetime="2016-01-01/2024-09-30",
                target_resolution=self.resolution,
                dask_scheduler_port=10022,
                dask_dashboard_address="127.0.0.1:37386",
                S2_mask_snow=True,
                S2_cloud_classification=True,
                S2_cloud_classification_device="cuda",
                S1_assets=["vv", "vh"],
                S2_apply_snow_mask=True,
                S2_apply_cloud_mask=True,
                num_workers=40,
            )

            return da
        except Exception as e:
            logger.error(f"Error in loading Sentinel data for index {idx}: {e}")
            return None

    def process_and_save(self, idx):
        """
        Processes and saves Sentinel data for a given index.

        Parameters:
            idx (int): Index of the grid cell to process.
        """
        try:
            logger.info(f"> Start processing Minicube {idx}...")

            da = self.load_sentle(idx)
            if da is None:
                return

            output_zarr_path = f"{self.s2_minicube_path}/{idx}_{self.resolution}_512_20152024_equi7_NA.zarr"

            # Delete the folder if it exists
            if os.path.exists(output_zarr_path):
                logger.info(f"Deleting existing Minicube folder: {output_zarr_path}")
                shutil.rmtree(output_zarr_path)

            # Save the dataset
            sentle.save_as_zarr(da, path=output_zarr_path)
            logger.info(f"> Successfully saved Minicube {idx} at {output_zarr_path}")

        except Exception as e:
            logger.error(f"An error occurred while processing index {idx}: {e}")

    def start_download(self, indices):
        """
        Starts Sentinel-2 downloads for multiple grid indices concurrently.

        Parameters:
            indices (list of int): List of grid indices to process.
        """
        start_time = time.time()
        logger.info(f"Starting Sentinel-2 downloads for {len(indices)} grid(s)...")

        # Use concurrent execution to process multiple grid cells
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.process_and_save, indices)

        elapsed_time = time.time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")


# Example usage
if run_download:
    try:
        downloader = S2Downloader(region_id=region, env_path=env_path)
        indices_to_process = [0, 1, 2, 3, 4]  # Example: Replace with actual indices
        downloader.start_download(indices_to_process)
        logging.info("Sentinel-2 downloading and processing completed successfully.")
    except Exception as e:
        logging.error(f"Error in Sentinel-2 downloading and processing: {e}")
