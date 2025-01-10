import geopandas as gpd
from sentle import sentle
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
import concurrent.futures
import logging
import time
import sys
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler('log_s2_downloading.log')  # Logs to a file
    ]
)
logger = logging.getLogger()

def load_sentle(grid_path, idx, res):
    """
    Load Sentinel data using the Sentle library for a given grid.

    Parameters:
        grid_path (str): Path to the grid shapefile.
        idx (int): Index of the grid cell to process.
        res (int): Resolution of the grid.

    Returns:
        da: Processed Sentinel dataset.
    """
    try:
        intersected_gdf_equi7 = gpd.read_file(grid_path)
        id_n = idx + 1
        bounds = intersected_gdf_equi7[idx:id_n].geometry.iloc[0].bounds
        bound_left = int(bounds[0])
        bound_bottom = int(bounds[1])
        bound_right = int(bounds[2])
        bound_top = int(bounds[3])
        equi7_crs = intersected_gdf_equi7.crs
        logger.info(f"Resolution: {res}")

        # Process the Sentinel data using Sentle
        da = sentle.process(
            target_crs=equi7_crs,
            bound_left=bound_left,
            bound_bottom=bound_bottom,
            bound_right=bound_right,
            bound_top=bound_top,
            datetime="2016-01-01/2024-09-30",
            target_resolution=res,
            dask_scheduler_port=10022,
            dask_dashboard_address='127.0.0.1:37386',
            S2_mask_snow=True,
            S2_cloud_classification=True,
            S2_cloud_classification_device="cuda",  # Runs on CUDA (GPU)
            S1_assets=["vv", "vh"],
            S2_apply_snow_mask=True,
            S2_apply_cloud_mask=True,
            num_workers=40,  # Parallel workers
        )
        return da
    except Exception as e:
        logger.error(f"Error in load_sentle function for index {idx}: {e}")
        raise

def process_and_save(grid_path, idx, res):
    """
    Process and save Sentinel data for a given index.

    Parameters:
        grid_path (str): Path to the grid shapefile.
        idx (int): Index of the grid cell to process.
        res (int): Resolution of the grid.
    """
    try:
        logger.info(f"> Start processing Minicube {idx}...")
        da = load_sentle(grid_path=grid_path, idx=idx, res=res)
        logger.info(f"> Saving Minicube {idx}...")
        output_zarr_path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_{res}_512_20152024_equi7_NA.zarr"
        sentle.save_as_zarr(da, path=output_zarr_path)
        logger.info(f"> Successfully saved Minicube {idx} at {output_zarr_path}")
    except Exception as e:
        logger.error(f"An error occurred for index {idx}: {e}")

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        int: Parsed index number from the command-line arguments.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <index>")
        sys.exit(1)
    return int(sys.argv[1])

def main(env_path, idx):
    """
    Main function to process a specific grid index.

    Parameters:
        idx (int): Index of the grid cell to process.
    """
    start_time = time.time()  # Capture the start time

    try:
        # Load the Environment variables
        # env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        # Path to Sentinel2 Minicubes folder
        path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_10_512_20152024_equi7_NA.zarr"

        # Check if the folder exists, then delete it along with all its subfolders
        if os.path.exists(path):
            logger.info(f"Deleting folder and subfolders at: {path}")
            shutil.rmtree(path)
        else:
            logger.info(f"Folder does not exist: {path}")

        # NOTE: Adjust CUDA_VISIBLE_DEVICES based on your cluster or machine setup
        # For example, set the GPU ID(s) you want to use: e.g., "0,1" for multi-GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")

        # Ensure the 'REGION' environment variable is set
        region = os.getenv('REGION')
        if region is None:
            raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

        logger.info(f"Working on USDA Region {region}...")
        region_id = str(region).zfill(2)

        # Set resolution of the grid to load
        resolution = 10
        pixel_size = 512

        # Path to grid shapefile
        grid_path = f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersetion.shp"

        # Process and save the data for the specified index
        process_and_save(grid_path, idx, resolution)

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time for index {idx}: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Sentinel-2 Data Downloader")
    parser.add_argument("--env", required=True, help="Path to the .env file")
    parser.add_argument("index", type=int, help="Index for the grid to process (from SLURM_ARRAY_TASK_ID)")
    
    # Parse arguments
    args = parser.parse_args()
     
    main(env_path=Path(args.env), metadata_output=args.index)
