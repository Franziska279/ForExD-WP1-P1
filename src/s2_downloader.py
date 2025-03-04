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
import argparse
import zarr

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
            datetime="2016-01-01/2024-12-31",
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

def process_and_save(grid_path, idx, res, px):
    """Process and save Sentinel data as a compressed Zarr ZIP."""
    try:
        logger.info(f"Starting processing Minicube {idx}...")

        # Load Sentinel data
        da = load_sentle(grid_path=grid_path, idx=idx, res=res)

        # Define paths
        output_dir = os.getenv("SENTINEL2_MINICUBES_DIR")
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        zip_path = f"{output_dir}/{idx}_{res}_{px}_2016_2024_equi7_NA.zarr.zip"

        # Save as Zarr ZIP
        zip_store = zarr.ZipStore(zip_path, mode="w")
        da.to_zarr(store=zip_store, mode="w", compute=True)
        zip_store.close()

        logger.info(f"✅ Successfully saved Minicube {idx} as ZIP at {zip_path}")

    except Exception as e:
        logger.error(f"❌ Error processing index {idx}: {e}")



def main(env_path, buffer, idx):
    """Main function to process a specific grid index."""
    start_time = time.time()
    try:
        load_dotenv(dotenv_path=env_path)

        # ✅ Select available GPU dynamically
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())
        else:
            logger.warning("CUDA not available, running on CPU.")

        region = os.getenv("REGION")
        if not region:
            raise ValueError("Missing 'REGION' in .env file!")
        region_id = str(region).zfill(2)

        logger.info(f"Processing region {region}, grid index {idx}...")

        resolution = 10
        pixel_size = 512
        grid_path = os.path.join(os.getenv('EQUI7_GRIDS_DIR'), os.getenv('INTERSECTED_GRIDS').format(resolution=resolution, pixel_size=pixel_size, region_id=region_id, buffer=buffer))
        
        # Process and save data
        process_and_save(grid_path, idx, resolution, pixel_size)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Finished processing index {idx} in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sentinel-2 Data Downloader")
    parser.add_argument("--env", required=True, help="Path to the .env file")
    parser.add_argument("--buffer", required=True, type=int, help="Spatial buffer size")
    parser.add_argument("--index", required=True, type=int, help="Grid index from SLURM array")

    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    main(env_path=Path(args.env), buffer=args.buffer, idx=args.index)
