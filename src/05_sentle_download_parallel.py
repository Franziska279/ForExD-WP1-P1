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
        logging.FileHandler('sentinel_downloading_0.log')  # Logs to a file
    ]
)
logger = logging.getLogger()

def load_sentle(grid_path, idx, res):
    """
    Load Sentinel data using the Sentle library for a given grid.
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
            S2_cloud_classification_device="cuda",
            S1_assets=["vv", "vh"],
            S2_apply_snow_mask=True,
            S2_apply_cloud_mask=True,
            #time_composite_freq="7d",
            # NOTE clemens: this can be set to 40
            num_workers=40,
        )
        return da
    except Exception as e:
        logger.error(f"Error in load_sentle function for index {idx}: {e}")
        raise

def process_and_save(grid_path, idx, res):
    """
    Process and save Sentinel data for a given index.
    """
    try:
        logger.info(f"> Load the Minicube {idx} ...")
        da = load_sentle(grid_path=grid_path, idx=idx, res=res)
        logger.info(f"> Save the Minicube {idx} ...")
        output_zarr_path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_{res}_512_20152024_equi7_NA.zarr"
        sentle.save_as_zarr(da, path=output_zarr_path)
        logger.info(f"> Successfully saved the Minicube {idx} at {output_zarr_path} ...")
    except Exception as e:
        logger.error(f"An error occurred for index {idx}: {e}")

def main(idx):
    start_time = time.time()  # Capture the start time

    try:
        # Load the Environment variables
        env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_10_512_20152024_equi7_NA.zarr"


        # Check if the folder exists, then delete it along with all its subfolders
        if os.path.exists(path):
            print(f"Deleting folder and subfolders at: {path}")
            shutil.rmtree(path)
        else:
            print(f"Folder does not exist: {path}")

        # NOTE clemens
        # if below does not work, try to set the CUDA_VISIBLE_DEVICES through the terminal before running the script
        # with `export CUDA_VISIBLE_DEVICES=2`
        
        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        logger.info(f"> Available CUDA devices: {torch.cuda.device_count()}")

        # Ensure the 'REGION' environment variable is set
        region = os.getenv('REGION')
        if region is None:
            raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

        print(f"Working on USDA Region {region} ...")
        region_id=str(region).zfill(2)

        # Set resolution of the grid that  i want to load
        resolution = 10
        pixel_size = 512
        # Path to grid_file
        grid_path = f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersetion.shp"
        #grid_path = f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{resolution}_512.shp"
        #intersected_gdf_equi7 = gpd.read_file(grid_path)
        #start_idx = 61
        #end_idx = len(intersected_gdf_equi7) - 1  # This is up to 3997  so i just tried with a shorter range for testing

       
        # # NOTE clemens
        # # do a simple for-loop here, the sentle process function is already parallelized
        #for idx in range(start_idx, end_idx + 1):
        process_and_save(grid_path, idx, resolution)


    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <number>")
        sys.exit(1)

    number = int(sys.argv[1])
    main(number)
   
