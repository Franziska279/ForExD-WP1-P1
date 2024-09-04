import geopandas as gpd
from sentle import sentle
import os
from dotenv import load_dotenv
from pathlib import Path
import torch
import concurrent.futures
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler('sentinel_processing_2.log')  # Logs to a file
    ]
)
logger = logging.getLogger()

def load_sentle(grid_path, idx, res):
    """
    Load Sentinel data using the Sentle library for a given grid.
    """
    try:
        intersected_gdf_equi7 = gpd.read_file(grid_path)
        bounds = idx + 1
        bounds = intersected_gdf_equi7[idx:bounds].geometry.iloc[0].bounds
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
            datetime="2015-01-01/2024-07-31",
            target_resolution=res,
            S2_mask_snow=True,
            S2_cloud_classification=True,
            S2_cloud_classification_device="cuda",
            S1_assets=["vv", "vh"],
            S2_apply_snow_mask=True,
            S2_apply_cloud_mask=True,
            time_composite_freq="7d",
            num_workers=5,
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

def main():
    start_time = time.time()  # Capture the start time

    try:
        # Load the Environment variables
        env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        logger.info(f"> Available CUDA devices: {torch.cuda.device_count()}")

        # Set resolution of the grid that  i want to load
        res = 10
        # Path to grid_file
        grid_path = f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{res}_512.shp"
        # intersected_gdf_equi7 = gpd.read_file(grid_path)
        start_idx = 0
        end_idx = 10# len(intersected_gdf_equi7) - 1  # This is up to 3997  so i just tried with a shorter range for testing

        # Use ThreadPoolExecutor with a limit of 5 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_and_save, grid_path, idx, res) for idx in range(start_idx, end_idx + 1)]
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                pass  # You can handle results if needed

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
