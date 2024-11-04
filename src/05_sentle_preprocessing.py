import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(1, '../Tools/')

from func_preprocessing import preprocess_sentinel_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler('sentle_restructure_2.log')  # Logs to a file
    ]
)
logger = logging.getLogger()


def main(idx):

    start_time = time.time()  # Capture the start time

    try:
        # Load the Environment variables
        env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_10_512_20152024_equi7_NA.zarr"
        nc_path = f"{os.getenv('SENTINEL2_CUBES_PP')}/{idx}_10_512_20152024_equi7_NA.nc"
        logger.info(f"> Load idx: {idx}")

        # Call the preprocess function
        preprocess_sentinel_data(idx, logger, path, nc_path)

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
   
