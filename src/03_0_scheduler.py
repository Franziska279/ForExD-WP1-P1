"""
S1CD Preprocessing and Relabeling Scheduler Script
=======================================

Author: Franziska Müller
Date: 28.06.2024

This script schedules and executes the preprocessing and relabeling of USDA disturbance data for Region 8 using radar data. 
The main tasks of the script are:

1. Setting Up Logging: Configures logging to record the progress and any errors encountered during execution.
2. Defining Constants: Specifies important constants such as file paths and maximum jobs for parallel processing.
3. Running Extraction Script: Executes a Python script to process individual files, 
    converting and relabeling data based on radar change detection.
4. Concurrent Processing: Utilizes concurrent processing to handle multiple files in parallel, improving efficiency.
5. Tracking Progress: Uses a progress bar to track and display the number of successfully processed files and errors.
6. Merging Results: After processing all files, calls a merge script to combine the results into a final output.

Results are saved in the specified directory with a confirmation message printed upon completion. 
The script ensures robust error handling and logs detailed information for troubleshooting.
"""

import os
import sys
import subprocess
import concurrent.futures
from tqdm import tqdm
import logging
from pathlib import Path
from dotenv import load_dotenv


# Logging setup
logging.basicConfig(filename='scheduler.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def run_extraction_script(input_file, INPUT_PATHS, IDS_USDA_PATH, TCC_PATH, TARGET_CRS, OUTPUT_DIR):
    input_path = os.path.join(INPUT_PATHS, input_file)
    try:
        # Command to execute 03_1_s1_polygon_extraction_relabeling.py
        cmd = ['python', '03_1_s1_polygon_extraction_relabeling.py', input_path, IDS_USDA_PATH, TCC_PATH, TARGET_CRS, OUTPUT_DIR]
        
        #subprocess.run(cmd, check=True)

        # Execute command, suppressing output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Log success
        logging.info(f"Success: {input_file}")
        return True
    except subprocess.CalledProcessError as e:
        # Log error
        logging.error(f"Error: {input_file} - {e}")
        return False
    except Exception as e:
        # Log other exceptions
        logging.error(f"Error: {input_file} - {str(e)}")
        return False

def main():

    # Load environment variables from a .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        raise FileNotFoundError(f"The .env file does not exist at {env_path}")
    
    # Retrieve the 'REGION' variable from the environment
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")
    
    # Ensure that the region ID is always formatted as a two-digit string
    region_id = str(region).zfill(2)

    # Define file paths
    tcc_dir = os.getenv('TCC_PATH')  # Fetch the TCC_PATH from environment variables
    if tcc_dir is None:
        raise ValueError("TCC_PATH environment variable is not set")
    tcc_dir = tcc_dir.rstrip('/') + '/'

    # Display the region being worked on
    print(f"Working on USDA Region {region_id} ...")

    # Constants
    MAX_JOBS = 20
    INPUT_PATHS = f"{os.getenv('SENTINEL1_TILES')}" 
    #"/Net/Groups/BGI/work_2/ForExD/WP1/Data/s1_change_detection_northamerica/"
    TCC_PATH = tcc_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_08.tif"
    #IDS_USDA_PATH = f"{os.getenv('RESULTS')}/region{region_id}_dca_filtered_ids_usda_polygons.shp" 
    file_output_path_equi7 = f"{os.getenv('RESULTS')}/region{region_id}_dca_filtered_ids_usda_polygons.shp"
    OUTPUT_FILENAME = f"radar_enhanced_forest_disturbance_mapping_region_{region_id}.shp"
    SHAPEFILES_DIR = f"{os.getenv('RESULTS')}/03_s1cd_polygons/"
    OUTPUT_DIR = f"{os.getenv('RESULTS')}/radar_results/"
    TARGET_CRS = "EPSG:4326"
    #TARGET_CRS = os.getenv('EQUI7_NA_EPSG')

    # Create the directory if it doesn't exist
    os.makedirs(SHAPEFILES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Get list of input files
    input_files = [f for f in os.listdir(INPUT_PATHS) if not f.endswith('.py')]
    total_files = len(input_files)
    
    # Progress bar setup
    with tqdm(total=total_files) as pbar:
        success_count = 0
        error_count = 0
        
        # Use ThreadPoolExecutor to parallelize execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
            # Submit jobs
            futures = []
            for input_file in input_files:
                futures.append(executor.submit(run_extraction_script, input_file, INPUT_PATHS, file_output_path_equi7, TCC_PATH, TARGET_CRS, SHAPEFILES_DIR))
                
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    error_count += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Success: {success_count}, Errors: {error_count}")
    
    # After all jobs are finished, call the merge script
    print("All jobs completed. Calling merge script...")
    try:
        subprocess.run(['python', '03_2_merge_s1cd_files.py', SHAPEFILES_DIR, OUTPUT_DIR, OUTPUT_FILENAME, TARGET_CRS], check=True)
        print("Merge script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing merge script: {e}")

if __name__ == "__main__":
    main()
