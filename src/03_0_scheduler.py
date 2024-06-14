                
import os
import sys
import subprocess
import concurrent.futures
from tqdm import tqdm
import logging

# Constants
MAX_JOBS = 20
INPUT_PATHS = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/s1_change_detection_northamerica/"
IDS_USDA_PATH = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"
OUTPUT_FILENAME = "radar_enhanced_forest_disturbance_mapping.shp"

# Logging setup
logging.basicConfig(filename='scheduler.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def run_extraction_script(input_file):
    input_path = os.path.join(INPUT_PATHS, input_file)
    try:
        # Command to execute 03_1_s1_polygon_extraction_relabeling.py
        cmd = ['python', '03_1_s1_polygon_extraction_relabeling.py', input_path, IDS_USDA_PATH]
        
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
    # # Get list of input files
    # input_files = [f for f in os.listdir(INPUT_PATHS) if not f.endswith('.py')]
    # total_files = len(input_files)
    
    # # Progress bar setup
    # with tqdm(total=total_files) as pbar:
    #     success_count = 0
    #     error_count = 0
        
    #     # Use ThreadPoolExecutor to parallelize execution
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
    #         # Submit jobs
    #         futures = []
    #         for input_file in input_files:
    #             futures.append(executor.submit(run_extraction_script, input_file))
                
    #         # Process results as they complete
    #         for future in concurrent.futures.as_completed(futures):
    #             if future.result():
    #                 success_count += 1
    #             else:
    #                 error_count += 1
                
    #             # Update progress bar
    #             pbar.update(1)
    #             pbar.set_description(f"Success: {success_count}, Errors: {error_count}")
    
    # After all jobs are finished, call the merge script
    print("All jobs completed. Calling merge script...")
    try:
        subprocess.run(['python', '03_2_merge_s1cd_files.py'], check=True)
        print("Merge script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing merge script: {e}")

if __name__ == "__main__":
    main()
