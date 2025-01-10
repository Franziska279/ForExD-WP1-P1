import os
import subprocess
import time

INPUT_DIR = "/work/sy58xupo-CleaningSpace/Data/S1_TILES_R8/s1_change_detection_northamerica/"
OUTPUT_DIR = "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/"
LOG_DIR = "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/src/logs/"
SBATCH_SCRIPT = "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/src/s1cd_jobs.sh"

MAX_CONCURRENT_JOBS = 20

def submit_jobs():
    """Submits SLURM jobs for all input files, limiting to 20 concurrent submissions."""
    input_files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    
    job_ids = []
    current_batch = []

    for file in input_files:
        input_path = os.path.join(INPUT_DIR, file)
        log_file = os.path.join(LOG_DIR, f"{file}.log")
        
        # Prepare the sbatch command
        sbatch_command = f"sbatch --output={log_file} {SBATCH_SCRIPT} {input_path} {OUTPUT_DIR}"
        
        # Submit the SLURM job
        result = subprocess.run(sbatch_command.split(), stdout=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            current_batch.append(job_id)
            print(f"Submitted job {job_id} for file: {file}")
        else:
            print(f"Failed to submit job for file: {file}")
        
        # If we've reached the maximum number of concurrent jobs, wait for the batch to finish
        if len(current_batch) >= MAX_CONCURRENT_JOBS:
            print(f"Waiting for {MAX_CONCURRENT_JOBS} jobs to complete before submitting more.")
            # Sleep for a few seconds before checking the job status (you may customize this logic)
            time.sleep(10)  # Wait for 10 seconds (or adjust as needed)

            # Here, you could check if jobs in `current_batch` are finished, but for simplicity, we wait a fixed time

            # Clear the batch (in reality, you'd monitor job status to clear this batch)
            current_batch = []

    return job_ids

def monitor_jobs(job_ids):
    """Monitors SLURM job completion."""
    while True:
        pending_jobs = subprocess.run(
            ["squeue", "--jobs", ",".join(job_ids)],
            stdout=subprocess.PIPE, text=True
        )
        if "JOBID" not in pending_jobs.stdout:
            print("All jobs completed.")
            break
        print("Jobs still running... Checking again in 300 seconds.")
        time.sleep(300)


def merge_shapefiles(input_dir):
    """
    Merge all shapefiles in the specified directory into a single GeoDataFrame.
    """
    logging.info(f"Merging shapefiles from {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
    gdf_list = []
    
    for file in tqdm(files, desc="Merging shapefiles"):
        filepath = os.path.join(input_dir, file)
        gdf = gpd.read_file(filepath)
        
        # Ensure CRS is set
        if gdf.crs is None:
            raise ValueError(f"CRS not defined for file: {filepath}. Please define CRS for all shapefiles.")
        
        gdf_list.append(gdf)
    
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    logging.info("Shapefiles merged successfully.")
    return merged_gdf   

def merge_metadata(input_dir):
    """
    Merge all shapefiles in the specified directory into a single GeoDataFrame.
    """
    logging.info(f"Merging shapefiles from {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    gdf_list = []

    for file in tqdm(files, desc="Merging shapefiles"):
        filepath = os.path.join(input_dir, file)
        gdf = gpd.read_file(filepath)
        
        # Ensure CRS is set
        if gdf.crs is None:
            raise ValueError(f"CRS not defined for file: {filepath}. Please define CRS for all shapefiles.")
        
        gdf_list.append(gdf)

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    logging.info("Shapefiles merged successfully.")
    return merged_gdf 

def calculate_and_filter_area(gdf):
    """
    Calculate area in square kilometers and filter out polygons larger than 15 km².
    """
    logging.info("Calculating area and filtering polygons...")
    gdf = gdf.to_crs("EPSG:4326")
    
    # Reproject to a CRS with meters (e.g., EPSG:3857) for accurate area calculation
    projected_gdf = gdf.to_crs('EPSG:27705')
    
    # Calculate the area in square meters and convert to km²
    projected_gdf['area_km2'] = projected_gdf.geometry.area / 1e6
    
    # Add the calculated area to the original GeoDataFrame and filter
    gdf['area_km2'] = projected_gdf['area_km2']
    filtered_gdf = gdf[gdf['area_km2'] <= 15]
    
    logging.info("Area calculated and polygons filtered.")
    return filtered_gdf

def save_result(gdf, output_dir, output_filename):

    """
    Save the resulting GeoDataFrame to a new shapefile.
    """
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Saving result to {output_path}...")
    gdf.to_file(output_path)
    logging.info("Result saved successfully.")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Example usage
    shapefile_dir= "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/03_s1cd_polygons/"
    metada_dir = "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/metadata/"
    metada_file = "/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/metadata.shp"

    print("Submitting SLURM jobs...")
    job_ids = submit_jobs()

    print("Monitoring SLURM jobs...")
    monitor_jobs(job_ids)

   # Step to merge after all files have been processed
    logging.info("All individual files processed. Starting merge of shapefiles...")
    merged_gdf = merge_shapefiles(shapefile_dir)
        
    # Calculate area and filter the merged GeoDataFrame
    filtered_gdf = calculate_and_filter_area(merged_gdf)
    
    # Save the final merged and filtered result
    save_result(filtered_gdf, self.output_dir, output_filename='s1dm.shp')

    # Step to merge after all files have been processed
    logging.info("All individual files processed. Starting merge of metadata...")
    merge_metadata_from_csv(metada_dir, metada_file)

    

if __name__ == "__main__":
    main()
