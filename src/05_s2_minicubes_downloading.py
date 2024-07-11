import os
import sys
import calendar
import logging
import concurrent.futures
import geopandas as gpd
import xarray as xr
from retry import retry
import earthnet_minicuber as emc
from mypythonlib import myfunctions, phenolopy
import os
import xarray as xr
from tqdm import tqdm
import xarray as xr
import numpy as np
import os
import sys
import logging  
import geopandas as gpd
import concurrent.futures
import calendar
from retry import retry

# # Hide warnings (many since some xarray class uses some deprecated python function on 3.9)
# import warnings
# warnings.filterwarnings('ignore')

# # Define retry decorator with maximum attempts and wait between retries
# @retry(Exception, tries=5, delay=5, backoff=2)
# def download_minicube(lon, lat, year, month, output_folder, idx):
#     filename = os.path.join(output_folder, f"{idx}_{year}_{month:02d}.nc")
#     if os.path.exists(filename):
#         print(f"Minicube for {year}-{month:02d} already exists. Skipping...")
#         return

#     print(f"Loading Minicube for {year}-{month:02d}")
#     last_day = calendar.monthrange(year, month)[1]
    
#     specs = {
#         "lon_lat": (lon, lat),
#         "xy_shape": (1024, 1024),
#         "resolution": 20,
#         "time_interval": f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
#         "providers": [
#             {
#                 "name": "s2",
#                 "kwargs": {
#                     "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"], 
#                     "best_orbit_filter": True, "five_daily_filter": True, "brdf_correction": True, 
#                     "cloud_mask": True, "cloud_mask_rescale_factor": 2, "aws_bucket": "planetary_computer"
#                 }
#             }
#         ]
#     }

#     mc = emc.load_minicube(specs, compute=True)
    
#     # Save minicube to a NetCDF file
#     print(f"Saving minicube for {year}-{month:02d} to {filename}")
#     comp = dict(zlib=True, complevel=9)
#     encoding = {var: comp for var in mc.data_vars}
#     mc.to_netcdf(filename, encoding=encoding)

# def generate_output_folder(idx, dir):
#     output_folder = os.path.join(dir, str(idx))
#     os.makedirs(output_folder, exist_ok=True)
#     return output_folder

# def minicuber_download(idx, intersecting_grid_gdf_events_unique, output_folder):
#     first_polygon = intersecting_grid_gdf_events_unique.geometry.iloc[idx]
#     lon = first_polygon.centroid.x
#     lat = first_polygon.centroid.y
#     start_year = 2015
#     last_year = 2023

#     # Number of concurrent workers should match the cpus-per-task value in the SLURM script
#     with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:  # Adjust max_workers as needed
#         futures = []
#         for year in range(start_year, last_year + 1):
#             for month in range(1, 13):
#                 futures.append(
#                     executor.submit(download_minicube, lon, lat, year, month, output_folder, idx)
#                 )

#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 logging.debug(f"Failed to download a minicube. Error: {e}")

#     print("End download")

# def merge_minicubes(idx, output_folder):
#     expected_years = range(2015, 2024)
#     expected_files = [f"{idx}_{year}_{month:02d}.nc" for year in expected_years for month in range(1, 13)]

#     nc_files = [file for file in os.listdir(output_folder) if file.endswith('.nc')]

#     missing_files = [file for file in expected_files if file not in nc_files]
#     if missing_files:
#         print(f"Missing files: {missing_files}")
#         print("Breaking off the merging.")
#         return
    
#     merged_filename = os.path.join(output_folder, f"{idx}_merged.nc")
#     print(f"Merging all NetCDF files into {merged_filename}")

#     files_to_merge = [os.path.join(output_folder, file) for file in nc_files]
#     merged_ds = xr.open_mfdataset(files_to_merge, combine='nested', concat_dim='time')
#     merged_ds.to_netcdf(merged_filename)
#     print("End merging")
#     print("\nStart preprocessing:")
#     preprocess_and_reduce_minicube(merged_ds, idx, output_folder)


def preprocess_and_reduce_minicube(ds, idx, output_folder):

    ds = ds.rename_dims({'lon': 'x', 'lat': 'y'})
    ds.rio.write_crs("epsg:4326", inplace=True)  # Set the coordinate reference system to EPSG:4326

    print(f"Remove clouds, shadows, and snow")
    # Calculate the fraction of 1 values in s2_mask for each time step
    mask_fraction = ds.s2_mask.where(ds.s2_mask == 0, 1).sum(dim=['x', 'y']) / (len(ds.x) * len(ds.y))

    # Filter the data to include only bands with names starting with 's2_B'
    filtered_data = ds[[b for b in ds.variables if 's2_B' in b]]

            # Mask out pixels where s2_mask is not equal to 0
    filtered_data = filtered_data.where(ds.s2_mask == 0)

    # Remove cirrus clouds from the entire dataset
    filtered_data = phenolopy.remove_cirrus_clouds(filtered_data)

    print(f"Remove outliers")
    # Remove outliers using a specified method
    remove_outliers = phenolopy.remove_outliers(filtered_data, method='median', user_factor=2, z_pval=0.07)

    print(f"Compute various vegetation indices")
    # Compute various vegetation indices and add them to the dataset
    print('NDVI')
    remove_outliers['ndvi'] = myfunctions.ndvi(remove_outliers)
    print('NBR')
    remove_outliers['nbr'] = myfunctions.nbr(remove_outliers)
    print('NDWI')
    remove_outliers['ndwi'] = myfunctions.ndwi(remove_outliers)
    print('NDRE')
    remove_outliers['ndre'] = myfunctions.ndre(remove_outliers)
    print('TCW')
    remove_outliers['tcw'] = myfunctions.tcw(remove_outliers)
    print('TCG')
    remove_outliers['tcg'] = myfunctions.tcg(remove_outliers)
    print('TCB')
    remove_outliers['tcb'] = myfunctions.tcb(remove_outliers)
    print('NDMI')
    remove_outliers['ndmi'] = myfunctions.ndmi(remove_outliers)
    print('NIRV')
    remove_outliers['nirv'] = myfunctions.nirv(remove_outliers)
    print('kNDVI')
    remove_outliers['kndvi'] = myfunctions.kndvi(remove_outliers, sigma=0.02)
    print('DRS/NDRS')
    remove_outliers['drs'] = myfunctions.drs(remove_outliers)
    remove_outliers['ndrs'] = myfunctions.ndrs(remove_outliers)
    print('kDRS/kNDRS')
    remove_outliers['kdrs'] = myfunctions.kdrs(remove_outliers, sigma=0.02)
    remove_outliers['kndrs'] = myfunctions.kndrs(remove_outliers)

    print(f"Sort temporal axis")
    remove_outliers = remove_outliers.sortby('time')

    print(f"Interpolate linearly over time")
    # Interpolate missing values linearly over time
    interpolated_data = remove_outliers.interpolate_na(dim='time', method='linear')

    print(f"Smooth data with Savitzky-Golay (window_length=15, polyorder=3)")
    # Smooth the data using the Savitzky-Golay filter
    smoothed_data = phenolopy.smooth(ds=interpolated_data, method='savitsky', window_length=15, polyorder=3)

    print(f"Calculate Anomaly for each variable")
    for var in smoothed_data.data_vars:
            print(var)
            smoothed_data[var+'_anom']     = smoothed_data[var].groupby('time.week')-smoothed_data[var].groupby('time.week').median()

    # Save the preprocessed data to a NetCDF file
    print(f"Compress data for saving to complevel=9")
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in smoothed_data.data_vars}
    print(encoding)
    outputpath = os.path.join(output_folder, f"{idx}_merged_compressed.nc")
    smoothed_data.to_netcdf(outputpath, encoding=encoding)

    print(f"Saved to path = {outputpath}")



# Configure logging
logging.basicConfig(filename='minicube_downloader.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def generate_output_folder(gdf, idx, dir):
    disturbance_type = gdf.dist_type.iloc[idx]
    year = gdf.year.iloc[idx]
    output_folder = f"/{dir}/{disturbance_type}/{year}/{idx}/"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Define retry decorator with maximum attempts and wait between retries
@retry(Exception, tries=5, delay=5, backoff=2)
def download_minicube(lon, lat, year, month, output_folder, idx):
    filename = os.path.join(output_folder, f"{idx}_{year}_{month:02d}.nc")
    if os.path.exists(filename):
        logging.info(f"Minicube for {year}-{month:02d} already exists. Skipping...")
        return

    logging.info(f"Loading Minicube for {year}-{month:02d}")
    last_day = calendar.monthrange(year, month)[1]
    
    specs = {
        "lon_lat": (lon, lat),
        "xy_shape": (256, 256),
        "resolution": 20,
        "time_interval": f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        "providers": [
            {
                "name": "s2",
                "kwargs": {"bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"], 
                           "best_orbit_filter": True, "five_daily_filter": True, "brdf_correction": True, 
                           "cloud_mask": True, "cloud_mask_rescale_factor": 2, "aws_bucket": "planetary_computer"}
            }
        ]
    }

    mc = emc.load_minicube(specs, compute=True)
    
    logging.info(f"Saving minicube for {year}-{month:02d} to {filename}")
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in mc.data_vars}
    mc.to_netcdf(filename, encoding=encoding)

def minicuber_download(idx, intersecting_grid_gdf_events_unique, output_folder):
    first_polygon = intersecting_grid_gdf_events_unique.geometry.iloc[idx]
    lon = first_polygon.centroid.x
    lat = first_polygon.centroid.y
    year = int(intersecting_grid_gdf_events_unique['year'].iloc[idx])    
    start_year =  year - 2
    last_year =  year + 2

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for year in range(start_year, last_year + 1):
            for month in range(1, 13):
                futures.append(
                    executor.submit(download_minicube, lon, lat, year, month, output_folder, idx)
                )

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Failed to download a minicube. Error: {e}")

    logging.info("End download")

def merge_minicubes(idx, output_folder, expected_years_range):
    # Expected years range
    expected_years = expected_years_range
    expected_files = [f"{idx}_{year}_{month:02d}.nc" for year in expected_years for month in range(1, 13)]
    nc_files = [file for file in os.listdir(output_folder) if file.endswith('.nc')]

    # Check if all expected files are present
    missing_files = [file for file in expected_files if file not in nc_files]
    if missing_files:
        print(f"Missing files: {missing_files}")
        print(f"Breaking off the merging ... ")
    else:
        valid_files = []
        empty_files = []

        for file in nc_files:
            file_path = os.path.join(output_folder, file)
            try:
                ds = xr.open_dataset(file_path, chunks={})
                if ds.dims:  # Check if the dataset has any dimensions
                    valid_files.append(file)
                else:
                    empty_files.append(file)
            except Exception as e:
                print(f"Error opening file {file}: {e}")
                empty_files.append(file)
        
        if empty_files:
            print(f"Empty files: {empty_files}")
            print(f"Removing empty files from the merging list.")
            
            # Delete the original NetCDF files
            for file in empty_files:
                file_path = os.path.join(output_folder, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

        if valid_files:
            merged_filename = os.path.join(output_folder, f"{idx}_merged.nc")
            print(f"Merging valid NetCDF files into {merged_filename}")

            # Specify coords='minimal' to handle differing coordinates
            file_paths = [os.path.join(output_folder, file) for file in valid_files]
            
            # Use dask for parallel processing and lazy loading
            datasets = [xr.open_dataset(fp, chunks={}) for fp in tqdm(file_paths, desc="Opening files")]
            
            # Perform lazy concatenation
            merged_ds = xr.concat(datasets, dim='time', coords='minimal')

            # Write the merged dataset to disk
            merged_ds.to_netcdf(merged_filename, compute=True)
            print("End merging")
            
            # Delete the original NetCDF files
            for file in valid_files:
                file_path = os.path.join(output_folder, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

            print("All valid NetCDF files have been deleted. Only the merged file remains.")

        else:
            print("No valid files found for merging.")

        print("\n Start preprocessing:")
        # preprocess_and_reduce_minicube(merged_ds, idx, output_folder)


def main():
    if len(sys.argv) != 4:
        print("Usage: python minicube_download_preprocessing_pipeline.py SLURM_ARRAY_TASK_ID INPUT_FILE OUTPUT_DIR")
        sys.exit(1)

    slurm_array_task_id = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    logging.info(f"Running Task ID {slurm_array_task_id}")

    index = int(slurm_array_task_id)
    gdf_grid = gpd.read_file(input_file)
    year = int(gdf_grid['year'].iloc[index])    
    start_year =  year - 2
    last_year =  year + 2  
    expected_years =   range(start_year, last_year+1)
    output_folder = generate_output_folder(gdf_grid, index, output_dir)
    minicuber_download(index, gdf_grid, output_folder)
    merge_minicubes(index, output_folder, expected_years)

    logging.info('Done')

if __name__ == "__main__":
    main()
