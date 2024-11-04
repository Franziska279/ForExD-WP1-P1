import os
import xarray as xr
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
import dask

# Set up logging
logging.basicConfig(
    filename='file_loading.log',  # Specify the log file name
    level=logging.INFO,            # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Path to the folder containing the files
folder_path = '/net/projects/forexd/WP1/Data/S2_Cubes_IDS_R8_Preprocessed/'

# Get a list of all .nc files in the folder
nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

# Function to process a single file
def process_file(file_path):
    try:
        # Try loading the NetCDF file using Dask
        ds = xr.open_dataset(file_path, chunks='auto')  # Enable Dask by specifying chunks
        
        # Check all variables in the dataset
        for var in ds.data_vars:
            # Attempt to load the variable data
            _ = ds[var].values  # Access the variable data to trigger any potential errors

        # Log success message
        logging.info(f"Successfully loaded file: {os.path.basename(file_path)}")

    except Exception as e:
        # Catch any general errors and log the filename and error message
        logging.error(f"Error loading file {os.path.basename(file_path)}: {e}")

# Iterate over all files in the folder with a progress bar
for filename in tqdm(nc_files, desc="Processing files", unit="file"):
    file_path = os.path.join(folder_path, filename)
    
    # Call the processing function
    process_file(file_path)

print("Processing complete. Check 'file_loading.log' for details.")
