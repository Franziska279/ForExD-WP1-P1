import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import warnings
from shapely.geometry import mapping
import rioxarray
from shapely import wkt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import sys
import pandas as pd
import logging
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import warnings
from shapely.geometry import mapping
import rioxarray
from shapely import wkt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import sys
import pandas as pd
import logging

import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rioxarray.exceptions import NoDataInBounds

def merge_disturbance_events(datasets):
    """
    Merge multiple datasets of disturbance events and calculate the mean over the same timesteps,
    only including variables that do not have 's2' or 'anom' in their names.

    Args:
        datasets (list): List of datasets for the disturbance events.

    Returns:
        merged_data (xarray.Dataset): Dataset containing the merged and averaged disturbance events.
    """
    filtered_datasets = []

    # Filter out variables that have 's2' or 'anom' in their names
    for dataset in datasets:
        filtered_vars = {var: dataset[var] for var in dataset.data_vars if 's2' not in var and 'anom' not in var}
        filtered_datasets.append(xr.Dataset(filtered_vars))

    # Concatenate the filtered datasets along the 'time' dimension and sort by time
    concatenated_data = xr.concat(filtered_datasets, dim='time').sortby('time')

    # Group by 'time' and calculate the mean
    merged_data = concatenated_data.groupby('time').mean()

    return merged_data


def combine_ids_refdm_medians(ids_median, refdm_median):
    # Define the row labels
    row_labels = [
        'drs', 'kdrs', 'kndrs', 'kndvi', 'nbr', 'ndmi', 'ndre', 
        'ndrs', 'ndvi', 'ndwi', 'nirv', 'tcb', 'tcg', 'tcw'
    ]

    # Define the column labels (1 through 20)
    column_labels = list(range(1, len(ids_median)))

    # Create the DataFrame with zeros (or any initial value you'd like)
    df_ids = pd.DataFrame(0, index=row_labels, columns=column_labels)
    df_refdm = pd.DataFrame(0, index=row_labels, columns=column_labels)

    # Assuming df is already created as shown in the previous example
    # ids_median is your list of xarray datasets

    for i, dataset in enumerate(ids_median):
        # Create a dictionary to store distances for the current dataset
        distances = {}

        # Loop over variables in the dataset
        for var in dataset:
            # Check if the variable name does not contain 'anom' or 's2'
            if 'anom' not in var and 's2' not in var:
                # Calculate the distance using your specific logic
                distance = calculate_distance(dataset, var)
                distances[var] = distance

        # Convert the distances dictionary to a Series and align it with the DataFrame's index
        # Note: Ensure the DataFrame `df` has matching row labels (variable names)
        df_ids[i + 1] = pd.Series(distances)


    for i, dataset in enumerate(refdm_median):
        # Create a dictionary to store distances for the current dataset
        distances = {}

        # Loop over variables in the dataset
        for var in dataset:
            # Check if the variable name does not contain 'anom' or 's2'
            if 'anom' not in var and 's2' not in var:
                # Calculate the distance using your specific logic
                distance = calculate_distance(dataset, var)
                distances[var] = distance

        # Convert the distances dictionary to a Series and align it with the DataFrame's index
        # Note: Ensure the DataFrame `df` has matching row labels (variable names)
        df_refdm[i + 1] = pd.Series(distances)

    # Transpose DataFrames and add 'Source' column
    df_ids_melted = df_ids.T
    df_ids_melted['Source'] = 'ids'
    df_refdm_melted = df_refdm.T
    df_refdm_melted['Source'] = 'refdm'

    # Combine DataFrames
    combined_df = pd.concat([df_ids_melted, df_refdm_melted])

    # Remove rows where the 'Source' column has 'Source'
    # First, identify rows where the index is 'Source'
    rows_to_remove = combined_df.index[combined_df.index == 'Source']
    combined_df_cleaned = combined_df.drop(index=rows_to_remove)


    melted_df = combined_df_cleaned.melt(id_vars=['Source'], var_name='Variable', value_name='Value')
    melted_df_sorted = melted_df.sort_values(by=['Source', 'Variable'], ascending=[True, False])
    melted_df_sorted['Variable'] = melted_df_sorted['Variable'].str.upper()
    
    return melted_df_sorted



def calculate_distance(data, var):
    """
    Calculate the trend and distance for a given variable.
    
    Parameters:
    - data: xarray.Dataset or pandas.DataFrame containing the time series data.
    - var: str, the variable name to analyze.
    
    Returns:
    - distance: The distance between the lowest and highest points in the time series.
    """
    # Convert xarray.Dataset to pandas.DataFrame if necessary
    if isinstance(data, xr.Dataset):
        # Check if the variable exists in the dataset
        if var not in data.data_vars:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
        # Extract the DataArray corresponding to the variable and convert it to a DataFrame
        df = data[var].to_dataframe().reset_index()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        # Ensure the variable exists in the DataFrame
        if var not in df.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
    else:
        raise ValueError("Unsupported data type. Provide xarray.Dataset or pandas.DataFrame.")
    
    # Drop rows with NaN values in the specified variable column
    df = df.dropna(subset=[var])
    
    # Ensure there are enough data points after dropping NaNs
    if len(df) > 1:
        # Calculate the distance between the lowest and highest points
        min_value = df[var].min()
        max_value = df[var].max()
        distance = abs(max_value - min_value)
        return distance
    else:
        return np.nan


    
def calculatePerfectSaison(mc, start_year, method='mean'):
    """
    Calculate a perfect seasonal time series and compare it with the original time series.

    Parameters:
        mc (xarray.Dataset): The original time series dataset.
        start_year (int): The starting year for the seasonal calculation.
        method (str): Method for calculating seasonal values ('mean', 'max', or 'min').

    Returns:
        difference (xarray.Dataset): The difference between the perfect seasonal and original time series.
        perfect_seasonal (xarray.Dataset): The perfect seasonal time series.
        normal_timeseries (xarray.Dataset): The original time series after smoothing.
    """
    
    print("Starting the calculation...")
    
    # Define constants
    num_years = 5
    num_weeks_per_year = 52
    
    # Custom preprocessing to resample the original dataframe to weekly frequency
    print("Resampling the original dataset...")
    mc_reprocessed = mc.resample(time="1W").mean()
    
    # Calculate the mean, max, or min values for each week of the year
    if method == 'mean':
        print("Calculating mean values...")
        ds_weekly_agg = mc_reprocessed.groupby('time.week').mean(dim='time')
    elif method == 'max':
        print("Calculating max values...")
        ds_weekly_agg = mc_reprocessed.groupby('time.week').max(dim='time')
    elif method == 'min':
        print("Calculating min values...")
        ds_weekly_agg = mc_reprocessed.groupby('time.week').min(dim='time')
    else:
        raise ValueError("Invalid method. Use 'mean', 'max', or 'min'.")
    
    # Create an empty list to store datasets for each year
    yearly_datasets = []

    for year in range(start_year, start_year + num_years):
        # Generate a list of datetime objects with weekly frequency for each year
        date_range = pd.date_range(start=f"{year}-01-01", periods=num_weeks_per_year, freq='W')

        # Repeat the mean/max/min values for each year
        num_weeks_total = num_weeks_per_year
        weekly_values_repeated = ds_weekly_agg.isel(week=slice(0, num_weeks_total)).rename({'week': 'time'})

        # Create a new dataset with the desired time values
        new_time_dataset = xr.Dataset(
            data_vars={
                'time': ('time', date_range)
            }
        )
        
        # Convert the 'time' data of weekly_values_repeated to match the data type of new_time_dataset
        weekly_values_repeated['time'] = new_time_dataset['time']

        # Use combine_first to concatenate the new dataset with weekly_values_repeated while ignoring NaN values
        yearly_dataset = weekly_values_repeated.combine_first(new_time_dataset)

        # Append the yearly dataset to the list
        yearly_datasets.append(yearly_dataset)

    # Concatenate all the yearly datasets along the 'time' dimension with NaN values ignored
    print("Concatenating yearly datasets...")
    perfect_seasonal = xr.concat(yearly_datasets, dim='time')

    # Custom preprocessing for smoothing
    print("Smoothing datasets...")
    # perfect_seasonal_smoothed = custom_preprocessing(perfect_seasonal, smooth_method='savitsky', smooth_window_length=2, smooth_polyorder=1)
    # normal_timeseries_smoothed = custom_preprocessing(mc_reprocessed, smooth_method='savitsky', smooth_window_length=2, smooth_polyorder=1)

    # Calculate the difference between perfect seasonal and original time series
    print("Calculating differences...")
    difference =  mc_reprocessed - perfect_seasonal
    
    print("Calculation completed.")

    return difference, perfect_seasonal, mc_reprocessed


def extract_info(index):
    # Remove the '.nc' extension
    parts = index.split('_')
    
    if len(parts) < 3:
        return None, None, None

    idx = parts[0]
    year = parts[1]
    dist_type = '_'.join(parts[2:])
    
    return idx, year, dist_type

# Beispielhafter Minicube-Loader
def load_minicubes(cube_file_path):
         
    mc = xr.open_dataset(cube_file_path, engine='netcdf4')

    # Rename dimensions from lat/lon to x/y
    #mc = mc.rename_vars({'lat': 'y', 'lon': 'x'})

    # Check if CRS is None and assign if needed
    if mc.rio.crs is None:
        mc.rio.write_crs("epsg:4326", inplace=True)
    # Set the CRS for the xarray dataset
    mc = mc.rio.write_crs("epsg:4326", inplace=True)

    return mc

def load_ids(path, disturbance_type):
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf_ids = gdf.rename(columns={'index_usda': 'USDA_IDX'})
    disturbance_ids = gdf_ids[gdf_ids['DCA_ID'] == disturbance_type]
    return disturbance_ids


def load_refdm(path, disturbance_type):
    # Load the shapefile using geopandas
    refdm_dataset = gpd.read_file(path)
    # Filter the dataset for DCA_ID 'drought'
    drought_refdm = refdm_dataset[refdm_dataset['DCA_ID'] == disturbance_type]
    dissolved_disturbance_refdm = drought_refdm.dissolve(by='USDA_IDX')
    dissolved_disturbance_refdm = dissolved_disturbance_refdm.reset_index()
    return dissolved_disturbance_refdm

# Function to get color for a given dist_type
def get_color(dist_type):
    return custom_colors.get(dist_type, '#000000')  # Default to black if dist_type not found

custom_colors = {
    'wind': '#1f77b4',      # tab:blue
    'fire': '#d62728',      # tab:red
    'defoliators': '#2ca02c',  # tab:green
    'drought': '#FFBA08', # tab:yellow
    'bark_beetle': '#714709'  # tab:brown
}

def extract_median_percentiles_distance_per_VI(data_ids, data_refdm):

    # Setup logging to capture skipped events
    logging.basicConfig(filename='skipped_events.log', level=logging.INFO)

    # Lists to store preprocessed data
    ids_median = []
    ids_q25 = []
    ids_q75 = []
    refdm_median = []
    refdm_q25 = []
    refdm_q75 = []

    # Iterate over USDA indexes in refdm_dist with a progress bar
    for idx, event in tqdm(data_refdm.iterrows(), total=len(data_refdm), desc="Processing Indexes"):
        try:
            print(idx)
            # Get the geometry for the current USDA_IDX
            geom_ids = data_ids[data_ids['USDA_IDX'] == event['USDA_IDX']]['geometry']
            geom_refdm = data_refdm[data_refdm['USDA_IDX'] == event['USDA_IDX']]['geometry']
            idx_part, year_part, dist_part = extract_info(event['cube_fn'])

            # Construct the file path for the corresponding NetCDF file
            cube_file_path = f"/Net/Groups/BGI/scratch/fmueller/Data/s2_region8_nc_256px_vi/{dist_part}/{year_part}/{event['cube_fn']}.nc"
            start_year = int(event['SURV_YEAR']) - 2

            # Load the minicube
            mc = load_minicubes(cube_file_path)

            # Convert the time dimension to a pandas DatetimeIndex
            time_index = pd.DatetimeIndex(mc.time.values)

            # Define the new year to start from
            new_year = 2018

            # Create a new DatetimeIndex with the updated year
            new_time_index = time_index.copy()
            new_time_index = new_time_index.to_series().apply(lambda dt: dt.replace(year=new_year) if dt.year < new_year else dt)

            # Convert back to xarray DataArray
            new_time = xr.DataArray(new_time_index.values, dims='time', name='time')

            mc['time'] = new_time
            mc = mc.sortby('time')

            # Clip the data
            clipped_data_ids = mc.rio.clip(geom_ids.geometry.apply(mapping), drop=True)
            clipped_data_refdm = mc.rio.clip(geom_refdm.geometry.apply(mapping), drop=True)

            mc.close()

            # Calculate the perfect season
            difference_max_ids, perfect_saision_max_ids, normal_timeseries_max_ids = calculatePerfectSaison(clipped_data_ids, start_year, method='max')
            difference_max_refdm, perfect_saision_max_refdm, normal_timeseries_max_refdm = calculatePerfectSaison(clipped_data_refdm, start_year, method='max')

            # Extract percentiles
            median_diff_ids = difference_max_ids.median(dim=['x', 'y'])
            q25_diff_ids = difference_max_ids.quantile(0.25, dim=['x', 'y'])
            q75_diff_ids = difference_max_ids.quantile(0.75, dim=['x', 'y'])

            median_diff_refdm = difference_max_refdm.median(dim=['x', 'y'])
            q25_diff_refdm = difference_max_refdm.quantile(0.25, dim=['x', 'y'])
            q75_diff_refdm = difference_max_refdm.quantile(0.75, dim=['x', 'y'])

            # Append the data to the respective lists
            ids_median.append(median_diff_ids)
            ids_q25.append(q25_diff_ids)
            ids_q75.append(q75_diff_ids)

            refdm_median.append(median_diff_refdm)
            refdm_q25.append(q25_diff_refdm)
            refdm_q75.append(q75_diff_refdm)

        except NoDataInBounds as e:
            # Log the event if NoDataInBounds error occurs and skip to the next iteration
            logging.info(f"Skipped event at index {idx} due to NoDataInBounds error: {e}")
            print(f"Skipped event at index {idx} due to NoDataInBounds error")

    # Notify if skipped events are logged
    if logging.getLogger().hasHandlers():
        print("Check 'skipped_events.log' for details on skipped events.")

    return ids_median, ids_q25, ids_q75, refdm_median, refdm_q25, refdm_q75


def main(ids_path, refdm_path, minicubes_shape_path, disturbance_type):
    logger = logging.getLogger(__name__)
    
    logger.info("Define Paths...")
    result_path = f'/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/distance/{disturbance_type}/'
    # Ensure the directory exists
    os.makedirs(result_path, exist_ok=True)
    r1 = f"ids_{disturbance_type}_median.nc"
    r2 = f"ids_{disturbance_type}_q25.nc"
    r3 = f"ids_{disturbance_type}_q75.nc"
    r4 = f"refdm_{disturbance_type}_median.nc"
    r5 = f"refdm_{disturbance_type}_q25.nc"
    r6 = f"refdm_{disturbance_type}_q75.nc"
    r7 = f"combined_{disturbance_type}_medians.csv"

    # Full paths for saving files
    path_r1 = os.path.join(result_path, r1)
    path_r2 = os.path.join(result_path, r2)
    path_r3 = os.path.join(result_path, r3)
    path_r4 = os.path.join(result_path, r4)
    path_r5 = os.path.join(result_path, r5)
    path_r6 = os.path.join(result_path, r6)
    path_r7 = os.path.join(result_path, r7)

    print(f"Load {disturbance_type} ids ...")
    ids = load_ids(path=ids_path, disturbance_type=disturbance_type)
    print(f"Load {disturbance_type} refdm ...")
    refdm = load_refdm(path=refdm_path, disturbance_type=disturbance_type)
    

    logger.info(f"Extract median, percentiles, slope and distance for each vegetation index ...")
    ids_median, ids_q25, ids_q75, refdm_median, refdm_q25, refdm_q75 = extract_median_percentiles_distance_per_VI(data_ids=ids, data_refdm=refdm) 

    dh_ids_median = merge_disturbance_events(ids_median) # Append the data to the list of preprocessed files
    dh_ids_q25 = merge_disturbance_events(ids_q25)
    dh_ids_q75 = merge_disturbance_events(ids_q75)

    dh_refdm_median = merge_disturbance_events(refdm_median) # Append the data to the list of preprocessed files
    dh_refdm_q25 = merge_disturbance_events(refdm_q25)
    dh_refdm_q75 = merge_disturbance_events(refdm_q75)

    dh_ids_median.to_netcdf(path_r1)
    dh_ids_q25.to_netcdf(path_r2)
    dh_ids_q75.to_netcdf(path_r3)
    dh_refdm_median.to_netcdf(path_r4)
    dh_refdm_q25.to_netcdf(path_r5)
    dh_refdm_q75.to_netcdf(path_r6)
    combined = combine_ids_refdm_medians(ids_median, refdm_median)
    combined.to_csv(path_r7)
    


    logger.info("Finished")

if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fire_vi_script.log"),
        logging.StreamHandler()
    ]
)
    
    if len(sys.argv) != 5:
        logging.error("Usage: python main.py <ids_path> <refdm_path> <minicubes_shape_path> <disturbance_type>")
        sys.exit(1)
    
    ids_path = sys.argv[1]
    refdm_path = sys.argv[2]
    minicubes_shape_path = sys.argv[3]
    disturbance_type = sys.argv[4]
    
    main(ids_path, refdm_path, minicubes_shape_path, disturbance_type)