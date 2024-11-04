import xarray as xr
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
import pandas as pd  # Assuming you're using pandas for timestamps
import os
from func_helper import parse_custom_colors
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from func_preprocessing import preprocess_sentinel_data
import logging

import argparse


def load_dissolved_refdm(refdm_path):
    """
    Load and process a GeoDataFrame from the given path. The processing includes converting
    necessary columns to numeric, dissolving geometries by ID_E and S1_YEAR, calculating 
    the duration for each ID_E, and returning the processed DataFrame.

    Parameters:
    refdm_path (str): Path to the GeoDataFrame file.

    Returns:
    GeoDataFrame: Processed GeoDataFrame with a 'Duration' column indicating the number 
                  of unique years for each ID_E.
    """
    # Load the GeoDataFrame
    refdm = gpd.read_file(refdm_path)
    

    logger.info("CRS:", refdm.crs)
    logger.info(f"Size of refdm_dataset: {len(refdm)}")

    # Convert columns to numeric, if not already
    refdm['SURVEY_Y'] = pd.to_numeric(refdm['SURVEY_Y'], errors='coerce')
    refdm['S1_YEAR'] = pd.to_numeric(refdm['S1_YEAR'], errors='coerce')
    
    # Dissolve geometries by ID_E and S1_YEAR
    dissolved_refdm = refdm.dissolve(by=['ID_E', 'S1_YEAR']).reset_index()
    
    # Group by ID_E and aggregate unique years
    unique_years_per_id = dissolved_refdm.groupby('ID_E')['S1_YEAR'].unique().reset_index()
    
    # Calculate the duration (number of unique years) for each ID_E
    unique_years_per_id['Duration'] = unique_years_per_id['S1_YEAR'].apply(len)
    
    # Merge the calculated duration with the main DataFrame
    dissolved_df = dissolved_refdm.merge(unique_years_per_id[['ID_E', 'Duration']], on='ID_E')
    
    # Dissolve geometries again by ID_E to ensure aggregation and reset index
    dissolved_df = dissolved_df.dissolve(by=['ID_E']).reset_index()
    logger.info(f"Size of unique refdm_dataset events: {len(dissolved_df)}")
    return dissolved_df


def load_ids_dataset(path):
    gdf_ids = gpd.read_file(path)
    logger.info("CRS:", gdf_ids.crs)
    logger.info(f"Size of gdf_ids: {len(gdf_ids)}")
    return gdf_ids

def add_minicube_index(intersected_grid, ids):

    # Function to get intersecting indices
    def get_intersecting_indices(geometry, grid):
        intersecting_indices = grid[grid.intersects(geometry)].index.tolist()
        return intersecting_indices

    # Apply the function to each row in reprojected_refdm
    ids['minicube_index'] = ids['geometry'].apply(lambda geom: get_intersecting_indices(geom, intersected_grid))

    # Add the 'cube_amount' column by counting the length of each list in 'minicube_index'
    ids['cube_amount'] = ids['minicube_index'].apply(len)

    return ids

def extract_unique_minicubes(refdm_filtered, ids_filtered, intersected_grid):
    def convert_to_list(value):
        """Convert string representation of list to an actual list or return value directly."""
        try:
            return ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            return []

    # Extract and flatten minicube values for refdm
    minicube_refdm = [
        int(item)
        for sublist in [convert_to_list(val) for val in refdm_filtered['mini_idx'].values]
        for item in sublist
    ]

    # print(f"REFDM: {minicube_refdm}")
    
    # Extract and flatten minicube values for ids
    minicube_ids = [
        int(item)
        for sublist in [convert_to_list(val) for val in ids_filtered['mini_idx'].values]
        for item in sublist
    ]
    # print(f"IDS: {minicube_ids}")

    # Combine both lists and get unique values
    unique_minicubes = list(set(minicube_refdm + minicube_ids))

    # Use .isin() to filter when unique_minicubes is a list
    grid = intersected_grid.loc[intersected_grid['FID'].isin(unique_minicubes)]

    return unique_minicubes, grid


def create_plots(combined_dataset, refdm_filtered, ids_filtered, grid, unique_minicubes, dca, ID, custom_colors, 
                 mean_ids, mean_refdm, year, output_dir, method=None, var='ndvi'):
    """
    Create two subplots: one for NDVI data and another for time series data. Save plots to a folder.

    Parameters:
    - combined_dataset: Dataset containing NDVI data
    - refdm_filtered: GeoDataFrame for REFDM boundaries
    - ids_filtered: GeoDataFrame for IDS boundaries
    - grid: GeoDataFrame for grid boundaries
    - unique_minicubes: Unique minicube IDs
    - dca: Data category for coloring
    - ID: Event ID
    - custom_colors: Dictionary of custom colors
    - mean_ids: Mean values for IDS
    - mean_refdm: Mean values for REFDM
    - year: Survey year to highlight
    - output_dir: Directory to save plots
    - method: Method used for aggregation ('mean', 'max', 'min', or None)
    - var: Variable to plot (default is 'ndvi')
    """
    
    time_index = 240  # Set the time index you want to plot
    ndvi_data = combined_dataset[var].isel(time=time_index)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))  # 1 row, 2 columns
    
    # --- First subplot: NDVI Data ---
    # Plot the boundaries of the geometries for refdm_filtered
    refdm_filtered.boundary.plot(ax=ax1, color='magenta', linewidth=2, label='S1DM')
    
    # Plot the boundaries of the geometries for ids_filtered
    ids_filtered.boundary.plot(ax=ax1, color='black', linewidth=3, linestyle='-', label='IDS')
    
    # Optional: Uncomment if you have a grid to plot
    grid.boundary.plot(ax=ax1, color='white', linewidth=3, linestyle=':')
    
    # Plot the NDVI data with a colormap
    ndvi_data.plot(ax=ax1, cmap='Greens', add_colorbar=True, cbar_kwargs={'shrink': 0.8})  # Shrink colorbar
    
    # Set axis labels
    ax1.set_xlabel('Longitude', fontsize=18)
    ax1.set_ylabel('Latitude', fontsize=18)
    
    # Set a title for the NDVI plot
    ax1.set_title(f' ', fontsize=16)  # Empty title
    
    # Add a legend (remove Minicube ID from legend)
    ax1.legend(loc='upper right', fontsize=10)  # Smaller legend
    
    # --- Second subplot: Time Series Data ---
    # Set the color for REFDM based on the DCA category using custom_colors
    refdm_color = custom_colors.get(dca, 'gray')  # Default to gray if dca not found
    
    # Add a red dotted line at y=0
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # Create a date range for the entire year
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    
    # Highlight the entire year with a gray box
    ax2.axvspan(start_date, end_date, color='gray', alpha=0.5, label='Survey Year')

    # Extract the time coordinates for the x-axis
    time = mean_ids['time'].values
    
    # Plot the mean for IDS (always black)
    ax2.plot(time, mean_ids[var], color='black', label='IDS ', linewidth=2)
    
    # Plot the mean for REFDM (use the custom color)
    ax2.plot(time, mean_refdm[var], color=refdm_color, label='S1DM ', linewidth=2)
    
    # Set plot title and labels
    ax2.set_xlabel("Disturbance Year", fontsize=18)
    ax2.set_ylabel(f"{var.upper()}", fontsize=18)
    
    # Add a legend
    ax2.legend(loc='lower right', fontsize=10)  # Smaller legend
    
    # Super title for both plots (centered)
    method_label = f"{method.capitalize()}" if method else "None"
    fig.suptitle(
        f"{dca.capitalize()} Event (Method: {method_label}) with ID_E={ID} on the Cubes {unique_minicubes}",
        ha='center', fontsize=28
    )
    
    # Adjust layout to ensure super title is centered
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to give space for the super title
    
    # Save the plot with a clear filename indicating method and ID
    plot_filename = os.path.join(output_dir, f"event_{ID}_method_{method_label}_{var}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close(fig)

    logger.info(f"Plot saved: {plot_filename}")

def calculatePerfectSaison(mc, start_year, min_year, method='mean'):
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
    
    # General start of calculation
    logger.info(f"Starting the calculation of {method} season...")

    # Define constants
    num_years = 8
    num_weeks_per_year = 52

    
    # Custom preprocessing to resample the original dataset to weekly frequency
    mc_reprocessed = mc.resample(time="1W").mean()

    # Filter the dataset for time after 2016 for aggregation
    mc_filtered = mc_reprocessed.sel(time=mc_reprocessed['time'].dt.year >= min_year)

    # Calculate the mean, max, or min values for each week of the year from 2017 onwards
    if method == 'mean':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).mean(dim='time')
    elif method == 'max':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).max(dim='time')
    elif method == 'min':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).min(dim='time')
    else:
        raise ValueError("Invalid method. Use 'mean', 'max', or 'min'.")

    # Create an empty list to store datasets for each year
    yearly_datasets = []

    # Iterate over all years in the original dataset (not just after 2016)
    for year in range(start_year, mc_reprocessed['time'].dt.year.max().item() + 1):
        # Generate a list of datetime objects with weekly frequency for each year
        date_range = pd.date_range(start=f"{year}-01-01", periods=num_weeks_per_year, freq='W')

        # Repeat the mean/max/min values for each week
        weekly_values_repeated = ds_weekly_agg.isel(week=slice(0, num_weeks_per_year)).rename({'week': 'time'})

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
    perfect_seasonal = xr.concat(yearly_datasets, dim='time')

    # Calculate the difference between the perfect seasonal and the original time series (over the whole timespan)
    difference = mc_reprocessed - perfect_seasonal
    
    logger.info("Calculation completed.")

    return difference, perfect_seasonal, mc_reprocessed



# Helper function to load and merge datasets
def load_and_merge_datasets(unique_minicubes, s2_minicube_folder, equi7_crs):
    """
    Load and merge datasets for given minicubes.
    
    Parameters:
    - unique_minicubes: List of unique minicubes.
    - s2_minicube_folder: Path to the folder containing S2 minicube files.
    - equi7_crs: Coordinate Reference System for the data.
    
    Returns:
    - combined_dataset: Merged dataset for all minicubes, or single dataset if only one is loaded.
    """
    datasets = []
    
    # Load datasets
    for minicube in tqdm(unique_minicubes, desc="Loading minicubes"):
        file_path = f"{s2_minicube_folder}/{minicube}_10_512_20152024_equi7_NA.nc"
        try:
            dataset = xr.open_dataset(file_path).sortby(['y', 'x'])
            if not dataset.rio.crs:
                dataset = dataset.rio.write_crs(equi7_crs)
            datasets.append(dataset)
        except FileNotFoundError:
            logger.info(f"File not found for minicube: {minicube}")
        except Exception as e:
            logger.info(f"Error loading minicube {minicube}: {e}")

    # Check the number of loaded datasets
    if not datasets:
        return None
    elif len(datasets) == 1:
        logger.info("Only one dataset loaded. Returning the dataset directly.")
        return datasets[0]  # Return the single dataset directly
    else:
        # Merge all loaded datasets by coordinates
        logger.info("Merging all loaded datasets by coordinates ...")
        return xr.combine_by_coords(datasets, combine_attrs="override")


# Helper function to clip datasets
def clip_datasets(combined_dataset, ids_geom, s1dm_geom):
    """
    Clip datasets using IDS and S1DM geometries.
    
    Parameters:
    - combined_dataset: The combined dataset to clip.
    - ids_geom: Geometry for IDS data.
    - s1dm_geom: Geometry for S1DM data.
    
    Returns:
    - clipped_ids: Clipped dataset for IDS.
    - clipped_s1dm: Clipped dataset for S1DM.
    """
    ids_shape = gpd.GeoSeries(ids_geom)
    s1dm_shape = gpd.GeoSeries(s1dm_geom)
    
    clipped_ids = combined_dataset.rio.clip(ids_shape.geometry.apply(mapping), drop=True)
    clipped_s1dm = combined_dataset.rio.clip(s1dm_shape.geometry.apply(mapping), drop=True)
    
    return clipped_ids, clipped_s1dm

# This method is less memory efficient as all datasets are loaded into memory before clipping.
def load_clip_merge_all_datasets(unique_minicubes, s2_minicube_folder, equi7_crs, ids_geom, s1dm_geom):
    """
    Load all datasets, then clip and merge them. This is less memory-efficient.
    
    Parameters:
    - unique_minicubes: List of unique minicubes.
    - s2_minicube_folder: Path to the folder containing S2 minicube files.
    - equi7_crs: Coordinate Reference System for the data.
    - ids_geom: Geometry for IDS data.
    - s1dm_geom: Geometry for S1DM data.
    
    Returns:
    - combined_clipped_ids: Merged clipped dataset for IDS.
    - combined_clipped_s1dm: Merged clipped dataset for S1DM.
    """
    datasets = []
    
    # Load datasets first
    for minicube in tqdm(unique_minicubes, desc="Loading minicubes"):
        file_path = f"{s2_minicube_folder}/{minicube}_10_512_20152024_equi7_NA.nc"
        try:
            dataset = xr.open_dataset(file_path).sortby(['y', 'x'])
            if not dataset.rio.crs:
                dataset = dataset.rio.write_crs(equi7_crs)
            datasets.append(dataset)
        except FileNotFoundError:
            logger.info(f"File not found for minicube: {minicube}")
        except Exception as e:
            logger.info(f"Error loading minicube {minicube}: {e}")

    # Clip all datasets after loading
    logger.info(f"Clip all datasets ({len(datasets)}) after loading")
    clipped_ids_datasets = [dataset.rio.clip(gpd.GeoSeries(ids_geom).geometry.apply(mapping), drop=True) for dataset in datasets]
    clipped_s1dm_datasets = [dataset.rio.clip(gpd.GeoSeries(s1dm_geom).geometry.apply(mapping), drop=True) for dataset in datasets]

    # Merge all clipped datasets by coordinates
    logger.info(f"Merge all clipped datasets ({len(datasets)}) by coordinates ")
    combined_clipped_ids = xr.combine_by_coords(clipped_ids_datasets, combine_attrs="override")
    combined_clipped_s1dm = xr.combine_by_coords(clipped_s1dm_datasets, combine_attrs="override")

    return combined_clipped_ids, combined_clipped_s1dm


# Helper function to calculate statistics
def calculate_statistics(clipped_data, method=None, year_difference=0):
    """
    Calculate the statistics (mean, max, min, series) for the clipped data.
    
    Parameters:
    - clipped_data: The clipped dataset.
    - method: The method for calculating statistics (None, "mean", "max", "min").
    - year_difference: Years to shift the time.
    
    Returns:
    - calculated_data: The calculated dataset.
    """
    # if method is None:
    #     result = clipped_data.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')
    # else:
    #     result = calculatePerfectSaison(clipped_data, start_year=2016, method=method)[0].mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')

    if method is None:
        result = clipped_data
    else:
        result = calculatePerfectSaison(clipped_data, start_year=2016, min_year=2017, method=method)[0].interpolate_na(dim='time', method='linear')

     # Shift time series to align with the target year
    result['time'] = pd.to_datetime(result['time'].values) + pd.DateOffset(years=year_difference)


    
    return result


def save_dataset(dataset, filepath):
    """
    Save dataset to NetCDF file.
    
    Parameters:
    - dataset: Dataset to save.
    - filepath: Path to the NetCDF file.
    """
    try:
        # Check if the file already exists
        if os.path.exists(filepath):
            # Attempt to open the existing dataset
            existing_ds = xr.open_dataset(filepath)
            # Combine existing and new datasets along the 'time' dimension
            combined_ds = xr.concat([existing_ds, dataset], dim='time')
            # Close the existing dataset
            existing_ds.close()  
            
            # Save the combined dataset
            combined_ds.to_netcdf(filepath, mode='w')
            logger.info(f"Succesfully saved NetCDF to {filepath}")
        else:
            # Save the new dataset directly
            dataset.to_netcdf(filepath)

    except PermissionError:
        logger.info(f"PermissionError: Cannot write to {filepath}. Attempting to delete and save again.")
        # If a PermissionError occurs, delete the file and try saving again
        if os.path.exists(filepath):
            try:
                os.remove(filepath)  # Delete the existing file
                logger.info(f"Deleted {filepath}. Attempting to save again.")
                dataset.to_netcdf(filepath)  # Try to save the new dataset
            except Exception as delete_error:
                logger.info(f"Failed to delete {filepath}: {delete_error}")
    except Exception as e:
        logger.info(f"An error occurred while saving the dataset: {e}")


# Main processing function
def process_all_ids(filtered_df, s1dm, ids_mincubes, dca, s2_minicube_folder, intersected_grid, equi7_crs, custom_colors, output_dir):
    """
    Process all IDs and calculate the necessary statistics.
    
    Parameters:
    - filtered_df: DataFrame with filtered IDs.
    - s1dm: GeoDataFrame for S1DM.
    - ids_mincubes: GeoDataFrame for IDS.
    - dca: Data category.
    - s2_minicube_folder: Path to S2 minicube folder.
    - intersected_grid: Grid for intersection.
    - equi7_crs: CRS for the data.
    - custom_colors: Dictionary of custom colors.
    - output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = {
        'mean_ids': os.path.join(output_dir, f'{dca}_merged_mean_ids.nc'),
        'mean_s1dm': os.path.join(output_dir, f'{dca}_merged_mean_s1dm.nc'),
        'max_ids': os.path.join(output_dir, f'{dca}_merged_max_ids.nc'),
        'max_s1dm': os.path.join(output_dir, f'{dca}_merged_max_s1dm.nc'),
        'min_ids': os.path.join(output_dir, f'{dca}_merged_min_ids.nc'),
        'min_s1dm': os.path.join(output_dir, f'{dca}_merged_min_s1dm.nc'),
        'series_ids': os.path.join(output_dir, f'{dca}_merged_series_ids.nc'),
        'series_s1dm': os.path.join(output_dir, f'{dca}_merged_series_s1dm.nc')
    }

    for ID in filtered_df['ID_E'].unique():

        logger.info(f"Processing ID_E: {ID}")
        
        # Filter GeoDataFrames for current ID
        ids_filtered = ids_mincubes[ids_mincubes['ID_E'] == ID]
        s1dm_filtered = s1dm[s1dm['ID_E'] == ID]
        unique_minicubes, _ = extract_unique_minicubes(s1dm_filtered, ids_filtered, intersected_grid)

        # Load and merge datasets
        combined_dataset = load_and_merge_datasets(unique_minicubes, s2_minicube_folder, equi7_crs)
        if combined_dataset is None:
            continue

        try:
            # Try to clip datasets using IDS and S1DM geometries
            clipped_ids, clipped_s1dm = clip_datasets(combined_dataset, ids_filtered.geometry, s1dm_filtered.geometry)
            
            # Interpolate missing values
            clipped_ids = clipped_ids.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')
            clipped_s1dm = clipped_s1dm.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')

        except Exception as e:
            # Log the error and continue processing other variables
            logger.error(f"Error during clipping for ID_E: {ID}. Dropping variable. Error: {str(e)}")
            continue  # Continue to the next ID or variable

    # Continue with further processing of clipped_ids and clipped_s1dm...

        # Calculate statistics (mean, max, min, series)
        target_year = 2018  # Example target year, adjust as needed
        survey_year = ids_filtered['SURVEY_Y'].iloc[0]
        logger.info(survey_year)
        year_difference = target_year - survey_year
        logger.info(year_difference)
        # Shift time series to align with the target year
        #difference['time'] = pd.to_datetime(difference['time'].values) + pd.DateOffset(years=year_difference)


        mean_ids = calculate_statistics(clipped_ids, method="mean", year_difference=year_difference)
        mean_s1dm = calculate_statistics(clipped_s1dm, method="mean", year_difference=year_difference)
        
        max_ids = calculate_statistics(clipped_ids, method="max", year_difference=year_difference)
        max_s1dm = calculate_statistics(clipped_s1dm, method="max", year_difference=year_difference)
        
        min_ids = calculate_statistics(clipped_ids, method="min", year_difference=year_difference)
        min_s1dm = calculate_statistics(clipped_s1dm, method="min", year_difference=year_difference)

        series_ids = calculate_statistics(clipped_ids, method=None, year_difference=year_difference)
        series_s1dm = calculate_statistics(clipped_s1dm, method=None, year_difference=year_difference)

        # Save datasets
        save_dataset(mean_ids, files['mean_ids'])
        save_dataset(mean_s1dm, files['mean_s1dm'])
        save_dataset(max_ids, files['max_ids'])
        save_dataset(max_s1dm, files['max_s1dm'])
        save_dataset(min_ids, files['min_ids'])
        save_dataset(min_s1dm, files['min_s1dm'])
        save_dataset(series_ids, files['series_ids'])
        save_dataset(series_s1dm, files['series_s1dm'])


        # Save plots for all methods
        create_plots(combined_dataset, s1dm_filtered, ids_filtered, intersected_grid, unique_minicubes, 
                     dca, ID, custom_colors, mean_ids, mean_s1dm, target_year, output_dir, method="mean", var='ndvi')
        create_plots(combined_dataset, s1dm_filtered, ids_filtered, intersected_grid, unique_minicubes, 
                     dca, ID, custom_colors, max_ids, max_s1dm, target_year, output_dir, method="max", var='ndvi')
        create_plots(combined_dataset, s1dm_filtered, ids_filtered, intersected_grid, unique_minicubes, 
                     dca, ID, custom_colors, min_ids, min_s1dm, target_year, output_dir, method="min", var='ndvi')
        create_plots(combined_dataset, s1dm_filtered, ids_filtered, intersected_grid, unique_minicubes, 
                     dca, ID, custom_colors, series_ids, series_s1dm, target_year, output_dir, method=None, var='ndvi')

    logger.info("Processing complete.")


def ensure_output_directory(output_dir):
    """Ensure that the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)

def load_datasets(files):
    """Load datasets from provided file paths."""
    datasets = {}
    for key, path in files.items():
        datasets[key] = xr.open_dataset(path)
    return datasets

def save_netcdf(dataset, path):
    """Save a dataset to a NetCDF file."""
    dataset.to_netcdf(path)
    logger.info(f"Saved dataset to {path}")

def calculate_resampled_stats(dataset, stat_type, time_resample='W'):
    """Calculate resampled statistics based on aggregation method."""
    if stat_type == 'mean':
        return dataset.sortby('time').resample(time=time_resample).mean()
    elif stat_type == 'median':
        return dataset.sortby('time').resample(time=time_resample).median()
    else:
        raise ValueError(f"Unsupported stat type: {stat_type}")

def calculate_quantiles(dataset, time_resample='W'):
    """Calculate the 25th and 75th quantiles for a dataset."""
    quantile_25 = dataset.sortby('time').resample(time=time_resample).quantile(0.25)
    quantile_75 = dataset.sortby('time').resample(time=time_resample).quantile(0.75)
    return quantile_25, quantile_75

def calculate_std(dataset, time_resample='W'):
    """Calculate standard deviation for a dataset."""
    return dataset.sortby('time').resample(time=time_resample).std(dim='time')

import os

def process_aggregation(datasets, output_dir, aggregation):
    """Process and calculate statistics based on aggregation method."""
    
    def save_with_aggregation(dataset_key, aggregation_type):
        """Calculate and save statistics for the given dataset key."""
        if dataset_key in datasets:
            if aggregation_type == 'mean':
                logger.info(f"Calculating weekly mean and standard deviation for {dataset_key}...")
                final_mean = calculate_resampled_stats(datasets[dataset_key], 'mean')
                final_std = calculate_std(datasets[dataset_key])
                save_netcdf(final_mean, os.path.join(output_dir, f'final_mean_{dataset_key}.nc'))
                save_netcdf(final_std, os.path.join(output_dir, f'final_std_{dataset_key}.nc'))

            elif aggregation_type == 'median':
                logger.info(f"Calculating weekly median for {dataset_key}...")
                final_median = calculate_resampled_stats(datasets[dataset_key], 'median')
                quantile_25, quantile_75 = calculate_quantiles(datasets[dataset_key])
                save_netcdf(final_median, os.path.join(output_dir, f'final_median_{dataset_key}.nc'))
                save_netcdf(quantile_25, os.path.join(output_dir, f'quantile_25_{dataset_key}.nc'))
                save_netcdf(quantile_75, os.path.join(output_dir, f'quantile_75_{dataset_key}.nc'))

    # Process each dataset type according to the aggregation method
    if aggregation == 'mean':
        save_with_aggregation('mean_ids', 'mean')
        save_with_aggregation('mean_s1dm', 'mean')  # Changed from mean_refdm to mean_s1dm
        
        # Save max, min, and series datasets with original filenames preserved
        save_with_aggregation('max_ids', 'mean')    # You can adjust the aggregation if needed
        save_with_aggregation('max_s1dm', 'mean')   # Changed from max_refdm to max_s1dm
        save_with_aggregation('min_ids', 'mean')     # You can adjust the aggregation if needed
        save_with_aggregation('min_s1dm', 'mean')    # Changed from min_refdm to min_s1dm
        save_with_aggregation('series_ids', 'mean')  # You can adjust the aggregation if needed
        save_with_aggregation('series_s1dm', 'mean') # Changed from series_refdm to series_s1dm

    elif aggregation == 'median':
        save_with_aggregation('mean_ids', 'median')
        save_with_aggregation('mean_s1dm', 'median')  # Changed from mean_refdm to mean_s1dm

        # Save max, min, and series datasets with median calculations
        save_with_aggregation('max_ids', 'median')    # You can adjust the aggregation if needed
        save_with_aggregation('max_s1dm', 'median')   # Changed from max_refdm to max_s1dm
        save_with_aggregation('min_ids', 'median')     # You can adjust the aggregation if needed
        save_with_aggregation('min_s1dm', 'median')    # Changed from min_refdm to min_s1dm
        save_with_aggregation('series_ids', 'median')  # You can adjust the aggregation if needed
        save_with_aggregation('series_s1dm', 'median') # Changed from series_refdm to series_s1dm

    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    logger.info(f"{aggregation.capitalize()} calculations completed and saved.")

def calculate_aggregation(dca, aggregation, input_dir, output_dir):
    """
    Calculate aggregation (mean, median) and optionally standard deviation for each dataset.
    
    Parameters:
    - filtered_df: DataFrame with filtered data.
    - refdm: GeoDataFrame for REFDM.
    - ids_mincubes: GeoDataFrame for IDS.
    - dca: Data category for coloring or processing.
    - s2_minicube_folder: Path to the folder containing S2 minicube files.
    - intersected_grid: Grid for intersection.
    - equi7_crs: Coordinate Reference System (CRS) for the data.
    - custom_colors: Dictionary of custom colors for plotting.
    - aggregation: The type of aggregation ('mean', 'median').
    - output_dir: Directory where the results will be saved (optional).
    """
    
    # Set default output directory if not provided
    if input_dir is None:
        logger.error(f'Input directory {input_dir} not found.')
        #output_dir = f'/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/results/intermediate_results/{dca}/'
    # Ensure the output directory exists
    ensure_output_directory(output_dir)

    os.makedirs(input_dir, exist_ok=True)
    files = {
        'mean_ids': os.path.join(input_dir, f'{dca}_merged_mean_ids.nc'),
        'mean_s1dm': os.path.join(input_dir, f'{dca}_merged_mean_s1dm.nc'),
        'max_ids': os.path.join(input_dir, f'{dca}_merged_max_ids.nc'),
        'max_s1dm': os.path.join(input_dir, f'{dca}_merged_max_s1dm.nc'),
        'min_ids': os.path.join(input_dir, f'{dca}_merged_min_ids.nc'),
        'min_s1dm': os.path.join(input_dir, f'{dca}_merged_min_s1dm.nc'),
        'series_ids': os.path.join(input_dir, f'{dca}_merged_series_ids.nc'),
        'series_s1dm': os.path.join(input_dir, f'{dca}_merged_series_s1dm.nc')
    }

    # Load datasets
    datasets = load_datasets(files)

    # Process aggregation based on the provided method
    process_aggregation(datasets, output_dir, aggregation)

    logger.info(f"{aggregation.capitalize()} and additional statistics (if applicable) calculations completed and saved.")


import os

def plot_median_and_quantiles_with_trends(final_median_ids, quantile_25_ids, quantile_75_ids, 
                                          final_median_s1dm, quantile_25_s1dm, quantile_75_s1dm, 
                                          var, dca, custom_colors, key, output_dir):
    """
    Plot median and quantiles with a gray region after 2018, red horizontal line at y=0, 
    and trend lines for IDS and S1DM datasets. Save the plot as a PNG file.
    Only data from 2016 onwards is plotted.
    """
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract the time coordinates for the x-axis
    time = final_median_ids['time'].values

    # Filter the data to include only time >= 2016
    mask = time >= np.datetime64('2016-01-01')
    
    time_filtered = time[mask]
    quantile_25_ids_filtered = quantile_25_ids[var].values[mask]
    quantile_75_ids_filtered = quantile_75_ids[var].values[mask]
    median_ids_filtered = final_median_ids[var].values[mask]
    
    quantile_25_s1dm_filtered = quantile_25_s1dm[var].values[mask]
    quantile_75_s1dm_filtered = quantile_75_s1dm[var].values[mask]
    median_s1dm_filtered = final_median_s1dm[var].values[mask]

    # Set the color for S1DM based on the DCA category using custom_colors
    s1dm_color = custom_colors.get(dca, 'gray')  # Default to gray if dca not found

    # Plot the median and quantiles for IDS (always black)
    ax.plot(time_filtered, median_ids_filtered, color='black', label='IDS Median', linewidth=2)
    ax.fill_between(time_filtered, 
                    quantile_25_ids_filtered, 
                    quantile_75_ids_filtered, 
                    color='black', alpha=0.2, label='IDS Quantiles')

    # Plot the median and quantiles for S1DM
    ax.plot(time_filtered, median_s1dm_filtered, color=s1dm_color, label='S1DM Median', linewidth=2)
    ax.fill_between(time_filtered, 
                    quantile_25_s1dm_filtered, 
                    quantile_75_s1dm_filtered, 
                    color=s1dm_color, alpha=0.2, label='S1DM Quantiles')

    # Add gray span for time in 2018
    ax.axvspan(np.datetime64('2018-01-01'), np.datetime64('2018-12-31'), color='#FFD6D6', alpha=0.3, label='Disturbance Year')

    # Add red horizontal line at y=0
    ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # Set plot title and labels
    ax.set_title(f"Time Series of {var.upper()} ({dca.capitalize()}; Calculation: {key}; Aggregation: Median)", fontsize=24)
    ax.set_xlabel("Disturbance Year", fontsize=20)
    ax.set_ylabel(f"{var.upper()}", fontsize=20)

    # Add a legend
    ax.legend(loc='lower right')
    output_file = os.path.join(output_dir, f'median_{var}_{key}.png')

    # Save the plot as a PNG file
    plt.savefig(output_file, format='png', dpi=400)
    
    # Show the plot
    plt.tight_layout()  # Adjust the layout for better fit
    plt.show()


def plot_mean_and_std_with_trends(final_mean_ids, final_std_ids, final_mean_s1dm, final_std_s1dm, 
                                  var, dca, custom_colors, key, output_dir):
    """
    Plot mean and standard deviation with a gray region after 2018, red horizontal line at y=0, 
    and trend lines for IDS and S1DM datasets. Save the plot as a PNG file.
    Only data from 2016 onwards is plotted.
    """
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract the time coordinates for the x-axis
    time = final_mean_ids['time'].values

    # Filter the data to include only time >= 2016
    mask = time >= np.datetime64('2016-01-01')

    time_filtered = time[mask]
    mean_ids_filtered = final_mean_ids[var].values[mask]
    std_ids_filtered = final_std_ids[var].values[mask]

    mean_s1dm_filtered = final_mean_s1dm[var].values[mask]
    std_s1dm_filtered = final_std_s1dm[var].values[mask]

    # Set the color for S1DM based on the DCA category using custom_colors
    s1dm_color = custom_colors.get(dca, 'gray')  # Default to gray if dca not found

    # Plot the mean and std for IDS (always black)
    ax.plot(time_filtered, mean_ids_filtered, color='black', label='IDS Mean', linewidth=2)
    ax.fill_between(time_filtered, 
                    mean_ids_filtered - std_ids_filtered, 
                    mean_ids_filtered + std_ids_filtered, 
                    color='black', alpha=0.2, label='IDS Std')

    # Plot the mean and std for S1DM
    ax.plot(time_filtered, mean_s1dm_filtered, color=s1dm_color, label='S1DM Mean', linewidth=2)
    ax.fill_between(time_filtered, 
                    mean_s1dm_filtered - std_s1dm_filtered, 
                    mean_s1dm_filtered + std_s1dm_filtered, 
                    color=s1dm_color, alpha=0.2, label='S1DM Std')

    # Add gray span for time in 2018
    ax.axvspan(np.datetime64('2018-01-01'), np.datetime64('2018-12-31'), color='#FFD6D6', alpha=0.3, label='Disturbance Year')

    # Add red horizontal line at y=0
    ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # Set plot title and labels
    ax.set_title(f"Time Series of {var.upper()} ({dca.capitalize()}; Calculation: {key}; Aggregation: Mean)", fontsize=24)
    ax.set_xlabel("Disturbance Year", fontsize=20)
    ax.set_ylabel(f"{var.upper()}", fontsize=20)

    # Add a legend
    ax.legend(loc='lower right')

    output_file = os.path.join(output_dir, f'mean_{var}_{key}.png')

    # Save the plot as a PNG file
    plt.savefig(output_file, format='png', dpi=400)
    
    # Show the plot
    plt.tight_layout()  # Adjust the layout for better fit
    plt.show()


def load_netcdf(filepath):
    """Load a NetCDF file using xarray."""
    if os.path.exists(filepath):
        return xr.open_dataset(filepath)
    else:
        print(f"File {filepath} not found.")
        return None

def get_file_paths(dataset_key, aggregation, output_dir):
    """
    Generate file paths for the NetCDF files based on the key (mean, max, min, series)
    and aggregation method (mean, median).
    
    Parameters:
    - dataset_key: The dataset key (mean, max, min, series).
    - aggregation: The aggregation type (mean, median).
    - output_dir: Directory where the NetCDF files are stored.
    
    Returns:
    - A dictionary containing file paths for both ids and s1dm datasets.
    """
    file_paths = {}

    if aggregation == 'mean':
        output_dir = f'{output_dir}/{aggregation}/'
        # For mean aggregation, load the mean and standard deviation files
        file_paths['mean_ids'] = os.path.join(output_dir, f'final_mean_{dataset_key}_ids.nc')
        file_paths['std_ids'] = os.path.join(output_dir, f'final_std_{dataset_key}_ids.nc')
        file_paths['mean_s1dm'] = os.path.join(output_dir, f'final_mean_{dataset_key}_s1dm.nc')
        file_paths['std_s1dm'] = os.path.join(output_dir, f'final_std_{dataset_key}_s1dm.nc')

    elif aggregation == 'median':
        output_dir = f'{output_dir}/{aggregation}/'
        # For median aggregation, load the median and quantile files
        file_paths['median_ids'] = os.path.join(output_dir, f'final_median_{dataset_key}_ids.nc')
        file_paths['quantile_25_ids'] = os.path.join(output_dir, f'quantile_25_{dataset_key}_ids.nc')
        file_paths['quantile_75_ids'] = os.path.join(output_dir, f'quantile_75_{dataset_key}_ids.nc')
        file_paths['median_s1dm'] = os.path.join(output_dir, f'final_median_{dataset_key}_s1dm.nc')
        file_paths['quantile_25_s1dm'] = os.path.join(output_dir, f'quantile_25_{dataset_key}_s1dm.nc')
        file_paths['quantile_75_s1dm'] = os.path.join(output_dir, f'quantile_75_{dataset_key}_s1dm.nc')

    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    return file_paths

def load_files_by_key_and_aggregation(dataset_key, aggregation, output_dir):
    """
    Load NetCDF files based on the dataset key (mean, max, min, series) and aggregation type (mean, median).
    
    Parameters:
    - dataset_key: The dataset key (mean, max, min, series).
    - aggregation: The aggregation type (mean, median).
    - output_dir: The directory where NetCDF files are stored.
    
    Returns:
    - A dictionary containing loaded datasets for both ids and s1dm.
    """
    file_paths = get_file_paths(dataset_key, aggregation, output_dir)
    loaded_datasets = {}

    # Load the datasets based on the file paths
    for key, path in file_paths.items():
        dataset = load_netcdf(path)
        if dataset:
            loaded_datasets[key] = dataset

    return loaded_datasets


# Configure logging at the module level
logger = logging.getLogger()

def setup_logging(dca):
    """Configure logging for the application."""
    log_file = f'{dca}_merging.log'  # Define log file based on the DCA variable
    logging.basicConfig(
        level=logging.INFO,  # Set the desired logging level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to the console
            logging.FileHandler(log_file)  # Logs to a file
        ]
    )


def main(dca, number):

    setup_logging(dca)
    # Load environment variables from the .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    # Retrieve and parse custom color settings from environment variables
    custom_colors_json = os.getenv('COLORS')
    custom_colors = parse_custom_colors(custom_colors_json)

    # Retrieve environment variables
    grid_folder = os.getenv('EQUI7_GRIDS')
    logger.info(f"Equi7 grids folder: {grid_folder}")

    # Check if the folder exists
    if not os.path.isdir(grid_folder):
        raise FileNotFoundError(f"The folder {grid_folder} does not exist.")

    # Retrieve the CRS (Coordinate Reference System) for Equi7 NA
    equi7_crs = os.getenv('EQUI7_NA_EPSG')

    # Ensure the 'REGION' environment variable is set
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

    # Format region ID as a two-digit string
    region_id = str(region).zfill(2)

    # Parameters for the grid
    resolution = 10
    pixel_size = 512

    ids_equi7_path = f"{os.getenv('RESULTS')}/region{region_id}_dca_filtered_ids_usda_polygons_espg_27705.shp"
    refdm_equi7_path = f"{os.getenv('RESULTS')}/radar_results/radar_enhanced_forest_disturbance_mapping_region_{region_id}_epsg_27705.shp"
    path_grid = f"{grid_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}.shp"
    path_conves = f"{os.getenv('RESULTS')}/radar_results/convex_hulls_refdm_region_{region_id}_epsg_4326.shp"
    path_intersetion_grid = f"{grid_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersetion.shp"

    figure_output_path = f"{os.getenv('FIGURES')}"
    if not os.path.exists(figure_output_path):
        os.makedirs(figure_output_path)

    s2_minicube_folder = os.getenv('SENTINEL2_CUBES_PP')
    logger.info(f"Sentinel 2 NetCDF folder: {s2_minicube_folder}")

    # logger.info("Load the Forest Disturbances ...")
    # logger.info("Intersected Grid ...")
    # intersected_grid = gpd.read_file(path_intersetion_grid)
    # logger.info("Adding Minicube index to ids ...")
    # ids_gdf = load_ids_dataset(ids_equi7_path)
    # ids_mincubes = add_minicube_index(intersected_grid, ids_gdf)
    # logger.info("Load Refdm ...")
    # s1dm = load_dissolved_refdm(refdm_equi7_path)
    # data = s1dm.copy()

    # filtered_df = data[data['DCA_ID'] == dca]
    # filtered_df = filtered_df[filtered_df['SURVEY_Y'] != 2016]
    # filtered_df = filtered_df[filtered_df['S1_YEAR'] != 2016]
    # filtered_df = filtered_df.reset_index()
    # logger.info(len(filtered_df))
    # rows_before = filtered_df[filtered_df['ID_E'] == number].index[0]
    # rows_after = len(filtered_df) - rows_before - 1
    # filtered_df = filtered_df[rows_before:-1]
    # logger.info(f"Events remaining: {rows_after}")

    # s1dm = s1dm.rename(columns={'minicube_i': 'mini_idx'})
    # ids_mincubes = ids_mincubes.rename(columns={'minicube_index': 'mini_idx'})

    output_dir = f'/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/timeseries/{dca}/'
    ensure_output_directory(output_dir)

    #process_all_ids(filtered_df, s1dm, ids_mincubes, dca, s2_minicube_folder, intersected_grid, equi7_crs, custom_colors, output_dir=output_dir)
    
    # agg='median'
    # new_output_dir = f'{output_dir}{agg}/'
    # calculate_aggregation(dca, 'median', input_dir=output_dir, output_dir=new_output_dir)
    # agg='mean'
    # new_output_dir = f'{output_dir}{agg}/'
    # calculate_aggregation(dca, 'mean', input_dir=output_dir, output_dir=new_output_dir)


    # # Define the dataset keys you want to iterate over
    # dataset_keys = ['mean', 'max', 'min', 'series']
    # aggregations = ['mean', 'median']  # Aggregations to loop through (mean, median)
    # Define the dataset keys you want to iterate over
    dataset_keys = ['max']
    aggregations = ['mean']  # Aggregations to loop through (mean, median)


    #input_dir_base = '/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/results/intermediate_results/'
    output_dir_base = f'{output_dir}/timeseries_merged_vegetation_indecies/'

    # Iterate over dataset keys and aggregations
    for dataset_key in dataset_keys:
        for aggregation in aggregations:
            input_dir = f'{output_dir}/'
            logger.info(input_dir)
            # Load files for the current dataset key and aggregation
            files = load_files_by_key_and_aggregation(dataset_key=dataset_key, aggregation=aggregation, output_dir=input_dir)
            logger.info(files)
            # Map aggregation types to their corresponding file keys
            var_file_key = f"{aggregation}_ids"  # e.g., 'mean_ids' or 'median_ids'
            var_file = files[var_file_key]  # Get the appropriate file based on the aggregation

            # Retrieve the variable names that do not start with 'B'
            # Retrieve the variable names that do not start with 'B' or 'S2'
            variable_names = [var for var in var_file.data_vars if not var.startswith('B') and not var.startswith('S2')]


            # Print the variable names (optional)
            print(f"Variable names that do not start with 'B' for {dataset_key} and {aggregation}:", variable_names)

            # Iterate over each variable name
            for var in variable_names:
                # Check if we are working with 'mean' or 'median' aggregation
                if aggregation == 'mean':

                    output_dir = f'{output_dir_base}/{aggregation}/{dataset_key}'
                    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

                    # Plot mean and standard deviation
                    plot_mean_and_std_with_trends(
                        final_mean_ids=files['mean_ids'],
                        final_std_ids=files['std_ids'],
                        final_mean_s1dm=files['mean_s1dm'],
                        final_std_s1dm=files['std_s1dm'],
                        var=var,
                        dca=dca,
                        custom_colors=custom_colors,
                        key=dataset_key,
                        output_dir=output_dir
                    )
                elif aggregation == 'median':

                    output_dir = f'{output_dir_base}{dca}/{aggregation}/{dataset_key}'
                    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

                    # Plot median and quantiles
                    plot_median_and_quantiles_with_trends(
                        final_median_ids=files['median_ids'],
                        quantile_25_ids=files['quantile_25_ids'],
                        quantile_75_ids=files['quantile_75_ids'],
                        final_median_s1dm=files['median_s1dm'],
                        quantile_25_s1dm=files['quantile_25_s1dm'],
                        quantile_75_s1dm=files['quantile_75_s1dm'],
                        var=var,
                        dca=dca,
                        custom_colors=custom_colors,
                        key=dataset_key,
                        output_dir=output_dir
                    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process DCA disturbance agent.")
    parser.add_argument('dca', type=str, help='Disturbance DCA agent (e.g., defoliators)')
    parser.add_argument('number', type=int, help='An integer number related to the disturbance agent')

    # Parse arguments
    args = parser.parse_args()
    
    # Call main with the parsed arguments
    main(args.dca, args.number)