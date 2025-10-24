import ast
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib import cm
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import xarray as xr
from shapely.geometry import mapping, shape, MultiPolygon, box, Point
from affine import Affine
import rasterio
from shapely import wkt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches  # Import for custom legend
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm  # Import tqdm for the progress bar
import numpy as np
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap
import rioxarray
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
from scipy.stats import zscore
from scipy.stats import zscore
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter
from rioxarray.exceptions import NoDataInBounds  # Import NoDataInBounds exception
from pathlib import Path
import os
import itertools
import logging
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

# Set font sizes for various components
plt.rcParams.update({
    'font.size': 16,           # Global font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 24,      # X and Y label font size
    'xtick.labelsize': 16,     # X tick label font size
    'ytick.labelsize': 16,     # Y tick label font size
})

# Importing specific functions from the module
from func_preprocessing import restructure_dataset, remove_outliers, smooth
from func_indecies import ndvi, nbr, ndwi, ndre, tcw, tcg, tcb, ndmi, nirv, kndvi, drs, ndrs, kdrs, kndrs
from func_helper import parse_custom_colors, format_label
from func_file_io import load_data


def load_netcdf(filepath):
    """Load a NetCDF file using xarray."""
    if os.path.exists(filepath):
        return xr.open_dataset(filepath)
    else:
        print(f"File {filepath} not found.")
        return None

def get_unique_IDX_D(event_idx, data):
    return  data.iloc[event_idx]['IDX_D']

def subset(idx, data):
    
    # Select subsets from s1dm_gdf and ids_gdf based on IDX_D
    return data[data['IDX_D'] == idx]

import numpy as np
import pandas as pd
import xarray as xr

# Function to filter the data for the reference years
def filter_data_by_years(ids_mean_difference, s1dm_mean_difference, ref_year, ref_years_filter=1):
    """
    Filter IDS and S1DM data for the reference year and the following year.
    
    Args:
        ids_mean_difference (xarray.Dataset): IDS dataset.
        s1dm_mean_difference (xarray.Dataset): S1DM dataset.
        ref_year (int): Reference year for filtering.
    
    Returns:
        xarray.Dataset: Filtered IDS and S1DM datasets.
    """
    years_to_filter = [ref_year, ref_year + ref_years_filter]
    ids_filtered = ids_mean_difference.sel(time=ids_mean_difference['time'].dt.year.isin(years_to_filter))
    s1dm_filtered = s1dm_mean_difference.sel(time=s1dm_mean_difference['time'].dt.year.isin(years_to_filter))
    
    return ids_filtered, s1dm_filtered

# Function to compute the difference between IDS and S1DM for a given variable (NBR or NDVI)
def compute_difference(ids_filtered, s1dm_filtered, variable):
    """
    Compute the difference between IDS and S1DM for a given variable (NBR or NDVI).
    
    Args:
        ids_filtered (xarray.Dataset): Filtered IDS dataset.
        s1dm_filtered (xarray.Dataset): Filtered S1DM dataset.
        variable (str): The variable ('nbr' or 'ndvi') to compute the difference for.
    
    Returns:
        xarray.DataArray: Difference array for the selected variable.
    """
    difference = ids_filtered[variable] - s1dm_filtered[variable]
    return difference

# Function to find the timestamp with the maximum absolute difference for a given variable (NBR or NDVI)
def find_max_diff_time(difference, variable):
    """
    Find the timestamp with the maximum absolute difference for a given variable (NBR or NDVI).
    
    Args:
        difference (xarray.DataArray): Difference array for the selected variable.
        variable (str): The variable ('nbr' or 'ndvi') to compute the max difference for.
    
    Returns:
        np.datetime64: Timestamp with the largest absolute difference.
    """
    max_diff_time = difference.isel(
        time=difference.argmax(dim='time')
    ).time
    return max_diff_time

# Function to find the closest timestamp index in bbox_difference
def find_closest_time_index(bbox_time, target_time):
    """
    Find the index of the closest timestamp in the bbox dataset to the target timestamp.
    
    Args:
        bbox_time (numpy.ndarray): Array of timestamps in the bbox dataset.
        target_time (np.datetime64): The target timestamp to find the closest match for.
    
    Returns:
        int: Index of the closest timestamp.
    """
    target_timestamp = np.datetime64(target_time)
    
    # Calculate the absolute differences in time
    time_diffs = np.abs(bbox_time - target_timestamp)
    
    # Find the index of the smallest time difference
    closest_idx = np.argmin(time_diffs)
    
    return closest_idx

# Main function to get the closest timestamp index and difference for a specific variable (NBR or NDVI)
def get_max_diff_timestamp_index(ids_mean_difference, s1dm_mean_difference, bbox_difference, ref_year, ref_years_filter, variable):
    """
    Get the index of the closest timestamp with the largest difference for a specific variable (NBR or NDVI).
    
    Args:
        ids_mean_difference (xarray.Dataset): IDS dataset.
        s1dm_mean_difference (xarray.Dataset): S1DM dataset.
        bbox_difference (xarray.Dataset): The dataset containing the NBR and NDVI data.
        ref_year (int): The reference year for filtering.
        variable (str): The variable ('nbr' or 'ndvi') to compute the max difference for.
    
    Returns:
        dict: Dictionary containing the closest timestamp index and the maximum time difference.
    """
    # Step 1: Filter the data for the reference year
    ids_filtered, s1dm_filtered = filter_data_by_years(ids_mean_difference, s1dm_mean_difference, ref_year, ref_years_filter)
    
    # Step 2: Compute the difference for the selected variable
    difference = compute_difference(ids_filtered, s1dm_filtered, variable)
    
    # Step 3: Find the timestamp with the maximum absolute difference for the selected variable
    max_diff_time = find_max_diff_time(difference, variable)

    # Calculate the maximum difference value (optional, if you want the value itself)
    max_diff_value = difference.values.max()

    # Step 4: Find the closest timestamp index in the bbox dataset
    time_idx = find_closest_time_index(bbox_difference['time'].values, max_diff_time.values)
    
    # Output the results
    result = {
        'time_idx': time_idx,
        'max_diff_time': max_diff_time,
        'max_diff_value': max_diff_value
    }
    
    return result


import os
import matplotlib.pyplot as plt
import numpy as np

def plot_spatial_variable_with_boundaries(merged_bbox, ids, s1dm, time_index, variable, ax, fig):
    """
    Plot the data for a specific time index and overlay IDS and S1DM boundaries on the provided axis.
    
    Args:
        merged_bbox (xarray.DataArray): The dataset containing the data.
        ids (GeoDataFrame): IDS geometries.
        s1dm (GeoDataFrame): S1DM geometries.
        time_index (int): Index for the specific time to plot.
        variable (str): The variable to plot ('nbr' or 'ndvi').
        ax (matplotlib.axes.Axes): The axis to plot the spatial data.
    
    Returns:
        None
    """
    # Extract the timestamp from the merged_bbox
    timestamp = str(merged_bbox['time'].isel(time=time_index).values)
    date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d')  # Format as "YYYY-MM-DD"

    # Plot the data for the specified time index with a diverging colormap
    data = merged_bbox[variable].isel(time=time_index)
    im = data.plot(ax=ax, cmap='bwr', add_colorbar=False)

    # Add a colorbar and set the label
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label(f'', fontsize=0)


    # Overlay IDS and S1DM boundaries
    ids.boundary.plot(ax=ax, color='orange', linewidth=2, label='IDS')
    s1dm.boundary.plot(ax=ax, color='black', linewidth=2, label='S1DM')

    # Set plot title and labels
    ax.set_title(f"{date_str}", fontsize=22)
    ax.axis('off')  # Turn off the axes for better visual aesthetics

    # Add legend
    ax.legend(loc='upper right', fontsize=14)

# Function to plot the time series for NBR data
def plot_time_series(ax, ids_mean_difference, s1dm_mean_difference, ref_year, variable, dca, idx_d):
    """
    Plot the time series of NBR for both IDS and S1DM datasets on the provided axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot the time series data.
        ids_mean_difference (xarray.Dataset): IDS dataset containing NBR values.
        s1dm_mean_difference (xarray.Dataset): S1DM dataset containing NBR values.
        ref_year (int): The reference year to highlight in the time series plot.
        variable (str): The variable ('nbr' or 'ndvi') to plot.

    Returns:
        None
    """
    # Plot NBR for IDS and S1DM with thicker lines
    ids_mean_difference[variable].plot(ax=ax, label='IDS', color='orange', linewidth=2)
    s1dm_mean_difference[variable].plot(ax=ax, label='S1DM', color='b', linewidth=2)

    # Highlight the reference year
    ax.axvspan(np.datetime64(f'{ref_year}-01-01'), np.datetime64(f'{ref_year}-12-31'),
               color='grey', alpha=0.2, label=f"Reference Year: {ref_year}")

    # Set plot title with dynamic idx_d and dca, ensuring underscores are escaped properly and variables are uppercase
    ax.set_title(r"$\text{" + variable.upper() + "}_{\text{Anomaly}}$" + r" for $E_{\text{" + dca + "}}$ : " + r"$\mathit{" + idx_d.replace('_', r'\_')  + "}$", fontsize=22)

    ax.set_xlabel(r"$T_{\text{years}}$", fontsize=14)
    ax.set_ylabel(f"", fontsize=0)

    # Increase tick label size and reduce number of ticks on x and y axes
    ax.tick_params(axis='both', which='major', labelsize=16)  # Larger font size for ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce the number of y-axis ticks

    # Make gridlines transparent
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

    # Add legend
    ax.legend(fontsize=12)

# Main function to combine spatial plot and time series plot in one figure
def plot_spatial_and_timeseries(bbox_difference, ids, s1dm, ids_mean_difference, s1dm_mean_difference, ref_year, variable, dca, idx_d, time_index, save_path):
    """
    Plot the spatial plot of NBR data with IDS and S1DM boundaries on the left and the time series on the right, and save the figure.
    
    Args:
        bbox_difference (xarray.Dataset): Dataset containing NBR data for the spatial plot.
        ids (GeoDataFrame): IDS geometries for the boundary plot.
        s1dm (GeoDataFrame): S1DM geometries for the boundary plot.
        ids_mean_difference (xarray.Dataset): IDS dataset for the time series.
        s1dm_mean_difference (xarray.Dataset): S1DM dataset for the time series.
        ref_year (int): Reference year for the time series plot.
        variable (str): The variable ('nbr' or 'ndvi') to plot.
        time_index (int): The time index for the spatial plot.
        save_path (str): Path where the figure should be saved.
    
    Returns:
        None
    """
    # Create a figure with two subplots: one for the spatial plot and one for the time series plot
    fig, axes = plt.subplots(1, 2, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 2]})  # 1/3 for left plot, 2/3 for right plot

    # Plot the spatial plot on the left (index 0)
    plot_spatial_variable_with_boundaries(bbox_difference, ids, s1dm, time_index, variable, axes[0], fig)

    # Plot the time series on the right (index 1)
    plot_time_series(axes[1], ids_mean_difference, s1dm_mean_difference, ref_year, variable, dca, idx_d)

    # Adjust layout
    plt.subplots_adjust(wspace=0.15)  # Reduce the space between the plots

    # Add a central label between the two plots for the colorbar and y-axis label
    fig.text(0.38, 0.5, r"$\text{" + variable.upper() + "}" + r"_{\text{Anomaly}}$", ha='center', va='center', fontsize=20, rotation=90)

    # Save the figure as a PNG file
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

    #plt.show()
    # Close the figure properly
    plt.close(fig)
    
    print(f"Figure saved as: {save_path}")


def get_unique_minicube_FID(i, df, s1dm, ids, grid):
    """
    Get a DataFrame of unique FIDs from intersections of s1dm and ids with the grid.
    
    Parameters:
        i (int): The index to use for selecting the IDX_D value from df.
        df (DataFrame): DataFrame containing IDX_D column.
        s1dm (GeoDataFrame): GeoDataFrame with s1dm data and geometries.
        ids (GeoDataFrame): GeoDataFrame with ids data and geometries.
        grid (GeoDataFrame): GeoDataFrame representing the grid with FID and geometry columns.
    
    Returns:
        DataFrame: A DataFrame with unique FIDs from the intersections.
    """
    # Get the IDX_D value for the given index
    idx_d_value = df.loc[i, 'IDX_D']

    # Subset s1dm and ids based on IDX_D
    s1dm_subset = s1dm[s1dm['IDX_D'] == idx_d_value]
    ids_subset = ids[ids['IDX_D'] == idx_d_value]

    # Convert CRS to match
    s1dm_subset = s1dm_subset.to_crs(grid.crs)
    ids_subset = ids_subset.to_crs(grid.crs)

    # Perform spatial joins to get intersecting FIDs
    intersection = gpd.sjoin(s1dm_subset, grid[['FID', 'geometry']], how='inner', predicate='intersects')
    intersection_ids = gpd.sjoin(ids_subset, grid[['FID', 'geometry']], how='inner', predicate='intersects')

    # Extract unique FIDs from both intersections
    intersected_fids = intersection[['FID']].drop_duplicates().reset_index(drop=True)
    intersected_fids_ids = intersection_ids[['FID']].drop_duplicates().reset_index(drop=True)

    # Combine the two DataFrames and ensure uniqueness
    combined_fids = pd.concat([intersected_fids, intersected_fids_ids]).drop_duplicates().reset_index(drop=True)

    return combined_fids, idx_d_value


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

import numpy as np

def calculatePerfectSaison(mc, start_year, method='mean', percentile_value=90):

    """
    Calculate a perfect seasonal time series and compare it with the original time series.

    Parameters:
        mc (xarray.Dataset): The original time series dataset.
        start_year (int): The starting year for the seasonal calculation.
        method (str): Method for calculating seasonal values ('mean', 'max', 'min', 'median', or 'percentile').
        percentile_value (int or float): The percentile value (default is 90) for the percentile method.

    Returns:
        difference (xarray.Dataset): The difference between the perfect seasonal and original time series.
        perfect_seasonal (xarray.Dataset): The perfect seasonal time series.
        normal_timeseries (xarray.Dataset): The original time series after smoothing.
    """

    if mc is None or not isinstance(mc, xr.Dataset):
        raise ValueError("Input dataset 'mc' is invalid or None.")
    
    # General start of calculation
    print(f"Starting the calculation of {method} season...")

    # Define constants
    num_years = 8
    num_weeks_per_year = 52
    min_year = 2016  # We only calculate mean/min/max/median from 2017 onwards
    
    # Custom preprocessing to resample the original dataset to weekly frequency
    mc_reprocessed = mc.resample(time="1W").mean()

    # Filter the dataset for time after 2016 for aggregation
    mc_filtered = mc_reprocessed.sel(time=mc_reprocessed['time'].dt.year >= min_year)

    # Calculate the mean, max, min, median, or percentile values for each week of the year from 2017 onwards
    if method == 'mean':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).mean(dim='time')
    elif method == 'max':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).max(dim='time')
    elif method == 'min':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).min(dim='time')
    elif method == 'median':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).median(dim='time')
    elif method == 'percentile':
        # Handle the percentile calculation
        if not isinstance(percentile_value, (int, float)):
            raise ValueError("Percentile value must be an int or float.")
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).reduce(np.percentile, q=percentile_value, dim='time')
    else:
        raise ValueError("Invalid method. Use 'mean', 'max', 'min', 'median', or 'percentile'.")

    # Create an empty list to store datasets for each year
    yearly_datasets = []

    # Iterate over all years in the original dataset (not just after 2016)
    for year in range(start_year, mc_reprocessed['time'].dt.year.max().item() + 1):
        # Generate a list of datetime objects with weekly frequency for each year
        date_range = pd.date_range(start=f"{year}-01-01", periods=num_weeks_per_year, freq='W')

        # Repeat the mean/max/min/median/percentile values for each week
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
    
    print("Calculation completed.")

    return difference, perfect_seasonal, mc_reprocessed

from shapely.geometry import box
import geopandas as gpd
import pandas as pd

def generate_square_bbox_with_buffer(s1dm, ids, buffer=0.1):
    """
    Generate a square bounding box around the geometries of two GeoDataFrames
    (s1dm and ids) with an optional buffer.

    Args:
        s1dm (GeoDataFrame): The first GeoDataFrame.
        ids (GeoDataFrame): The second GeoDataFrame.
        buffer (float): The buffer to apply to the bounding box (as a fraction of its size).
                        Default is 0.1 (10% of the size).

    Returns:
        GeoDataFrame: A GeoDataFrame containing the square bounding box geometry.
    """
    # Concatenate the GeoDataFrames for IDS and S1DM
    combined_gdf = pd.concat([ids, s1dm], ignore_index=True)

    # Create a multipolygon by combining all geometries into one
    multipolygon = combined_gdf.geometry.unary_union

    # Get the bounding box of the multipolygon
    xmin, ymin, xmax, ymax = multipolygon.bounds

    # Calculate the width and height of the bounding box
    width = xmax - xmin
    height = ymax - ymin

    # Make the bounding box square by adjusting the larger dimension
    if width > height:
        offset = (width - height) / 2
        ymin -= offset
        ymax += offset
    elif height > width:
        offset = (height - width) / 2
        xmin -= offset
        xmax += offset

    # Apply a buffer to the square bounding box
    buffer_amount = max(width, height) * buffer
    xmin -= buffer_amount
    ymin -= buffer_amount
    xmax += buffer_amount
    ymax += buffer_amount

    # Create a square bounding box geometry
    square_bbox = box(xmin, ymin, xmax, ymax)

    # Save the square bounding box to a separate GeoDataFrame
    bbox_gdf = gpd.GeoDataFrame({'geometry': [square_bbox]}, crs=ids.crs)
    return bbox_gdf

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

def aggregate_group(group, aggregation_mode="earliest", reference_column="S1_YEAR"):
    """
    Aggregates a group of geometries and associated data.

    Parameters:
        group (DataFrame): The group to aggregate.
        aggregation_mode (str): "earliest" or "nearest".
            - "earliest": Uses the earliest S1_YEAR.
            - "nearest": Uses the S1_YEAR closest to SURVEY_Y.
        reference_column (str): The column to use as reference for aggregation, either "S1_YEAR" or "SURVEY_Y".

    Returns:
        pd.Series: Aggregated results for the group.
    """
    # Merge geometries
    merged_geometry = unary_union(group.geometry)

    if aggregation_mode == "earliest":
        # Use the earliest S1_YEAR
        chosen_year = group.S1_YEAR.min()
    elif aggregation_mode == "nearest":
        # Use the S1_YEAR closest to SURVEY_Y
        survey_year = group.iloc[0].SURVEY_Y  # Assuming all rows in the group have the same SURVEY_Y
        chosen_year = group.loc[(group.S1_YEAR - survey_year).abs().idxmin(), "S1_YEAR"]
    else:
        raise ValueError("Invalid aggregation_mode. Choose 'earliest' or 'nearest'.")

    # Keep all other columns by taking the first row's values
    other_columns = group.iloc[0].drop(["geometry", "S1_YEAR", "IDX_D"])  # Exclude these explicitly

    return pd.Series({"geometry": merged_geometry, reference_column: chosen_year, **other_columns.to_dict()})

# Perform the groupby operation with the desired aggregation mode

def aggregate_geodataframe(gdf, aggregation_mode="earliest"):
    """
    Aggregates a GeoDataFrame by IDX_D with a configurable aggregation mode.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame.
        aggregation_mode (str): "earliest" or "nearest".

    Returns:
        GeoDataFrame: Aggregated GeoDataFrame.
    """
    aggregated = (
        gdf.groupby("IDX_D").apply(lambda group: aggregate_group(group, aggregation_mode)).reset_index(drop=False)
    )
    return gpd.GeoDataFrame(aggregated, geometry="geometry", crs=gdf.crs)

# Define a function to group by and count based on a chosen reference year
def calculate_disturbance_counts(gdf, reference_year="S1_YEAR"):
    """
    Groups by DCA_ID and a chosen reference year, and counts the number of rows for each combination.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame.
        reference_year (str): The column to use for the year ("S1_YEAR" or "SURVEY_Y").

    Returns:
        DataFrame: Counts of disturbances grouped by DCA_ID and the reference year.
    """
    if reference_year not in gdf.columns:
        raise ValueError(f"Reference year column '{reference_year}' not found in the GeoDataFrame.")

    disturbance_counts = gdf.groupby(["DCA_ID", reference_year]).size()
    return disturbance_counts.reset_index(name="Count")


def process_geodataframe(
    gdf, 
    aggregation_mode="nearest", 
    reference_year_column="SURVEY_Y"
):
    """
    Prozessiert ein GeoDataFrame, indem es aggregiert und Störungen nach Jahr zählt.

    Args:
        gdf (GeoDataFrame): Das Eingabe-GeoDataFrame.
        aggregation_mode (str): Aggregationsmodus ('nearest' oder 'earliest').
        reference_year_column (str): Der Name der Spalte, die als Referenzjahr verwendet wird.

    Returns:
        DataFrame: DataFrame mit Störungen nach Jahr gezählt.
    """
    # Konvertiere die Spalten in numerische Datentypen
    gdf['S1_YEAR'] = pd.to_numeric(gdf['S1_YEAR'], errors='coerce')
    gdf['SURVEY_Y'] = pd.to_numeric(gdf['SURVEY_Y'], errors='coerce')

    # Überprüfe auf NaN-Werte, die durch fehlerhafte Konvertierungen entstehen können
    if gdf[['S1_YEAR', 'SURVEY_Y']].isnull().any().any():
        print("Warnung: Es gibt ungültige Werte in S1_YEAR oder SURVEY_Y!")

    # Aggregiere das GeoDataFrame
    aggregated_gdf = aggregate_geodataframe(gdf, aggregation_mode=aggregation_mode)

    # Zähle Störungen basierend auf dem Referenzjahr
    disturbance_counts_df = calculate_disturbance_counts(
        aggregated_gdf, reference_year=reference_year_column
    )

    # Ändere den Namen der Referenzjahr-Spalte in 'Year'
    disturbance_counts_df = disturbance_counts_df.rename(
        columns={reference_year_column: "Year"}
    )

    return disturbance_counts_df, aggregated_gdf


import logging
import os

# Create a logger
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the log directory exists
    logger = logging.getLogger('minicube_logger')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

def calculate_save_netcdf_events_anomalys(ids_gdf, s1dm_gdf, grid, data_path, figure_path, s2_minicube_folder, equi7_crs, tcc_file, logger):
    """
    Führt die Aggregation und Gruppierung mit mehreren Varianten durch und speichert die Ergebnisse.
    1. Aggregation 'earliest' mit Referenzjahr 'S1_YEAR'.
    2. Aggregation 'nearest' mit Referenzjahr 'S1_YEAR'.
    3. Aggregation 'nearest' mit Referenzjahr 'SURVEY_Y'.

    Args:
        ids_gdf (GeoDataFrame): Eine Liste von IDs oder Schlüsseln für die Verarbeitung.
        s1dm_gdf (GeoDataFrame): Das Eingabe-GeoDataFrame.
        data_path (str): Pfad zum Speichern der Daten.
        figure_path (str): Pfad zum Speichern der Abbildungen.
    """

    # List of aggregation modes and reference years
    aggregation_modes = ["earliest"] #"earliest", 
    reference_years = ['S1_YEAR', 'SURVEY_Y']
    dca_keys = ["defoliators", "wind", "bark_beetle"]#, "fire", "drought"]
    year_keys = list(range(2016, 2022))
    method_keys = ['percentile']
    variables = ['ndvi']

    tcc = rioxarray.open_rasterio(tcc_file)
    # Transform to the desired CRS (equi7_crs)
    if s1dm_gdf.crs != equi7_crs:
        s1dm_gdf = s1dm_gdf.to_crs(equi7_crs)
        logger.info(f"🌍 Transformed S1DM to CRS {equi7_crs}.")

    if ids_gdf.crs != equi7_crs:
        ids_gdf = ids_gdf.to_crs(equi7_crs)
        logger.info(f"🌍 Transformed IDS to CRS {equi7_crs}.")

    # Iterate over all combinations of aggregation modes and reference years
    for agg_mode in aggregation_modes:
        for ref_year in reference_years:
            # log_file = f'./logs/s1dm_s2_minicube_timeseries_agg_{agg_mode}_ref_{ref_year}.log'
            # logger = setup_logger(log_file)
            # Perform aggregation
            logger.info(f"Processing aggregation_mode={agg_mode} with reference_year={ref_year}...")
            
            disturbance_counts_df, s1dm_gdf_aggregated = process_geodataframe(
                s1dm_gdf, 
                aggregation_mode=agg_mode, 
                reference_year_column=ref_year
                )
            
            for dca_key, year_key in itertools.product(dca_keys, year_keys):
            
                logger.info("\n" + "=" * 70)
                logger.info(f"   🚀 Starting Processing for DCA_ID: {dca_key}")
                logger.info(f"   📅 Year: {year_key}")
                logger.info("=" * 70 + "\n")

                # Filter the GeoDataFrame based on DCA_ID and reference year
                gdf = s1dm_gdf_aggregated[
                    (s1dm_gdf_aggregated['DCA_ID'] == dca_key) &
                    (s1dm_gdf_aggregated[ref_year] == year_key)
                ]

                # Extract unique IDX_D values
                unique_idx_d_values = gdf['IDX_D'].unique()
                unique_idx_d_df = pd.DataFrame(unique_idx_d_values, columns=['IDX_D'])

                # Iterate through unique IDX_D values
                logger.info(f"🔍 Found {len(unique_idx_d_df)} unique IDX_D values for DCA_ID: {dca_key}")
                for idx, row in unique_idx_d_df.iterrows():
                    logger.info(f"   👉 Processing IDX_D {idx + 1}/{len(unique_idx_d_df)}: {row['IDX_D']}")
                    
                    try:
                        # Get minicubes for the group
                        cubes, idx_d = get_unique_minicube_FID(
                            i=idx,
                            df=unique_idx_d_df,
                            s1dm=s1dm_gdf,
                            ids=ids_gdf,
                            grid=grid
                            )

                        # Subset the data
                        s1dm = subset(idx_d, s1dm_gdf)
                        ids = subset(idx_d, ids_gdf)
                        square_bbox = generate_square_bbox_with_buffer(s1dm, ids, buffer=0.02)

                        # Validate geometries
                        if ids.geometry.is_empty.any() or not ids.geometry.is_valid.all():
                            logger.warning(f"⚠️ Invalid or missing geometries in IDS for {idx_d}. Skipping...")
                            continue  # Skip to next IDX_D
                        if s1dm.geometry.is_empty.any() or not s1dm.geometry.is_valid.all():
                            logger.warning(f"⚠️ Invalid or missing geometries in S1DM for {idx_d}. Skipping...")
                            continue  # Skip to next IDX_D

                        ids_mc, s1dm_mc, bbox_mc = [], [], []

                        # Process minicubes
                        for index, row in cubes.iterrows():
                            try:
                                i = row['FID']
                                logger.info(f"      🔄 Processing minicube FID {i}...")

                                path = f"{s2_minicube_folder}/{i}_10_512_20152024_equi7_NA.nc"
                                mc = load_netcdf(path)

                                if not mc.rio.crs:
                                    mc = mc.rio.write_crs(equi7_crs)

                                
                                min_lon, max_lon = mc['x'].min(), mc['x'].max()
                                min_lat, max_lat = mc['y'].min(), mc['y'].max()
                                tcc_subset = tcc.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))
                                normalized_subset_interp = tcc_subset.interp(x=mc.coords['x'], y=mc.coords['y'], method='nearest')

                                # Apply mask with NaN instead of 0
                                masked_mc = mc.where(normalized_subset_interp > 0.2)
                                logger.info(f"      🌳 TCC subset successfully processed and applied for FID {i}.")
                                # Check for required data variables
                                # if all(var in masked_mc.data_vars for var in ['nbr', 'ndvi', 'kndvi']):
                                #     masked_mc = masked_mc[['nbr', 'ndvi','kndvi']]
                                # else:
                                #     logger.warning(f"⚠️ Missing required variables ['nbr', 'ndvi', 'kndvi'] in FID {i}. Skipping...")
                                #     continue

                                # Clip IDS and S1DM geometries
                                clipped_ids, clipped_s1dm, clipped_bbox = None, None, None
                                ids_shape = gpd.GeoSeries(ids.geometry)
                                s1dm_shape = gpd.GeoSeries(s1dm.geometry)
                                square_bbox_shape = gpd.GeoSeries(square_bbox.geometry)

                                try:
                                    clipped_ids = masked_mc.rio.clip(ids_shape.geometry.apply(mapping), drop=True)
                                    if clipped_ids:
                                        ids_mc.append(clipped_ids)
                                    logger.info(f"         ✅ Successfully clipped IDS for FID {i}.")
                                except NoDataInBounds:
                                    logger.warning(f"         ⚠️ No data found in bounds for IDS in FID {i}. Skipping IDS.")

                                try:
                                    clipped_s1dm = masked_mc.rio.clip(s1dm_shape.geometry.apply(mapping), drop=True)
                                    if clipped_s1dm:
                                        s1dm_mc.append(clipped_s1dm)
                                    logger.info(f"         ✅ Successfully clipped S1DM for FID {i}.")
                                except NoDataInBounds:
                                    logger.warning(f"         ⚠️ No data found in bounds for S1DM in FID {i}. Skipping S1DM.")

                                try:
                                    clipped_bbox = masked_mc.rio.clip(square_bbox_shape.geometry.apply(mapping), drop=True)
                                    if clipped_bbox:
                                        bbox_mc.append(clipped_bbox) 
                                    logger.info(f"         ✅ Successfully clipped Box for FID {i}.")
                                except NoDataInBounds:
                                    logger.warning(f"         ⚠️ No data found in bounds for S1DM in FID {i}. Skipping Box.")

                                # Ensure 'time' dimension is present
                                if 'time' not in masked_mc.dims:
                                    logger.warning(f"⚠️ No 'time' dimension in NetCDF for FID {i}. Skipping...")
                                    continue

                            except Exception as e:
                                logger.warning(f"❌ Error processing FID {i}: {e}")
                                continue

                        # Log the count of clipped datasets
                        logger.info(f"   📊 Processed minicubes summary for IDX_D {idx_d}:")
                        logger.info(f"         📌 Nₖ(IDS) minicubes: {len(ids_mc)}")
                        logger.info(f"         📌 Nₖ(S1DM) minicubes count: {len(s1dm_mc)}")
                        logger.info(f"         📌 Nₖ(Box) minicubes count: {len(bbox_mc)}")

                        # Skip if no valid minicubes
                        if not ids_mc or not s1dm_mc:
                            logger.warning(f"⚠️ No valid data for IDX_D {idx_d}. Skipping...")
                            continue

                        # Merge datasets
                        try:
                            merged_ids = xr.merge(ids_mc)
                            merged_s1dm = xr.merge(s1dm_mc)
                            merged_bbox = xr.merge(bbox_mc)
                            logger.info(f"   ✅ Successfully merged datasets for IDX_D {idx_d}.")
                        except ValueError as e:
                            logger.error(f"❌ Failed to merge datasets for IDX_D {idx_d}: {e}")
                            continue
                        
                        # Loop through the season calculation methods
                        for mode in method_keys:
                            logger.info(f"   🌀 Calculating Perfect Season using method: {mode}")

                            # Define paths for saving data and figures
                            save_data_dir = os.path.join(data_path, f"Event_Timeseries_Anomalys/Ref_Col_{ref_year}/Aggregation_{agg_mode}/Anomaly_{mode}/{dca_key}/{year_key}/")
                            os.makedirs(save_data_dir, exist_ok=True)

                            save_fig_dir = os.path.join(figure_path, f"Event_Timeseries_Anomalys/Ref_Col_{ref_year}/Aggregation_{agg_mode}/Anomaly_{mode}/{dca_key}/{year_key}/")
                            os.makedirs(save_fig_dir, exist_ok=True)
                            try:
                                # Calculate Perfect Season for IDS and S1DM
                                ids_diff, ids_perfect, ids_mc_reprocessed = calculatePerfectSaison(merged_ids, 2016, mode)
                                s1dm_diff, s1dm_perfect, s1dm_mc_reprocessed = calculatePerfectSaison(merged_s1dm, 2016, mode)
                                bbox_diff, bbox_perfect, bbox_reprocessed = calculatePerfectSaison(merged_bbox, 2016, mode)

                                # Compute the mean differences over spatial dimensions and interpolate missing values
                                ids_mean_diff = ids_diff.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')
                                s1dm_mean_diff = s1dm_diff.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')

                                # Create output file paths based on method
                                output_path_ids = os.path.join(save_data_dir, f"ids_event_{idx_d}_{dca_key}_{year_key}_{mode}_diff.nc")
                                output_path_s1dm = os.path.join(save_data_dir, f"s1dm_event_{idx_d}_{dca_key}_{year_key}_{mode}_diff.nc")

                                # Save NetCDF files
                                logger.info(f"      📂 Saving IDS differences to: {output_path_ids}")
                                ids_mean_diff.to_netcdf(output_path_ids)

                                logger.info(f"      📂 Saving S1DM differences to: {output_path_s1dm}")
                                s1dm_mean_diff.to_netcdf(output_path_s1dm)

                                logger.info(f"   ✅ Successfully calculated and saved outputs for method: {mode}")

                                 # Plotting loop for each variable
                                for variable in variables:
                                    logger.info(f"   🔄 Processing variable: {variable}")
                                    result = get_max_diff_timestamp_index(ids_mean_diff, s1dm_mean_diff, bbox_diff, year_key,ref_years_filter=1, variable=variable)
                                    time_index = result['time_idx']
                                    logger.info(f"    📅 Index in {ref_year} for the largest {variable.upper()} difference: {time_index}")

                                    # Create output file paths based on method
                                    output_path_fig = os.path.join(save_fig_dir, f"timeseries_spatial_event_{idx_d}_{dca_key}_{ref_year}_{year_key}_{mode}_diff_{variable}.png")
                                    
                                    plot_spatial_and_timeseries(
                                        bbox_difference=bbox_diff,
                                        ids=ids,
                                        s1dm=s1dm,
                                        ids_mean_difference=ids_mean_diff,
                                        s1dm_mean_difference=s1dm_mean_diff,
                                        ref_year=year_key,
                                        variable=variable,
                                        dca=dca_key,
                                        idx_d=idx_d,
                                        time_index=time_index,
                                        save_path=output_path_fig
                                    )
                                    logger.info(f"    📂 📊 Saved plot for variable: {variable} at: {output_path_fig}")

                            except Exception as e:
                                logger.error(f"   ❌ Error calculating season for method {mode}: {e}")
                                continue

                    except Exception as e:
                        logger.error(f"❌ Critical error during processing of IDX_D {idx_d}: {e}")
                        continue


    print("Processing complete for all combinations.")


def main():
    # Load environment variables from the .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    #env_path = Path('/work/sy58xupo-cleaning/sy58xupo-CleaningSpace-1736389214/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    # Retrieve environment variables
    grid_folder = os.getenv('EQUI7_GRIDS_DIR')
    print(f"Equi7 grids folder: {grid_folder}")

    # Check if the folder exists
    if not os.path.isdir(grid_folder):
        raise FileNotFoundError(f"The folder {grid_folder} does not exist.")


    # Ensure the 'REGION' environment variable is set
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

    # Format region ID as a two-digit string
    region_id = str(region).zfill(2)

    # Parameters for the grid
    resolution = 10
    pixel_size = 512

    netcdf_data_path = os.getenv('WP1_DATA')
    print(f"Netcdf Data folder: {netcdf_data_path}")
    s1dm_s1_figure_path = os.getenv('WP1_FIGURE')
    print(f"Netcdf Figures folder: {s1dm_s1_figure_path}")

    figures_path = os.getenv('FIGURES_DIR')
    print(f"Figures folder: {figures_path}")

    ids_path = f"{os.getenv('RESULTS_DIR')}/region_{region_id}_dca_filtered_ids_usda_polygons.shp"
    # refdm_path = f"{os.getenv('RESULTS')}/radar_enhanced_forest_disturbance_mapping_region_{region_id}.shp"
    # path_grid = f"{grid_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}.shp"
    # path_conves = f"{os.getenv('RESULTS')}/radar_results/convex_hulls_refdm_region_{region_id}_epsg_4326.shp"
    #path_intersetion_grid = f"{grid_folder}/intersected_grid_{resolution}_{pixel_size}_region_{region_id}.shp"
   

    # Retrieve the CRS (Coordinate Reference System) for Equi7 NA
    equi7_crs = os.getenv('EQUI7_NA_EPSG')


    figure_output_path = f"{os.getenv('WP1_FIGURE')}/timeseries_tcc_masking/"
    if not os.path.exists(figure_output_path):
            os.makedirs(figure_output_path)

    data_output_path = f"{os.getenv('WP1_DATA')}/timeseries_tcc_masking/"
    if not os.path.exists(data_output_path):
            os.makedirs(data_output_path)
            
    tcc_file = f"{os.getenv('TCC_DIR')}/wp1_nlcd_tcc_conus_2016_20m_EPSG_4326_cropped_normalized_region_08.tif"

    s2_minicube_folder = os.getenv('SENTINEL2_CUBES_PP_DIR')
    print(f"Sentinel 2 NetCDF folder: {s2_minicube_folder}")

    

    spatial_buffer = [500]

    for buffer in spatial_buffer:
        print(f"Processing for buffer: {buffer}")
        path_intersetion_grid = f"{grid_folder}/grid_equi7_10_512_region_08_intersetion.shp"
        print("Load the Forest Disturbances ...")
        grid_gdf = load_data(path_intersetion_grid)
        
        # Reference file for the buffer
        s1dm_path = f"{os.getenv('RESULTS_DIR')}/radar_enhanced_forest_disturbance_mapping_region_{region_id}_buffer_{buffer}_s1dm.shp"

        s1dm_gdf = load_data(s1dm_path)
        ids_gdf = load_data(ids_path)
        

        # Create individual log file for each buffer
        log_file = f'./log_s1dm_s2_buffer_{buffer}.log'
        logger = setup_logger(log_file)
        
        # Log the length of refdm_gdf
        s1dm_length = len(s1dm_gdf)
        logger.info(f"Buffer: {buffer} - Length of refdm_gdf: {s1dm_length}")
        buffer_data_path = os.path.join(netcdf_data_path, f"buffer_{buffer}")
        buffer_figure_path = os.path.join(s1dm_s1_figure_path, f"buffer_{buffer}")

        
        # Continue processing as needed
        # Example: Additional operations and logging
        logger.info(f"Starting processing for buffer {buffer}")
        calculate_save_netcdf_events_anomalys(ids_gdf=ids_gdf, 
                                            s1dm_gdf=s1dm_gdf, 
                                            grid=grid_gdf, 
                                            data_path=buffer_data_path, 
                                            figure_path=s1dm_s1_figure_path,
                                            s2_minicube_folder=s2_minicube_folder, 
                                            equi7_crs=equi7_crs,
                                            tcc_file=tcc_file,
                                            logger=logger)




if __name__ == "__main__":
    
    main()