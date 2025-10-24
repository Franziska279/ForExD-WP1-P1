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

import geopandas as gpd

def update_convex_with_minicubes(convex_path, minicube_bounds_file, output_path=None):
    """
    Update convex polygons with intersecting minicubes.

    Parameters:
    - convex_path: path to convex shapefile
    - minicube_bounds_file: path to minicube shapefile
    - output_path: optional path to save updated convex shapefile

    Returns:
    - convex_clean: GeoDataFrame with intersecting minicube info
    """
    # Load shapefiles
    minicube = gpd.read_file(minicube_bounds_file)
    convex = gpd.read_file(convex_path).to_crs(minicube.crs)

    # Ensure minicube has an index column
    if "FIA" not in minicube.columns:
        minicube = minicube.reset_index().rename(columns={"index": "FIA"})

    # Initialize column to store lists of intersecting minicube IDs
    convex["mini_FIA"] = pd.Series([None] * len(convex), dtype="object")

    # Iterate over convex rows and find overlapping minicubes
    for idx, row in convex.iterrows():
        intersects = minicube[minicube.geometry.intersects(row.geometry)]
        convex.at[idx, "mini_FIA"] = intersects["FIA"].tolist()

    # Drop polygons where fire or drought is True / nonzero
    convex_clean = convex[~convex['DCA_ID'].isin(["fire", "drought"])].copy()

    # Count rows where mini_FIA is empty or None
    no_intersection_count_clean = convex_clean["mini_FIA"].apply(lambda x: not x).sum()
    print(f"Number of convex polygons with no intersecting minicubes after removing fire/drought: "
          f"{no_intersection_count_clean} / {len(convex_clean)}")

    # Count the number of intersecting minicubes for each convex polygon
    convex_clean["num_intersections"] = convex_clean["mini_FIA"].apply(lambda x: len(x) if x else 0)

    # Count how many polygons have 0, 1, 2, 3, etc. intersections
    intersection_counts = convex_clean["num_intersections"].value_counts().sort_index()
    print("Number of convex polygons by number of intersecting minicubes:")
    print(intersection_counts)

    # Save updated convex file
    save_path = output_path if output_path else convex_path
    convex_clean.to_file(save_path)
    print(f"✅ Updated convex shapefile saved to {save_path}")

def contains_target_strlist(s, target):
    try:
        lst = ast.literal_eval(s)  # wandelt den String in eine echte Liste um
        return target in lst
    except:
        return False
    
def extract_ndvi_for_polygons(ndvi_cube, gdf_polygons):
    """
    Mask NDVI cube with the polygons in gdf_polygons.
    Returns a DataArray with only the pixels inside polygons.
    """
    geometries = [mapping(geom) for geom in gdf_polygons.geometry]
    ndvi_masked = ndvi_cube.rio.clip(geometries, all_touched=True, drop=False)
    return ndvi_masked

def crop_apply_tcc(minicube_row, dataset, tcc, equi7_crs):
    # Get the geometry of the minicube
    minicube_geom = minicube_row.geometry

    # Clip TCC to the minicube polygon
    tcc_crop = tcc.rio.clip([mapping(minicube_geom)], all_touched=True)


    #print("Cropped TCC:", tcc_crop)

    # 4️⃣ Apply TCC mask
    dataset.rio.write_crs(equi7_crs, inplace=True)

    # Optional: if you know the affine transform of the NDVI cube, assign it too
    # ndvi_cube.rio.write_transform(affine_transform, inplace=True)

    # Ensure TCC crop has CRS too
    tcc_crop.rio.write_crs(tcc.rio.crs, inplace=True)

    # Now reproject TCC to match NDVI cube
    tcc_aligned = tcc_crop.rio.reproject_match(dataset)

    # Apply TCC mask
    ndvi_cube_masked = dataset.where(tcc_aligned >= 0.3)
    
    return ndvi_cube_masked

# def save_or_update_nc(dataarray, minicube_name, out_path):
#     try:
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         if os.path.exists(out_path):
#             existing = xr.open_dataarray(out_path)
#             combined = xr.combine_by_coords([existing, dataarray], combine_attrs="override")
#             combined.to_netcdf(out_path, mode="w")
#             existing.close()
#             print(f"[Updated] Minicube {minicube_name}")
#         else:
#             dataarray.to_netcdf(out_path)
#             print(f"[Created] Minicube {minicube_name}")
#     except PermissionError:
#         print(f"[SKIPPED] Permission denied for {out_path}")


def save_or_update_nc(dataarray, idx_d, out_path):
    """
    Save NDVI snippet to NetCDF, merge spatially if file already exists.
    Replaces only new pixels/timesteps, keeps existing ones.
    Deletes old file and writes new.
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Optional: set folder permissions
        os.chmod(os.path.dirname(out_path), 0o777)

        if os.path.exists(out_path):
            # Load existing file
            existing = xr.open_dataarray(out_path)
            
            # Combine: replace new values, keep existing where not present
            combined = existing.combine_first(dataarray)
            
            # Remove old file first
            os.remove(out_path)
            
            # Save new combined
            combined.to_netcdf(out_path)
            existing.close()
            print(f"[Updated] Minicube {idx_d}")
        else:
            dataarray.to_netcdf(out_path)
            print(f"[Created] Minicube {idx_d}")
            
        # Optional: set permissions on new file
        #os.chmod(out_path, 0o666)

    except PermissionError:
        print(f"[SKIPPED] Permission denied for {out_path}")

import numpy as np

# def calculatePerfectSaison(mc, start_year, method='mean', percentile_value=90):

#     """
#     Calculate a perfect seasonal time series and compare it with the original time series.

#     Parameters:
#         mc (xarray.Dataset): The original time series dataset.
#         start_year (int): The starting year for the seasonal calculation.
#         method (str): Method for calculating seasonal values ('mean', 'max', 'min', 'median', or 'percentile').
#         percentile_value (int or float): The percentile value (default is 90) for the percentile method.

#     Returns:
#         difference (xarray.Dataset): The difference between the perfect seasonal and original time series.
#         perfect_seasonal (xarray.Dataset): The perfect seasonal time series.
#         normal_timeseries (xarray.Dataset): The original time series after smoothing.
#     """

#     if mc is None or not isinstance(mc, xr.Dataset):
#         raise ValueError("Input dataset 'mc' is invalid or None.")
    
#     # General start of calculation
#     print(f"Starting the calculation of {method} season...")

#     # Define constants
#     num_years = 8
#     num_weeks_per_year = 52
#     min_year = 2016  # We only calculate mean/min/max/median from 2017 onwards
    
#     # Custom preprocessing to resample the original dataset to weekly frequency
#     mc_reprocessed = mc.resample(time="1W").mean()

#     # Filter the dataset for time after 2016 for aggregation
#     mc_filtered = mc_reprocessed.sel(time=mc_reprocessed['time'].dt.year >= min_year)

#     # Calculate the mean, max, min, median, or percentile values for each week of the year from 2017 onwards
#     if method == 'mean':
#         ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).mean(dim='time')
#     elif method == 'max':
#         ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).max(dim='time')
#     elif method == 'min':
#         ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).min(dim='time')
#     elif method == 'median':
#         ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).median(dim='time')
#     elif method == 'percentile':
#         # Handle the percentile calculation
#         if not isinstance(percentile_value, (int, float)):
#             raise ValueError("Percentile value must be an int or float.")
#         ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).reduce(np.percentile, q=percentile_value, dim='time')
#     else:
#         raise ValueError("Invalid method. Use 'mean', 'max', 'min', 'median', or 'percentile'.")

#     # Create an empty list to store datasets for each year
#     yearly_datasets = []

#     # Iterate over all years in the original dataset (not just after 2016)
#     for year in range(start_year, mc_reprocessed['time'].dt.year.max().item() + 1):
#         # Generate a list of datetime objects with weekly frequency for each year
#         date_range = pd.date_range(start=f"{year}-01-01", periods=num_weeks_per_year, freq='W')

#         # Repeat the mean/max/min/median/percentile values for each week
#         weekly_values_repeated = ds_weekly_agg.isel(week=slice(0, num_weeks_per_year)).rename({'week': 'time'})

#         # Create a new dataset with the desired time values
#         new_time_dataset = xr.Dataset(
#             data_vars={
#                 'time': ('time', date_range)
#             }
#         )

#         # Convert the 'time' data of weekly_values_repeated to match the data type of new_time_dataset
#         weekly_values_repeated['time'] = new_time_dataset['time']

#         # Use combine_first to concatenate the new dataset with weekly_values_repeated while ignoring NaN values
#         yearly_dataset = weekly_values_repeated.combine_first(new_time_dataset)

#         # Append the yearly dataset to the list
#         yearly_datasets.append(yearly_dataset)

#     # Concatenate all the yearly datasets along the 'time' dimension with NaN values ignored
#     perfect_seasonal = xr.concat(yearly_datasets, dim='time')

#     # Calculate the difference between the perfect seasonal and the original time series (over the whole timespan)
#     difference = mc_reprocessed - perfect_seasonal
    
#     print("Calculation completed.")

#     return difference, perfect_seasonal, mc_reprocessed






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
    
    print(f"Starting the calculation of {method} season...")

    # Define constants
    num_weeks_per_year = 52
    min_year = 2017  # We only calculate mean/min/max/median from 2017 onwards
    
    # --- Resample original dataset to weekly frequency
    mc_reprocessed = mc.resample(time="1W").mean()

    # Fill missing weeks with forward fill, then backward fill (no NaNs)
    mc_reprocessed = mc_reprocessed.ffill(dim="time").bfill(dim="time")

    # --- Filter dataset for time after 2016 for aggregation
    mc_filtered = mc_reprocessed.sel(time=mc_reprocessed['time'].dt.year >= min_year)

    # --- Calculate weekly aggregates
    if method == 'mean':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).mean(dim='time')
    elif method == 'max':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).max(dim='time')
    elif method == 'min':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).min(dim='time')
    elif method == 'median':
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).median(dim='time')
    elif method == 'percentile':
        if not isinstance(percentile_value, (int, float)):
            raise ValueError("Percentile value must be an int or float.")
        ds_weekly_agg = mc_filtered.groupby(mc_filtered['time'].dt.isocalendar().week).reduce(
            np.percentile, q=percentile_value, dim='time'
        )
    else:
        raise ValueError("Invalid method. Use 'mean', 'max', 'min', 'median', or 'percentile'.")

    # --- Build perfect seasonal dataset year by year
    yearly_datasets = []
    for year in range(start_year, mc_reprocessed['time'].dt.year.max().item() + 1):
        # Generate fixed weekly dates
        date_range = pd.date_range(start=f"{year}-01-01", periods=num_weeks_per_year, freq='W')

        # Repeat seasonal weekly values across the year
        weekly_values_repeated = ds_weekly_agg.isel(week=slice(0, num_weeks_per_year)).rename({'week': 'time'})
        weekly_values_repeated = weekly_values_repeated.assign_coords(time=date_range)

        yearly_datasets.append(weekly_values_repeated)

    perfect_seasonal = xr.concat(yearly_datasets, dim='time')

    # --- Calculate difference
    difference = mc_reprocessed - perfect_seasonal
    
    print("Calculation completed.")
    return difference, perfect_seasonal, mc_reprocessed
