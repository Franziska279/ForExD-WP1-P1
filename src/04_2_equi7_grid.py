import geopandas as gpd
from sentle import sentle
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import logging
from dotenv import load_dotenv
import os
from pathlib import Path
import torch
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution
import sys, os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distributed.client")
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Assuming you have appropriate imports for `torch`, `load_sentle`, `restructure_preprocessing_data`, etc.

# Configure logging
logging.basicConfig(filename='processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global locks for progress bar updates
load_lock = Lock()
save_lock = Lock()

# Set up logging to only log to a file
logging.basicConfig(filename='minicube_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure no print statements go to the terminal
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)

def remove_outliers(ds, method='median', user_factor=2, z_pval=0.05):
    """
    Takes an xarray dataset and removes outliers within the timeseries on a per-pixel basis.
    Outliers are set to NaN. Works on datasets with or without existing NaN values.
    
    Parameters:
    -----------
    ds : xarray Dataset
        A multi-dimensional array with time, x, y, and band dimensions.
    method : str, optional
        Outlier detection method ('median' or 'zscore'). Default is 'median'.
    user_factor : float, optional
        Factor to multiply the threshold cutoff for detecting outliers. Default is 2.
    z_pval : float, optional
        P-value for the zscore method. Default is 0.05.
    
    Returns:
    --------
    ds : xarray Dataset
        The original dataset with detected outliers replaced with NaN.
    """
    
    # Notify user
    print(f"Outlier removal method: {method} with a user factor of: {user_factor}")
    
    # Check that input is a Dataset and has the necessary dimensions
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input is not an xarray Dataset.")
    
    if 'time' not in ds.dims:
        raise ValueError("Dataset does not contain a 'time' dimension.")
    
    if user_factor <= 0:
        raise ValueError("User factor must be greater than 0.")
    
    if method == 'zscore' and z_pval not in [0.1, 0.05, 0.01]:
        raise ValueError("Invalid z_pval value. Choose from 0.1, 0.05, or 0.01.")
    
    # Calculate cutoff values per pixel (std of pixel time-series multiplied by user_factor)
    cutoffs = ds.std('time') * user_factor
    
    if method == 'median':
        # Determine rolling window size
        num_years = len(ds['time']) / 365
        win_size = int(num_years / 7)  # Adjust this based on your data
        win_size = max(3, win_size + 1 if win_size % 2 == 0 else win_size)
        
        print(f"Generated rolling window size: {win_size}")
        
        # Calculate rolling median
        ds_med = ds.rolling(time=win_size, center=True).median()
        
        # Fill in NaN values in the rolling median with the original dataset
        ds_med = ds_med.where(~ds_med.isnull(), ds)
        
        # Calculate the absolute difference from the median
        ds_diff = abs(ds - ds_med)
        
        # Identify outliers where the difference exceeds the cutoff
        outlier_mask = ds_diff > cutoffs
        
    elif method == 'zscore':
        # Determine the critical z-score value based on p-value
        crit_val = {0.01: 2.3263, 0.05: 1.6449, 0.1: 1.2816}[z_pval]
        
        # Calculate z-scores along the time dimension
        zscores = ds.apply(zscore, nan_policy='omit', axis=0)
        
        # Identify outliers based on the z-score
        outlier_mask = abs(zscores) > crit_val
    
    # Shift data by one time step forward and backward for neighbor comparison
    left_neighbors = ds.shift(time=1)
    right_neighbors = ds.shift(time=-1)
    
    # Calculate mean and maximum of neighboring values
    neighbor_mean = (left_neighbors + right_neighbors) / 2
    neighbor_max = np.maximum(left_neighbors, right_neighbors)
    
    # Further refine the outlier mask using neighbor comparison
    outlier_mask = outlier_mask & ((ds < (neighbor_mean - cutoffs)) | (ds > (neighbor_max + cutoffs)))
    
    # Replace outliers with NaN in the original dataset
    ds = ds.where(~outlier_mask, np.nan)
    
    # Check for NaN values in the dataset after outlier removal
    if ds.isnull().any():
        print("> Warning: dataset contains NaN values. You may want to interpolate next.")
    
    # Notify user of successful outlier removal
    print("> Outlier removal successful.\n")
    
    return ds


def smooth(ds, method='savitsky', window_length=3, polyorder=1, sigma=1, mode='nearest'):  
    """
    Takes an xarray dataset containing vegetation index variable and smoothes timeseries
    timeseries on a per-pixel basis. The resulting dataset contains a smoother timeseries. 
    Recommended that no nan values present in dataset.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    method: str
        The smoothing algorithm to apply to the dataset. The savitsky method uses the robust
        savitsky-golay smooting technique, as per TIMESAT. Symmetrical gaussian applies a simple 
        symmetrical gaussian. Asymmetrical gaussian applies an asymmetrical gaussian, resulting in
        a flatter peak. Double logistic applies two seperate logistic functions to give a flatter 
        peak based on TIMESAT. Default is savitsky.
    window_length: int
        The length of the filter window (i.e., the number of coefficients). Value must 
        be a positive odd integer. The larger the window length, the smoother the dataset.
        Default value is 3 (as per TIMESAT).
    polyorder: int
        The order of the polynomial used to fit the samples. Must be a odd number (int) and
        less than window_length.
    sigma: int
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter 
        must be provided as a single number between 1-9.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset as input into the function, with smoothed data in the
        veg_index variable.
    """
    
    # notify user
    print('Smoothing method: {0} with window length: {1}, polyorder: {2}  and mode: {3}.'.format(method, window_length, polyorder, mode))
    
    # check if type is xr dataset
    if type(ds) != xr.Dataset:
        raise TypeError('> Not a dataset. Please provide a xarray dataset.')
        
    # check if time dimension is in dataset
    if 'time' not in list(ds.dims):
        raise ValueError('> Time dimension not in dataset. Please ensure dataset has a time dimension.')
    
    # # check if dataset contains veg_index variable
    # if 'veg_index' not in list(ds.data_vars):
    #     raise ValueError('> Vegetation index (veg_index) not in dataset. Please generate veg_index first.')
                        
    # check if dataset is 2D or above
    #if len(ds['s2_B01'].shape) == 1:
    #    raise Exception('> Remove outliers does not operate on 1D datasets. Ensure it has an x, y and time dimension.')
        
    # check if window length provided
    if window_length <= 0 or not isinstance(window_length, int):
        raise TypeError('> Window_length is <= 0 and/or not an integer. Please provide a value of 0 or above.')
        
    # check if user factor provided
    if polyorder <= 0 or not isinstance(polyorder, int):
        raise TypeError('> Polyorder is <= 0 and/or not an integer. Please provide a value of 0 or above.')
        
    # check if polyorder less than window_length
    if polyorder > window_length:
        raise TypeError('> Polyorder is > than window_length. Must be less than window_length.')
        
    # check if sigma is between 1 and 9
    if sigma < 1 or sigma > 9:
        raise TypeError('> Sigma is < 1 or > 9. Must be between 1 - 9.')
        
    # perform smoothing based on user selected method     
    if method in ['savitsky', 'symm_gaussian', 'asymm_gaussian', 'double_logistic']:
        if method == 'savitsky':
            
            # create savitsky smoother func
            def smoother(da, window_length, polyorder, mode):
                return da.apply(savgol_filter, mode=mode, window_length=window_length, polyorder=polyorder, axis=0)
            
            # create kwargs dict
            kwargs = {'window_length': window_length, 'polyorder': polyorder, 'mode': mode}

        elif method == 'symm_gaussian':
            
            # create gaussian smoother func
            def smoother(da, sigma):
                return da.apply(gaussian_filter, sigma=sigma)
            
            # create kwargs dict
            kwargs = {'sigma': sigma}

        elif method == 'asymm_gaussian':
            raise ValueError('> Asymmetrical gaussian not yet implemented.')
            
        elif method == 'double_logistic':
            raise ValueError('> Double logistic not yet implemented.')
                
        # create template and map func to dask chunks
        temp = xr.full_like(ds, fill_value=np.nan)
        ds = xr.map_blocks(smoother, ds, template=temp, kwargs=kwargs)
        
    else:
        raise ValueError('Provided method not supported. Please use savtisky.')
        
    # check if any nans exist in dataset after resample and tell user
    if bool(ds.isnull().any()):
        print('> Warning: dataset contains nan values. You may want to interpolate next.')

    # notify user
    print('> Smoothing successful.\n')

    return ds

def PSSRa (event):
    return event.B07 / event.B04

def rvi(event):
    return event.B08 / event.B04

def nbr(event):
    # Calculate the components that make up the NBR calculation
    band_diff = event.B08 - event.B12
    band_sum = event.B08 + event.B12

    # Calculate NBR and store it as a measurement in the original dataset
    return  band_diff / band_sum

def ndvi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum
    
def ndvi_re(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B8A - event.B04
    band_sum = event.B8A + event.B04

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum


def ndre(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B09 - event.B05
    band_sum = event.B09 + event.B05

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum


def ndwi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B03 - event.B08
    band_sum = event.B03 + event.B08

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

# Wetness = 0.1509 (Band 2) + 0.1973 (Band 3) + 0.3279 (Band 4) + 0.3406 (Band 8) – 0.7112 (Band 11) – 0.4572 (Band 12)
def tcw(event):
    tcw = 0.1509 * event.B02 + 0.1973 * event.B03 + 0.3279 * event.B04 + 0.3406 * event.B08 - 0.7112 * event.B11 - 0.4572 * event.B12
    return tcw

# Greenness = – 0.2848 (Band 2) – 0.2435 (Band 3) – 0.5436 (Band 4) + 0.7243 (Band 8) + 0.0840 (Band 11) – 0.1800 (Band 12)
def tcg(event):
    tcg = -0.2848 * event.B02 - 0.2435 * event.B03 - 0.5436 * event.B04 + 0.7243 * event.B08 + 0.0840 * event.B11 - 0.1800 * event.B12
    return tcg

# Brightness = 0.3037 (Band 2) + 0.2793 (Band 3) + 0.4743 (Band 4) + 0.5585 (Band 8) + 0.5082 (Band 11) + 0.1863 (Band 12)
def tcb(event):
    tcb = 0.3037 * event.B02 + 0.2793 * event.B03 + 0.4743 * event.B04 + 0.5585 * event.B08 + 0.5082 * event.B11 + 0.1863 * event.B12
    return tcb

def drs(event):
   
    red_band_power = event.B04 ** 2
    nir_band_power = event.B12 ** 2
    band_sqrt = np.sqrt(red_band_power + nir_band_power)

    return band_sqrt

def ndrs(event):

    drs_values = event.drs

    # Calculate the minimum and maximum values
    min_value = np.min(drs_values)
    max_value = np.max(drs_values)

    # Normalize the values to the range [0, 1]
    normalized_values = (drs_values - min_value) / (max_value - min_value)

    return normalized_values

def ndmi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B11
    band_sum = event.B08 + event.B11

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

def nirv(event, C=0.08):

    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04
    
    p2 = event.B08

    # Calculate NDVI and store it as a measurement in the original dataset
    ndvi = band_diff / band_sum

    nirv = (ndvi - C)* p2

    return nirv
 
def kndvi(event, sigma):
    # Extract red and near-infrared band values
    red_band_value = event.B04
    nir_band_value = event.B12
    
    # Calculate the squared difference
    squared_difference = (nir_band_value - red_band_value)
    
    # Calculate the divisor
    divisor = (2 * sigma)
    
    # Calculate the expression inside tanh
    expression = (squared_difference / divisor)**2
    
    # Calculate and return the kndvi using the hyperbolic tangent
    kndvi = np.tanh(expression)
    
    return kndvi

def kndvi05(event):

    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04

    mid = band_diff / band_sum

    tan = np.tanh(mid ** 2) 
    
    return tan

def k(r, s, sigma):
    return np.exp(-((r - s) ** 2) / (2 * sigma ** 2))


def kdrs(event, sigma):
    red = event.B04
    swir = event.B12

    return 2 * (1 + k(red, swir, sigma))


def kndrs(event):

    kDRS_values = event.kdrs

    # Calculate the minimum and maximum values
    min_value = np.min(kDRS_values)
    max_value = np.max(kDRS_values)

    # Normalize the values to the range [0, 1]
    normalized_values = (kDRS_values - min_value) / (max_value - min_value)

    return normalized_values

def restructure_preprocessing_data(ds):
    """
    Process and preprocess a dataset containing satellite data.
    
    This function performs several preprocessing steps, including:
    - Restructuring the dataset
    - Removing outliers
    - Computing various vegetation indices
    - Sorting and rechunking the dataset
    - Interpolating missing values
    - Smoothing the data
    - Calculating anomalies
    - Renaming dimensions
    - Sorting variables alphabetically

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing satellite data with multiple bands and a 'time' dimension.

    Returns
    -------
    xarray.Dataset
        The preprocessed dataset with various vegetation indices, anomalies, and adjusted dimensions.
    """
    
    # Create a dictionary to store the new variables
    variables = {}

    # Extract the band values as a variable
    band_values = ds.coords['band'].values

    # Iterate over each band and extract corresponding data
    for band in band_values:
        # Extract the DataArray for the current band
        band_data = ds.sel(band=band)
        
        # Add the band data to the dictionary with the band name as the key
        variables[band] = band_data

    # Create a new Dataset from the dictionary of variables
    new_ds = xr.Dataset(
        variables,
        coords={'time': ds.coords['time'], 'x': ds.coords['x'], 'y': ds.coords['y'], 'band': ('band', band_values)}
    )

    # Drop the 'band' coordinate from the dataset
    new_ds = new_ds.drop_vars('band')

    # Remove outliers using a specified method
    print("Removing outliers")
    ds = remove_outliers(new_ds, method='median', user_factor=2, z_pval=0.07)

    # Compute various vegetation indices and add them to the dataset
    print("Computing vegetation indices")
    ds['ndvi'] = ndvi(ds)
    ds['nbr'] = nbr(ds)
    ds['ndwi'] = ndwi(ds)
    ds['ndre'] = ndre(ds)
    ds['tcw'] = tcw(ds)
    ds['tcg'] = tcg(ds)
    ds['tcb'] = tcb(ds)
    ds['ndmi'] = ndmi(ds)
    ds['nirv'] = nirv(ds)
    ds['kndvi'] = kndvi(ds, sigma=0.02)
    ds['drs'] = drs(ds)
    ds['ndrs'] = ndrs(ds)
    ds['kdrs'] = kdrs(ds, sigma=0.02)
    ds['kndrs'] = kndrs(ds)

    # Sort the dataset by the time dimension
    print("Sorting dataset by time")
    ds = ds.sortby('time')

    # Rechunk the dataset along the 'time' dimension
    print("Rechunking dataset")
    ds = ds.chunk({'time': -1})

    # Interpolate missing values linearly over time
    print("Interpolating missing values")
    interpolated_data = ds.interpolate_na(dim='time', method='linear')

    # Smooth the data
    print("Smoothing data")
    smoothed_data = smooth(interpolated_data, method='savitsky', window_length=7, polyorder=2)

    # Rechunk the smoothed data
    rechunked_data = smoothed_data.chunk({'time': 1, 'x': 256, 'y': 256})


    # Sort variables alphabetically
    print("Sorting variables alphabetically")
    sorted_vars = sorted(rechunked_data.data_vars)
    sorted_ds = rechunked_data[sorted_vars]

    # Save the preprocessed data to a NetCDF file
    print(f"Compress data for saving to complevel=9")
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in sorted_ds.data_vars}
    print(encoding)

    print("Processing complete")

    return sorted_ds


def load_sentle(grid_path, idx, res):
    """
    Load Sentinel data using the Sentle library for a given grid.
    """
    intersected_gdf_equi7 = gpd.read_file(grid_path)
    bounds = idx + 1
    bounds = intersected_gdf_equi7[idx:bounds].geometry.iloc[0].bounds
    bound_left = int(bounds[0])
    bound_bottom = int(bounds[1])
    bound_right = int(bounds[2])
    bound_top = int(bounds[3])
    equi7_crs = intersected_gdf_equi7.crs
    print(f"Resolution: {res}")

    da = sentle.process(
        target_crs=equi7_crs,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        datetime="2015-01-01/2024-07-31",
        target_resolution=res,
        S2_mask_snow=True,
        S2_cloud_classification=True,
        S2_cloud_classification_device="cuda",
        S1_assets=["vv", "vh"],
        S2_apply_snow_mask=True,
        S2_apply_cloud_mask=True,
        time_composite_freq="7d",
        num_workers=40,
    )
    return da



def save_minicube(ds, output_zarr_path, idx):
    """
    Function to save the Minicube data to a zarr file.
    This will run in parallel using ThreadPoolExecutor.
    """
    try:
        ds.to_zarr(output_zarr_path)
        return True  # Indicate success
    except Exception as e:
        logger.error(f"An error occurred while saving Minicube {idx}: {e}", exc_info=True)
        return False  # Indicate failure


def main():

    #sys.stdout = open(os.devnull, 'w')
    try:
        # Load the Environment variables
        env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
        
        res = 10
        grid_path=f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{res}_512.shp"

        start_idx = 1
        end_idx = 2
        for idx in range(start_idx, end_idx + 1):
            print(f"Load the Minicube {idx} ...\n")
            da = load_sentle(grid_path=grid_path, idx = idx, res=res)
            print("Preprocess the Minicube ...\n")
            ds = restructure_preprocessing_data(da)
            print("Save the Minicube ...\n")
            # sentle.save_as_zarr(da, path="/net/projects/forexd/WP1/Data/S2_Cubes_IDS_R8/20_res_minicube_test.zarr")
            output_zarr_path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_{res}_512_20152024_equi7_NA_corr.zarr"
            ds.to_zarr(output_zarr_path)
            print(f"Sucessfully saved the Minicube {idx} at {output_zarr_path} ...")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()