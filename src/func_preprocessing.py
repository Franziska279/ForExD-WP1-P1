import numpy as np
import xarray as xr
from scipy.stats import zscore
import bottleneck as bn
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from numcodecs import Blosc
import os
import xarray as xr

from func_indecies import (ndvi, nbr, ndwi, ndre, 
                          tcw, tcg, tcb, ndmi, nirv, 
                          kndvi, drs, ndrs, kdrs, kndrs)

def restructure_dataset(ds):
    # Assuming 'ds' is your dataset
    # Create a dictionary to store variables for each band
    variables = {}

    # Loop through each band and create a separate variable for it
    for band in ds.band.values:
        # Select the data for the current band and drop the 'band' coordinate
        variables[band] = ds['sentle'].sel(band=band).drop_vars('band')

    # Create a new dataset with time, x, and y as coordinates, and band data as variables
    ds_restructured = xr.Dataset(variables, coords={'time': ds['time'], 'x': ds['x'], 'y': ds['y']})

    # Check the structure of the new dataset
    return ds_restructured

def remove_outliers(ds, method='median', user_factor=2, z_pval=0.05):
    """
    Simplified function to remove outliers from xarray datasets using 'median' or 'zscore'.
    """
    print(f'Outlier removal method: {method} with a user factor of: {user_factor}')
    
    # Check if input is xarray Dataset and contains time dimension
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Input must be an xarray Dataset.')
    if 'time' not in ds.dims:
        raise ValueError('Dataset must have a time dimension.')
    if user_factor <= 0:
        raise ValueError('User factor must be greater than 0.')

    # Calculate the cutoff (standard deviation times user_factor) per pixel
    cutoffs = ds.std(dim='time') * user_factor

    if method == 'median':
        # Calculate a rolling median using bottleneck for faster performance
        num_years = len(ds['time']) / 365  # Assuming daily data
        win_size = max(3, int(num_years / 7))  # Ensure window size is at least 3 and odd
        if win_size % 2 == 0:
            win_size += 1

        print(f'Rolling window size: {win_size}')

        # Compute rolling median
        ds_med = ds.rolling(time=win_size, center=True, min_periods=1).construct('window').reduce(bn.nanmedian)

        # Calculate absolute difference between original data and rolling median
        ds_diffs = abs(ds - ds_med)

        # Identify outliers based on cutoff
        outlier_mask = ds_diffs > cutoffs

    elif method == 'zscore':
        crit_val = {0.01: 2.3263, 0.05: 1.6449, 0.1: 1.2816}.get(z_pval, None)
        if crit_val is None:
            raise ValueError('Invalid p-value for zscore method.')

        # Calculate z-scores for each pixel's time series
        zscores = xr.apply_ufunc(zscore, ds, input_core_dims=[['time']], vectorize=True)

        # Identify outliers based on z-scores
        outlier_mask = abs(zscores) > crit_val

    else:
        raise ValueError(f'Method {method} not supported.')

    # Replace outliers with NaNs
    ds_cleaned = ds.where(~outlier_mask, np.nan)

    # Notify user of NaN presence
    if ds_cleaned.isnull().any():
        print('> Warning: dataset contains NaN values.')

    print('> Outlier removal successful.\n')
    return ds_cleaned

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
        raise ValueError('Provided method not supported. Please use savitsky.')
        
    # check if any nans exist in dataset after resample and tell user
    if bool(ds.isnull().any()):
        print('> Warning: dataset contains nan values. You may want to interpolate next.')

    # notify user
    print('> Smoothing successful.\n')

    return ds


def preprocess_sentinel_data(idx, logger, path, output_path):
    """
    Preprocess Sentinel-2 data for a specific index.
    
    Parameters
    ----------
    idx: str
        The index to load and process data for.
    logger: logging.Logger
        The logger to record information during processing.
    path: str
        The path to the Sentinel-2 Zarr data.
    output_path: str
        The path to save the preprocessed NetCDF file.
    """
    
    ds = xr.open_zarr(path)
    
    # Restructure the dataset
    da = restructure_dataset(ds)    

    logger.info(f" > Compute various vegetation indices")
    # Compute various vegetation indices and add them to the dataset
    da['ndvi'] = ndvi(da)
    da['nbr'] = nbr(da)
    da['ndwi'] = ndwi(da)
    da['ndre'] = ndre(da)
    da['tcw'] = tcw(da)
    da['tcg'] = tcg(da)
    da['tcb'] = tcb(da)
    da['ndmi'] = ndmi(da)
    da['nirv'] = nirv(da)
    da['kndvi'] = kndvi(da, sigma=0.02)
    da['drs'] = drs(da)
    da['ndrs'] = ndrs(da)
    da['kdrs'] = kdrs(da, sigma=0.02)
    da['kndrs'] = kndrs(da)


    logger.info(" - Sort temporal axis")
    data = da.sortby('time')

    logger.info(f" > Remove outliers with user_factor=1.9")
    outliers = remove_outliers(data, method='median', user_factor=1.9, z_pval=0.05)

    logger.info(f" > Interpolate linearly over time")
    rechunked_data = outliers.chunk({'time': -1})
    # Perform interpolation
    interpolated_data = rechunked_data.interpolate_na(dim='time', method='linear')

    logger.info(f" > Smooth data with Savitzky-Golay (window_length=15, polyorder=3)")
    # Smooth the data using Savitzky-Golay filter
    smoothed_data = smooth(ds=interpolated_data, method='savitsky', window_length=15, polyorder=3)

    logger.info(f" > Compress data for saving with Blosc (lz4, complevel=9)")
    # Apply compression for saving, using Blosc for better performance
    compressor = Blosc(cname="lz4", clevel=9, shuffle=Blosc.SHUFFLE)
    encoding = {
        var: {
            "compressor": compressor,
            "write_empty_chunks": False
        } for var in smoothed_data.data_vars
    }

    logger.info(f" -- Encoding configuration: {encoding}")

    logger.info(f" - Chunk data for optimal storage")
    # Convert to Dask array with optimal chunk sizes
    smoothed_data = smoothed_data.chunk({'time': 30, 'y': 512, 'x': 512})
    
    logger.info(f"Saving data at: {output_path}")
    # Save the smoothed data to a NetCDF file
    smoothed_data.to_netcdf(output_path, mode='w')
    logger.info(f"Successfully saved at: {output_path}")
