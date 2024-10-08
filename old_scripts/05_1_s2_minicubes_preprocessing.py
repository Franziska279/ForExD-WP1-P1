import os
import sys
import logging
import geopandas as gpd
import xarray as xr
from mypythonlib import myfunctions, phenolopy
import os
import xarray as xr
from tqdm import tqdm
import xarray as xr
import os
import sys
import logging  
import geopandas as gpd

# Configure logging
logging.basicConfig(filename='minicube_preprocessing.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def generate_output_folder(gdf, idx, dir):
    disturbance_type = gdf.dist_type.iloc[idx]
    year = gdf.year.iloc[idx]
    output_folder = f"/{dir}/{disturbance_type}/{year}/{idx}/"
    os.makedirs(output_folder, exist_ok=True)
    input_nc = f"/{dir}/{disturbance_type}/{year}/{idx}/{idx}_merged.nc"
    output_filename = f"{idx}_{year}_{disturbance_type}.nc"

    return output_folder, input_nc, output_filename



def preprocess_and_reduce_minicube(ds, idx, output_folder, output_filename):

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

    # Smooth the data using the Savitzky-Golay filter
    #resampled_data = phenolopy.resample(interpolated_data, interval='1W', reducer='mean')

    print(f"Smooth data with Savitzky-Golay (window_length=15, polyorder=3)")
    # Smooth the data using the Savitzky-Golay filter
    smoothed_data = phenolopy.smooth(ds=interpolated_data, method='savitsky', window_length=15, polyorder=3)

    print(f"Calculate Anomaly for each variable")
    for var in smoothed_data.data_vars:
            print(var)
            smoothed_data[var+'_anom']     = smoothed_data[var].groupby('time.week')-smoothed_data[var].groupby('time.week').median()

    # Rename dimensions from lat/lon to x/y
    smoothed_data = smoothed_data.rename_vars({'lat': 'y', 'lon': 'x'})

    # Check if CRS is None and assign if needed
    if smoothed_data.rio.crs is None:
        smoothed_data.rio.write_crs("epsg:4326", inplace=True)
    # Set the CRS for the xarray dataset
    smoothed_data = smoothed_data.rio.write_crs("epsg:4326", inplace=True)

    # Save the preprocessed data to a NetCDF file
    print(f"Compress data for saving to complevel=9")
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in smoothed_data.data_vars}
    print(encoding)
    outputpath = os.path.join(output_folder, output_filename)
    smoothed_data.to_netcdf(outputpath, encoding=encoding)

    print(f"Saved to path = {outputpath}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python minicube_download_preprocessing_pipeline.py SLURM_ARRAY_TASK_ID INPUT_FILE OUTPUT_DIR")
        sys.exit(1)

    slurm_array_task_id = sys.argv[1]
    input_file = sys.argv[2]
    input_dir = sys.argv[3]

    logging.info(f"Running Task ID {slurm_array_task_id}")

    index = int(slurm_array_task_id)
    gdf_grid = gpd.read_file(input_file)

    output_folder, file_path_nc, output_filename = generate_output_folder(gdf_grid, index, input_dir)
    merged_ds = xr.open_dataset(file_path_nc)
    preprocess_and_reduce_minicube(merged_ds, index, output_folder, output_filename)

    logging.info('Done')

if __name__ == "__main__":
    main()
