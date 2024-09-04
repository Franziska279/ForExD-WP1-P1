import os
import netCDF4
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm

def get_nc_file_boundaries(root_dir, output_file):
    geometries = []
    filenames = []
    errors = 0  # Counter for errors

    # Get a list of all .nc files
    nc_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(root_dir) for file in files if file.endswith(".nc")]

    # Loop through the files with a progress bar
    for file_path in tqdm(nc_files, desc="Processing .nc files"):
        filename = Path(file_path).name
        if "_merged" in filename:
            continue  # Skip files with '_merged' in their name
        try:
            with xr.open_dataset(file_path) as ds:
                x_min = ds['x'].min().item()
                x_max = ds['x'].max().item()
                y_min = ds['y'].min().item()
                y_max = ds['y'].max().item()
                geom = box(x_min, y_min, x_max, y_max)
                geometries.append(geom)
                filenames.append(filename)
        except Exception as e:
            errors += 1
            print(f"Error processing file {file_path}: {e}")

    gdf = gpd.GeoDataFrame({'geometry': geometries, 'filename': filenames}, crs="EPSG:4326")
    gdf.to_file(output_file)

    print(f"Processing completed with {errors} errors.")



# Function to extract year, dist_type, and USDA_IDX from filename
def extract_info(filename):
    # Remove the '.nc' extension
    base_name = filename.replace('.nc', '')
    parts = base_name.split('_')
    
    if len(parts) < 3:
        return None, None, None
    
    USDA_IDX = base_name
    idx = parts[0]
    year = parts[1]
    dist_type = '_'.join(parts[2:])
    
    return USDA_IDX, idx, year, dist_type


def main():
    print("Starting main function...")

    folder_dir = '/Net/Groups/BGI/scratch/fmueller/Data/s2_region8_nc_256px_vi/'
    result_path = '/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/s2_minicube_bounderies_all.shp'
    
    print("Calling get_nc_file_boundaries...")
    get_nc_file_boundaries(folder_dir, result_path)
    
    print(f"Reading file from {result_path}...")
    gdf = gpd.read_file(result_path)

    print("Dropping rows with 'merged' in the filename...")
    # gdf = gdf[~gdf['filename'].str.contains('merged', na=False)]

    # print(f"Remaining rows after dropping: {len(gdf)}")

    print("Extracting USDA_IDX, year, and dist_type from filename...")
    gdf['USDA_IDX'], gdf['idx'], gdf['year'], gdf['dist_type'] = zip(*gdf['filename'].apply(extract_info))

    print("Renaming 'filename' column to 'cube_fn'...")
    gdf.rename(columns={'filename': 'cube_fn'}, inplace=True)

    print("Saving updated GeoDataFrame...")
    gdf.to_file(result_path)

    print("Finished")

if __name__ == "__main__":
    main()