import sys
import os
import geopandas as gpd
from shapely.geometry import box
import os
import logging
from pathlib import Path
import geopandas as gpd
from dotenv import load_dotenv
import concurrent.futures
import xarray as xr
import rioxarray
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Lock
from concurrent.futures import as_completed
from affine import Affine
import rasterio
from shapely.geometry import box, shape
import numpy as np
from func_data_preprocessing import extract_s1cd_filename_part, calculate_area_in_km2_s1cd
from func_file_io import load_data
import shutil
import pandas as pd
import geopandas as gpd
import time
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box, shape
import pandas as pd
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Global variables to hold environment values
region = None
region_id = None
tcc_dir = None
input_dir = None
output_dir = None
shapefile_dir = None
s1dm_dir = None
output_path_metadata = None
ids_usda_path = '/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/region_08_dca_filtered_ids_usda_polygons.shp'
buffer_years = 2
target_crs = "EPSG:4326"

def set_up_logging():
    """Set up logging to a file with timestamps for tracking the process."""
    logging.basicConfig(
        filename='log_s1cd_processor_2.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def load_env_variables(env_path):
    """Load required environment variables from a .env file."""
    global region, region_id, tcc_dir, input_dir, output_dir, shapefile_dir, s1dm_dir, output_path_metadata, ids_usda_path

    load_dotenv(dotenv_path=env_path)

    # Load environment variables and validate
    region = os.getenv('REGION')
    if not region:
        raise ValueError("The 'REGION' environment variable is not set.")
    region_id = str(region).zfill(2)

    tcc_dir = os.getenv('TCC_PATH')
    input_dir = os.getenv('SENTINEL1_TILES')
    output_dir = os.getenv('RESULTS')

    # Ensure all required paths are set
    if not all([tcc_dir, input_dir, output_dir]):
        raise ValueError("Missing required environment variables: TCC_PATH, SENTINEL1_TILES, or RESULTS")

    # Create required output directories
    shapefile_dir = Path(f"{output_dir}/03_s1cd_polygons")
    shapefile_dir.mkdir(parents=True, exist_ok=True)
    s1dm_dir = Path(f"{output_dir}/s1dm")
    s1dm_dir.mkdir(parents=True, exist_ok=True)
    output_path_metadata = Path(f"{output_dir}/metadata")
    output_path_metadata.mkdir(parents=True, exist_ok=True)
    ids_usda_path = '/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/results_clean/region_08_dca_filtered_ids_usda_polygons.shp'

    # Set target CRS (Coordinate Reference System)
    target_crs = "EPSG:4326"

def process_file(input_file, output_dir, env_path):
    """Process the input file by initializing environment variables, setting up logging, and running the extraction process."""
    load_env_variables(env_path)
    set_up_logging()

    logging.info(f"Processing file: {input_file}")

    try:
        if run_extraction_script(input_file, output_dir):
            logging.info(f"Successfully processed {input_file}")
            return True
        else:
            logging.error(f"Failed to process {input_file}")
            return False
    except Exception as e:
        logging.error(f"Error processing {input_file}: {e}")
        return False

def run_extraction_script(input_file, output_dir):
    """
    Run the extraction process for a single input file.
    Includes applying a TCC mask, extracting polygons, and filtering.
    """
    tcc_path = os.path.join(tcc_dir, f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_normalized_region_08.tif")

    try:
        # Extract polygons from raster and process
        if extract_polygons_from_raster(input_file, tcc_path, target_crs, output_dir):
            logging.info(f"Extraction successful for {input_file}")
            return True
        else:
            logging.error(f"Extraction failed for {input_file}")
            return False
    except Exception as e:
        logging.error(f"Error during extraction for {input_file}: {str(e)}")
        return False

def extract_polygons_from_raster(input_file, tcc_path, target_crs, output_dir):
    """Extract polygons from a raster file after applying a TCC mask."""
    try:
        s1_year = extract_year_from_s1cd_filename(input_file)
        filename = os.path.basename(input_file)
        logging.info(f"Processing file: {input_file} for year {s1_year}")

        # Load and preprocess the dataset
        logging.info(f"Load and preprocess the dataset")
        dataset_s1 = load_and_preprocess_dataset(input_file)
        
        # Apply TCC mask to dataset
        logging.info(f"Apply TCC mask to dataset")
        dataset_tcc = apply_tcc_mask(dataset_s1, tcc_path)
        
        # Extract polygons from mask
        logging.info(f"Extract polygons from mask")
        dataset = extract_polygons_from_mask(filename, dataset_tcc)

        # Process and filter polygons
        logging.info(f"Process and filter polygons")
        process_and_filter_polygons(input_file,dataset, s1_year, buffer_years, target_crs, output_dir)

        logging.info(f"Polygon extraction successful for {input_file}, saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error during polygon extraction for {input_file}: {e}")
        return False

def apply_tcc_mask(dataset, tcc_path):
    """
    Applies a Tree Canopy Cover (TCC) mask to the dataset.
    
    Parameters:
    - dataset (xarray.DataArray): The input dataset.
    - tcc_path (str): Path to the TCC 2017 raster file.
    
    Returns:
    - xarray.DataArray: The masked dataset.
    """
    logging.info("Opening TCC file...")
    try:
        tcc = rioxarray.open_rasterio(tcc_path)
    except Exception as e:
        logging.error(f"Error opening TCC file: {e}")
        return None
    
    # Extract spatial extent
    min_lon, max_lon = dataset['x'].min(), dataset['x'].max()
    min_lat, max_lat = dataset['y'].min(), dataset['y'].max()

    # Subset the TCC data
    logging.info("Selecting subset from TCC data...")
    subset = tcc.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))
    
    # Check if the subset is empty
    if subset.isnull().all():
        logging.error(f"Subset extraction for coordinates {min_lon}-{max_lon}, {max_lat}-{min_lat} is empty.")
        return None

    # Interpolate and apply mask
    normalized_subset_interp = subset.interp(x=dataset.coords['x'], y=dataset.coords['y'], method='nearest')
    masked_dataset = dataset.where(normalized_subset_interp > 0.3, 0).fillna(0)
    
    return masked_dataset

def extract_polygons_from_mask(filename, masked_data_array):
    """
    Extracts polygons from a masked data array and stores them in a GeoDataFrame 
    with year and tile metadata extracted from the filename.

    Parameters:
    - filename (str): The filename containing year and tile information.
    - masked_data_array (xarray.DataArray): The masked data array.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing extracted polygons with year and tile metadata.
    """
    logging.info(f"Extracting polygons from masked data array for {filename}.")
    
    year = int(filename.split('_year_')[-1].split('_')[0])
    tile_name = filename[13:23]  # Extract tile name based on naming convention

    results_gdf = gpd.GeoDataFrame(columns=['geometry', 'S1_YEAR', 'S1_TILE'], crs="EPSG:4326")
    min_x, max_x = masked_data_array['x'].min().item(), masked_data_array['x'].max().item()
    min_y, max_y = masked_data_array['y'].min().item(), masked_data_array['y'].max().item()
    
    # Create bounding box GeoDataFrame
    bounds_gdf = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs="EPSG:4326")

    cropped_mask = masked_data_array.squeeze("band")
    transform = Affine.translation(cropped_mask.x[0], cropped_mask.y[0]) * Affine.scale(cropped_mask.x[1] - cropped_mask.x[0], cropped_mask.y[1] - cropped_mask.y[0])

    binary_mask = (cropped_mask['layer'] > 0).astype(np.uint8)
    extracted_shapes = list(rasterio.features.shapes(binary_mask.values, transform=transform))
    polygons = [shape(geom) for geom, value in extracted_shapes if value == 1]

    polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=cropped_mask.spatial_ref)
    polygons_gdf['S1_YEAR'] = year
    polygons_gdf['S1_TILE'] = tile_name

    results_gdf = pd.concat([results_gdf, polygons_gdf], ignore_index=True)
    logging.info(f"Polygon extraction completed successfully for {filename}.")
    return results_gdf

def process_and_filter_polygons(file, dataset, s1_year, year_buffer, target_crs, output_dir):
    """
    Process USDA polygons, perform spatial joins, and filter out polygons larger than a threshold.
    Save the final result to a shapefile.
    """
    try:
        logging.debug(f"USDA path: {ids_usda_path}")
        ids_usda_gdf = load_data(ids_usda_path)
        ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(0.005)  # 500m buffer

        # # Überprüfen und Setzen des CRS
        # if ids_usda_gdf.crs is None:
        #     ids_usda_gdf.set_crs("EPSG:4326", inplace=True)  # Standard-CRS setzen, falls nicht definiert

        # # Reprojektieren in metrisches CRS
        # ids_usda_gdf = ids_usda_gdf.to_crs("EPSG:3857")  # Metrisches CRS für Buffer-Operationen

        # # Buffer-Operation (500m)
        # ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(500)

        # # Zurück zu WGS 84, falls erforderlich
        # ids_usda_gdf = ids_usda_gdf.to_crs("EPSG:4326")
    except Exception as e:
        logging.error(f"Error loading USDA polygons: {e}")
        return
    
    logging.info(f"Step 2: Filtering USDA polygons within ±{year_buffer} years of {s1_year}...")
    try:
        ids_usda_filtered = ids_usda_gdf[
            (ids_usda_gdf['SURVEY_Y'] >= s1_year - year_buffer) & 
            (ids_usda_gdf['SURVEY_Y'] <= s1_year + year_buffer)
        ]
        ids_area_gdf = calculate_area_in_km2_s1cd(ids_usda_filtered)
        ids_area = ids_area_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error filtering USDA polygons: {e}")
        return

    logging.info("Step 3: Calculating area before intersection...")

    polygons_gdf = dataset
    try:
        # Compute total area before intersection
        area_gdf = calculate_area_in_km2_s1cd(polygons_gdf)
        area_before_intersection = area_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error calculating area before intersection: {e}")
        return
    
    logging.info("Step 4: Performing spatial join to find intersecting polygons...")
    try:
        intersecting_gdf = gpd.sjoin(polygons_gdf, ids_usda_filtered, predicate='intersects')
        intersecting_gdf = intersecting_gdf.rename(columns={'index_right': 'S1CD_IDX'})
    except Exception as e:
        logging.error(f"Error during spatial join: {e}")
        return
    
    logging.info("Step 5: Calculating area after intersection...")
    try:
        # Compute total area after intersection
        #area_after_intersection = intersecting_gdf['geometry'].area.sum()
        area_after_gdf = calculate_area_in_km2_s1cd(intersecting_gdf)
        area_after_intersection = area_after_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error calculating area after intersection: {e}")
        return

    logging.info("Step 5,1: Add metadata to the table...")
    # Add metadata to the table
    tile_name = extract_s1cd_filename_part(os.path.basename(file))
    metadata_entries = [{
        'Tile': tile_name,
        'Year': s1_year,
        'IDS Area': ids_area,
        'Area Before Intersection': area_before_intersection,
        'Area After Intersection': area_after_intersection  
    }]
    metadata_df = pd.DataFrame(metadata_entries)

    # Define output path for the shapefile
    meta_path = os.path.join(output_path_metadata, f"metadata_table_{tile_name}.csv")

    # Save the GeoDataFrame as a shapefile
    metadata_df.to_csv(meta_path)
    logging.info(f"Metadata table saved as shapefile to {meta_path}")

    logging.info("Step 6: Aggregating geometries by USDA_IDX...")
    try:
        aggregated_gdf = intersecting_gdf.dissolve(by='IDX_D')
        aggregated_gdf.reset_index(inplace=True)
    except Exception as e:
        logging.error(f"Error during geometry aggregation: {e}")
        return

    logging.info("Step 7: Reprojecting and saving output shapefile...")
    try:
        aggregated_gdf.set_crs(epsg=4326, inplace=True)
        aggregated_gdf = aggregated_gdf.to_crs(target_crs)
        os.makedirs(shapefile_dir, exist_ok=True)
        shapefile_path = os.path.join(shapefile_dir, f"{tile_name}.shp")
        aggregated_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    except Exception as e:
        logging.error(f"Error saving shapefile: {e}")


def extract_year_from_s1cd_filename(input_path):
    """
    Extract the year from the filename, assuming a specific pattern '_year_'.
    Returns None if parsing fails.
    """
    try:
        filename = os.path.basename(input_path)
        return int(filename.split('_year_')[-1].split('_')[0])
    except (IndexError, ValueError) as e:
        logging.error(f"Error extracting year from filename {input_path}: {e}")
        return None

def drop_unnecessary_vars(dataset):
    """Drop unnecessary variables from the dataset."""
    return dataset.drop_vars(["x_bnds", "y_bnds"], errors='ignore')

def rename_variables(dataset):
    """Rename variables for consistency."""
    if 'unnamed' in dataset.variables:
        dataset = dataset.rename({'unnamed': 'layer'})
    if 'X' in dataset.variables and 'Y' in dataset.variables:
        dataset = dataset.rename({'X': 'x', 'Y': 'y'})
    return dataset

def reproject_to_wgs84(dataset):
    """Reproject the dataset to WGS 84 CRS."""
    
    crs_azimuthal_equidistant = "+proj=aeqd +lat_0=52 +lon_0=-97.5 +x_0=8264722.17686 +y_0=4867518.35323 +datum=WGS84 +units=m +no_defs"
    crs_wgs84 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    dataset.rio.write_crs(crs_azimuthal_equidistant, inplace=True)

    return dataset.rio.reproject(crs_wgs84)

def load_and_preprocess_dataset(input_file):
    """Load and preprocess the raster dataset."""
    dataset = xr.open_dataset(input_file)
    dataset = drop_unnecessary_vars(dataset)
    dataset = rename_variables(dataset)
    dataset = reproject_to_wgs84(dataset)
    return dataset 


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    env_path = sys.argv[3]
    env_path = '/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/environment/.env'  # falls die .env-Datei im gleichen Verzeichnis liegt


    process_file(input_file, output_dir, env_path)
