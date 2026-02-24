import geopandas as gpd
from shapely.geometry import box, shape
import rasterio.features
import numpy as np
import os
import logging
import xarray as xr
import rioxarray
import pandas as pd
from tqdm import tqdm
from func_file_io import load_data
from affine import Affine
from pyproj import Transformer
from func_data_preprocessing import extract_s1cd_filename_part, calculate_area_in_km2_s1cd

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

def drop_unnecessary_vars(data):
    """Drop unnecessary variables from the dataset."""
    data = data.drop_vars(["x_bnds", "y_bnds"], errors='ignore')
    return data

def rename_variables(data):
    """Rename variables for consistency."""
    if 'unnamed' in data.variables:
        data = data.rename({'unnamed': 'layer'})
    if 'X' in data.variables and 'Y' in data.variables:
        data = data.rename({'X': 'x', 'Y': 'y'})
    return data

def reproject_to_wgs84(data):
    """Reproject the dataset to WGS 84 CRS."""
    
    crs_azimuthal_equidistant = "+proj=aeqd +lat_0=52 +lon_0=-97.5 +x_0=8264722.17686 +y_0=4867518.35323 +datum=WGS84 +units=m +no_defs"
    crs_wgs84 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    data.rio.write_crs(crs_azimuthal_equidistant, inplace=True)

    return data.rio.reproject(crs_wgs84)

def load_and_preprocess_dataset(input_file):
    """Load and preprocess the raster dataset."""
    dataset = xr.open_dataset(input_file)
    dataset = drop_unnecessary_vars(dataset)
    dataset = rename_variables(dataset)
    dataset = reproject_to_wgs84(dataset)
    return dataset 

def apply_tcc_mask(dataset, tcc_raster_path, threshold):
    """
    Applies a Tree Canopy Cover (TCC) mask to the dataset based on the TCC raster file.

    The function subsets the TCC raster based on the dataset's spatial extent, interpolates 
    the TCC data to match the dataset's grid, and applies a mask to retain only the areas 
    where the TCC value exceeds a specified threshold.

    Parameters:
    - dataset (xarray.DataArray): The input dataset to be masked.
    - tcc_raster_path (str): Path to the TCC 2017 raster file.

    Returns:
    - xarray.DataArray: The masked dataset where TCC values are above the threshold.
      Returns `None` if an error occurs at any step.
    """
    
    # Step 1: Load the TCC raster data from the provided file path
    logging.info("Loading TCC raster data from: %s", tcc_raster_path)
    
    try:
        tcc_data = rioxarray.open_rasterio(tcc_raster_path)
    except Exception as e:
        logging.error("Failed to open TCC raster: %s", e)
        return None

      # CRS of both datasets
    ds_crs = dataset.rio.crs
    tcc_crs = tcc_data.rio.crs

    if ds_crs is None or tcc_crs is None:
        logging.error("Missing CRS information in dataset or TCC raster.")
        return None

    # Extract bounds from dataset
    dataset_lon_min = float(dataset['x'].min())
    dataset_lon_max = float(dataset['x'].max())
    dataset_lat_min = float(dataset['y'].min())
    dataset_lat_max = float(dataset['y'].max())

    logging.info("Dataset bounds (original CRS %s): x=[%f,%f], y=[%f,%f]",
                 ds_crs, dataset_lon_min, dataset_lon_max,
                 dataset_lat_min, dataset_lat_max)

    # Transform bounds into TCC CRS
    transformer = Transformer.from_crs(ds_crs, tcc_crs, always_xy=True)

    x_min_tcc, y_min_tcc = transformer.transform(dataset_lon_min, dataset_lat_min)
    x_max_tcc, y_max_tcc = transformer.transform(dataset_lon_max, dataset_lat_max)
    x_min_tcc2, y_max_tcc2 = transformer.transform(dataset_lon_min, dataset_lat_max)
    x_max_tcc2, y_min_tcc2 = transformer.transform(dataset_lon_max, dataset_lat_min)

    # Get overall bounding box (since reprojection can flip axes)
    x_min = min(x_min_tcc, x_min_tcc2, x_max_tcc, x_max_tcc2)
    x_max = max(x_min_tcc, x_min_tcc2, x_max_tcc, x_max_tcc2)
    y_min = min(y_min_tcc, y_min_tcc2, y_max_tcc, y_max_tcc2)
    y_max = max(y_min_tcc, y_min_tcc2, y_max_tcc, y_max_tcc2)

    logging.info("Transformed bounds in TCC CRS (%s): x=[%f,%f], y=[%f,%f]",
                 tcc_crs, x_min, x_max, y_min, y_max)

     # Subset TCC raster in its CRS
    try:
        tcc_subset = tcc_data.sel(
            x=slice(x_min, x_max),
            y=slice(y_max, y_min)  # note reversed y order
        )
    except Exception as e:
        logging.error("Failed to subset TCC raster: %s", e)
        return None

    if tcc_subset.rio.bounds() is None or tcc_subset.size == 0:
        logging.error("TCC subset is empty after reprojection.")
        return None

    # Reproject TCC subset to dataset CRS & grid
    try:
        tcc_reproj = tcc_subset.rio.reproject_match(dataset)
    except Exception as e:
        logging.error("Failed to reproject TCC subset to dataset grid: %s", e)
        return None

    # Apply mask
    mask = tcc_reproj.squeeze() > threshold
    masked = dataset.where(mask)

    return masked

    # # Step 2: Extract the spatial bounds of the dataset (longitude and latitude ranges)
    # dataset_lon_min, dataset_lon_max = dataset['x'].min(), dataset['x'].max()
    # dataset_lat_min, dataset_lat_max = dataset['y'].min(), dataset['y'].max()

    # logging.info("Dataset spatial bounds - Longitude: [%f, %f], Latitude: [%f, %f]",
    #             dataset_lon_min, dataset_lon_max, dataset_lat_min, dataset_lat_max)

    # # Step 3: Subset the TCC raster to the region of interest (dataset spatial bounds)
    # logging.info("Extracting TCC data subset within dataset bounds...")
    # tcc_subset = tcc_data.sel(x=slice(dataset_lon_min, dataset_lon_max),
    #                           y=slice(dataset_lat_max, dataset_lat_min))

    # # Step 4: Check if the subset is valid, return None if empty
    # if tcc_subset.isnull().all():
    #     logging.error("TCC data subset is empty for the given bounds.")
    #     return None

    # # Step 5: Interpolate the TCC data to match the dataset's grid (coordinate system)
    # logging.info("Interpolating TCC data to match dataset grid...")
    # try:
    #     interpolated_tcc = tcc_subset.interp(x=dataset.coords['x'], y=dataset.coords['y'], method='nearest')
    # except Exception as e:
    #     logging.error("Interpolation failed: %s", e)
    #     return None

    # # Step 6: Apply the TCC mask to the dataset
    # # Only retain values where the TCC is above the specified threshold (0.3 in this case)
    # logging.info("Masking dataset based on TCC values...")
    # threshold = 0.3
    # masked_dataset = dataset.where(interpolated_tcc > threshold, 0).fillna(0)

    # # Step 7: Return the masked dataset
    # logging.info("Masking complete. Returning masked dataset.")
    
    # return masked_dataset

def extract_polygons_from_mask(filename, masked_data_array):
    """
    Extracts polygons from a masked data array, storing them in a GeoDataFrame 
    with metadata (year and tile) parsed from the filename.

    Parameters:
    - filename (str): The filename containing year and tile information.
    - masked_data_array (xarray.DataArray): The masked data array with 'x' and 'y' coordinates 
      and a 'layer' attribute representing the mask layer.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing extracted polygons with year and tile metadata.
    """
    
    # Step 1: Parse the year and tile name from the filename
    year = int(filename.split('_year_')[-1].split('_')[0])
    tile_name = filename[13:23]  # Extract tile name based on the assumed naming convention
    
    # Step 2: Initialize an empty GeoDataFrame to store the polygons
    results_gdf = gpd.GeoDataFrame(columns=['geometry', 'S1_YEAR', 'S1_TILE'], crs="EPSG:4326")
    
    # Step 3: Extract the bounding box coordinates (min and max) from the masked data array
    min_x, max_x = masked_data_array['x'].min().item(), masked_data_array['x'].max().item()
    min_y, max_y = masked_data_array['y'].min().item(), masked_data_array['y'].max().item()
    
    # Step 4: Create a GeoDataFrame to represent the bounding box of the data array
    bounds_gdf = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs="EPSG:4326")
    
    # Step 5: Drop the 'band' dimension only if it exists
    if "band" in masked_data_array.dims:
        cropped_mask = masked_data_array.squeeze("band")
    else:
        cropped_mask = masked_data_array
    
    # Step 6: Define the affine transformation to map array indices to geospatial coordinates
    transform = (
        Affine.translation(cropped_mask.x[0], cropped_mask.y[0]) * 
        Affine.scale(cropped_mask.x[1] - cropped_mask.x[0], cropped_mask.y[1] - cropped_mask.y[0])
    )
    
    # Step 7: Create a binary mask where values greater than zero are set to 1 (True)
    binary_mask = (cropped_mask['layer'] > 0).astype(np.uint8)
    
    # Step 8: Extract polygon shapes from the binary mask, using the affine transformation
    extracted_shapes = list(rasterio.features.shapes(binary_mask.values, transform=transform))
    
    # Step 9: Filter and keep only the polygons with a value of 1 (representing valid mask regions)
    polygons = [shape(geom) for geom, value in extracted_shapes if value == 1]
    
    # Step 10: Create a GeoDataFrame from the list of extracted polygons
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=cropped_mask.spatial_ref)
    
    # Step 11: Add metadata columns (year and tile) to the GeoDataFrame
    polygons_gdf['S1_YEAR'] = year
    polygons_gdf['S1_TILE'] = tile_name
    
    # Step 12: Append the new polygons to the results GeoDataFrame
    results_gdf = pd.concat([results_gdf, polygons_gdf], ignore_index=True)
    
    # Return the final GeoDataFrame containing all extracted polygons with metadata
    return results_gdf

def process_and_filter_usda_polygons(dataset, usda_data_path, reference_year, year_buffer, 
                                    buffer_distance, file_path, target_crs, output_shapefile, output_metadata, tile_name):
    """
    Process and filter USDA polygons based on a reference year and buffer distance, 
    then performs spatial operations to calculate area before and after intersection 
    with the polygons dataset. Outputs metadata and results to specified directories.

    Parameters:
    - dataset (GeoDataFrame): The input polygons dataset to be processed.
    - usda_data_path (str): Path to the USDA polygons data (shapefile).
    - reference_year (int): The reference year to filter USDA polygons based on 'SURVEY_Y'.
    - year_buffer (int): Number of years before and after the reference year to include.
    - buffer_distance (float): The buffer distance (in meters) to apply to USDA polygons.
    - file_path (str): Path to the input file to extract tile information for metadata.
    - target_crs (str): The desired coordinate reference system (CRS) to reproject the results.
    - output_dir (str): Directory where the output shapefile and metadata will be saved.

    Returns:
    - None
    """
    logging.info("Step 1: Loading USDA polygons and buffering them...")

    # Load the USDA data
    try:
        usda_gdf = load_data(usda_data_path)
        
        # Ensure CRS is defined, if not set it to WGS84 (EPSG:4326)
        if usda_gdf.crs is None:
            usda_gdf.set_crs(target_crs, inplace=True)
        
        # Reproject USDA data to a metric CRS for buffer operations (EPSG:3857)
        usda_gdf = usda_gdf.to_crs("EPSG:3857")

        # Apply the buffer operation to USDA polygons (buffer_distance in meters)
        usda_gdf['geometry'] = usda_gdf['geometry'].buffer(buffer_distance)
        
        # Reproject back to WGS84 (EPSG:4326)
        usda_gdf = usda_gdf.to_crs(target_crs)
    except Exception as e:
        logging.error(f"Error loading USDA polygons: {e}")
        return

    logging.info(f"Step 2: Filtering USDA polygons within ±{year_buffer} years of {reference_year}...")

    # Filter USDA polygons based on the reference year and the year buffer
    try:
        filtered_usda_gdf = usda_gdf[
            (usda_gdf['SURVEY_Y'] >= reference_year - year_buffer) & 
            (usda_gdf['SURVEY_Y'] <= reference_year + year_buffer)
        ]
        # Calculate the area of the filtered USDA polygons
        filtered_usda_area_gdf = calculate_area_in_km2_s1cd(filtered_usda_gdf)
        total_usda_area = filtered_usda_area_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error filtering USDA polygons: {e}")
        return

    logging.info("Step 3: Calculating total area before intersection...")

    # Compute the total area of the input polygons dataset
    try:
        input_area_gdf = calculate_area_in_km2_s1cd(dataset)
        area_before_intersection = input_area_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error calculating area before intersection: {e}")
        return

    logging.info("Step 4: Performing spatial join to find intersecting polygons...")

    # Perform spatial join to find polygons that intersect the filtered USDA polygons
    try:
        intersecting_polygons_gdf = gpd.sjoin(dataset, filtered_usda_gdf, predicate='intersects')
        intersecting_polygons_gdf = intersecting_polygons_gdf.rename(columns={'index_right': 'S1CD_IDX'})
    except Exception as e:
        logging.error(f"Error during spatial join: {e}")
        return

    logging.info("Step 5: Calculating total area after intersection...")

    # Calculate the area of the intersecting polygons
    try:
        intersecting_area_gdf = calculate_area_in_km2_s1cd(intersecting_polygons_gdf)
        area_after_intersection = intersecting_area_gdf['area'].sum()
    except Exception as e:
        logging.error(f"Error calculating area after intersection: {e}")
        return

    logging.info("Step 6: Aggregating geometries by 'IDX_D'...")

    # Aggregate geometries by 'IDX_D' (dissolve operation)
    try:
        aggregated_gdf = intersecting_polygons_gdf.dissolve(by='IDX_D')
        aggregated_gdf.reset_index(inplace=True)
    except Exception as e:
        logging.error(f"Error during geometry aggregation: {e}")
        return

    logging.info("Step 7: Reprojecting and saving output shapefile...")

    # Reproject the aggregated geometries to the target CRS
    try:
        aggregated_gdf.set_crs(epsg=4326, inplace=True)  # Ensure it starts from EPSG:4326
        aggregated_gdf = aggregated_gdf.to_crs(target_crs)  # Reproject to target CRS
        
        # Save the GeoDataFrame to a shapefile
        aggregated_gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        logging.info(f"Shapefile saved successfully to {output_shapefile}")
    except Exception as e:
        logging.error(f"Error saving shapefile: {e}")
        return

    # Save metadata to a CSV file
    try:
        metadata_entries = [{
            'Tile': tile_name,
            'Year': reference_year,
            'IDS Area': total_usda_area,
            'Area Before Intersection': area_before_intersection,
            'Area After Intersection': area_after_intersection  
        }]
        metadata_df = pd.DataFrame(metadata_entries)
        
        # Ensure the directories exist
        os.makedirs(os.path.dirname(output_metadata), exist_ok=True)
        
        # Save the metadata to CSV
        metadata_df.to_csv(output_metadata, index=False)
        logging.info(f"Metadata table saved as CSV to {output_metadata}")
    except Exception as e:
        logging.error(f"Error saving metadata CSV: {e}")
        return

    logging.info("Processing completed successfully.")

def merge_shapefiles(input_dir, target_crs):
    """
    Merges all shapefiles in the specified directory into a single GeoDataFrame.

    Parameters:
    - input_dir (str): Path to the directory containing shapefiles.

    Returns:
    - GeoDataFrame: The merged GeoDataFrame containing all shapefile data.
    """
    logging.info(f"Starting to merge shapefiles from directory: {input_dir}")

    # List all shapefiles in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
    if not files:
        logging.warning(f"No shapefiles found in directory: {input_dir}. Nothing to merge.")
        return None

    gdf_list = []

    # Iterate through each shapefile and read them into a GeoDataFrame
    for file in tqdm(files, desc="Merging shapefiles", unit="file"):
        filepath = os.path.join(input_dir, file)
        logging.info(f"Reading shapefile: {filepath}")

        try:
            gdf = gpd.read_file(filepath)
        except Exception as e:
            logging.error(f"Error reading shapefile {file}: {e}")
            continue
        
        # Check if CRS is defined, raise an error if it's not
        if gdf.crs is None:
            logging.warning(f"CRS not defined for file: {filepath}. Assigning a default CRS.")
            # Here, you might want to define a default CRS based on your data context:
            gdf.set_crs(target_crs, inplace=True)

        gdf_list.append(gdf)

    # Merge the list of GeoDataFrames into a single GeoDataFrame
    try:
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        logging.info(f"Merged {len(gdf_list)} shapefiles successfully.")
    except Exception as e:
        logging.error(f"Error merging shapefiles: {e}")
        return None

    # Optionally, call an additional method to clean/merge geometries and keep relevant columns
    # merged_gdf = merge_geometries_and_keep_columns(merged_gdf)

    logging.info(f"Shapefiles merged successfully. Total records: {len(merged_gdf)}")
    return merged_gdf

def calculate_and_filter_area(gdf, target_crs):
    """
    Calculate area in square kilometers and filter out polygons larger than 15 km².
    """
    logging.info("Calculating area and filtering polygons...")
    gdf = gdf.to_crs(target_crs)
    
    # Reproject to a CRS with meters (e.g., EPSG:3857) for accurate area calculation
    projected_gdf = gdf.to_crs('EPSG:27705')
    
    # Calculate the area in square meters and convert to km²
    projected_gdf['area_km2'] = projected_gdf.geometry.area / 1e6
    
    # Add the calculated area to the original GeoDataFrame and filter
    gdf['area_km2'] = projected_gdf['area_km2']
    filtered_gdf = gdf[gdf['area_km2'] <= 15]
    
    logging.info("Area calculated and polygons filtered.")
    return filtered_gdf
