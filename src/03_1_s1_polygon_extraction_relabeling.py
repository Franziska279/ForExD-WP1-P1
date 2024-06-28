"""
S1 Change Detection Data Processing Script
==========================================

Author: Franziska Müller
Date: 28.06.2024

This script processes S1 Change Detection data from NetCDF files, applies a Tree Canopy Cover (TCC) mask,
extracts polygons from the data, and filters these polygons based on USDA survey data. The resulting polygons
are saved to a shapefile.

Steps:
1. Preprocess the NetCDF dataset.
2. Apply a TCC mask to the preprocessed dataset.
3. Extract polygons from the masked dataset.
4. Filter and aggregate the extracted polygons based on USDA survey data.
5. Save the resulting polygons to a shapefile.

The script accepts the following parameters as arguments:
- input_path: Path to the input NetCDF file.
- ids_usda_path: Path to the CSV file containing USDA polygons.

"""

import os
import argparse
import rioxarray
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely import wkt
import rasterio.features
from affine import Affine
from shapely.geometry import box, shape

def extract_filename_part(filename):
    """Extracts a specific part of the filename up to the 10th underscore-separated part."""
    parts = filename.split('_')
    extracted_part = '_'.join(parts[0:10])
    return extracted_part

def preprocess_dataset(input_file, filename):

    # Step 1: Load dataset with filename
    print("Step  1: Load data from:", filename)
    dataset = xr.open_dataset(input_file)
    print("     Dataset loaded successfully.")
    # Get a list of variable names in the dataset
    variables = list(dataset.variables)
    print("     Variables in the dataset:", variables)

    # Step 2: Define the Coordinate Reference Systems (CRS)
    print("Step 2: Define the Coordinate Reference Systems (CRS)")
    crs_azimuthal_equidistant = "+proj=aeqd +lat_0=52 +lon_0=-97.5 +x_0=8264722.17686 +y_0=4867518.35323 +datum=WGS84 +units=m +no_defs"
    crs_wgs84 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'


    # Step 3: Define the variables to drop
    print("Step 3: Define the variables to drop")
    variables_to_remove = ["x_bnds", "y_bnds"]

    # Step 4: Drop 'x_bnds' and 'y_bnds' if they exist in the dataset
    print("Step 4: Drop 'x_bnds' and 'y_bnds' if they exist in the dataset")
    for variable in variables_to_remove:
        if variable in dataset.variables:
            dataset = dataset.drop_vars(variable)
    print("     Variables 'x_bnds' and 'y_bnds' dropped successfully.")

    # Step 5: Update the list of variables after dropping
    print("Step 5: Update the list of variables after dropping")
    current_variable_list = list(dataset.variables)

    # Step 6: Rename 'unnamed' variable to 'layer' if it exists
    print("Step 6: Rename 'unnamed' variable to 'layer' if it exists")
    if 'unnamed' in dataset.variables:
        dataset = dataset.rename({'unnamed': 'layer'})
        print("     Variable 'unnamed' renamed to 'layer'.")
    else:
        print("     No variable named 'unnamed' found.")

    # Step 7: Rename 'X' and 'Y' variables to 'x' and 'y' if both exist
    print("Step 7: Rename 'X' and 'Y' variables to 'x' and 'y' if both exist")
    if 'X' in dataset.variables and 'Y' in dataset.variables:
        dataset = dataset.rename({'X': 'x', 'Y': 'y'})
        print("     Variables 'X' and 'Y' renamed to 'x' and 'y'.")
    else:
        print("     No variables named 'X' and 'Y' found.")

    # Step 8: Update the list of variables after renaming
    print("Step 8: Update the list of variables after renaming")
    current_variable_list = list(dataset.variables)
    print("     Current variables:", current_variable_list)

    # Step 9: Write the azimuthal equidistant CRS to the dataset
    print("Step 9: Write the azimuthal equidistant CRS to the dataset")
    dataset.rio.write_crs(crs_azimuthal_equidistant, inplace=True)

    # Step 10: Reproject the dataset to the WGS 84 projection (EPSG:4326)
    print("Step 10: Reproject the dataset to the WGS 84 projection (EPSG:4326)")
    dataset_wgs84 = dataset.rio.reproject(crs_wgs84)

    return dataset_wgs84


def apply_tcc_mask(dataset):

    # Step 1: Define the path to the TIF file
    print("Step 1: Defining the path to the TIF file...")
    TCC_path_2017 = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"

    # Step 2: Open the entire TIF file
    print("Step 2: Opening the entire TIF file...")
    tcc_2017 = rioxarray.open_rasterio(TCC_path_2017, decode_coords="all", masked=True)

    # Step 3: Extract the spatial extent from the NetCDF file
    print("Step 3: Extracting spatial extent from the NetCDF file...")
    min_lon, max_lon = dataset['x'].min(), dataset['x'].max()
    min_lat, max_lat = dataset['y'].min(), dataset['y'].max()

    # Step 4: Select the subset using xarray's indexing capabilities
    print("Step 4: Selecting the subset using xarray's indexing capabilities...")
    subset = tcc_2017.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))

    # Step 5: Calculate the minimum value of the subset
    print("Step 5: Calculating the minimum value of the subset...")
    min_value = subset.min() if not subset.isnull().all() else 0

    # Step 6: Calculate the normalized subset
    print("Step 6: Calculating the normalized subset...")
    normalized_subset = (subset - min_value) / (subset.max() - min_value) if subset.max() != min_value else subset

    # Step 7: Set values equal to 1 to 0
    print("Step 7: Setting values equal to 1 to 0...")
    normalized_subset = normalized_subset.where(normalized_subset != 1, 0)

    # Step 8: Reindex the normalized subset to match the coordinates of dataset_wgs84
    print("Step 8: Reindexing the normalized subset...")
    normalized_subset = normalized_subset.reindex(x=dataset.coords['x'], method='nearest')
    normalized_subset = normalized_subset.reindex(y=dataset.coords['y'], method='nearest')

    # Step 9: Apply masking based on the normalized subset
    print("Step 9: Applying masking based on the normalized subset...")
    masked_mc = dataset.where(normalized_subset > 0.3, 0).fillna(0)

    return masked_mc


def process_nc_file(masked_mc, filename):
    # Step 1: Extract the year and tile information from the file name
    print("Step 1: Extracting the year and tile information from the file name...")
    s1_year = int(filename.split('_year_')[-1].split('_')[0])
    tile_name = filename[13:23]
    print(f"    Processing year: {s1_year}")
    print(f"    Processing tile: {tile_name}")

    # Step 2: Initialize an empty GeoDataFrame to accumulate results
    print("Step 2: Initializing the GeoDataFrame to store results...")
    all_polygons_gdf = gpd.GeoDataFrame(columns=['geometry', 'S1_YEAR', 'S1_TILE'])

    # Step 3: Extract the bounds of the masked data array
    print("Step 3: Extracting the bounds of the masked data array...")
    min_lon, max_lon = masked_mc['x'].min().item(), masked_mc['x'].max().item()
    min_lat, max_lat = masked_mc['y'].min().item(), masked_mc['y'].max().item()

    # Step 4: Create a GeoDataFrame with the bounds of the masked data array
    print("Step 4: Creating a GeoDataFrame with the bounds of the masked data array...")
    bounds_gdf = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)])
    print(f"    Bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")

    # Step 5: Drop the 'band' dimension from the masked data array
    print("Step 5: Dropping the 'band' dimension from the masked data array...")
    masked_mc_cropped = masked_mc.squeeze("band")

    # Step 6: Get the geospatial information from the data array
    print("Step 6: Extracting geospatial information from the data array...")
    transform = (Affine.translation(masked_mc_cropped.x[0], masked_mc_cropped.y[0]) * 
                    Affine.scale(masked_mc_cropped.x[1] - masked_mc_cropped.x[0], 
                                masked_mc_cropped.y[1] - masked_mc_cropped.y[0]))
    print(f"Affine transform: {transform}")

    # Step 7: Convert the cropped data into a mask with a valid data type
    print("Step 7: Converting the cropped data into a mask...")
    mask = (masked_mc_cropped['layer'] > 0).astype(np.uint8)

    # Step 8: Extract geometry shapes from the mask
    print("Step 8: Extracting shapes from the mask...")
    shapes = list(rasterio.features.shapes(mask, transform=transform))
    print(f"    Extracted {len(shapes)} shapes.")

    # Step 9: Create a list to store individual polygons
    print("Step 9: Creating individual polygons...")
    polygons_list = [shape(geom) for geom, value in shapes if value == 1]
    print(f"    Extracted {len(polygons_list)} polygons.")

    # Step 10: Create a GeoDataFrame from individual polygons
    print("Step 10: Creating a GeoDataFrame from individual polygons...")
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons_list, crs=masked_mc_cropped.spatial_ref)

    # Step 11: Add columns with information from the current loop
    print("Step 11: Adding year and tile information to the GeoDataFrame...")
    polygons_gdf['S1_YEAR'] = s1_year
    polygons_gdf['S1_TILE'] = tile_name

    # Step 12: Append the GeoDataFrame to the result GeoDataFrame
    print("Step 12: Appending the current GeoDataFrame to the results...")
    all_polygons_gdf = all_polygons_gdf.append(polygons_gdf, ignore_index=True)
    print("     Polygon extraction and GeoDataFrame update completed.")

    return all_polygons_gdf


def process_and_filter_polygons(ids_usda_path, polygons_gdf, s1_year, filename):
    # Step 1: Load the ids_usda file
    print("Step 1: Loading the ids_usda file...")
    ids_usda = pd.read_csv(ids_usda_path)
    # Step 2: Convert the WKT geometries to Shapely geometries
    print("     Converting WKT geometries...")
    ids_usda['geometry'] = ids_usda['geometry'].apply(wkt.loads)
    # Step 3: Convert the DataFrame to a GeoDataFrame
    print("     Converting DataFrame to GeoDataFrame...")
    ids_usda_gdf = gpd.GeoDataFrame(ids_usda, geometry='geometry')

    # Step 2: Add a buffer of 500m around the ids_usda geometries
    print("Step 2: Adding a 500m buffer around the ids_usda geometries...")
    ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(0.005)
    
    # Step 3: Filter elements from ids_usda within a +-1 year buffer of s1_year
    print(f"Step 3: Filtering ids_usda for elements within a +-1 year buffer of {s1_year}...")
    ids_usda_filtered = ids_usda_gdf[(ids_usda_gdf['SURVEY_YEAR'] >= s1_year - 1) & (ids_usda_gdf['SURVEY_YEAR'] <= s1_year + 1)]
    print(f"Filtered down from {len(ids_usda_gdf)} to {len(ids_usda_filtered)} entries within the 2 year buffer.")

    # Step 4: Spatially join polygons_gdf and ids_usda_filtered
    print("Step 4: Performing spatial join to find intersecting polygons...")
    intersecting_polygons_gdf = gpd.sjoin(polygons_gdf, ids_usda_filtered, predicate='intersects')
    intersecting_polygons_gdf = intersecting_polygons_gdf.rename(columns={'index_right': 'S1CD_INDEX', 'index_usda': 'USDA_INDEX'})
    print(f"Found {len(intersecting_polygons_gdf)}/{len(polygons_gdf)} intersecting polygons.")

    # Step 5: Rename columns to be less than or equal to 10 characters
    print("Step 5: Renaming columns to be less than or equal to 10 characters...")
    rename_mapping = {
        'SURVEY_YEAR': 'SURV_YEAR',
        'REGION_ID': 'REG_ID',
        'DAMAGE_TYPE': 'DAM_TYPE',
        'DAMAGE_TYPE_CODE': 'DAM_TYPE_CD',
        'DCA_CODE': 'DCA_CD',
        'DA_Code_USDA': 'DA_CD_USDA',
        'PERCENT_AFFECTED': 'PCT_AFFECT',
        'S1CD_INDEX': 'S1CD_IDX',
        'USDA_INDEX': 'USDA_IDX'
    }
    intersecting_polygons_gdf = intersecting_polygons_gdf.rename(columns=rename_mapping)

    # Step 6: Aggregating geometries based on their shared USDA_IDX
    print("Step 6: Aggregating geometries based on their shared USDA_IDX ...")
    merged_gdf = intersecting_polygons_gdf.dissolve(by='USDA_IDX')
    merged_gdf.reset_index(inplace=True)
    print(f"Total number of remaining rows: {len(merged_gdf)}/{len(intersecting_polygons_gdf)}")

    # Set the coordinate reference system (CRS) if it's not already set
    merged_gdf.set_crs(epsg=4326, inplace=True)
    target_crs = 'EPSG:4326'
    merged_gdf = merged_gdf.to_crs(target_crs)
   

    # Step 7: Save the intersecting polygons as a shapefile or CSV
    print("Step 7: Saving the intersecting polygons as a shapefile...")
    output_folder = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/03_s1cd_polygons/"
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the output file paths
    extracted_part = extract_filename_part(os.path.basename(filename))
    shapefile_path = os.path.join(output_folder, f"{extracted_part}.shp")
    
    # Save as shapefile
    merged_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"Intersecting polygons saved as shapefile: {shapefile_path}")


def main(input_path, ids_usda_path):
    
    """
    Main function to orchestrate the processing steps.

    Parameters:
    - input_path (str): Path to the input NetCDF file.
    - ids_usda_path (str): Path to the CSV file containing USDA polygons.
    """
    filename = os.path.basename(input_path)
    s1_year = int(filename.split('_year_')[-1].split('_')[0])

    print(f"Starting processing for file: {filename}")
    

    print("Starting the preprocessing of the S1 Change detected raster file...")
    preprocessed_dataset = preprocess_dataset(input_path, filename)
    print("Sarting the masking with the TreeCanopyCover CONUS 2017...")
    masked_dataset = apply_tcc_mask(preprocessed_dataset)
    print("Starting the polygon extraction process...")
    polygons_gdf = process_nc_file( masked_dataset, filename)
    print("Starting the filtering process...")
    process_and_filter_polygons(ids_usda_path, polygons_gdf, s1_year, filename)
    print("\nPreprocessing, masking, extracting polygons, filtering  and aggregating completed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process S1 Change detection data and filter polygons.")
    parser.add_argument('input_path', type=str, help="Path to the input NetCDF file.")
    parser.add_argument('ids_usda_path', type=str, help="Path to the CSV file containing USDA polygons.")

    args = parser.parse_args()

    main(args.input_path, args.ids_usda_path)

    # input_path = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/s1_change_detection_northamerica/EQUI7_NA020M_E084N024T3_rqatrend_VH_A_thresh_3.0_year_2016_cluster_compressed.nc"
    # ids_usda_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"



    # main(input_path, ids_usda_path)
