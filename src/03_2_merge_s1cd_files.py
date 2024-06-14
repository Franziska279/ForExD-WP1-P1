import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import shutil

# Constants
SHAPEFILES_DIR = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/03_s1cd_polygons/"
OUTPUT_DIR = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/"
OUTPUT_FILENAME = "radar_enhanced_forest_disturbance_mapping.shp"
TARGET_CRS = "EPSG:4326"

def merge_shapefiles(input_dir):
    """Merge all shapefiles in the specified directory into a single GeoDataFrame."""
    print(f"Merging shapefiles from {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
    gdf_list = []
    
    for file in tqdm(files, desc="Merging shapefiles"):
        filepath = os.path.join(input_dir, file)
        gdf = gpd.read_file(filepath)
        # Set the coordinate reference system (CRS) if it's not already set
        #gdf.set_crs(epsg=4326, inplace=True)

        # Ensure CRS is set
        if gdf.crs is None:
            raise ValueError(f"CRS not defined for file: {filepath}. Please define CRS for all shapefiles.")
        
        # Set CRS explicitly
        #gdf.crs = TARGET_CRS
        gdf_list.append(gdf)
    
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    print("Shapefiles merged successfully.")
    return merged_gdf

def calculate_and_filter_area(gdf):
    """Calculate area in square kilometers and filter out polygons larger than 15 km²."""
    print("Calculating area and filtering polygons...")
    target_crs = 'EPSG:4326'
    gdf = gdf.to_crs(target_crs)
    
    # Step 2: Reproject to a CRS with meters (e.g., EPSG:3857)
    projected_gdf = gdf.to_crs('EPSG:3857')
    
    # Step 3: Calculate the area in square meters
    projected_gdf['area_m2'] = projected_gdf.geometry.area
    
    # Step 4: Convert the area to square kilometers
    projected_gdf['area_km2'] = projected_gdf['area_m2'] / 1e6
    
    # Step 5: Assign the calculated area to a new column in the original GeoDataFrame
    gdf['area_km2'] = projected_gdf['area_km2']

    # Filter out polygons larger than 15 km²
    filtered_gdf = gdf[gdf['area_km2'] <= 15]
    
    print("Area calculated and polygons filtered.")
    return filtered_gdf


def save_result(gdf, output_dir, output_filename):
    """Save the resulting GeoDataFrame to a new shapefile."""
    print(f"Saving result to {os.path.join(output_dir, output_filename)}...")
    output_path = os.path.join(output_dir, output_filename)
    gdf.to_file(output_path)
    print("Result saved successfully.")

def main():
    print("Starting processing...")
    
    # Merge shapefiles
    merged_gdf = merge_shapefiles(SHAPEFILES_DIR)
    
    # Calculate area and filter
    filtered_gdf = calculate_and_filter_area(merged_gdf)
    
    # Save result
    save_result(filtered_gdf, OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Remove the original shapefiles directory and all its contents
    try:
        shutil.rmtree(SHAPEFILES_DIR)
        print(f"Successfully removed directory and all contents: {SHAPEFILES_DIR}")
    except OSError as e:
        print(f"Error removing directory and all contents: {SHAPEFILES_DIR} - {e}")


    print("Processing completed.")

if __name__ == "__main__":
    main()

