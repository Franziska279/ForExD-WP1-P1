"""
Shapefile Merging and Filtering Script
======================================

Author: Franziska Müller
Date: 28.06.2024

This script merges multiple shapefiles from a specified directory into a single GeoDataFrame,
calculates the area of each polygon, filters out polygons larger than 15 km², and saves the
filtered results to a new shapefile. Additionally, it removes the original shapefiles directory
after processing.

Steps:
1. Merge all shapefiles in the specified directory into a single GeoDataFrame.
2. Calculate the area of each polygon in square kilometers.
3. Filter out polygons larger than 15 km².
4. Save the resulting GeoDataFrame to a new shapefile.
5. Remove the original shapefiles directory and its contents.

The script accepts the following parameters as arguments:
- shapefiles_dir: Directory containing the shapefiles to be merged.
- output_dir: Directory where the final output will be saved.
- output_filename: Name of the output shapefile.
- target_crs: Target coordinate reference system (CRS) for the GeoDataFrame.
"""

import os
import sys
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import shutil

def merge_shapefiles(input_dir):
    """Merge all shapefiles in the specified directory into a single GeoDataFrame."""
    print(f"Merging shapefiles from {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
    gdf_list = []
    
    for file in tqdm(files, desc="Merging shapefiles"):
        filepath = os.path.join(input_dir, file)
        gdf = gpd.read_file(filepath)
        
        # Ensure CRS is set
        if gdf.crs is None:
            raise ValueError(f"CRS not defined for file: {filepath}. Please define CRS for all shapefiles.")
        
        gdf_list.append(gdf)
    
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    print("Shapefiles merged successfully.")
    return merged_gdf


def calculate_and_filter_area(gdf, target_crs):
    """Calculate area in square kilometers and filter out polygons larger than 15 km²."""
    
    print("Calculating area and filtering polygons...")
    gdf = gdf.to_crs(target_crs)
    
    # Reproject to a CRS with meters (e.g., EPSG:3857)
    projected_gdf = gdf.to_crs('EPSG:3857')
    
    # Calculate the area in square meters
    projected_gdf['area_m2'] = projected_gdf.geometry.area
    
    # Convert the area to square kilometers
    projected_gdf['area_km2'] = projected_gdf['area_m2'] / 1e6
    
    # Assign the calculated area to a new column in the original GeoDataFrame
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

def main(shapefiles_dir, output_dir, output_filename, target_crs):
    """
    Main function to orchestrate the processing steps.

    Parameters:
    - shapefiles_dir: Directory containing the shapefiles to be merged.
    - output_dir: Directory where the final output will be saved.
    - output_filename: Name of the output shapefile.
    - target_crs: Target coordinate reference system (CRS) for the GeoDataFrame.
    """

    print("Starting processing...")
    
    # Merge shapefiles
    merged_gdf = merge_shapefiles(shapefiles_dir)
    
    # Calculate area and filter
    filtered_gdf = calculate_and_filter_area(merged_gdf, target_crs)
    
    # Save result
    save_result(filtered_gdf, output_dir, output_filename)
    
    # Remove the original shapefiles directory and all its contents
    try:
        shutil.rmtree(shapefiles_dir)
        print(f"Successfully removed directory and all contents: {shapefiles_dir}")
    except OSError as e:
        print(f"Error removing directory and all contents: {shapefiles_dir} - {e}")

    print("Processing completed.")


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python script.py <shapefiles_dir> <output_dir> <output_filename> <target_crs>")
        sys.exit(1)
    
    shapefiles_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_filename = sys.argv[3]
    target_crs = sys.argv[4]
    
    main(shapefiles_dir, output_dir, output_filename, target_crs)