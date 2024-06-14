"""
USDA Disturbance Data Processing Script
=======================================

Author: Franziska Müller
Date: 24.05.2024

This script performs data processing tasks on IDS USDA disturbance data for Region 8.
It follows a series of steps to filter, manipulate, and save the data for further analysis.

Steps:
1. Loading CSV File: Load USDA disturbance data from a CSV file.
2. Converting WKT Geometries: Convert WKT geometries to Shapely geometries.
3. Converting to GeoDataFrame: Convert DataFrame to GeoDataFrame for spatial operations.
4. Filtering Data: Filter disturbance records based on criteria such as timeframe and disturbance type.
5. Renaming Columns: Rename columns for clarity and consistency.
6. Exploding Multipolygons: Explode multipolygons into individual polygons for analysis.
7. Calculation area in km2
8. filtering the areas over 15km2
9. Saving Results: Save processed data to a CSV file.


Results are saved in the specified directory with a confirmation message printed upon completion.
"""

# Import necessary libraries
import pandas as pd
import geopandas as gpd
from shapely import wkt
import os

def calculate_area_in_km2(gdf):
    # Step 1: Set target CRS to EPSG:4326
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
    
    # Step 6: Return the modified GeoDataFrame
    return gdf


def main():
    # Step 1: Read the CSV file into a DataFrame
    print("Step 1: Loading CSV file...")
    input_file = "/Net/Groups/BGI/work_2/ForExD/USDA/tables_new/CONUS_Region8_dissolved.csv"
    df = pd.read_csv(input_file)

    # Step 2: Convert the WKT geometries to Shapely geometries
    print("Step 2: Converting WKT geometries...")
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Step 3: Convert the DataFrame to a GeoDataFrame
    print("Step 3: Converting DataFrame to GeoDataFrame...")
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set the coordinate reference system (CRS) if it's not already set
    gdf.set_crs(epsg=4326, inplace=True)

    # Step 4: Filter for disturbances recorded between 2016 and 2020
    print("Step 4: Filtering disturbances recorded between 2016 and 2020...")
    gdf_timeframe = gdf[(gdf['SURVEY_YEAR'] > 2016) & (gdf['SURVEY_YEAR'] <= 2020)]

    # Filter out specific disturbance types
    excluded_types = ['other', 'multi_damage', 'other_abiotic', 'other_biotic']
    rslt_df = gdf_timeframe[~gdf_timeframe['DCA_ID'].isin(excluded_types)].copy()

    # Step 5: Rename the column 'Unnamed: 0' to 'index_usda'
    print("Step 5: Rename columns for clarity and consistency")
    rslt_df.rename(columns={'Unnamed: 0': 'index_usda'}, inplace=True)

    # Step 6: Explode multipolygons into individual polygons
    print("Step 6: Exploding multipolygons into individual polygons...")
    exploded_df = rslt_df.explode(index_parts=True)

    # Reset the index to ensure a clean index
    exploded_df.reset_index(drop=True, inplace=True)

    # Step 7: Generate new index_usda values
    print("Step 7: Generating new index_usda values...")
    exploded_df['index_usda'] = exploded_df.apply(lambda row: f"{row['DCA_ID']}_{row['SURVEY_YEAR']}_{row.name}", axis=1)

    # Step 8: Calculating area in km²
    print("Step 8: Calculating area in km²")
    gdf_with_area = calculate_area_in_km2(exploded_df)

    # Step 9: Remove elements larger than 15km²
    print("Step 9: Remove elements larger than 15km²")
    filtered_gdf = gdf_with_area[gdf_with_area['area_km2'] <= 15]

    # Print the number of remaining elements
    print(f"Number of elements after filtering: {len(filtered_gdf)}")


    # Define the absolute path to the "results" folder
    results_folder = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results"

    # Define the output CSV file path
    output_file = os.path.join(results_folder, "region8_dca_filtered_ids_usda_polygons.csv")

    # Step 10: Save the exploded DataFrame to a CSV file in the results folder
    print(f"Step 10: Saving results to: {output_file}...")
    filtered_gdf.to_csv(output_file, index=False)

    # Display the path of the saved CSV file
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
