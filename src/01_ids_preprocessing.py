"""
IDS USDA Disturbance Data Processing Script
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
7. Calculating Area in km²: Calculate the area of each polygon in square kilometers.
8. Filtering Large Areas: Filter out polygons with an area larger than 15 km².
9. Saving Results: Save processed data to a CSV file.

Results are saved in the specified directory with a confirmation message printed upon completion.
"""

# Import necessary libraries
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
import pandas as pd
import os

def calculate_area_in_km2(gdf):
    """
    Calculate the area of each polygon in the GeoDataFrame in square kilometers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame with geometries in WKT format.

    Returns:
    GeoDataFrame: GeoDataFrame with an added column for area in square kilometers.
    """
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


def remove_overlapping_entries(df, year_col='SURVEY_YEAR', geometry_col='geometry'):
    """
    Remove entries that overlap spatially within a given temporal window.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    year_col (str): Name of the column containing the survey year.
    geometry_col (str): Name of the column containing geometry data.

    Returns:
    DataFrame: DataFrame with overlapping entries removed.
    """
    df = df.copy()
    overlaps = set()

    # Iterate over each row to check for overlaps with a progress bar
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking overlaps"):
        year = row[year_col]
        geom = row[geometry_col]

        # Define the time window for overlap check
        time_window = (df[year_col] >= (year - 10)) & (df[year_col] <= (year + 5))

        # Check for spatial overlaps within the time window
        #spatial_overlaps = df[time_window][df[geometry_col].intersects(geom)]
        spatial_overlaps = df.loc[time_window & df[geometry_col].intersects(geom)]
        
        # Add the indices of the overlapping rows to the overlaps set
        if len(spatial_overlaps) > 1:
            overlaps.add(i)
            overlaps.update(spatial_overlaps.index)

    # Drop the overlapping rows
    df = df.drop(index=overlaps)
    return df


def main():

    # Define the paths to folders
    input_file = "/Net/Groups/BGI/work_2/ForExD/USDA/tables_new/CONUS_Region8_dissolved.csv"
    results_folder = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results"

    # Step 1: Read the CSV file into a DataFrame
    print("Step 1: Loading CSV file...")
    df = pd.read_csv(input_file)

    # Step 2: Convert the WKT geometries to Shapely geometries
    print("Step 2: Converting WKT geometries...")
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Step 3: Convert the DataFrame to a GeoDataFrame
    print("Step 3: Converting DataFrame to GeoDataFrame...")
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set the coordinate reference system (CRS) if it's not already set
    gdf.set_crs(epsg=4326, inplace=True)

    # Step 4: Check for overlaps
    print("Step 4: Checking for temporal and spatial overlaps...")
    gdf_no_overlap = remove_overlapping_entries(gdf)

    print(f"Number of elements: {len(gdf)}")
    print(f"Number of elements after removing temporal and spatial overlaps: {len(gdf_no_overlap)}")

    # Step 5: Filter for disturbances recorded between 2016 and 2020
    print("Step 5: Filtering disturbances correct recorded between 2016 and 2020...")
    gdf_filtered = gdf_no_overlap[(gdf_no_overlap['SURVEY_YEAR'] > 2016) & (gdf_no_overlap['SURVEY_YEAR'] <= 2020)]

    # Filter out specific disturbance types
    excluded_types = ['other', 'multi_damage', 'other_abiotic', 'other_biotic']
    filtered_df = gdf_filtered[~gdf_filtered['DCA_ID'].isin(excluded_types)].copy()

    print(f"Number of elements within the timeframe and disturbance types: {len(filtered_df)}")

    # Step 6: Rename the column 'Unnamed: 0' to 'index_usda'
    print("Step 6: Renaming columns for clarity and consistency")
    filtered_df.rename(columns={'Unnamed: 0': 'index_usda'}, inplace=True)

    # Step 7: Explode multipolygons into individual polygons
    print("Step 7: Exploding multipolygons into individual polygons...")
    exploded_df = filtered_df.explode(index_parts=True)

    # Reset the index to ensure a clean index
    exploded_df.reset_index(drop=True, inplace=True)

    print(f"Number of individual elements: {len(exploded_df)}")

    # Step 8: Generate new index_usda values
    print("Step 8: Generating new index_usda values...")
    exploded_df['index_usda'] = exploded_df.apply(lambda row: f"{row['DCA_ID']}_{row['SURVEY_YEAR']}_{row.name}", axis=1)

    # Step 9: Calculating area in km²
    print("Step 9: Calculating area in km²")
    gdf_with_area = calculate_area_in_km2(exploded_df)

    # Step 10: Remove elements larger than 15km²
    print("Step 10: Removing elements larger than 15km²")
    gdf_area = gdf_with_area[gdf_with_area['area_km2'] <= 15]
    gdf_area.reset_index(drop=True, inplace=True)

    # Print the number of remaining elements
    print(f"Number of elements after filtering: {len(gdf_area)}")

    # Define the output CSV file path
    output_file = os.path.join(results_folder, "region8_dca_filtered_ids_usda_polygons.csv")

    # Step 11: Save the exploded DataFrame to a CSV file in the results folder
    print(f"Step 11: Saving results to: {output_file} ...")
    gdf_area.to_csv(output_file, index=False)

    # Display the path of the saved CSV file
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
