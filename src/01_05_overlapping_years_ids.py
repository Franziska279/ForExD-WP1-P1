import pandas as pd
import geopandas as gpd
from shapely import wkt

def check_overlap_seperate_years_indexing(df, pre_year, post_year, year_col='SURVEY_YEAR', geometry_col='geometry'):
    """
    Remove elements that overlap with others within the specified pre_year and post_year range.
    """
    df = df.copy()
    overlaps = set()
    
    # Create a spatial index for the GeoDataFrame
    sindex = df.sindex

    # Iterate over each row to check for overlaps
    for i, row in df.iterrows():
        year = row[year_col]
        geom = row[geometry_col]

        # Define the time window for overlap check
        time_window = (df[year_col] >= (year - pre_year)) & (df[year_col] <= (year + post_year))
        
        # Filter the DataFrame based on the time window
        df_time_window = df[time_window]
        
        # Use the spatial index to find potential overlaps
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = df.iloc[possible_matches_index]

        # Ensure that possible matches are within the time window
        possible_matches = possible_matches[time_window.loc[possible_matches.index]]

        # Check for spatial overlaps within the time window
        spatial_overlaps = possible_matches[possible_matches[geometry_col].intersects(geom)]

        # Add the indices of the overlapping rows to the overlaps set
        if len(spatial_overlaps) > 1:
            overlaps.add(i)
            overlaps.update(spatial_overlaps.index)

    # Drop the overlapping rows
    df = df.drop(index=overlaps)
    return overlaps


def check_overlap_specific_year(df, year_offset, year_col='SURVEY_YEAR', geometry_col='geometry'):
    """
    Identify elements that overlap with others exactly `year_offset` years before the current year.
    
    Args:
        df (GeoDataFrame): The input GeoDataFrame.
        year_offset (int): The number of years before the current year to check for overlaps.
        year_col (str): The column name for the year of the event.
        geometry_col (str): The column name for the geometry.
    
    Returns:
        set: A set of indices of rows that have overlaps.
    """
    df = df.copy()
    overlaps = set()
    
    # Create a spatial index for the GeoDataFrame
    sindex = df.sindex

    # Iterate over each row to check for overlaps
    for i, row in df.iterrows():
        year = row[year_col]
        geom = row[geometry_col]

        # Define the specific year to check for overlaps
        specific_year = year - year_offset
        
        # Filter the DataFrame to include only rows from the specific year
        df_specific_year = df[df[year_col] == specific_year]
        
        if df_specific_year.empty:
            continue
        
        # Use the spatial index to find potential overlaps within the filtered specific year dataframe
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = df_specific_year[df_specific_year.index.isin(possible_matches_index)]

        # Check for spatial overlaps in the specific year
        spatial_overlaps = possible_matches[possible_matches[geometry_col].intersects(geom)]

        # Exclude the current row itself in case of year_offset == 0
        if year_offset == 0:
            spatial_overlaps = spatial_overlaps[spatial_overlaps.index != i]

        # Add the indices of the overlapping rows to the overlaps set
        if len(spatial_overlaps) > 0:
            overlaps.add(i)
            overlaps.update(spatial_overlaps.index)

    # Return the indices of the overlapping rows
    return overlaps



def main():
    # Step 1: Read the CSV file into a DataFrame
    print("Step 1: Loading CSV file...")
    input_file = "/Net/Groups/BGI/work_2/ForExD/USDA/tables/CONUS_Region8_dissolved.csv"
    df = pd.read_csv(input_file)

    # Step 2: Convert the WKT geometries to Shapely geometries
    print("Step 2: Converting WKT geometries...")
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Step 3: Convert the DataFrame to a GeoDataFrame
    print("Step 3: Converting DataFrame to GeoDataFrame...")
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set the coordinate reference system (CRS) if it's not already set
    gdf.set_crs(epsg=4326, inplace=True)
    print(f"Total records: {len(gdf)}")

    # Step 4: Check for overlaps
    print("Step 4: Checking for temporal and spatial overlaps...")
    results = {}

    # print("0")
    # gdf_no_overlap_0 = check_overlap_seperate_years_indexing(gdf, 0, 0)
    # results['gdf_no_overlap_0'] = len(gdf_no_overlap_0)

    # print("1")
    # gdf_no_overlap_pre_1 = check_overlap_seperate_years_indexing(gdf, 1, 0)
    # results['gdf_no_overlap_pre_1'] = len(gdf_no_overlap_pre_1)

    # print("2")
    # gdf_no_overlap_pre_2 = check_overlap_seperate_years_indexing(gdf, 2, 0)
    # results['gdf_no_overlap_pre_2'] = len(gdf_no_overlap_pre_2)

    # print("3")
    # gdf_no_overlap_pre_3 = check_overlap_seperate_years_indexing(gdf, 3, 0)
    # results['gdf_no_overlap_pre_3'] = len(gdf_no_overlap_pre_3)

    # print("4")
    # gdf_no_overlap_pre_4 = check_overlap_seperate_years_indexing(gdf, 4, 0)
    # results['gdf_no_overlap_pre_4'] = len(gdf_no_overlap_pre_4)

    # print("5")
    # gdf_no_overlap_pre_5 = check_overlap_seperate_years_indexing(gdf, 5, 0)
    # results['gdf_no_overlap_pre_5'] = len(gdf_no_overlap_pre_5)

    # print("6")
    # gdf_no_overlap_post_1 = check_overlap_seperate_years_indexing(gdf, 0, 1)
    # results['gdf_no_overlap_post_1'] = len(gdf_no_overlap_post_1)

    # print("7")
    # gdf_no_overlap_post_2 = check_overlap_seperate_years_indexing(gdf, 0, 2)
    # results['gdf_no_overlap_post_2'] = len(gdf_no_overlap_post_2)

    # print("8")
    # gdf_no_overlap_post_3 = check_overlap_seperate_years_indexing(gdf, 0, 3)
    # results['gdf_no_overlap_post_3'] = len(gdf_no_overlap_post_3)

    # print("9")
    # gdf_no_overlap_post_4 = check_overlap_seperate_years_indexing(gdf, 0, 4)
    # results['gdf_no_overlap_post_4'] = len(gdf_no_overlap_post_4)

    # print("10")
    # gdf_no_overlap_post_5 = check_overlap_seperate_years_indexing(gdf, 0, 5)
    # results['gdf_no_overlap_post_5'] = len(gdf_no_overlap_post_5)


    print("0")
    gdf_no_overlap_0 = check_overlap_specific_year(gdf, 0)
    results['gdf_no_overlap_0'] = len(gdf_no_overlap_0)

    print("1")
    gdf_no_overlap_pre_1 = check_overlap_specific_year(gdf, 1)
    results['gdf_no_overlap_pre_1'] = len(gdf_no_overlap_pre_1)

    print("2")
    gdf_no_overlap_pre_2 = check_overlap_specific_year(gdf, 2)
    results['gdf_no_overlap_pre_2'] = len(gdf_no_overlap_pre_2)

    print("3")
    gdf_no_overlap_pre_3 = check_overlap_specific_year(gdf, 3)
    results['gdf_no_overlap_pre_3'] = len(gdf_no_overlap_pre_3)

    print("4")
    gdf_no_overlap_pre_4 = check_overlap_specific_year(gdf, 4)
    results['gdf_no_overlap_pre_4'] = len(gdf_no_overlap_pre_4)

    print("5")
    gdf_no_overlap_pre_5 = check_overlap_specific_year(gdf, 5)
    results['gdf_no_overlap_pre_5'] = len(gdf_no_overlap_pre_5)

    print("6")
    gdf_no_overlap_post_1 = check_overlap_specific_year(gdf, -1)
    results['gdf_no_overlap_post_1'] = len(gdf_no_overlap_post_1)

    print("7")
    gdf_no_overlap_post_2 = check_overlap_specific_year(gdf,-2)
    results['gdf_no_overlap_post_2'] = len(gdf_no_overlap_post_2)

    print("8")
    gdf_no_overlap_post_3 = check_overlap_specific_year(gdf,-3)
    results['gdf_no_overlap_post_3'] = len(gdf_no_overlap_post_3)

    print("9")
    gdf_no_overlap_post_4 = check_overlap_specific_year(gdf,-4)
    results['gdf_no_overlap_post_4'] = len(gdf_no_overlap_post_4)

    print("10")
    gdf_no_overlap_post_5 = check_overlap_specific_year(gdf,-5)
    results['gdf_no_overlap_post_5'] = len(gdf_no_overlap_post_5)


    # Save the results to a CSV file
    results_df = pd.DataFrame(list(results.items()), columns=['Overlap_Type', 'Count'])
    output_file = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/overlap_specific_years.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
