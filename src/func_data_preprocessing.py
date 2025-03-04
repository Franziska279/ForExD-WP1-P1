import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import xarray as xr
import os
import geopandas as gpd
import rasterio
from shapely.geometry import box, shape
from affine import Affine
import pandas as pd
import numpy as np
import rioxarray
from func_file_io import load_data


def rename_columns_and_process_data(gdf):
    """
    Rename columns to ensure no column name exceeds 10 characters, filter and process data.
    
    Parameters:
    - gdf (GeoDataFrame): The input geospatial dataframe.
    
    Returns:
    - GeoDataFrame: Processed and filtered GeoDataFrame.
    """
    gdf = gdf.rename(columns={col: col[:10] for col in gdf.columns})  # Truncate column names to 10 chars
    gdf = gdf[gdf['PERCENT_AF'].isna() | gdf['PERCENT_AF'].str.contains("Severe", case=False, na=False)]
    gdf = gdf.drop(columns=['PERCENT_AF', 'HOST', 'HOST_CODE', 'DCA_CODE', 'DAMAGE_TYP', 'DAMAGE_T_1', 'cluster_id'])
    gdf = gdf[gdf['SURVEY_YEA'] > 2009]  # Filter data based on SURVEY_YEA
    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    return gdf

def merge_and_iterate(gdf, max_iterations=10):
    """
    Iteratively merges intersecting geometries with the same DCA_ID and SURVEY_YEA
    until no further merges occur.
    
    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame with geometries to merge.
    - max_iterations (int): Maximum iterations to attempt.
    
    Returns:
    - GeoDataFrame: The fully merged GeoDataFrame.
    """
    prev_len = len(gdf)
    for iteration in range(max_iterations):
        print(f"> Iteration {iteration + 1}")
        gdf = merge_intersecting_geometries(gdf)
        if len(gdf) == prev_len:
            print(">> No more changes detected, stopping iterations.")
            break
        prev_len = len(gdf)
    print(f"   Final number of merged records: {len(gdf)}")
    return gdf

def merge_intersecting_geometries(gdf):
    """
    Merge geometries that spatially intersect and share the same DCA_ID & SURVEY_YEA.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame.
    
    Returns:
    - GeoDataFrame: GeoDataFrame with merged intersecting geometries.
    """
    gdf = gdf.reset_index(drop=True)
    spatial_intersections = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects", lsuffix="left", rsuffix="right")
    spatial_intersections = spatial_intersections[
        (spatial_intersections["SURVEY_YEA_left"] == spatial_intersections["SURVEY_YEA_right"]) &
        (spatial_intersections["DCA_ID_left"] == spatial_intersections["DCA_ID_right"])
    ]
    spatial_intersections = spatial_intersections.loc[:, ~spatial_intersections.columns.str.endswith('_right')]
    spatial_intersections.columns = spatial_intersections.columns.str.replace('_left', '', regex=False)
    
    gdf["group"] = -1
    group_id = 0
    for idx, row in spatial_intersections.iterrows():
        if gdf.at[row.name, "group"] == -1:
            gdf.at[row.name, "group"] = group_id
        intersecting_idx = spatial_intersections.index[spatial_intersections.index == row.name]
        gdf.loc[intersecting_idx, "group"] = gdf.at[row.name, "group"]
        group_id += 1

    return gdf.dissolve(by=["group"], aggfunc="first").reset_index(drop=True)

def remove_temporal_overlaps(gdf, year_column='SURVEY_YEA', geometry_column='geometry', year_range=5):
    """
    Remove entries that overlap temporally and spatially within the specified year range.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame.
    - year_column (str): Column name for the survey year.
    - geometry_column (str): Column name for geometry.
    - year_range (int): Temporal window to consider overlaps.
    
    Returns:
    - GeoDataFrame: GeoDataFrame with overlapping entries removed.
    """
    overlapping_indices = set()
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc=f"Removing overlaps within ±{year_range} years"):
        time_window = gdf[year_column].between(row[year_column] - year_range, row[year_column] + year_range)
        overlaps = gdf[time_window & gdf[geometry_column].intersects(row[geometry_column])]
        if len(overlaps) > 1:
            overlapping_indices.update(overlaps.index)
    return gdf.drop(index=overlapping_indices)

def keep_and_analyze_overlaps(gdf, id_column='ID_E', year_column='SURVEY_YEA', geometry_column='geometry', year_range=2):
    """
    Keep spatially overlapping entries, analyze overlaps, and add ID_O for overlapping IDs.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame.
    - id_column (str): ID column to use for overlap analysis.
    - year_column (str): Survey year column.
    - geometry_column (str): Geometry column for spatial analysis.
    - year_range (int): Year range within which overlaps are considered.
    
    Returns:
    - GeoDataFrame: Filtered and analyzed GeoDataFrame with overlap information.
    """
    gdf['ID_O'] = None
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc=f"Finding overlaps within ±{year_range} years"):
        time_window = gdf[year_column].between(row[year_column] - year_range, row[year_column] + year_range)
        overlaps = gdf.loc[time_window & gdf[geometry_column].intersects(row[geometry_column]) & (gdf[id_column] != row[id_column])]
        if not overlaps.empty:
            gdf.at[index, 'ID_O'] = overlaps[id_column].tolist()

    gdf_overlap_analyzed = gdf.dropna(subset=['ID_O'])
    gdf_overlap_analyzed['Longest_Duration'] = None
    gdf_overlap_analyzed['DCA_ID_Count'] = None
    gdf_overlap_analyzed['DCA_ID_List'] = None
    for idx, row in tqdm(gdf_overlap_analyzed.iterrows(), total=gdf_overlap_analyzed.shape[0], desc="Analyzing overlaps"):
        overlap_ids = row['ID_O']
        if overlap_ids:
            overlap_data = gdf_overlap_analyzed[gdf_overlap_analyzed[id_column].isin(overlap_ids)]
            gdf_overlap_analyzed.at[idx, 'Longest_Duration'] = overlap_data[year_column].max() - overlap_data[year_column].min() + 1
            dca_ids = overlap_data['DCA_ID'].tolist()
            gdf_overlap_analyzed.at[idx, 'DCA_ID_Count'] = len(dca_ids)
            gdf_overlap_analyzed.at[idx, 'DCA_ID_List'] = dca_ids

    return gdf_overlap_analyzed[gdf_overlap_analyzed.apply(lambda row: row['DCA_ID_List'] == [row['DCA_ID']], axis=1)]


def filter_and_enrich_overlaps(gdf, year_col='SURVEY_YEA', id_col='ID_E', dca_col='DCA_ID'):
    """
    Filter and enrich overlaps based on DCA_ID match, and explode overlap IDs.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame.
    - year_col (str): Survey year column.
    - id_col (str): ID column.
    - dca_col (str): DCA_ID column.
    
    Returns:
    - GeoDataFrame: Filtered and enriched GeoDataFrame.
    """
    exploded_df = gdf.explode('ID_O').drop(columns=['Longest_Duration', 'DCA_ID_Count', 'DCA_ID_List'])
    year_lookup = gdf.set_index(id_col)[year_col].to_dict()
    dca_lookup = gdf.set_index(id_col)[dca_col].to_dict()
    
    exploded_df['O_Year'] = exploded_df['ID_O'].map(year_lookup)
    exploded_df['O_DCA_ID'] = exploded_df['ID_O'].map(dca_lookup)
    exploded_df['O_Y_diff'] = exploded_df['O_Year'] - exploded_df[year_col]
    
    return exploded_df


def filter_disturbance_data(data, excluded_dca_types, start_year=2015, end_year=2021):
    """
    Filters the input data for specific time, size, disturbance types, and overlapping conditions.
    Removes overlapping entries with specified mismatches.
    
    Parameters:
    - data: GeoDataFrame containing the original data.
    - excluded_dca_types: List of disturbance types to exclude.
    - start_year: Start year for filtering.
    - end_year: End year for filtering.

    Returns:
    - combined_gdf: Filtered and cleaned GeoDataFrame.
    """
    
    # Step 1: Split Data into Overlapping and Non-overlapping Entries
    without_id_o = data[data['ID_O'].isnull()].copy()
    with_id_o = data[data['ID_O'].notnull()].copy()
    
    # Step 2: Apply Filters to Non-overlapping Entries
    non_overlap_filtered = without_id_o[
        (without_id_o['SURVEY_Y'] > start_year) & 
        (without_id_o['SURVEY_Y'] <= end_year) & 
        (~without_id_o['DCA_ID'].isin(excluded_dca_types))
    ].copy()

    # Calculate area and filter by size for non-overlapping entries
    non_overlap_filtered = calculate_area_in_km2(non_overlap_filtered)
    non_overlap_filtered = non_overlap_filtered[non_overlap_filtered['area_km2'] <= 15]

    # Step 3: Apply Filters to Overlapping Entries
    # Filter by time, type, and size for overlapping entries
    overlap_filtered = with_id_o[
        (with_id_o['SURVEY_Y'] > start_year) & 
        (with_id_o['SURVEY_Y'] <= end_year) & 
        (~with_id_o['DCA_ID'].isin(excluded_dca_types))
    ].copy()
    overlap_filtered = calculate_area_in_km2(overlap_filtered)
    overlap_filtered = overlap_filtered[overlap_filtered['area_km2'] <= 15]

    # Ensure 'ID_E' and 'ID_O' are integers
    overlap_filtered['ID_E'] = overlap_filtered['ID_E'].astype(int)
    overlap_filtered['ID_O'] = overlap_filtered['ID_O'].astype(int)
    overlap_filtered = overlap_filtered[overlap_filtered['ID_E'] != overlap_filtered['ID_O']]

    # Step 4: Identify and Remove Invalid Overlapping Matches
    # Find mismatches in DCA_ID between overlapping pairs
    mismatch_df = overlap_filtered[
        (overlap_filtered['DCA_ID'] != overlap_filtered['O_DCA_ID'])
    ]
    # Identify invalid overlap entries (size, type, time, or DCA mismatch)
    invalid_ids = set(mismatch_df['ID_E']).union(set(mismatch_df['ID_O']))
    valid_overlap_df = overlap_filtered[
        ~overlap_filtered['ID_E'].isin(invalid_ids) &
        ~overlap_filtered['ID_O'].isin(invalid_ids)&
        (overlap_filtered['ID_E'] != overlap_filtered['ID_O'])  # Exclude self overlaps
    ]

    # Step 5: Combine Non-overlapping and Valid Overlapping Entries
    combined_df = pd.concat([non_overlap_filtered, valid_overlap_df], ignore_index=True)
    
    # Step 6: Assign Unique Identifier and Return
    combined_df['IDX_D'] = combined_df.apply(lambda row: f"{row['DCA_ID']}_{row['SURVEY_Y']}_{row.name}", axis=1)
    combined_gdf = gpd.GeoDataFrame(combined_df, geometry='geometry')

    print(f"Final combined data size: {len(combined_gdf)}")
    return combined_gdf

def calculate_area_in_km2(gdf):
    """
    Calculate the area of each polygon in the GeoDataFrame in square kilometers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame with geometries.

    Returns:
    GeoDataFrame: GeoDataFrame with an added column for area in square kilometers.
    """
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    projected_gdf = gdf.to_crs('EPSG:3857')
    projected_gdf['area_km2'] = projected_gdf.geometry.area / 1e6
    gdf['area_km2'] = projected_gdf['area_km2']

    return gdf

def calculate_area_in_km2_s1cd(gdf):
    """
    Calculate the area of each polygon in the GeoDataFrame in square kilometers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame with geometries.

    Returns:
    GeoDataFrame: GeoDataFrame with an added column for area in square kilometers.
    """
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    projected_gdf = gdf.to_crs('EPSG:3857')
    projected_gdf['area'] = projected_gdf.geometry.area / 1e6
    gdf['area'] = projected_gdf['area']

    return gdf


######### S1CD Preprocessing   #####################

# Function to extract the year from the filename
def extract_year_from_filename(input_path):
    """
    Extract the year from the filename, which is expected to follow a '_year_' pattern.
    
    Parameters:
    - input_path (str): Path to the input file.
    
    Returns:
    - int: The extracted year.
    """
    filename = os.path.basename(input_path)
    try:
        s1_year = int(filename.split('_year_')[-1].split('_')[0])
        return s1_year
    except (IndexError, ValueError) as e:
        print(f"Error extracting year from filename {filename}: {e}")
        return None

# Function to extract a specific part of the filename
def extract_s1cd_filename_part(filename):
    """
    Extract the first 10 parts of the filename, split by underscores.
    
    Parameters:
    - filename (str): The input filename.
    
    Returns:
    - str: Extracted part of the filename.
    """
    parts = filename.split('_')
    return '_'.join(parts[:10])

# Function to load the dataset and preprocess
def load_and_preprocess_dataset(input_file):
    """
    Load and preprocess the S1 Change Detection raster file.
    
    Parameters:
    - input_file (str): Path to the input raster file.
    - filename (str): Name of the file used for logging purposes.
    
    Returns:
    - xarray.Dataset: Preprocessed dataset.
    """
    filename = os.path.basename(input_file)
    print(f"Loading and preprocessing dataset: {filename}")
    # dataset = xr.open_dataset(input_file)  # Assume load_data is a function to load the dataset
    # print('loaded .............')
    # dataset = drop_unnecessary_vars(dataset)
    # dataset = rename_variables(dataset)
    # dataset = reproject_to_wgs84(dataset)
    # return dataset

# Helper function to drop unnecessary variables
def drop_unnecessary_vars(dataset):
    """Drop unnecessary variables from the dataset."""
    variables_to_remove = ["x_bnds", "y_bnds"]
    dataset = dataset.drop_vars([var for var in variables_to_remove if var in dataset.variables])
    return dataset

# Helper function to rename variables
def rename_variables(dataset):
    """Rename variables for consistency."""
    if 'unnamed' in dataset.variables:
        dataset = dataset.rename({'unnamed': 'layer'})
    if 'X' in dataset.variables and 'Y' in dataset.variables:
        dataset = dataset.rename({'X': 'x', 'Y': 'y'})
    return dataset

# Helper function to reproject dataset to WGS84
def reproject_to_wgs84(dataset):
    """Reproject the dataset to the WGS 84 CRS."""
    crs_azimuthal_equidistant = "+proj=aeqd +lat_0=52 +lon_0=-97.5 +datum=WGS84 +units=m"
    crs_wgs84 = 'EPSG:4326'
    dataset.rio.write_crs(crs_azimuthal_equidistant, inplace=True)
    return dataset.rio.reproject(crs_wgs84)

def apply_tcc_mask(dataset, TCC_path_2017):

    # Step 1: Define the path to the TIF file
    print("Step 1: Get the TIF file...")
    #TCC_path_2017 = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"

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


# Function to extract polygons from the dataset
def extract_polygons_from_mask(masked_mc, filename):
    """
    Extract polygons from the masked dataset using a binary mask.
    
    Parameters:
    - masked_mc (xarray.DataArray): The masked data array.
    - filename (str): Filename used to extract year and tile info.
    
    Returns:
    - GeoDataFrame: A GeoDataFrame containing the extracted polygons.
    """
    s1_year = extract_year_from_filename(filename)
    tile_name = filename[13:23]
    
    # Get bounds of the dataset
    min_lon, max_lon, min_lat, max_lat = masked_mc['x'].min().item(), masked_mc['x'].max().item(), masked_mc['y'].min().item(), masked_mc['y'].max().item()
    
    # Create a mask and extract polygons
    transform = Affine.translation(masked_mc.x[0], masked_mc.y[0]) * Affine.scale(masked_mc.x[1] - masked_mc.x[0], masked_mc.y[1] - masked_mc.y[0])
    mask = (masked_mc['layer'] > 0).astype(np.uint8)
    shapes = list(rasterio.features.shapes(mask, transform=transform))
    polygons_list = [shape(geom) for geom, value in shapes if value == 1]

    # Convert to GeoDataFrame
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons_list, crs=masked_mc.spatial_ref)
    polygons_gdf['S1_YEAR'], polygons_gdf['S1_TILE'] = s1_year, tile_name

    return polygons_gdf

# Function to filter and save the polygons based on USDA data
def filter_and_save_polygons(ids_usda_path, polygons_gdf, s1_year, filter_years, filename, target_crs, output_folder):
    """
    Filter polygons based on USDA survey data and spatial intersection, then save to shapefile.
    
    Parameters:
    - ids_usda_path (str): Path to the USDA shapefile.
    - polygons_gdf (GeoDataFrame): Polygons to be filtered.
    - s1_year (int): Sentinel-1 year for filtering.
    - filter_years (int): Number of years to filter around the given year.
    - filename (str): Input filename for metadata.
    - target_crs (str): Target CRS.
    - output_folder (str): Directory to save the resulting shapefile.
    
    Returns:
    - GeoDataFrame: Filtered and merged polygons.
    """
    ids_usda_gdf = gpd.read_file(ids_usda_path)
    ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(0.005)
    
    # Filter USDA data based on year range
    year_range = (ids_usda_gdf['SURVEY_Y'] >= s1_year - filter_years) & (ids_usda_gdf['SURVEY_Y'] <= s1_year + filter_years)
    filtered_ids_usda = ids_usda_gdf[year_range]
    
    # Spatially join polygons with filtered USDA data
    intersecting_polygons = gpd.sjoin(polygons_gdf, filtered_ids_usda, how='inner', predicate='intersects')

    # Aggregate polygons and reproject to target CRS
    merged_polygons = intersecting_polygons.dissolve(by='IDX_D')
    merged_polygons = merged_polygons.to_crs(target_crs)

    # Save to shapefile
    os.makedirs(output_folder, exist_ok=True)
    shapefile_path = os.path.join(output_folder, f"{extract_s1cd_filename_part(filename)}.shp")
    merged_polygons.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"Intersecting polygons saved as shapefile: {shapefile_path}")

    return merged_polygons