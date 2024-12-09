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


def process_and_merge_disturbances(gdf, max_iterations=10):
    """
    Iteratively merges intersecting polygons with matching attributes until no further changes.
    """
    gdf = gdf[gdf['PERCENT_AFFECTED'].isna() | gdf['PERCENT_AFFECTED'].str.contains("Severe", case=False, na=False)]
    gdf = gdf.drop(columns=['PERCENT_AFFECTED', 'HOST', 'HOST_CODE', 'DCA_CODE', 'DAMAGE_TYPE_CODE', 'DAMAGE_TYPE', 'cluster_id'])
    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    gdf = gdf.rename(columns={'Unnamed: 0': 'ID_E'})

    for iteration in range(max_iterations):
        print(f"> Iteration {iteration + 1}")
        spatial_intersections_att = gpd.sjoin(gdf, gdf, how="left", predicate="intersects", lsuffix="left", rsuffix="right",
            on_attribute=['DCA_ID', 'SURVEY_YEAR', 'REGION_ID', 'DA_Code_USDA'])
        #print(spatial_intersections_att.columns)
        merged_data = spatial_intersections_att.dissolve(by="ID_E_right", aggfunc="min").dissolve(by="ID_E_left", aggfunc="min").reset_index(drop=True)
        merged_data = merged_data.drop(columns=['index_right'])
        merged_data = merged_data.rename(columns={'ID_E_left': 'ID_E'})
        merged_data['ID_E'] = range(len(merged_data))
        merged_data = merged_data.reset_index(drop=True)

        if len(merged_data) == len(gdf):
            print(">> No more changes detected, stopping iterations.")
            break
        gdf = merged_data

    #print(f"Columns: {gdf.columns}")
    print(f"   Final number of merged records: {len(gdf)}")
    return gdf

def remove_overlapping_entries(df, year_column='SURVEY_YEAR', geometry_column='geometry', year_range=5):
    """
    Remove overlapping entries within a specified temporal window.
    """
    df = df.copy()
    overlapping_indices = set()
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Removing overlaps within ±{year_range} years"):
        time_window = (df[year_column].between(row[year_column] - year_range, row[year_column] + year_range))
        overlaps = df[time_window & df[geometry_column].intersects(row[geometry_column])]
        if len(overlaps) > 1:
            overlapping_indices.update(overlaps.index)

    return df.drop(index=overlapping_indices)

def keep_overlapping_entries(df, id_column='ID_E', year_column='SURVEY_YEAR', geometry_column='geometry', year_range=2):
    """
    Keep entries that overlap spatially within a specified temporal window, adding 'ID_O' for overlapping IDs.
    """
    df = df.copy()
    df['ID_O'] = None
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Finding overlaps within ±{year_range} years"):
        time_window = df[year_column].between(row[year_column] - year_range, row[year_column] + year_range)
        overlaps = df.loc[time_window & df[geometry_column].intersects(row[geometry_column])]
        if not overlaps.empty:
            df.at[index, 'ID_O'] = overlaps[id_column].tolist()

    return df.dropna(subset=['ID_O'])

def analyze_overlaps(gdf_overlap, id_col='ID_E', year_col='SURVEY_YEAR', dca_id_col='DCA_ID'):
    """
    Analyze overlapping entries to determine the longest duration of overlap 
    and count the unique DCA_IDs for each entry.
    """
    gdf_overlap['Longest_Duration'] = None
    gdf_overlap['DCA_ID_Count'] = None
    gdf_overlap['DCA_ID_List'] = None

    for idx, row in tqdm(gdf_overlap.iterrows(), total=gdf_overlap.shape[0], desc="Analyzing overlaps"):
        overlap_ids = row['ID_O']
        if overlap_ids:
            overlap_data = gdf_overlap[gdf_overlap[id_col].isin(overlap_ids)]
            gdf_overlap.at[idx, 'Longest_Duration'] = overlap_data[year_col].max() - overlap_data[year_col].min() + 1
            dca_ids = overlap_data[dca_id_col].tolist()
            gdf_overlap.at[idx, 'DCA_ID_Count'] = len(dca_ids)
            gdf_overlap.at[idx, 'DCA_ID_List'] = dca_ids

    return gdf_overlap

def analyze_and_enrich_overlaps(df, year_col='SURVEY_YEAR', id_col='ID_E', dca_col='DCA_ID'):
    """
    Explode the 'ID_O' column and enrich with 'O_Year', 'O_DCA_ID', and year differences.
    """
    exploded_df = df.explode('ID_O').drop(columns=['Longest_Duration', 'DCA_ID_Count', 'DCA_ID_List'])
    year_lookup = df.set_index(id_col)[year_col].to_dict()
    dca_lookup = df.set_index(id_col)[dca_col].to_dict()

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