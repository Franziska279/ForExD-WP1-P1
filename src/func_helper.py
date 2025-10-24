import json
from scipy.stats import zscore
import xarray as xr
import numpy as np
import pandas as pd
from func_file_io import load_data
from shapely.validation import make_valid
from scipy.stats import ttest_rel

########################################################################################
#######                             Helper Functions                             #######
########################################################################################
def format_label(label):
    """
    Format a label by capitalizing each word and replacing underscores with spaces.

    Parameters:
    label (str): The input label with underscores.

    Returns:
    str: The formatted label with each word capitalized and underscores replaced by spaces.
    """
    return ' '.join(word.capitalize() for word in label.split('_'))


def format_label_count(dca_id, count):
    """
    Format the label to include the count of events.

    Parameters:
    dca_id (str): The DCA_ID to be formatted.
    count (int): The count of events to be included in the label.

    Returns:
    str: The formatted label including the count of events in parentheses.
    """
    return f'{dca_id.replace("_", " ").title()} ({count})'


def parse_custom_colors(colors_json):
    """
    Parse a JSON string to extract custom color mappings.

    Parameters:
    colors_json (str): A JSON string containing color mappings where keys are color names
                       and values are color codes.

    Returns:
    dict: A dictionary containing color mappings extracted from the JSON string. 
          Returns an empty dictionary if the input JSON string is empty or None.
    """
    # Check if the JSON string is provided
    if colors_json:
        try:
            # Attempt to parse the JSON string into a Python dictionary
            custom_colors = json.loads(colors_json)
        except json.JSONDecodeError:
            # Handle JSON decoding errors (e.g., invalid JSON format)
            print("Error: Invalid JSON format.")
            custom_colors = {}
    else:
        # Default to an empty dictionary if the JSON string is empty or None
        custom_colors = {}

    return custom_colors


def load_and_extract_region(path, region_id):
    """
    Loads the shapefile for the entire USA, extracts the specified region by region ID,
    and returns only the largest part of the region.
    
    Args:
        path (str): Path to the shapefile.
        region_id (str or int): The region ID to filter and retrieve.

    Returns:
        GeoDataFrame: Extracted region data for the largest part of the specified region ID.
    """
    # Load the shapefile data from the specified path
    usa_shape = load_data(path)
    
    # Filter for the specified region using the region_id
    region = usa_shape[usa_shape['REGION'] == region_id]
    
    # Explode geometries to ensure each part of multi-part geometries is separate,
    # then select only the largest part (first in the sequence)
    region_conus = region.explode(index_parts=False).iloc[0:1]
    
    return region_conus

import geopandas as gpd

def load_and_extract_region_crs(path, region_id, target_crs='EPSG:4326'):
    """
    Loads the shapefile for the entire USA, extracts the specified region by region ID,
    reprojects it to the target CRS, and returns only the largest part of the region.
    
    Args:
        path (str): Path to the shapefile.
        region_id (str or int): The region ID to filter and retrieve.
        target_crs (str): The CRS to reproject the region to (default is 'EPSG:27705').

    Returns:
        GeoDataFrame: Extracted region data for the largest part of the specified region ID.
    """
    # Load the shapefile data from the specified path
    usa_shape = gpd.read_file(path)
    
    # Filter for the specified region using the region_id
    region = usa_shape[usa_shape['REGION'] == region_id]
    
    # Explode geometries to ensure each part of multi-part geometries is separate,
    # then select only the largest part (first in the sequence)
    region_conus = region.explode(index_parts=False).iloc[0:1]
    
    # Reproject the region to the target CRS
    region_conus = region_conus.to_crs(target_crs)
    
    return region_conus


# def get_mainland(gdf_path, region_id):
#     """
#     Extracts the mainland parts of specified regions from the given GeoDataFrame.
    
#     Parameters:
#         gdf_path (str): Path to the input GeoDataFrame file.
#         region_ids (list): List of region IDs to process.
    
#     Returns:
#         GeoDataFrame: Cleaned GeoDataFrame with only the mainland parts of specified regions.
#     """
#     gdf = gpd.read_file(gdf_path)
#     cleaned_parts = []

#     for region_id in region_id:
#         region_gdf = gdf[gdf['REGION'] == region_id]
#         exploded = region_gdf.explode(index_parts=True)
#         exploded['area'] = exploded.area
        
#         # Select the largest mainland part
#         mainland_part = exploded.loc[exploded['area'].idxmax()]
#         cleaned_region = exploded[exploded['area'] == mainland_part['area']]
#         cleaned_parts.append(cleaned_region)

#     cleaned_parts_gdf = gpd.GeoDataFrame(pd.concat(cleaned_parts, ignore_index=True))

#     # Combine cleaned mainland regions with the rest of the dataset
#     mainland_gdf = pd.concat([gdf[~gdf['REGION'].isin(region_ids)], cleaned_parts_gdf], ignore_index=True)

#     return mainland_gdf



def get_mainland(gdf_path, regions_to_clean=['05', '08']):
    """
    Keep only the largest polygon (mainland) for specified regions in a GeoDataFrame.

    Parameters:
        gdf_path (str): Path to the input GeoDataFrame file.
        regions_to_clean (list): List of region codes to clean (keep only largest polygon).

    Returns:
        GeoDataFrame: Updated GeoDataFrame with small parts removed from specified regions.
    """
    gdf = gpd.read_file(gdf_path)
    # Keep all regions except '10'
    gdf = gdf[gdf['REGION'] != '10']
    cleaned_parts = []

    for region in regions_to_clean:
        region_gdf = gdf[gdf['REGION'] == region]
        exploded = region_gdf.explode(index_parts=True)
        exploded['area'] = exploded.geometry.area

        # Keep only the largest polygon
        largest_poly = exploded.loc[exploded['area'].idxmax()]
        cleaned_parts.append(largest_poly.to_frame().T)  # convert Series to DataFrame

    # Combine cleaned parts with the rest of the GeoDataFrame
    remaining = gdf[~gdf['REGION'].isin(regions_to_clean)]
    updated_gdf = pd.concat([remaining, pd.concat(cleaned_parts, ignore_index=True)], ignore_index=True)
    updated_gdf = gpd.GeoDataFrame(updated_gdf, geometry='geometry', crs=gdf.crs)

    return updated_gdf



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


def calculate_minimum_outerline_area(gdf):
    """
    Calculate the area of the minimum outerline (convex hull) of each multipolygon 
    in the GeoDataFrame in square kilometers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame with geometries.

    Returns:
    GeoDataFrame: GeoDataFrame with added column 'area_km2' for area in square kilometers.
    """
    # Calculate the convex hull for each geometry (minimum outerline)
    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.convex_hull if geom else None)

    # Calculate the area of the convex hulls in km² using the provided function
    gdf = calculate_area_in_km2(gdf)

    return gdf


import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

def merge_geometries_and_keep_columns(gdf):
    """
    Merge geometries by 'IDX_D' and 'S1_YEAR' into a single geometry (MultiPolygon) and
    keep the first value for all other columns.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame with geometries and other columns.
    
    Returns:
    - GeoDataFrame: A new GeoDataFrame with merged geometries and first values of other columns.
    """
    grouped_gdf = gdf.groupby(['IDX_D', 'S1_YEAR']).apply(
        lambda group: group.unary_union  # Merge the geometries within each group
    ).reset_index(name='geometry')

    # Step 2: For other columns, keep the first value
    for column in gdf.columns:
        if column not in ['IDX_D', 'S1_YEAR', 'geometry']:  # Skip 'IDX_D', 'S1_YEAR', and 'geometry'
            # Ensure the aggregation keeps the first value for each group
            grouped_gdf[column] = gdf.groupby(['IDX_D', 'S1_YEAR'])[column].first().values

    # Step 3: Convert to GeoDataFrame and ensure geometries are MultiPolygons if they aren't already
    grouped_gdf = gpd.GeoDataFrame(grouped_gdf, geometry='geometry')

    # Ensure the geometries are MultiPolygons if they aren't already
    grouped_gdf['geometry'] = grouped_gdf['geometry'].apply(
        lambda geom: MultiPolygon([geom]) if not isinstance(geom, MultiPolygon) else geom
    )

    # Step 4: Set CRS (coordinate reference system) if needed
    grouped_gdf.set_crs(gdf.crs, allow_override=True, inplace=True)

    return grouped_gdf



from shapely.ops import unary_union

def remove_drought(gdf):
    return gdf[gdf['DCA_ID'] != 'drought']

def remove_dca_ids(gdf, ids_to_remove):
    """
    Remove rows from a GeoDataFrame based on one or more DCA_ID values.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame.
    ids_to_remove : str or list of str
        DCA_ID(s) to remove. Can be a single string or a list of strings.

    Returns
    -------
    GeoDataFrame
        Filtered GeoDataFrame with specified DCA_ID(s) removed.
    """
    if isinstance(ids_to_remove, str):
        ids_to_remove = [ids_to_remove]
    return gdf[~gdf['DCA_ID'].isin(ids_to_remove)]


def closest_s1_year(group):
    # Ensure that both 'S1_YEAR' and 'SURVEY_Y' are numeric
    group['S1_YEAR'] = pd.to_numeric(group['S1_YEAR'], errors='coerce')
    group['SURVEY_Y'] = pd.to_numeric(group['SURVEY_Y'], errors='coerce')
    
    # Calculate the absolute difference between 'S1_YEAR' and 'SURVEY_Y'
    group['S1_Y_diff'] = (group['S1_YEAR'] - group['SURVEY_Y']).abs()
    
    # Find the index of the row with the minimum difference
    closest_idx = group['S1_Y_diff'].idxmin()
    
    return group.loc[closest_idx, 'S1_YEAR']


def add_signal_duration_column(refdm_gdf, crs=None):
    """
    Fügt eine Signal-Dauer-Spalte basierend auf der Häufigkeit von IDX_D hinzu und aggregiert
    nach IDX_D, so dass jedes IDX_D nur einmal erscheint. Dabei wird die Geometrie der verschiedenen
    IDX_D-Ereignisse zu einem MultiPolygon zusammengeführt. Alle Spalten, die für jedes IDX_D denselben Wert haben,
    bleiben erhalten, während Spalten mit unterschiedlichen Werten entsprechend aggregiert werden.

    Parameter:
    - refdm_gdf (GeoDataFrame): Das GeoDataFrame mit der Spalte IDX_D.
    - crs (str oder dict, optional): Das CRS, das der Geometrie zugewiesen werden soll.

    Rückgabe:
    - refdm_gdf_aggregated (GeoDataFrame): Aggregiertes GeoDataFrame mit zusätzlicher 'signal_duration'-Spalte.
    """

    # Zähle, wie oft jedes IDX_D vorkommt
    signal_duration_counts = refdm_gdf.groupby('IDX_D').size().reset_index(name='signal_duration')
    
    # Wir erstellen eine Aggregationsstrategie für alle Spalten
    aggregation = {
        'geometry': unary_union  # Aggregiert Geometrien zu einem MultiPolygon
    }
    
    # Aggregiere für alle anderen Spalten
    for column in refdm_gdf.columns:
        if column != 'IDX_D' and column != 'geometry':  # Alle anderen Spalten außer IDX_D und geometry
            if refdm_gdf[column].nunique() == 1:  # Wenn alle Werte in der Spalte gleich sind
                aggregation[column] = 'first'  # Behalte den ersten Wert
            else:
                aggregation[column] = 'first'  # Ansonsten könnte man z.B. 'first' oder 'mean' verwenden
    
    # Aggregiere das GeoDataFrame nach IDX_D
    refdm_gdf_aggregated = refdm_gdf.groupby('IDX_D').agg(aggregation).reset_index()
    
    # Add the 'S1_YEAR' closest to 'SURVEY_Y' for each 'IDX_D'
    refdm_gdf_aggregated['S1_YEAR'] = refdm_gdf.groupby('IDX_D').apply(closest_s1_year).values
    
    # Setze die Geometriespalte explizit
    refdm_gdf_aggregated = refdm_gdf_aggregated.set_geometry('geometry')

    # Merge die gezählten Signal-Dauern in das aggregierte GeoDataFrame
    refdm_gdf_aggregated = refdm_gdf_aggregated.merge(signal_duration_counts, on='IDX_D', how='left')
    
    # Falls ein CRS übergeben wurde, weise es der Geometrie zu
    if crs:
        refdm_gdf_aggregated.set_crs(crs, allow_override=True, inplace=True)
    
    return refdm_gdf_aggregated


def calculate_overlap_percentages(ids, s1dm):
    """Calculate the overlap percentage between IDS and S1DM geometries."""
    # Ensure CRS matches
    if ids.crs != s1dm.crs:
        s1dm = s1dm.to_crs(ids.crs)

    # Filter for common IDX_D values
    common_idx_d = set(ids["IDX_D"]).intersection(s1dm["IDX_D"])
    ids_common = ids[ids["IDX_D"].isin(common_idx_d)]
    s1cd_common = s1dm[s1dm["IDX_D"].isin(common_idx_d)]

    # Store results
    results = []
    for idx_d in common_idx_d:
        ids_poly = ids_common[ids_common["IDX_D"] == idx_d].geometry.union_all()
        s1cd_poly = s1cd_common[s1cd_common["IDX_D"] == idx_d].geometry.union_all()
        
        ids_area = ids_poly.area
        s1cd_area = s1cd_poly.area
        intersection_area = ids_poly.intersection(s1cd_poly).area
        
        percentage_ids = (intersection_area / ids_area) * 100 if ids_area > 0 else 0
        percentage_s1cd = (intersection_area / s1cd_area) * 100 if s1cd_area > 0 else 0
        
        dca_id = ids_common[ids_common["IDX_D"] == idx_d]["DCA_ID"].iloc[0]
        results.append({"DCA_ID": dca_id, "percentage_ids": percentage_ids, "percentage_s1cd": percentage_s1cd})

    return pd.DataFrame(results)




# ==============================================================
# HELPER FUNCTIONS SIMILARITY
# ==============================================================

def compute_overlap_jaccard(geom_candidate, geom_manual):
    """
    Compute Overlap (%) and Jaccard index between candidate and manual geometries.

    Parameters
    ----------
    geom_candidate : shapely.geometry
        Candidate geometry (IDS or S1DM)
    geom_manual : shapely.geometry
        Reference manual geometry

    Returns
    -------
    overlap_pct : float
        Percent of manual area overlapping with candidate
    jaccard : float
        Jaccard index (intersection/union)
    """
    geom_candidate = make_valid(geom_candidate)
    geom_manual = make_valid(geom_manual)

    inter = geom_candidate.intersection(geom_manual)
    union = geom_candidate.union(geom_manual)

    area_inter = inter.area
    area_union = union.area
    area_manual = geom_manual.area

    overlap_pct = (area_inter / area_manual) * 100 if area_manual > 0 else 0
    jaccard = (area_inter / area_union) if area_union > 0 else 0

    return overlap_pct, jaccard


def paired_ttest_significance(x, y, alpha1=0.05, alpha2=0.1, alternative="greater"):
    """
    One-sided paired t-test between S1DM and IDS values.

    Parameters
    ----------
    x, y : array-like
        Sample values (e.g., S1DM vs IDS metrics)
    alpha1 : float
        Primary significance threshold (default=0.05)
    alpha2 : float
        Secondary significance threshold (default=0.1)
    alternative : {"greater","less"}
        One-sided test direction

    Returns
    -------
    pvalue : float
        p-value from paired t-test
    sig_symbol : str
        "*" if p < alpha1, "+" if p < alpha2, else ""
    """
    t_res = ttest_rel(x, y, alternative=alternative)
    if t_res.pvalue < alpha1:
        sig = "*"
    elif t_res.pvalue < alpha2:
        sig = "+"
    else:
        sig = ""
    return t_res.pvalue, sig

