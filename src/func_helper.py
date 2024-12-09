import json
from scipy.stats import zscore
import xarray as xr
import numpy as np
from func_file_io import load_data

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
