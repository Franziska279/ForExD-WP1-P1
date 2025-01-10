import os
import logging
import numpy as np
import xarray as xr
import rioxarray
from affine import Affine
import geopandas as gpd
from shapely.geometry import shape, box
import rasterio.features

def load_and_preprocess_dataset(input_file):
    """Load and preprocess the raster dataset."""
    dataset = xr.open_dataset(input_file)
    dataset = dataset.drop_vars(["x_bnds", "y_bnds"], errors='ignore')
    if 'unnamed' in dataset.variables:
        dataset = dataset.rename({'unnamed': 'layer'})
    if 'X' in dataset.variables and 'Y' in dataset.variables:
        dataset = dataset.rename({'X': 'x', 'Y': 'y'})
    dataset.rio.write_crs("+proj=aeqd +lat_0=52 +lon_0=-97.5 +datum=WGS84", inplace=True)
    return dataset.rio.reproject("EPSG:4326")

def apply_tcc_mask(dataset, tcc_path):
    """Applies a TCC mask to the dataset."""
    try:
        tcc = rioxarray.open_rasterio(tcc_path)
        subset = tcc.sel(x=slice(dataset['x'].min(), dataset['x'].max()),
                         y=slice(dataset['y'].max(), dataset['y'].min()))
        normalized_subset = subset.interp(x=dataset.coords['x'], y=dataset.coords['y'], method='nearest')
        return dataset.where(normalized_subset > 0.3, 0).fillna(0)
    except Exception as e:
        logging.error(f"Error applying TCC mask: {e}")
        return None

def extract_polygons_from_mask(filename, masked_data_array):
    """Extract polygons from masked data."""
    try:
        year = int(filename.split('_year_')[-1].split('_')[0])
        tile_name = filename[13:23]
        transform = Affine.translation(masked_data_array.x[0], masked_data_array.y[0]) * \
                    Affine.scale(masked_data_array.x[1] - masked_data_array.x[0], masked_data_array.y[1] - masked_data_array.y[0])
        binary_mask = (masked_data_array.squeeze("band")['layer'] > 0).astype(np.uint8)
        polygons = [shape(geom) for geom, value in rasterio.features.shapes(binary_mask.values, transform=transform) if value == 1]
        return gpd.GeoDataFrame({'geometry': polygons, 'S1_YEAR': year, 'S1_TILE': tile_name}, crs="EPSG:4326")
    except Exception as e:
        logging.error(f"Error extracting polygons: {e}")
        return gpd.GeoDataFrame()

def process_and_filter_polygons(dataset, ids_usda_path, s1_year, year_buffer, target_crs, output_dir, filename):
    """Processes and filters polygons."""
    try:
        ids_usda_gdf = gpd.read_file(ids_usda_path).to_crs("EPSG:3857")
        ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(500)
        ids_usda_gdf = ids_usda_gdf.to_crs("EPSG:4326")
        filtered_usda = ids_usda_gdf[(ids_usda_gdf['SURVEY_Y'] >= s1_year - year_buffer) & 
                                      (ids_usda_gdf['SURVEY_Y'] <= s1_year + year_buffer)]
        intersecting = gpd.sjoin(dataset, filtered_usda, predicate='intersects').dissolve(by='IDX_D').reset_index()
        intersecting = intersecting.to_crs(target_crs)
        os.makedirs(output_dir, exist_ok=True)
        intersecting.to_file(os.path.join(output_dir, f"{filename}.shp"), driver="ESRI Shapefile")
    except Exception as e:
        logging.error(f"Error during polygon processing: {e}")
