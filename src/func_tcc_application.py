import os
import logging
import numpy as np
import xarray as xr
import rioxarray
from affine import Affine
import geopandas as gpd
from shapely.geometry import shape, box
import rasterio.features

from func_helper import load_and_extract_region

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



# def create_downsampled_tcc_map(input_tiff, region_shapefile_path, region_id, temp_netcdf, final_netcdf):
#     """
#     Downsamples and crops a forest canopy cover map to a specified region, saving the result as a NetCDF file.
#     """
#     logging.info("Starting the downsampling and cropping process")
    
#     try:

#         # Load raster
#         forest_cover = rioxarray.open_rasterio(input_tiff, masked=True).squeeze()

#         # Ensure CRS
#         forest_cover = forest_cover.rio.write_crs("EPSG:4326")

#         # Load region geometry
#         #region_geometry = load_and_extract_region(region_shapefile_path, region_id).unary_union
#         #region_gdf = region_gdf.to_crs(forest_cover.rio.crs)  # Ensure same CRS
        
#         # Load and ensure CRS of forest cover map
#         #forest_cover = rioxarray.open_rasterio(input_tiff, masked=True).squeeze()
#         #region_geometry = region_geometry.to_crs(forest_cover.rio.crs)

#         #forest_cover = forest_cover.rio.write_crs("EPSG:4326")
        
#         # Downsample the raster data
#         downsample_factor = 100
#         forest_cover = forest_cover.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()
        
#         forest_cover.to_netcdf(temp_netcdf)

#         # Optional weiterverarbeiten
#         processed_data = xr.open_dataset(temp_netcdf).rename({'__xarray_dataarray_variable__': 'tcc'})
#         if 'spatial_ref' in processed_data:
#             processed_data = processed_data.drop_vars('spatial_ref')

#         # Alle Werte >= 1 zu NaN machen
#         processed_data['tcc'] = processed_data['tcc'].where(processed_data['tcc'] < 1, np.nan)

#         # Speichern
#         processed_data.to_netcdf(final_netcdf, mode='w')

#         print(f"Successfully saved the final NetCDF file to {final_netcdf}")
    
#     except Exception as e:
#         logging.error(f"Error during processing: {e}")
#         return None
    
#     finally:
#         # Cleanup intermediate file
#         if os.path.exists(temp_netcdf):
#             os.remove(temp_netcdf)
#             logging.info(f"Deleted intermediate file: {temp_netcdf}")


import os
import logging
import geopandas as gpd
import rioxarray
import xarray as xr
import numpy as np

def create_downsampled_tcc_map(input_tiff, region_shapefile_path, region_id, temp_netcdf, final_netcdf,
                               target_crs="EPSG:4326", clip_crs="EPSG:27705", downsample_factor=100):
    """
    Downsamples and crops a forest canopy cover map to a specified region, saving the result as a NetCDF file.
    
    Parameters:
    - input_tiff: Path to input raster TIFF
    - region_shapefile_path: Path to region shapefile
    - region_id: ID of the region to clip
    - temp_netcdf: Path to temporary NetCDF file
    - final_netcdf: Path to final NetCDF file
    - target_crs: CRS for output NetCDF (default EPSG:4326)
    - clip_crs: CRS to use for clipping (default EPSG:27705)
    - downsample_factor: Factor to coarsen raster
    """
    logging.info("Starting the downsampling and cropping process")
    
    try:
        # 1️⃣ Load raster
        forest_cover = rioxarray.open_rasterio(input_tiff, masked=True).squeeze()

        # 2️⃣ Ensure raster CRS
        forest_cover = forest_cover.rio.write_crs(clip_crs)

        # 3️⃣ Load region and select only desired region
        region_gdf = gpd.read_file(region_shapefile_path)
        region = region_gdf[region_gdf['REGION'] == region_id]

        # 4️⃣ Reproject region to raster CRS
        region_proj = region.to_crs(clip_crs)

        # 5️⃣ Combine all polygons into one
        region_geometry = region_proj.unary_union

        # 6️⃣ Clip raster to region
        forest_cover = forest_cover.rio.clip([region_geometry], clip_crs, drop=True, from_disk=True)

        # 7️⃣ Downsample raster
        forest_cover = forest_cover.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()

        # 8️⃣ Convert to Dataset and rename variable
        processed_data = forest_cover.to_dataset(name='tcc')

        # 9️⃣ Remove spatial_ref if exists
        if 'spatial_ref' in processed_data:
            processed_data = processed_data.drop_vars('spatial_ref')

        # 🔟 Set target CRS (EPSG:4326)
        processed_data = processed_data.rio.write_crs(clip_crs)  # aktuell Meter-CRS
        processed_data = processed_data.rio.reproject(target_crs)  # reproject to EPSG:4326

        # 1️⃣1️⃣ Replace all values >= 1 with NaN
        processed_data['tcc'] = processed_data['tcc'].where(processed_data['tcc'] < 1, np.nan)

        # 1️⃣2️⃣ Save final NetCDF
        processed_data.to_netcdf(final_netcdf, mode='w')
        logging.info(f"Successfully saved the final NetCDF file to {final_netcdf}")

        return processed_data
    
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return None
    
    finally:
        # Cleanup intermediate file
        if os.path.exists(temp_netcdf):
            os.remove(temp_netcdf)
            logging.info(f"Deleted intermediate file: {temp_netcdf}")
