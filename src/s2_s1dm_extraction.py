#!/usr/bin/env python3
"""
Process NDVI time series for IDS and S1DM polygons,
calculate perfect seasonal difference, and save/update NetCDFs.
"""

import os
import warnings
from pathlib import Path

import geopandas as gpd
import xarray as xr
import rioxarray
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dotenv import load_dotenv
# -------------------------------------------------------------------
# Import your existing helper functions
# -------------------------------------------------------------------
from func_helpers_s2 import (
    save_or_update_nc,
    calculatePerfectSaison,
    crop_apply_tcc,
    extract_ndvi_for_polygons,
    contains_target_strlist,
    update_convex_with_minicubes
)
from func_helper import parse_custom_colors

# -------------------------------------------------------------------
# Suppress warnings for missing CRS
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# -------------------------------------------------------------------
# Workflow parameters
# -------------------------------------------------------------------
def load_environment_variables():
    """
    Load environment variables and prepare input/output paths.
    """
    from dotenv import load_dotenv
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    region = os.getenv("REGION")
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set in .env file.")
    
    buffer_size = 500
    tcc_year = 2016
    region_id = str(region).zfill(2)
    equi7_crs = os.getenv("EQUI7_NA_EPSG")
    
    output_folder = f"{os.getenv('PROJECT_DATA_DIR')}/region_{region_id}_buffer_{buffer_size}_ndvi_timeseries_ids_s1dm/"
    
    paths = {
        "ids": f"{os.getenv('RESULTS_DIR')}/region_{region_id}_dca_filtered_ids_usda_polygons_espg_27705.shp",
        "s1dm": f"{os.getenv('RESULTS_DIR')}/radar_enhanced_forest_disturbance_mapping_region_{region_id}_buffer_{buffer_size}_s1dm.shp",
        "convex": f"{os.getenv('RESULTS_DIR')}/convex_hulls_region_{region_id}_buffer_{buffer_size}_s1dm.shp",
        "minicube": f"{os.getenv('RESULTS_DIR')}/minicube_region_{region_id}_buffer_{buffer_size}_bounds.shp",
        "tcc": "tcc_equi7_reproj.tif", #f"{os.getenv('TCC_DIR')}/wp1_nlcd_tcc_conus_{tcc_year}_20m_EPSG_{equi7_crs}_cropped_normalized_region_{region_id}.tif",
        "s2_minicube_folder": os.getenv("SENTINEL2_CUBES_PP_DIR")
    }

    return region_id, equi7_crs, output_folder, paths

# -------------------------------------------------------------------
# Load shapefiles and convex updates
# -------------------------------------------------------------------
def load_and_prepare_data(paths, equi7_crs):
    """
    Load IDS, S1DM, convex, and minicube shapefiles and update convex hulls.
    """
    update_convex_with_minicubes(paths['convex'], paths['minicube'])
    
    ids = gpd.read_file(paths['ids']).to_crs(equi7_crs)
    s1dm = gpd.read_file(paths['s1dm']).to_crs(equi7_crs)
    convex = gpd.read_file(paths['convex']).to_crs(equi7_crs)
    minicube = gpd.read_file(paths['minicube']).to_crs(equi7_crs)
    
    print(f"Loaded shapefiles with CRS: IDS={ids.crs}, S1DM={s1dm.crs}, Convex={convex.crs}, Minicube={minicube.crs}")
    
    return ids, s1dm, convex, minicube

# -------------------------------------------------------------------
# Main processing workflow
# -------------------------------------------------------------------
def process_minicubes(
    minicube,
    convex,
    ids,
    s1dm,
    tcc,
    s2_minicube_folder,
    equi7_crs,
    output_folder,
    start_year=2016,
    method_seasonality="percentile",
    start_index=0,
    end_index=None,
    log_file="processed_minicubes.log"
):
    """
    Loop through minicubes and convex polygons to calculate NDVI differences.
    Allows processing a specific index range [start_index:end_index].
    """
    if end_index is None:
        end_index = len(minicube)

    with open(log_file, "a") as logf:  # append mode
        for i, minicube_row in tqdm(
            minicube.iloc[start_index:end_index].iterrows(),
            total=end_index - start_index,
            desc=f"Minicubes {start_index}–{end_index}"
        ):
            minicube_geom = minicube_row['geometry']
            nc_file = os.path.join(s2_minicube_folder, minicube_row['filename'])

            # Load NDVI cube
            cube = xr.open_dataset(nc_file)
            if not cube.rio.crs:
                cube = cube.rio.write_crs(equi7_crs)
            ndvi_cube = cube[['ndvi']]

            # Skip short time series
            if ndvi_cube['time'].size < 200:
                print(f"Skipping minicube {i}: only {ndvi_cube['time'].size} timesteps (< 200).")
                continue

            print(f"Processing minicube {i}")
            logf.write(f"{i}\n")  # log the minicube number immediately
            logf.flush()  # ensure it's written to disk

            ndvi_cube_masked = crop_apply_tcc(minicube_row, ndvi_cube, tcc, equi7_crs)

            convex_ids = convex[
                convex['mini_FIA'].apply(lambda x: contains_target_strlist(x, minicube_row['FIA']))
            ]['IDX_D']
            id_list = convex_ids.tolist()

            # Loop over all IDs
            for idx in id_list:
                ids_poly = ids[ids['IDX_D'] == idx]
                s1dm_poly = s1dm[s1dm['IDX_D'] == idx]

                if ids_poly.empty and s1dm_poly.empty:
                    continue  # skip if nothing to process

                dca_id = ids_poly["DCA_ID"].values[0] if not ids_poly.empty else s1dm_poly["DCA_ID"].values[0]
                base_dir = os.path.join(output_folder, str(dca_id), str(idx))
                os.makedirs(base_dir, exist_ok=True)

                # Clip NDVI for IDS polygons
                if not ids_poly.empty:
                    ndvi_ids = extract_ndvi_for_polygons(ndvi_cube_masked, ids_poly)
                    ids_diff, _, _ = calculatePerfectSaison(ndvi_ids, start_year, method_seasonality, percentile_value=90)
                    save_or_update_nc(ids_diff, idx, os.path.join(base_dir, f"{idx}_ndvi_ids.nc"))

                # Clip NDVI for S1DM polygons
                if not s1dm_poly.empty:
                    ndvi_s1dm = extract_ndvi_for_polygons(ndvi_cube_masked, s1dm_poly)
                    s1dm_diff, _, _ = calculatePerfectSaison(ndvi_s1dm, start_year, method_seasonality, percentile_value=90)
                    save_or_update_nc(s1dm_diff, idx, os.path.join(base_dir, f"{idx}_ndvi_s1dm.nc"))

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main():
    # Load environment variables and paths
    region_id, equi7_crs, output_folder, paths = load_environment_variables()

    # Load shapefiles
    ids, s1dm, convex, minicube = load_and_prepare_data(paths, equi7_crs)

    # Load TCC raster
    tcc = rioxarray.open_rasterio(paths['tcc'])
    # import rioxarray

    # tcc = rioxarray.open_rasterio(paths['tcc'], chunks={})  # open lazily
    # tcc = tcc.load()  # force load into memory


    # Run processing workflow
    process_minicubes(
        minicube=minicube,
        convex=convex,
        ids=ids,
        s1dm=s1dm,
        tcc=tcc,
        s2_minicube_folder=paths['s2_minicube_folder'],
        equi7_crs=equi7_crs,
        output_folder=output_folder,
        start_year=2016,
        method_seasonality="percentile",
        start_index=735,
        end_index=976 
    )

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()