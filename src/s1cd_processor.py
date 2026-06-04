"""
S1CDProcessor — Sentinel-1 Change Detection Processing
=======================================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Matches Sentinel-1 SAR change-detection tiles against filtered USDA IDS
disturbance polygons to produce the refined disturbance mapping (S1DM).

process_files() orchestrates the full pipeline:
  1. Collect S1 tile boundaries and save as a reference shapefile
  2. For each S1 tile (in parallel):
       a. Parse the survey year from the filename
       b. Load and preprocess the NetCDF tile
       c. Apply TCC forest mask (using the year prior to the S1 detection year)
       d. Vectorise change-detected pixels into polygons
       e. For each spatial buffer: intersect with IDS polygons and save
  3. Merge all per-tile shapefiles per buffer, filter by area, and save the
     final S1DM shapefile

All paths are set in __init__ from .env variables.
Logging is configured centrally in main.py — do not call basicConfig here.
"""

import os
import logging
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from shapely.geometry import box

from func_file_io import load_data
from func_data_preprocessing import get_tile_basename
from func_s1cd_preprocessing import (
    drop_coordinate_bounds,
    standardize_variable_names,
    merge_shapefiles_from_dir,
    intersect_s1cd_with_ids,
    vectorize_change_detections,
    mask_non_forest_pixels,
    load_s1cd_tile,
    parse_year_from_filename,
    filter_by_max_area,
)


class S1CDProcessor:
    """
    Matches Sentinel-1 change-detection tiles to USDA IDS polygons.
    Instantiate with env_path and processing parameters, then call process_files().
    """

    def __init__(self, env_path, buffer_years, spatial_buffer, max_jobs):
        if not Path(env_path).exists():
            raise FileNotFoundError(f".env file not found at {env_path}")
        load_dotenv(dotenv_path=env_path)

        region = os.getenv('REGION')
        if not region:
            raise ValueError("'REGION' environment variable is not set.")

        self.region_id      = str(region).zfill(2)
        self.target_crs     = os.getenv('TARGET_CRS')
        self.equi7_crs      = os.getenv('EQUI7_NA_EPSG')
        self.tcc_threshold  = float(os.getenv('TCC_THRESHOLD', 0.3))
        self.buffer_years   = buffer_years
        self.spatial_buffer = spatial_buffer
        self.max_jobs       = max_jobs

        # Input / output directories
        self.input_dir       = os.getenv('SENTINEL1_TILES_DIR')
        self.intermediate_dir = os.getenv('INTERMEDIATE_FILES_DIR')
        self.metadata_dir    = os.getenv('METADATA_FILES_DIR')

        # Derived file paths
        results_dir = os.getenv('RESULTS_DIR')
        self.ids_filtered_path    = os.path.join(results_dir, os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        self.s1_tiles_boundary_path = os.path.join(results_dir, os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id))

        # TCC path template — filled per tile in _process_single_tile
        tcc_crs_code = os.getenv('TCC_CRS', 'EPSG:27705').split(':')[1]
        self.tcc_raster_template = os.path.join(
            os.getenv('TCC_DIR'),
            os.getenv('TCC_NORMALIZED_RASTER_TEMPLATE').format(
                region_id=self.region_id, crs=tcc_crs_code, tcc_year='{tcc_year}'
            )
        )

        # Output path template for final S1DM shapefile per buffer
        self.s1dm_output_template = os.path.join(
            results_dir, os.getenv('S1DM_SHAPE_FILE').format(region_id=self.region_id, buffer='{buffer}')
        )

        # Ensure required directories exist
        for d in [self.input_dir, self.intermediate_dir, self.metadata_dir]:
            os.makedirs(d, exist_ok=True)

        logging.info(f"S1CDProcessor initialised — region {self.region_id}, buffers {self.spatial_buffer} m")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_files(self):
        """Run the full S1CD processing pipeline."""
        logging.info("=== S1CDProcessor: starting ===")

        self._collect_s1_tile_boundaries()

        input_files = sorted(f for f in os.listdir(self.input_dir) if not f.endswith('.py'))
        if not input_files:
            logging.warning("No input files found — aborting.")
            return

        logging.info(f"Processing {len(input_files)} tiles with {self.max_jobs} workers...")
        with ProcessPoolExecutor(max_workers=self.max_jobs) as executor:
            results = list(executor.map(self._process_single_tile, input_files))

        n_ok  = sum(results)
        n_err = len(results) - n_ok
        logging.info(f"Tile processing complete: {n_ok} succeeded, {n_err} failed.")
        if n_err:
            logging.warning(f"{n_err} tiles failed — check logs above for details.")

        self._merge_and_save_shapefiles()
        logging.info("=== S1CDProcessor: done ===")

    # ------------------------------------------------------------------
    # Per-tile processing (parallelised)
    # ------------------------------------------------------------------

    def _process_single_tile(self, file_name):
        """
        Full processing pipeline for one S1 tile:
          a. Parse survey year from filename
          b. Load and preprocess the NetCDF tile
          c. Apply TCC forest mask (year = s1_year - 1)
          d. Vectorise change-detected pixels into polygons
          e. Intersect with IDS polygons for each spatial buffer and save

        Returns True on success, False on any error.
        """
        input_path = os.path.join(self.input_dir, file_name)
        if not os.path.exists(input_path):
            logging.error(f"File not found: {input_path}")
            return False

        try:
            # a. Parse year from filename (S1 detection year)
            s1_year = parse_year_from_filename(input_path)
            logging.info(f"Processing {file_name} (year {s1_year})")

            # b. Load and preprocess
            s1cd_tile = load_s1cd_tile(input_path)

            # c. Apply TCC mask — use the year prior to the S1 detection year
            #    so the mask reflects pre-disturbance forest cover
            tcc_path = self.tcc_raster_template.format(tcc_year=s1_year - 1)
            if not os.path.exists(tcc_path):
                logging.warning(f"TCC file not found for year {s1_year - 1}: {tcc_path} — skipping tile.")
                return False
            s1cd_tile = mask_non_forest_pixels(s1cd_tile, tcc_path, threshold=self.tcc_threshold)

            # d. Vectorise: raster mask → polygon GeoDataFrame
            detections_gdf = vectorize_change_detections(file_name, s1cd_tile)

            # e. Intersect with IDS polygons for each buffer size
            tile_basename = get_tile_basename(file_name)
            for buffer in self.spatial_buffer:
                output_shp  = os.path.join(self.intermediate_dir, f"buffer_{buffer}", f"{tile_basename}.shp")
                output_meta = os.path.join(self.metadata_dir,     f"buffer_{buffer}", f"metadata_{tile_basename}.csv")
                os.makedirs(os.path.dirname(output_shp),  exist_ok=True)
                os.makedirs(os.path.dirname(output_meta), exist_ok=True)

                intersect_s1cd_with_ids(
                    detections_gdf, self.ids_filtered_path, s1_year,
                    self.buffer_years, buffer, input_path,
                    self.target_crs, output_shp, output_meta, tile_basename
                )

            logging.info(f"Tile {file_name} complete.")
            return True

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Post-processing: merge per-tile shapefiles into final S1DM output
    # ------------------------------------------------------------------

    def _merge_and_save_shapefiles(self):
        """
        For each spatial buffer:
          1. Merge all per-tile intermediate shapefiles into one GeoDataFrame
          2. Filter out polygons larger than 15 km² (area artefacts)
          3. Save as the final S1DM shapefile
          4. Delete the intermediate buffer directory
        """
        logging.info("Merging per-tile shapefiles into final S1DM outputs...")

        for buffer in self.spatial_buffer:
            buffer_dir = os.path.join(self.intermediate_dir, f'buffer_{buffer}')
            output_path = self.s1dm_output_template.format(buffer=buffer)

            merged_gdf = merge_shapefiles_from_dir(buffer_dir, self.target_crs)
            if merged_gdf is None or merged_gdf.empty:
                logging.warning(f"No shapefiles to merge for buffer {buffer} m — skipping.")
                continue

            filtered_gdf = filter_by_max_area(merged_gdf, self.target_crs)
            if filtered_gdf.empty:
                logging.warning(f"No polygons remain after area filter for buffer {buffer} m — skipping.")
                continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                filtered_gdf.to_file(output_path, driver='ESRI Shapefile')
                logging.info(f"S1DM shapefile saved ({buffer} m buffer) -> {output_path}")
            except Exception as e:
                logging.error(f"Error saving S1DM shapefile for buffer {buffer}: {e}")
                continue

            # Clean up intermediate per-tile files
            try:
                shutil.rmtree(buffer_dir)
                logging.info(f"Removed intermediate directory: {buffer_dir}")
            except OSError as e:
                logging.error(f"Could not remove {buffer_dir}: {e}")

        logging.info("Merge complete.")

    # ------------------------------------------------------------------
    # Tile boundary collection (used for the study-area figure)
    # ------------------------------------------------------------------

    def _collect_s1_tile_boundaries(self):
        """
        Read all S1 tile NetCDF files, extract their spatial bounding boxes in the
        original AEQD projection, and save as a shapefile for the study-area figure.

        Note: coordinates are read before reprojection, so the bounding boxes are
        in the raw Azimuthal Equidistant CRS of the S1 tiles, not EPSG:27705.
        """
        logging.info("Collecting S1 tile bounding boxes...")
        input_files = sorted(f for f in os.listdir(self.input_dir) if not f.endswith('.py'))

        bounds_list = []
        for file_name in input_files:
            file_path = os.path.join(self.input_dir, file_name)
            try:
                ds = xr.open_dataset(file_path)
                ds = drop_coordinate_bounds(ds)
                ds = standardize_variable_names(ds)

                lon_min, lon_max = float(ds['x'].min()), float(ds['x'].max())
                lat_min, lat_max = float(ds['y'].min()), float(ds['y'].max())
                bounds_list.append(box(lon_min, lat_min, lon_max, lat_max))

            except Exception as e:
                logging.error(f"Error reading bounds from {file_name}: {e}")

        if not bounds_list:
            logging.warning("No tile boundaries collected — boundary shapefile not created.")
            return

        tiles_gdf = gpd.GeoDataFrame(geometry=bounds_list, crs=self.equi7_crs)
        tiles_gdf = tiles_gdf.drop_duplicates().reset_index(drop=True)

        os.makedirs(os.path.dirname(self.s1_tiles_boundary_path), exist_ok=True)
        tiles_gdf.to_file(self.s1_tiles_boundary_path, driver="ESRI Shapefile")
        logging.info(f"Tile boundary shapefile saved -> {self.s1_tiles_boundary_path}")