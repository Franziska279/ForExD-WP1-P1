"""
TCCProcessor — NLCD Tree Canopy Cover Raster Preparation
=========================================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Prepares one year of NLCD Tree Canopy Cover (TCC) raster data for use as a
forest mask in the S1CD processing step. For each year, process() runs:

  1. Resample   -- change pixel resolution to 20 m (gdalwarp)
  2. Reproject  -- transform to the target CRS defined by TCC_CRS in .env
  3. Crop       -- clip to the bounding box of the study region
  4. Normalise  -- scale pixel values to [0, 1]
  5. Downsample -- reduce resolution further for overview plots, saved as NetCDF
  6. Plot       -- save a diagnostic figure with region boundary overlay

All file paths are constructed from .env templates (see environment/.env.example).
Logging is configured centrally in main.py -- do not call basicConfig here.
"""

import os
import logging
import numpy as np

import rasterio
import rioxarray

from dotenv import load_dotenv
from func_file_io import run_shell_command
from func_helper import load_region_boundary
from func_tcc_application import create_downsampled_tcc_map


class TCCProcessor:
    """
    Prepares one year of NLCD Tree Canopy Cover raster data for a given USFS region.
    Instantiate with env_path and tcc_year, then call process().
    """

    def __init__(self, env_path, tcc_year):
        load_dotenv(dotenv_path=env_path)

        self.region_id = str(os.getenv('REGION')).zfill(2)
        self.tcc_year  = tcc_year
        self.crs       = os.getenv('TCC_CRS')   # e.g. 'EPSG:27705'
        self.crs_code  = self.crs.split(':')[1]  # numeric part used in filename templates

        # Assemble file paths from .env directory and filename templates
        tcc_dir  = os.getenv('TCC_DIR')
        year_dir = os.path.join(tcc_dir, str(self.tcc_year))
        fmt = dict(tcc_year=self.tcc_year, region_id=self.region_id, crs=self.crs_code)

        self.region_shape_path        = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.input_raster_path        = os.path.join(year_dir, os.getenv('TCC_INPUT_RASTER').format(**fmt))
        self.resampled_raster_path    = os.path.join(year_dir, os.getenv('TCC_RESAMPLED_RASTER').format(**fmt))
        self.reprojected_raster_path  = os.path.join(year_dir, os.getenv('TCC_EPSG_RASTER').format(**fmt))
        self.cropped_raster_path      = os.path.join(year_dir, os.getenv('TCC_CROPPED_RASTER_TEMPLATE').format(**fmt))
        self.normalized_raster_path   = os.path.join(tcc_dir,  os.getenv('TCC_NORMALIZED_RASTER_TEMPLATE').format(**fmt))
        self.downsampled_tcc_path     = os.path.join(tcc_dir,  os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(**fmt))
        self.figure_bounds_path       = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_TCC_BOUNDS_TEMPLATE').format(**fmt))
        self.figure_bounds_shape_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_TCC_BOUNDS_SHAPE_TEMPLATE').format(**fmt))

        # Ensure all output directories exist
        for path in [
            self.region_shape_path, self.input_raster_path, self.resampled_raster_path,
            self.reprojected_raster_path, self.cropped_raster_path,
            self.normalized_raster_path, self.figure_bounds_path, self.figure_bounds_shape_path
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        logging.info(f"TCCProcessor initialised -- region {self.region_id}, year {self.tcc_year}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delete_if_exists(self, file_path):
        """Remove a file if it already exists, to allow clean overwrites."""
        if os.path.exists(file_path):
            logging.info(f"Removing existing file: {file_path}")
            os.remove(file_path)

    def _resample_raster(self, input_path, output_path, resolution):
        """Resample raster to the given pixel resolution (in metres) using gdalwarp."""
        logging.info(f"Resampling to {resolution} m -> {output_path}")
        self._delete_if_exists(output_path)
        run_shell_command(f"gdalwarp -tr {resolution[0]} {resolution[1]} {input_path} {output_path}")

    def _reproject_raster(self, input_path, output_path, crs):
        """Reproject raster to the given CRS using gdalwarp."""
        logging.info(f"Reprojecting to {crs} -> {output_path}")
        self._delete_if_exists(output_path)
        run_shell_command(f"gdalwarp -t_srs '{crs}' {input_path} {output_path}")

    def _crop_raster_to_bounds(self, input_path, output_path, minx, miny, maxx, maxy):
        """Crop raster to the given bounding box using gdalwarp."""
        logging.info(f"Cropping to [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}] -> {output_path}")
        self._delete_if_exists(output_path)
        run_shell_command(f"gdalwarp -te {minx} {miny} {maxx} {maxy} {input_path} {output_path}")

    def _get_region_bounds(self, fig_path):
        """
        Load the study-region boundary in TCC_CRS, save a diagnostic bounds figure,
        and return the bounding box as (minx, miny, maxx, maxy).
        """
        logging.info("Extracting region bounds...")
        region = load_region_boundary(self.region_shape_path, self.region_id, crs=self.crs)
        bounds = region.total_bounds
        #plot_region_bounds(region, *bounds, self.region_id, fig_path)
        logging.info(f"Region bounds: {bounds}")
        return bounds

    def _normalize_raster(self, input_path, output_path):
        """
        Scale pixel values to [0, 1] using the global min/max of the raster.
        Processes in rasterio block-sized chunks to avoid loading the full
        raster into memory at once.
        """
        self._delete_if_exists(output_path)

        # Pass 1 -- find global min and max across all blocks
        logging.info(f"Computing global min/max for {input_path}")
        global_min, global_max = float('inf'), float('-inf')
        with rasterio.open(input_path) as src:
            for _, window in src.block_windows(1):
                block = src.read(window=window)
                global_min = min(global_min, block.min())
                global_max = max(global_max, block.max())

        logging.info(f"Global min={global_min:.4f}, max={global_max:.4f}")
        if global_max == global_min:
            logging.warning("Min and max are identical -- skipping normalisation.")
            return

        # Pass 2 -- apply normalisation block by block and write output
        with rasterio.open(input_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32)
            with rasterio.open(output_path, 'w', **profile) as dst:
                for _, window in src.block_windows(1):
                    block = src.read(window=window)
                    normalised = ((block - global_min) / (global_max - global_min)).astype(np.float32)
                    dst.write(normalised, window=window)

        logging.info(f"Normalisation complete -> {output_path}")

    def _plot_raster_with_boundary(self, raster_path, fig_path):
        """Save a diagnostic figure of the TIF with the region boundary overlaid."""
        logging.info(f"Plotting raster with region boundary -> {fig_path}")
        region = load_region_boundary(self.region_shape_path, self.region_id, crs=self.crs)
        raster = rioxarray.open_rasterio(raster_path)
        #plot_tcc_region_bounds(raster, region, self.region_id, fig_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self):
        """
        Run the full TCC preparation pipeline for self.tcc_year:
          1. Resample input raster to 20 m pixel resolution
          2. Reproject to TCC_CRS (EPSG:27705 by default)
          3. Extract study-region bounding box and save bounds figure
          4. Crop reprojected raster to region bounding box
          5. Normalise pixel values to [0, 1]
          6. Downsample normalised raster and save as NetCDF for the S1CD masking step
          7. Save diagnostic figure with region boundary overlay
        """
        logging.info(f"TCCProcessor.process() -- region {self.region_id}, year {self.tcc_year}")

        self._resample_raster(self.input_raster_path, self.resampled_raster_path, (20, 20))
        self._reproject_raster(self.resampled_raster_path, self.reprojected_raster_path, self.crs)

        minx, miny, maxx, maxy = self._get_region_bounds(self.figure_bounds_path)
        self._crop_raster_to_bounds(self.reprojected_raster_path, self.cropped_raster_path, minx, miny, maxx, maxy)
        self._normalize_raster(self.cropped_raster_path, self.normalized_raster_path)

        create_downsampled_tcc_map(
            self.normalized_raster_path, self.region_shape_path,
            self.region_id, output_netcdf=self.downsampled_tcc_path
        )

        self._plot_raster_with_boundary(self.normalized_raster_path, self.figure_bounds_shape_path)
        logging.info(f"TCCProcessor.process() complete -- region {self.region_id}, year {self.tcc_year}")
