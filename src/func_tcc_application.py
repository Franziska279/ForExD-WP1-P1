"""
func_tcc_application.py — TCC Downsampling Utility
====================================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Contains create_downsampled_tcc_map(), the final step of the TCC pipeline.
Called by TCCProcessor.process() after the raster has been normalised.

The function clips the normalised TCC raster to the study region, downsamples
it spatially, and saves it as a NetCDF file. This lightweight NetCDF is what
the S1CD processor loads as its forest mask.
"""

import os
import logging
import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr

from func_helper import load_region_boundary


def create_downsampled_tcc_map(input_tiff, region_shapefile_path, region_id,
                                output_netcdf, target_crs="EPSG:4326",
                                clip_crs="EPSG:27705", downsample_factor=100):
    """
    Clip, downsample, and reproject a normalised TCC raster, saving the result
    as a NetCDF file for use as a forest mask in the S1CD processing step.

    Processing steps:
      1. Load the normalised TCC raster and assign clip_crs
      2. Clip to the study-region boundary
      3. Spatially downsample by downsample_factor (coarsen + mean)
      4. Convert to xarray Dataset, rename variable to 'tcc'
      5. Reproject to target_crs (EPSG:4326 by default)
      6. Set values >= 1 to NaN (artefacts from normalisation at boundaries)
      7. Save as NetCDF

    Parameters
    ----------
    input_tiff           : str   path to the normalised TCC GeoTIFF
    region_shapefile_path: str   path to the USFS administrative regions shapefile
    region_id            : str   USFS region code (e.g. '08')
    output_netcdf        : str   path for the output NetCDF file
    target_crs           : str   CRS for the output (default EPSG:4326)
    clip_crs             : str   CRS used for clipping (default EPSG:27705)
    downsample_factor    : int   spatial coarsening factor (default 100)

    Returns
    -------
    xr.Dataset or None if an error occurs.
    """
    logging.info(f"create_downsampled_tcc_map: starting for region {region_id}")

    try:
        # Load raster and assign the metric CRS used for clipping
        tcc = rioxarray.open_rasterio(input_tiff, masked=True).squeeze()
        tcc = tcc.rio.write_crs(clip_crs)

        # Load region boundary and reproject to match the raster CRS
        regions_gdf = gpd.read_file(region_shapefile_path)
        region_gdf  = regions_gdf[regions_gdf['REGION'] == region_id].to_crs(clip_crs)
        clip_geom   = region_gdf.unary_union

        # Clip raster to region boundary
        tcc = tcc.rio.clip([clip_geom], clip_crs, drop=True, from_disk=True)

        # Downsample: coarsen spatially and take the block mean
        tcc = tcc.coarsen(x=downsample_factor, y=downsample_factor, boundary='trim').mean()

        # Convert to Dataset and assign variable name
        tcc_ds = tcc.to_dataset(name='tcc')
        if 'spatial_ref' in tcc_ds:
            tcc_ds = tcc_ds.drop_vars('spatial_ref')

        # Reproject from metric CRS to geographic CRS (EPSG:4326)
        tcc_ds = tcc_ds.rio.write_crs(clip_crs).rio.reproject(target_crs)

        # Values >= 1 are normalisation boundary artefacts — set to NaN
        tcc_ds['tcc'] = tcc_ds['tcc'].where(tcc_ds['tcc'] < 1, np.nan)

        # Save output
        tcc_ds.to_netcdf(output_netcdf, mode='w')
        logging.info(f"Downsampled TCC NetCDF saved -> {output_netcdf}")

        return tcc_ds

    except Exception as e:
        logging.error(f"Error in create_downsampled_tcc_map: {e}")
        return None