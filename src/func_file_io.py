"""
func_file_io.py — File I/O Utilities
======================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Generic load/save helpers used across the pipeline. Supports CSV (with optional
WKT geometry), ESRI Shapefiles, and NetCDF files.
"""

import logging
import os
import shutil
import subprocess

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely import wkt


def load_data(file_path, crs="EPSG:4326"):
    """
    Load a data file into a GeoDataFrame or DataFrame depending on format and content.

    Supported formats:
      .csv  — loaded as DataFrame; if a 'geometry' column with WKT strings is
              present, converted to GeoDataFrame with the given CRS
      .shp  — loaded as GeoDataFrame via geopandas
      .nc   — loaded as xarray Dataset (chunked for memory efficiency)

    Parameters
    ----------
    file_path : str    path to the data file
    crs       : str    CRS to assign when reading a CSV with geometry (default EPSG:4326)

    Returns
    -------
    gpd.GeoDataFrame, pd.DataFrame, or xr.Dataset depending on file type.
    """
    if file_path.endswith('.csv'):
        logging.info(f"Loading CSV: {file_path}")
        df = pd.read_csv(file_path)
        if 'geometry' in df.columns:
            logging.info("Geometry column found — converting to GeoDataFrame")
            df['geometry'] = df['geometry'].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry='geometry').set_crs(crs)
        return df

    elif file_path.endswith('.shp'):
        logging.info(f"Loading shapefile: {file_path}")
        return gpd.read_file(file_path)

    elif file_path.endswith('.nc'):
        logging.info(f"Loading NetCDF: {file_path}")
        return xr.open_dataset(file_path, chunks={'time': 10, 'lat': 100, 'lon': 100})

    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .csv, .shp, or .nc")


def save_shapefile(data, file_path):
    """
    Save a GeoDataFrame to an ESRI Shapefile, overwriting any existing file.

    Creates the output directory if it does not exist. Removes all associated
    sidecar files (.shx, .dbf, .prj) before writing to avoid stale data.

    Parameters
    ----------
    data      : gpd.GeoDataFrame
    file_path : str   full path to the .shp output file
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    # Remove existing shapefile components to avoid leftover sidecar files
    if os.path.exists(file_path):
        logging.info(f"Overwriting existing shapefile: {file_path}")
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            sidecar = file_path.replace('.shp', ext)
            try:
                os.remove(sidecar)
            except OSError as e:
                logging.warning(f"Could not remove {sidecar}: {e}")

    logging.info(f"Saving shapefile → {file_path}")
    data.to_file(file_path, index=False)
    logging.info("Shapefile saved.")


def reproject(data, target_crs):
    """
    Reproject a GeoDataFrame to target_crs and return the result.

    Parameters
    ----------
    data       : gpd.GeoDataFrame
    target_crs : str   e.g. 'EPSG:27705'
    """
    logging.info(f"Reprojecting to {target_crs}")
    return data.to_crs(target_crs)


def reproject_and_save(data, target_crs, file_path):
    """Reproject data to target_crs and save as a shapefile."""
    save_shapefile(reproject(data, target_crs), file_path)


def save_gdf(gdf, output_dir, output_filename):
    """Save a GeoDataFrame to output_dir/output_filename."""
    output_path = os.path.join(output_dir, output_filename)
    logging.info(f"Saving result → {output_path}")
    gdf.to_file(output_path)


def delete_directory(input_dir):
    """Remove a directory and all its contents."""
    try:
        shutil.rmtree(input_dir)
        logging.info(f"Removed directory: {input_dir}")
    except OSError as e:
        logging.error(f"Error removing directory {input_dir}: {e}")


def run_shell_command(command):
    """Run a shell command via subprocess, raising on failure."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}\n{e}")


def load_tcc_dataset(tcc_nc_path):
    """
    Load a pre-processed Tree Canopy Cover (TCC) NetCDF file as an xarray Dataset.

    Parameters
    ----------
    tcc_nc_path : str   path to the TCC .nc file

    Returns
    -------
    xr.Dataset
    """
    logging.info(f"Loading TCC NetCDF: {tcc_nc_path}")
    return xr.open_dataset(tcc_nc_path)
