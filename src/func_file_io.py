# file_io.py
import pandas as pd
import os
import geopandas as gpd
from shapely import wkt
import subprocess
import shutil
import xarray as xr

def load_data(file_path, crs="EPSG:4326"):
    """
    Load data from a CSV file or shapefile. If the CSV contains geometry data in WKT format,
    it will be loaded as a GeoDataFrame; otherwise, it will be loaded as a regular DataFrame.

    Args:
        file_path (str): Path to the data file (CSV or shapefile).
        crs (str): Coordinate reference system for the GeoDataFrame, default is "EPSG:4326".

    Returns:
        gpd.GeoDataFrame or pd.DataFrame: Loaded data as a GeoDataFrame if geometry is present,
                                          otherwise as a regular DataFrame.
    """
    if file_path.endswith('.csv'):
        print(f"Loading CSV file from: {file_path}")
        df = pd.read_csv(file_path)

        # Check if a 'geometry' column is present in the CSV
        if 'geometry' in df.columns:
            print("Detected 'geometry' column in CSV; converting to GeoDataFrame...")
            df['geometry'] = df['geometry'].apply(wkt.loads)  # Convert WKT to geometries
            gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs(crs)  # Set CRS
            return gdf
        else:
            print("No 'geometry' column detected; loading as a regular DataFrame.")
            return df

    elif file_path.endswith('.shp'):
        print(f"Loading shapefile from: {file_path}")
        return gpd.read_file(file_path)
    
    elif file_path.endswith('.nc'):
        print(f"Loading NetCDF file from: {file_path}")
        # Load the NetCDF file as an xarray Dataset
        dataset = xr.open_dataset(file_path, chunks={'time': 10, 'lat': 100, 'lon': 100})
        return dataset

    else:
        raise ValueError("Unsupported file format. Please use a .csv or .shp file.")

def save_shapefile(data, file_path):
    """
    Save a GeoDataFrame to a shapefile, overwriting if it already exists.
    
    Args:
        data (gpd.GeoDataFrame): GeoDataFrame to save.
        file_path (str): Path where the shapefile should be saved.
    """
    # Create the directory for the file if it does not exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File already exists at {file_path}. Overwriting...")
        # Remove the existing shapefile and associated files
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            try:
                os.remove(file_path.replace('.shp', ext))
            except OSError as e:
                print(f"Error removing file {file_path.replace('.shp', ext)}: {e}")

    # Save the GeoDataFrame to a shapefile
    print(f"Saving GeoDataFrame to shapefile at: {file_path}")
    data.to_file(file_path, index=False)
    print("File saved successfully.")


def transform_crs(data, target_crs):
    """
    Transform the CRS of a GeoDataFrame to the specified CRS.
    
    Args:
        data (gpd.GeoDataFrame): GeoDataFrame with geometries to transform.
        target_crs (str): Target CRS in EPSG format (e.g., 'EPSG:27705').
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with transformed CRS.
    """
    print(f"Transforming CRS to {target_crs}...")
    data_transformed = data.to_crs(target_crs)
    print("CRS transformation complete.")
    return data_transformed

def save_transformed_shapefile(data, target_crs, file_path):
    """
    Transform a GeoDataFrame to the target CRS and save it as a shapefile.
    
    Args:
        data (gpd.GeoDataFrame): GeoDataFrame to transform and save.
        target_crs (str): Target CRS in EPSG format (e.g., 'EPSG:27705').
        file_path (str): Path to save the transformed shapefile.
    """
    print(f"Saving transformed GeoDataFrame to shapefile at: {file_path}")
    data_transformed = transform_crs(data, target_crs)
    data_transformed.to_file(file_path, index=False)
    print("Transformed shapefile saved successfully.")

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running command: {command}\nError: {e}")



def save_result(gdf, output_dir, output_filename):
    """
    Save the resulting GeoDataFrame to a new shapefile.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame to be saved.
    - output_dir (str): Directory where the output shapefile will be saved.
    - output_filename (str): Name of the output shapefile.
    """
    print(f"Saving result to {os.path.join(output_dir, output_filename)}...")
    output_path = os.path.join(output_dir, output_filename)
    gdf.to_file(output_path)
    print("Result saved successfully.")


def remove_directory(input_dir):
    """
    Remove the specified directory and all of its contents.

    Parameters:
    - input_dir (str): Directory to be removed.
    """
    try:
        shutil.rmtree(input_dir)
        print(f"Successfully removed directory and all contents: {input_dir}")
    except OSError as e:
        print(f"Error removing directory and all contents: {input_dir} - {e}")
