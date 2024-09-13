"""
Script to resampel, reproject, and crop a TreCoverCanopy raster file using GDAL.

Author: Franziska Müller
Date: 27.05.2024

This script performs the following steps:
1. Reproject the input raster file to EPSG:4326.
2. Change the resolution of the reprojected file to 20x20 meters.
3. Crop the resampled file based on the rough bounds of region 8.
4. Delete intermediate files.

Make sure to update the input file paths, output file paths, and shapefile path
before running the script.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

def reproject_to_epsg4326(input_file, output_file):
    """
    Reproject the input file to EPSG:4326.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the reprojected file.
    """
    command = f"gdalwarp -t_srs EPSG:4326 {input_file} {output_file}"
    subprocess.run(command, shell=True)
    print("Reprojection completed.")


def change_resolution(input_file, output_file, resolution):
    """
    Change the resolution of the input file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the output file with changed resolution.
        resolution (tuple): Resolution in meters (e.g., (20, 20)).
    """
    command = f"gdalwarp -tr {resolution[0]} {resolution[1]} {input_file} {output_file}"
    subprocess.run(command, shell=True)
    print("Resolution change completed.")


def crop_to_shapefile(input_file, output_file, minx, miny, maxx, maxy):
    """
    Crop the input file based on a shapefile.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the cropped output file.
        shapefile (str): Path to the shapefile for cropping.
    """

    command = f"gdalwarp -te {minx} {miny} {maxx} {maxy} {input_file} {output_file}"
    subprocess.run(command, shell=True)
    print("Cropping completed.")


def get_region_shape_bounds(filepath, region_nr, output_path):
    # Load the shapefile
    usa = gpd.read_file(filepath)
    
    # Filter for the specified region
    country = usa[usa['REGION'] == region_nr]
    
    # Explode multipolygons (if any) and get the first part
    region = country.explode(index_parts=False)[0:1]
    
    # Get the bounding box of the region
    bounds = region.total_bounds  # Returns (x_min, y_min, x_max, y_max)
    
    # Extract values from the bounding box
    x_min, y_min, x_max, y_max = bounds
    
    # Plot the region shape
    fig, ax = plt.subplots(figsize=(8, 8))
    region.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=1)
    
    # Plot the bounding box as a rectangle
    bbox = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         edgecolor='brown', facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(bbox)
    
    # Plot the bounding box corners with red dots
    ax.scatter([x_min, x_max, x_min, x_max], [y_min, y_min, y_max, y_max], color='red', zorder=5)
    
    # Set axis labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Region {region_nr} with Bounding Box')
    
    # Save the figure to the specified output path
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    # Return the bounding box coordinates
    return x_min, y_min, x_max, y_max


def main():

    # Define cropping bounds 
    # minx, miny, maxx, maxy = -110, 22, -75, 42

    print("Set up environment variables ...")
    # Load environment variables from a .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)
    # Get the region from environment variables and pad it with leading zero if necessary
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("REGION environment variable is not set")

    print(f"Working on USDA Region {region} ...")
    region_id = str(region).zfill(2)

    # Define file paths
    output_dir = os.getenv('TCC_PATH')  # Fetch the TCC_PATH from environment variables
    if output_dir is None:
        raise ValueError("TCC_PATH environment variable is not set")

    # Make sure output_dir ends with a slash (to avoid issues in path concatenation)
    output_dir = output_dir.rstrip('/') + '/'

    # Define file paths using the variables
    input_raster_file = output_dir + "nlcd_tcc_conus_2017_v2021-4.tif"
    output_file_resampled = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m.tif"
    output_file_epsg4326 = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326.tif"
    output_file_cropped = output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_{region_id}.tif"

    region_shape_path = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
    output_path_figure = output_dir + f"region_{region_id}_bounds_equi7.png"


    # Step 1: Change the resolution to 20x20 meters
    print("Step 1: Changing the resolution to 20x20 meters...")
    change_resolution(input_raster_file, output_file_resampled, (20, 20))

    # Step 2: Reproject the file to EPSG:4326
    print("Step 2: Reprojecting the file to EPSG:4326...")
    reproject_to_epsg4326(output_file_resampled, output_file_epsg4326)


    # Step 4: Crop the file based on the shapefile
    print(f"Step 4: Extract the Bound of the Region {region_id} shapefile...")
    minx, miny, maxx, maxy = get_region_shape_bounds(region_shape_path, region_id, output_path_figure)

    # Step 5: Crop the file based on the shapefile
    print("Step 5: Cropping the TCC file based on the shapefile bounds...")
    crop_to_shapefile(output_file_epsg4326, output_file_cropped, minx, miny, maxx, maxy)

    print("Preprocessing completed")

if __name__ == "__main__":

    main()


