"""
Script to resampel, reproject, and crop a TreCoverCanopy raster file using GDAL.

Author: Franziska Müller
Date: 27.05.2024

This script performs the following steps:
1. Reproject the input raster file to EPSG:4326.
2. Change the resolution of the reprojected file to 20x20 meters.
3. Crop the resampled file based on the bounds region 8.
4. Delete intermediate files.

Make sure to update the input file paths, output file paths, and shapefile path
before running the script.
"""

import subprocess
import os

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

def main():

    # Define cropping bounds 
    minx, miny, maxx, maxy = -100, 25, -75, 40

    # Define file paths
    output_dir = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/"

    input_raster_file = output_dir + "nlcd_tcc_conus_2017_v2021-4.tif"
    output_file_resampled = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m.tif"
    output_file_epsg4326 = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326.tif"
    output_file_cropped = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"

    # Step 1: Change the resolution to 20x20 meters
    print("Step 1: Changing the resolution to 20x20 meters...")
    change_resolution(input_raster_file, output_file_resampled, (20, 20))

    # Step 2: Reproject the file to EPSG:4326
    print("Step 2: Reprojecting the file to EPSG:4326...")
    reproject_to_epsg4326(output_file_resampled, output_file_epsg4326)
    
    # Step 3: Crop the file based on the shapefile
    print("Step 3: Cropping the file based on the shapefile...")
    crop_to_shapefile(output_file_epsg4326, output_file_cropped, minx, miny, maxx, maxy)

    # Step 4: Delete intermediate files
    print("Step 4: Deleting intermediate files...")
    #os.remove(output_file_epsg4326)
    os.remove(output_file_resampled)

    print("Preprocessing completed")

if __name__ == "__main__":

    main()


