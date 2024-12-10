import geopandas as gpd
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv
import rasterio
import numpy as np
from rasterio.enums import Resampling
import rioxarray
from matplotlib.patches import Patch
from func_plots import plot_region_bounds, plot_tcc_region_bounds
from func_file_io import load_data, save_shapefile, run_command
from func_helper import load_and_extract_region

class TCCProcessor:
    """
    TCCProcessor class to handle raster processing tasks such as resolution adjustment, reprojection, and cropping for a specific USDA region.
    """

    def __init__(self, env_path):
        # Load environment variables for configuration
        load_dotenv(dotenv_path=env_path)
        
        # Attributes for region and paths
        self.region = os.getenv('REGION')
        self.region_id = str(self.region).zfill(2)  # Pad region ID for consistent formatting
        
        # Define output directory and verify its presence
        self.output_dir = os.getenv('TCC_PATH')
        if self.output_dir is None:
            raise ValueError("TCC_PATH environment variable is not set")
        self.output_dir = self.output_dir.rstrip('/') + '/'
        
        # Define file paths for input and output TIFFs, shapefiles, and figures
        self.input_raster_file = self.output_dir + "nlcd_tcc_conus_2017_v2021-4.tif"
        self.output_file_resampled = self.output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m.tif"
        self.output_file_epsg4326 = self.output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326.tif"
        self.output_file_cropped_epsg4326 = self.output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_{self.region_id}.tif"
        self.normalized_output_file = self.output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_normalized_region_{self.region_id}.tif"
        self.region_shape_path_epsg4326 = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
        self.output_path_figure_epsg4326 = self.output_dir + f"bounds_epsg4326_region_{self.region_id}.png"
        self.tcc_output_path_figure_epsg4326 = self.output_dir + f"tcc_bounds_epsg4326_region_{self.region_id}.png"
        #self.tcc_output_path_figure_epsg27705 = self.output_dir + f"tcc_bounds_epsg27705_region_{self.region_id}.png"

    def __confirm_and_delete(self, file_path):
        """
        Confirm with the user whether to overwrite an existing file.
        If confirmed, delete the file before proceeding.
        
        Args:
            file_path (str): The path to the file that may be overwritten.
        
        Returns:
            bool: True if the user wants to overwrite, False otherwise.
        """
        if os.path.exists(file_path):
            while True:
                choice = input(f"File {file_path} already exists. Do you want to overwrite it? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    os.remove(file_path)  # Delete the file if it exists and user chooses to overwrite
                    print(f"Deleted existing file: {file_path}")
                    return True
                elif choice in ['n', 'no']:
                    print(f"Using existing file: {file_path}.")
                    return False
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        return True  # No file to overwrite

    def __reproject_to_crs(self, input_file, output_file, crs):
        """
        Reprojects a given raster file to a specified Coordinate Reference System (CRS).
        
        Args:
            input_file (str): Path to the input raster file.
            output_file (str): Path to the output reprojected raster file.
            crs (str): Target CRS in EPSG format (e.g., 'EPSG:4326').
        """
        if not self.__confirm_and_delete(output_file):
            return
        command = f"gdalwarp -t_srs '{crs}' {input_file} {output_file}"
        run_command(command)
        print("Reprojection completed.")

    def __change_resolution(self, input_file, output_file, resolution):
        """
        Changes the resolution of a raster file using gdalwarp.
        
        Args:
            input_file (str): Path to the input raster file.
            output_file (str): Path to the output resampled raster file.
            resolution (tuple): Target resolution (e.g., (20, 20) for 20m x 20m).
        """
        if not self.__confirm_and_delete(output_file):
            return
        command = f"gdalwarp -tr {resolution[0]} {resolution[1]} {input_file} {output_file}"
        run_command(command)
        print("Resolution change completed.")

    def __crop_to_bounds(self, input_file, output_file, minx, miny, maxx, maxy):
        """
        Crops a raster file to specified bounding coordinates.
        
        Args:
            input_file (str): Path to the input raster file.
            output_file (str): Path to the output cropped raster file.
            minx, miny, maxx, maxy (float): Bounding box coordinates.
        """
        if not self.__confirm_and_delete(output_file):
            return
        command = f"gdalwarp -te {minx} {miny} {maxx} {maxy} {input_file} {output_file}"
        run_command(command)
        print("Cropping completed.")

    def __get_region_shape_bounds(self, fig_path):
        """
        Loads the shapefile for the specified region, extracts and plots the bounding box coordinates.
        
        Returns:
            tuple: Bounds as (minx, miny, maxx, maxy) for cropping.
        """
        region = load_and_extract_region(self.region_shape_path_epsg4326, self.region_id)
        bounds = region.total_bounds
        x_min, y_min, x_max, y_max = bounds

        plot_region_bounds(region, x_min, y_min, x_max, y_max,
                        self.region_id, fig_path)

        return x_min, y_min, x_max, y_max



    def __normalize_raster_in_chunks(self, input_file, output_file, chunk_size=1024):
        """
        Normalizes the values of a raster file between 0 and 1 in chunks to avoid memory overload.
        
        Args:
            input_file (str): Path to the input cropped raster file.
            output_file (str): Path to the output normalized raster file.
            chunk_size (int): Size of the chunk (in pixels) to process at a time.
        """
        if not self.__confirm_and_delete(output_file):
            return
        
        with rasterio.open(input_file) as src:
            profile = src.profile  # Save metadata for later

            # Update profile for the normalized data
            profile.update(dtype=rasterio.float32)

            # Create the output file
            with rasterio.open(output_file, 'w', **profile) as dst:
                # Iterate over windows (chunks) of the raster
                for ji, window in src.block_windows(1):  # Process by blocks defined by the raster's internal tiling
                    # Read data for the current window
                    data = src.read(window=window)

                    # Normalize the data
                    normalized_data = np.empty_like(data, dtype=np.float32)
                    for i in range(data.shape[0]):  # Loop over each band
                        band = data[i]
                        min_val = band.min()
                        max_val = band.max()
                        print(f"Window {ji}, Band {i + 1} - Min: {min_val}, Max: {max_val}")

                        if max_val > min_val:
                            normalized_data[i] = (band - min_val) / (max_val - min_val)
                        else:
                            normalized_data[i] = band  # No normalization if min == max

                    # Write normalized data to the corresponding window
                    dst.write(normalized_data, window=window)

        print(f"Normalization completed in chunks. Saved to {output_file}.")

    def __normalize_raster(self, input_file, output_file):
        """
        Normalizes the values of a raster file between 0 and 1 and saves it to a new file.
        
        Args:
            input_file (str): Path to the input cropped raster file.
            output_file (str): Path to the output normalized raster file.
        """
        if not self.__confirm_and_delete(output_file):
            return
        
        with rasterio.open(input_file) as src:
            # Read the data as a 3D numpy array: (bands, height, width)
            data = src.read()
            profile = src.profile  # Save metadata for later

            # Initialize an array to store normalized data
            normalized_data = np.empty_like(data, dtype=np.float32)

            # Normalize each band independently
            for i in range(data.shape[0]):  # Loop over each band
                band = data[i]
                min_val = band.min()
                max_val = band.max()
                print(f"Band {i + 1} - Min: {min_val}, Max: {max_val}")

                # Normalize the band if min and max are different; otherwise, leave as-is
                if max_val > min_val:
                    normalized_data[i] = (band - min_val) / (max_val - min_val)
                else:
                    normalized_data[i] = band  # No normalization needed if min == max

            # Update the profile for normalized data type
            profile.update(dtype=rasterio.float32)

        # Save the normalized data to the destination path
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized_data)
                
        print(f"Normalization completed. Saved to {output_file}.")

    def __load_and_plot_tif_with_shape(self, tif_filepath, fig_path):
        """
        Loads a raster file and overlays it with the specified shapefile boundary.
        
        Args:
            tif_filepath (str): Path to the TIFF file.
        """

        region = load_and_extract_region(self.region_shape_path_epsg4326, self.region_id)
        data = rioxarray.open_rasterio(tif_filepath)
        plot_tcc_region_bounds(data, region, self.region_id, fig_path)




    # Public Method to Call Processing Sequence from Outside

    # Updated process method to include the normalization step
    def process(self):
        """
        Main processing sequence, callable from outside.
        
        This function performs:
            - Step 1: Changes raster resolution to 20x20 meters.
            - Step 2: Reprojects the raster file to EPSG:4326.
            - Step 3: Extracts the bounding box coordinates of the region.
            - Step 4: Crops the raster file based on the extracted bounding box.
            - Step 5: Normalizes the cropped raster.
            - Step 6: Plots the cropped and normalized raster with the region boundary overlay.
        """
        print(f"Working on USDA Region {self.region} ...")
        
        # Step 1: Change the resolution to 20x20 meters
        print("Step 1: Changing the resolution to 20x20 meters...")
        self.__change_resolution(self.input_raster_file, self.output_file_resampled, (20, 20))

        # Step 2: Reprojecting the file to EPSG:4326
        print("Step 2: Reprojecting the file to EPSG:4326...")
        self.__reproject_to_crs(self.output_file_resampled, self.output_file_epsg4326, 'EPSG:4326')

        # Step 3: Extracting bounds for Region
        print(f"Step 3: Extracting bounds for Region {self.region} with EPSG:4326...")
        minx, miny, maxx, maxy = self.__get_region_shape_bounds(self.output_path_figure_epsg4326)
        print(f">> Bounds: {minx}, {miny}, {maxx}, {maxy}")

        # Step 4: Cropping the raster based on the shapefile bounds
        print("Step 4: Cropping the raster based on the shapefile bounds...")
        self.__crop_to_bounds(self.output_file_epsg4326, self.output_file_cropped_epsg4326, minx, miny, maxx, maxy)

        # Step 5: Normalizing the cropped raster
        print("Step 5: Normalizing the cropped raster...")
        self.__normalize_raster_in_chunks(self.output_file_cropped_epsg4326, self.normalized_output_file)

        # Step 6: Plotting the final result
        print("Step 6: Plotting the final result...")
        self.__load_and_plot_tif_with_shape(self.normalized_output_file, self.tcc_output_path_figure_epsg4326)

        print("Preprocessing completed.")