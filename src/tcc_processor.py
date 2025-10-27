import geopandas as gpd
import matplotlib.pyplot as plt
import subprocess
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import rasterio
from rasterio.windows import Window
import numpy as np
from rasterio.enums import Resampling
import rioxarray
from matplotlib.patches import Patch
from func_plots import plot_region_bounds, plot_tcc_region_bounds
from func_file_io import load_data, save_shapefile, run_command
from func_helper import load_and_extract_region_crs
from func_tcc_application import create_downsampled_tcc_map

class TCCProcessor:
    """
    TCCProcessor class to handle raster processing tasks such as resolution adjustment, reprojection, and cropping for a specific USDA region.
    """
    def __init__(self, env_path, tcc_year):
        self._set_up_logging()
        load_dotenv(dotenv_path=env_path)

        # Load region information
        self.region = os.getenv('REGION')
        self.region_id = str(self.region).zfill(2)
        self.tcc_year = tcc_year # os.getenv('TCC_YEAR')
        self.crs = os.getenv('TCC_CRS')
        self.crs_number = self.crs.split(':')[1]

        # Load paths from .env and construct full file paths dynamically
        self.region_shape_path = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.input_raster_file = os.path.join(os.getenv('TCC_DIR'), f'{self.tcc_year}', os.getenv('TCC_INPUT_RASTER').format(tcc_year=self.tcc_year))
        self.output_file_resampled = os.path.join(os.getenv('TCC_DIR'), f'{self.tcc_year}', os.getenv('TCC_RESAMPLED_RASTER').format(tcc_year=self.tcc_year))
        self.output_file_epsg4326 = os.path.join(os.getenv('TCC_DIR'), f'{self.tcc_year}', os.getenv('TCC_EPSG_RASTER').format(tcc_year=self.tcc_year, crs=self.crs_number))
        self.output_file_cropped_epsg4326 = os.path.join(os.getenv('TCC_DIR'), f'{self.tcc_year}', os.getenv('TCC_CROPPED_RASTER_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))
        self.normalized_output_file = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_NORMALIZED_RASTER_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))
        self.temp_downsampled_tcc_netcdf = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_DOWNSAMPLED_TEMP_RASTER_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))
        self.downsampled_tcc_netcdf = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))
        self.figure_output_path_bounds = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_TCC_BOUNDS_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))
        self.figure_output_path_bounds_shape = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_TCC_BOUNDS_SHAPE_TEMPLATE').format(region_id=self.region_id, tcc_year=self.tcc_year, crs=self.crs_number))

        # Ensure required directories exist
        for path in [
            self.region_shape_path, self.input_raster_file, self.output_file_resampled,
            self.output_file_epsg4326, self.output_file_cropped_epsg4326, 
            self.normalized_output_file, self.figure_output_path_bounds, 
            self.figure_output_path_bounds_shape
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)


    def _set_up_logging(self):
        """Set up logging to file with timestamp."""
        logging.basicConfig(
            filename='log_tcc_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def __delete_if_exists(self, file_path):
        """Delete file if it already exists."""
        if os.path.exists(file_path):
            logging.info(f"Deleting existing file: {file_path}")
            os.remove(file_path)

    def __reproject_to_crs(self, input_file, output_file, crs):
        """Reprojects a raster file to the specified CRS."""
        logging.info(f"Reprojecting {input_file} to {crs} -> {output_file}")
        self.__delete_if_exists(output_file)
        command = f"gdalwarp -t_srs '{crs}' {input_file} {output_file}"
        run_command(command)
        logging.info(f"Reprojection completed: {output_file}")

    def __change_resolution(self, input_file, output_file, resolution):
        """Changes raster resolution."""
        logging.info(f"Changing resolution of {input_file} to {resolution} -> {output_file}")
        self.__delete_if_exists(output_file)
        command = f"gdalwarp -tr {resolution[0]} {resolution[1]} {input_file} {output_file}"
        run_command(command)
        logging.info(f"Resolution change completed: {output_file}")

    def __crop_to_bounds(self, input_file, output_file, minx, miny, maxx, maxy):
        """Crops raster to specified bounding box."""
        logging.info(f"Cropping {input_file} to bounds {minx}, {miny}, {maxx}, {maxy} -> {output_file}")
        self.__delete_if_exists(output_file)
        command = f"gdalwarp -te {minx} {miny} {maxx} {maxy} {input_file} {output_file}"
        run_command(command)
        logging.info(f"Cropping completed: {output_file}")

    def __get_region_shape_bounds(self, fig_path):
        """Extracts bounds of the region and plots them."""
        logging.info("Extracting region shape bounds...")
        region = load_and_extract_region_crs(self.region_shape_path, self.region_id, self.crs)
        bounds = region.total_bounds
        plot_region_bounds(region, *bounds, self.region_id, fig_path)
        logging.info(f"Region bounds extracted: {bounds}")
        return bounds

    def __normalize_raster(self, input_file, output_file):
        """
        Normalizes the values of a raster file between 0 and 1 (using global min/max values)
        and saves it to a new file. This avoids excessive memory usage by processing in chunks.

        Args:
            input_file (str): Path to the input raster file.
            output_file (str): Path to the output normalized raster file.
        """

        # Confirm and delete the output file if it already exists
        self.__delete_if_exists(output_file)

        logging.info(f"Step 1: Calculating global min and max for {input_file}")

        # Step 1: Determine global min and max values
        global_min, global_max = float('inf'), float('-inf')
        with rasterio.open(input_file) as src:
            for ji, window in src.block_windows(1):  # Read in chunks
                data = src.read(window=window)
                global_min = min(global_min, data.min())
                global_max = max(global_max, data.max())

        logging.info(f"Global Min: {global_min}, Global Max: {global_max}")

        if global_max == global_min:
            logging.warning("Global min and max are the same. No normalization will be performed.")
            return

        # Step 2: Normalize the data using global min and max
        logging.info(f"Step 2: Normalizing raster {input_file} -> {output_file}")

        with rasterio.open(input_file) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32)  # Update profile to float32 for normalized data

            with rasterio.open(output_file, 'w', **profile) as dst:
                for ji, window in src.block_windows(1):  # Process in manageable chunks
                    data = src.read(window=window)
                    normalized_data = (data - global_min) / (global_max - global_min)
                    dst.write(normalized_data.astype(np.float32), window=window)

        logging.info(f"Normalization completed. Saved to {output_file}.")



    def __load_and_plot_tif_with_shape(self, tif_filepath, fig_path):
        """Plots raster data with region boundary overlay."""
        logging.info(f"Plotting raster {tif_filepath} with region boundaries...")
        region = load_and_extract_region_crs(self.region_shape_path, self.region_id, self.crs)
        data = rioxarray.open_rasterio(tif_filepath)
        plot_tcc_region_bounds(data, region, self.region_id, fig_path)
        logging.info(f"Raster plot saved to: {fig_path}")

    def process(self):
        """Main processing sequence."""
        logging.info(f"Processing USDA Region {self.region}...")
        # # Step 1: Change resolution
        # self.__change_resolution(self.input_raster_file, self.output_file_resampled, (20, 20))

        # logging.info(f"Reprojecting the file to EPSG:4326...")
        # # Step 2: Reproject to EPSG:4326
        # self.__reproject_to_crs(self.output_file_resampled, self.output_file_epsg4326, self.crs)
        
        # logging.info(f"Extract region bounds")
        # # Step 3: Extract region bounds
        # minx, miny, maxx, maxy = self.__get_region_shape_bounds(self.figure_output_path_bounds)

        # logging.info(f"Crop raster to region bounds")
        # # Step 4: Crop raster to region bounds
        # self.__crop_to_bounds(self.output_file_epsg4326, self.output_file_cropped_epsg4326, minx, miny, maxx, maxy)

        # logging.info(f"Normalize raster")
        # # Step 5: Normalize raster
        # self.__normalize_raster(self.output_file_cropped_epsg4326, self.normalized_output_file)

        logging.info(f"Downsample normalized raster for plotting")
        # Step 6: Downsample normalized raster for plotting
        create_downsampled_tcc_map(self.normalized_output_file, self.region_shape_path, self.region_id, 
        temp_netcdf=self.temp_downsampled_tcc_netcdf, final_netcdf=self.downsampled_tcc_netcdf)

        # logging.info(f"Plot raster with region boundaries")
        # # Step 7: Plot raster with region boundaries
        # self.__load_and_plot_tif_with_shape(self.normalized_output_file, self.figure_output_path_bounds_shape)

        logging.info("Processing completed successfully.")
