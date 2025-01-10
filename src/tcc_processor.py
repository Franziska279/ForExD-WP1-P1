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
from func_helper import load_and_extract_region


class TCCProcessor:
    """
    TCCProcessor class to handle raster processing tasks such as resolution adjustment, reprojection, and cropping for a specific USDA region.
    """

    def __init__(self, env_path):
        self._set_up_logging()
        # Load environment variables for configuration
        load_dotenv(dotenv_path=env_path)

        # Attributes for region and paths
        self.region = os.getenv('REGION')
        self.region_id = str(self.region).zfill(2)  # Pad region ID for consistent formatting

        # Define output directory and verify its presence
        self.output_dir = os.getenv('TCC_PATH')
        if self.output_dir is None:
            logging.error("TCC_PATH environment variable is not set. Exiting.")
            raise ValueError("TCC_PATH environment variable is not set")
        self.output_dir = self.output_dir.rstrip('/') + '/'

        # Define file paths for input and output TIFFs, shapefiles, and figures
        #/work/sy58xupo-CleaningSpace/Data/CONUS/tcc/wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_27705.tif
        self.input_raster_file = self.output_dir + "nlcd_tcc_conus_2017_v2021-4.tif"
        self.output_file_resampled = self.output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m.tif"
        self.output_file_epsg4326 = self.output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326.tif"
        self.output_file_cropped_epsg4326 = self.output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_{self.region_id}.tif"
        self.normalized_output_file = self.output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_normalized_region_{self.region_id}.tif"
        self.region_shape_path_epsg4326 = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
        self.output_path_figure_epsg4326 = self.output_dir + f"bounds_epsg4326_region_{self.region_id}.png"
        self.tcc_output_path_figure_epsg4326 = self.output_dir + f"tcc_bounds_epsg4326_region_{self.region_id}.png"

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
        region = load_and_extract_region(self.region_shape_path_epsg4326, self.region_id)
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
        region = load_and_extract_region(self.region_shape_path_epsg4326, self.region_id)
        data = rioxarray.open_rasterio(tif_filepath)
        plot_tcc_region_bounds(data, region, self.region_id, fig_path)
        logging.info(f"Raster plot saved to: {fig_path}")

    def process(self):
        """Main processing sequence."""
        logging.info(f"\n")
        # logging.info(f"Processing USDA Region {self.region}...")
        # # Step 1: Change resolution
        # self.__change_resolution(self.input_raster_file, self.output_file_resampled, (20, 20))

        # logging.info(f"Reprojecting the file to EPSG:4326...")
        # # Step 2: Reproject to EPSG:4326
        # self.__reproject_to_crs(self.output_file_resampled, self.output_file_epsg4326, 'EPSG:4326')
        
        # logging.info(f"Extract region bounds")
        # # Step 3: Extract region bounds
        # minx, miny, maxx, maxy = self.__get_region_shape_bounds(self.output_path_figure_epsg4326)

        # logging.info(f"Crop raster to region bounds")
        # # Step 4: Crop raster to region bounds
        # self.__crop_to_bounds(self.output_file_epsg4326, self.output_file_cropped_epsg4326, minx, miny, maxx, maxy)

        logging.info(f"Normalize raster")
        # Step 5: Normalize raster
        self.__normalize_raster(self.output_file_cropped_epsg4326, self.normalized_output_file)
        logging.info(f"Plot raster with region boundaries")
        # Step 6: Plot raster with region boundaries
        self.__load_and_plot_tif_with_shape(self.normalized_output_file, self.tcc_output_path_figure_epsg4326)

        logging.info("Processing completed successfully.")
