# plotter.py
from func_plots import plot_study_area, plot_radar_reduction_potential, plot_d_area_ch_area_centroid_disturbances, plot_disturbance_signal_duration, plot_signal_counts_by_diff_year_combined, plot_percentages_histograms
from func_helper import parse_custom_colors, format_label, calculate_area_in_km2, calculate_minimum_outerline_area, add_signal_duration_column, remove_drought, closest_s1_year, merge_geometries_and_keep_columns
from func_file_io import load_data
from func_preprocessing import calculate_size_shift_difference
from func_tcc_application import create_downsampled_tcc_map
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box, shape

class Plotter:
    def __init__(self, env_path, spatial_buffer):
        self._set_up_logging()
        self._load_env_variables(env_path)
        self.spatial_buffer= spatial_buffer

    def plot(self):
        """
        Plots the processed data.
        
        :param data: The processed data to be plotted
        """

        # # Get figure file paths from environment variables with dynamic buffer and region ID replacement
        # figure_study_area_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_STUDY_AREA').format(region_id=self.region_id))
        # ids_gdf = load_data(self.ids_path)
        # plot_study_area(
        #         self.usa_filepath, 
        #         self.region_id, 
        #         self.tcc_downsampled, 
        #         self.s1_tiles_boundary_path,
        #         ids_gdf, 
        #         self.custom_colors, 
        #         figure_study_area_path,
        #         logging)

        for buffer in self.spatial_buffer:
            logging.info(f"Processing for buffer: {buffer}")
            
            # Get figure file paths from environment variables with dynamic buffer and region ID replacement
            figure_radar_reduction_potential_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_RADAR_REDUCTION').format(buffer=buffer))
            figure_year_lag_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_YEAR_LAG').format(buffer=buffer))
            figure_size_position_change_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_SIZE_CHANGE').format(buffer=buffer))
            figure_overlap_percentage_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_OVERLAP').format(buffer=buffer))

            s1dm_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1DM_SHAPE_FILE').format(region_id=self.region_id, buffer=buffer))
            
            ids_gdf = load_data(self.ids_path)
            s1dm_gdf = load_data(s1dm_path)

            refdm_gdf_yearly_aggregated = merge_geometries_and_keep_columns(s1dm_gdf)
            s1dm_frequency = add_signal_duration_column(refdm_gdf_yearly_aggregated, self.target_crs)
            s1dm_cleaned = remove_drought(s1dm_frequency)
            # logging.info('Claculate size shift difference')
            # gdf = calculate_size_shift_difference(ids_gdf, s1dm_cleaned)
            # # Minimum outerline areas
            # s1dm_convex = calculate_minimum_outerline_area(s1dm_cleaned)[['geometry', 'area_km2', 'DCA_ID']]
            # ids_convex = calculate_minimum_outerline_area(ids_gdf)[['geometry', 'area_km2', 'DCA_ID']]
            
            # Remove drought disturbances
            s1dm_no_drought = remove_drought(s1dm_cleaned)
            s1dm_no_drought_gdf = remove_drought(s1dm_gdf)
        

            # # Plot radar reduction potential
            # logging.info('Plot radar reduction potential')
            # plot_radar_reduction_potential(
            #     s1dm_frequency, 
            #     ids_gdf, 
            #     save_path=figure_radar_reduction_potential_path, 
            #     plot_reduction=False
            # )
            
            # # Plot size and position change
            # logging.info('Plot size and position change')
            # plot_d_area_ch_area_centroid_disturbances(
            #     gdf, 
            #     ids_gdf, 
            #     s1dm_convex, 
            #     ids_convex, 
            #     self.custom_colors, 
            #     save_path=figure_size_position_change_path
            # )
            
            # Plot signal counts
            logging.info('Plot signal counts')
            plot_signal_counts_by_diff_year_combined(
                s1dm_no_drought_gdf,
                s1dm_no_drought,
                self.custom_colors,
                save_path=figure_year_lag_path
            )
            
            # # Calculate and plot overlap percentages
            # logging.info('Calculate and plot overlap percentages')
            # plot_percentages_histograms(
            #     ids_gdf, 
            #     s1dm_no_drought_gdf, 
            #     self.custom_colors, 
            #     figure_overlap_percentage_path
            # )


    def _set_up_logging(self):
        """Set up logging to a file with timestamps for tracking the process."""
        logging.basicConfig(
            filename='log_plotter.log',
            level=logging.INFO, 
            format='%(asctime)s - %(message)s'
        )

    def _load_env_variables(self, env_path):
        """Load required environment variables from a .env file."""

        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        # Load environment variables and validate
        self.region = os.getenv('REGION')
        if not self.region:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)
        # Set target CRS (Coordinate Reference System)
        self.target_crs = os.getenv('TARGET_CRS')

        custom_colors_json = os.getenv('COLORS')
        self.custom_colors = parse_custom_colors(custom_colors_json)


        # Define file paths for shapefiles and output locations
        self.usa_filepath = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.ids_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        #self.tcc_downsampled = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(region_id=self.region_id))
        #self.s1_tiles_boundary_path =  os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id))