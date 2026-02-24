# plotter.py
from func_plots import analyze_and_plot_manual_significance, plot_signal_counts_by_diff_year, plot_study_area, plot_radar_reduction_potential, plot_d_area_ch_area_centroid_disturbances, plot_d_area_ch_area_centroid_disturbances_test, plot_disturbance_signal_duration, plot_signal_counts_by_diff_year_combined, plot_percentages_histograms, analyze_and_plot_manual_significance, plot_disturbance_layers, plot_disturbance_counts
from func_helper import parse_custom_colors, format_label, calculate_area_in_km2, calculate_minimum_outerline_area, add_signal_duration_column, remove_dca_ids, remove_drought, closest_s1_year, merge_geometries_and_keep_columns
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

        # Get figure file paths from environment variables with dynamic buffer and region ID replacement
        figure_study_area_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_STUDY_AREA').format(region_id=self.region_id))
        ids_gdf = load_data(self.ids_path)
        ids_clean = remove_dca_ids(ids_gdf, ["drought","fire"])
        plot_study_area(
                self.usa_filepath, 
                self.region_id, 
                self.tcc_downsampled, 
                self.s1_tiles_boundary_path,
                ids_clean, 
                self.custom_colors, 
                figure_study_area_path,
                logging)

        for buffer in self.spatial_buffer:
            logging.info(f"Processing for buffer: {buffer}")
            
            # Get figure file paths from environment variables with dynamic buffer and region ID replacement
            figure_radar_reduction_potential_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_RADAR_REDUCTION').format(buffer=buffer))
            figure_year_lag_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_YEAR_LAG').format(buffer=buffer))
            figure_size_position_change_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_SIZE_CHANGE').format(buffer=buffer))
            figure_overlap_percentage_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_OVERLAP').format(buffer=buffer))
            figure_radar_reduction_potential_year_dca_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_POTENTIAL_DCA_YEAR').format(buffer=buffer))
            figure_manual_significance_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_SIGNIFICANCE'))
            figure_manual_examples_path =  os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_MANUAL'))

            s1dm_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1DM_SHAPE_FILE').format(region_id=self.region_id, buffer=buffer))
            manual_base_folder = os.getenv('MANUAL_DIR')

            ids_gdf = load_data(self.ids_path)
            s1dm_gdf = load_data(s1dm_path)

            refdm_gdf_yearly_aggregated = merge_geometries_and_keep_columns(s1dm_gdf)
            s1dm_frequency = add_signal_duration_column(refdm_gdf_yearly_aggregated, self.target_crs)
            #s1dm_cleaned = remove_drought(s1dm_frequency)
            s1dm_cleaned = remove_dca_ids(s1dm_frequency, ["drought","fire"])
            s1dm_cleaned = s1dm_cleaned[s1dm_cleaned["DCA_ID"] != "fire"]
            logging.info('Claculate size shift difference')
            gdf = calculate_size_shift_difference(ids_gdf, s1dm_cleaned)
            # Minimum outerline areas
            s1dm_convex = calculate_minimum_outerline_area(s1dm_cleaned)[['geometry', 'area_km2', 'DCA_ID']]
            ids_convex = calculate_minimum_outerline_area(ids_gdf)[['geometry', 'area_km2', 'DCA_ID']]
            
            # Remove drought disturbances
            s1dm_no_drought = remove_dca_ids(s1dm_cleaned, ["drought","fire"]) #remove_drought(s1dm_cleaned)
            s1dm_no_drought_gdf = remove_dca_ids(s1dm_gdf, ["drought","fire"]) #remove_drought(s1dm_gdf)
        

            # Plot radar reduction potential
            logging.info('Plot radar reduction potential')
            plot_radar_reduction_potential(
                s1dm_frequency, 
                ids_gdf, 
                save_path=figure_radar_reduction_potential_path, 
                plot_reduction=True
            )
            

            plot_d_area_ch_area_centroid_disturbances_test(
                gdf, 
                ids_gdf, 
                s1dm_convex, 
                ids_convex, 
                self.custom_colors, 
                save_path=figure_size_position_change_path
            )
            
            # Plot signal counts
            logging.info('Plot signal counts')
            plot_signal_counts_by_diff_year(
                #s1dm_no_drought_gdf,
                s1dm_no_drought,
                self.custom_colors,
                save_path=figure_year_lag_path
            )
            
            # Calculate and plot overlap percentages
            logging.info('Calculate and plot overlap percentages')
            plot_percentages_histograms(
                ids_gdf, 
                s1dm_no_drought_gdf, 
                self.custom_colors, 
                figure_overlap_percentage_path
            )

            logging.info('Calculate and plot DCA, Year Potential')
            plot_disturbance_counts(s1dm_path, self.ids_path, 
                                    exclude_types=['fire', 'drought'], 
                                    ordered_types=['wind', 'bark_beetle', 'defoliators'],
                                    custom_colors=self.custom_colors, 
                                    output_file=figure_radar_reduction_potential_year_dca_path)
            
            # # Calculate the significance , manual Plot
            logging.info('Calculate the significanc')
            analyze_and_plot_manual_significance(self.ids_path, 
                                                 s1dm_path, 
                                                 manual_base_folder, 
                                                 os.getenv('RESULTS_DIR'),
                                                 figure_manual_significance_path)
            print('Plotted')
            # disturbance_files = {
            #                     "wind": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/Wind_0_larger_RGB_psscene_visual/composite_file_format.tif",
            #                         "idx": 0,
            #                         "date": "2018-04-16"
            #                     },
            #                     "defoliators": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/Defoliators_17_psscene_visual/composite_file_format.tif",
            #                         "idx": 17,
            #                         "date": "2021-05-14"
            #                     },
            #                     "bark_beetle": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/BarkBeetle_10_psscene_visual/composite_file_format.tif",
            #                         "idx": 10,
            #                         "date": "2018-04-28"
            #                     }
            #                 }
            
            # logging.info('Calculate Manual Examples')

            # plot_disturbance_layers(
            #     disturbance_files,
            #     manual_base_folder,
            #     ids_gdf,
            #     s1dm_gdf,
            #     figure_path=figure_manual_examples_path)




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
        crs = os.getenv('TCC_CRS')
        crs_number = crs.split(":")[-1] if crs else None
        self.tcc_downsampled = os.path.join(os.getenv('TCC_DIR'), 
                                            os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(region_id=self.region_id,
                                                                                         crs=crs_number,
                                                                                         tcc_year=2017))
        self.s1_tiles_boundary_path =  os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id))