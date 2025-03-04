# import os
# import geopandas as gpd
# import pandas as pd
# import logging
# from dotenv import load_dotenv
# from pathlib import Path
# from func_plots import plot_regions_disturbances
# from func_file_io import load_data, save_shapefile
# from func_helper import parse_custom_colors
# from func_data_preprocessing import (
#     process_and_merge_disturbances, remove_overlapping_entries, filter_disturbance_data,
#     keep_overlapping_entries, analyze_overlaps, analyze_and_enrich_overlaps
# )


# class IDSProcessor:
#     def __init__(self, env_path):
#         self._set_up_logging()
#         # Load environment variables and set up paths and parameters
#         load_dotenv(dotenv_path=env_path)
#         self.region = os.getenv('REGION')
#         self.region_id = str(self.region).zfill(2)
#         self.target_crs = "EPSG:27705"

#         # File paths and configurations based on environment variables
#         self.region_shape_path = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
#         self.ids_region_file_path = f"{os.getenv('IDS_REGIONS')}CONUS_Region{self.region}_dissolved.csv"
#         self.file_output_path = f"{os.getenv('RESULTS')}/region_{self.region_id}_dca_filtered_ids_usda_polygons.shp"
#         self.file_equi7_output_path = f"{os.getenv('RESULTS')}/region{self.region_id}_dca_filtered_ids_usda_polygons_espg_27705.shp"
#         self.figure_output_path = f"{os.getenv('FIGURES')}/p1_f1_disturbances_region_{self.region_id}.png"
#         self.custom_colors_json = os.getenv('COLORS')

#         # Ensure the directories exist, create them if they don't
#         os.makedirs(os.path.dirname(self.region_shape_path), exist_ok=True)
#         os.makedirs(os.path.dirname(self.ids_region_file_path), exist_ok=True)
#         os.makedirs(os.path.dirname(self.file_output_path), exist_ok=True)
#         os.makedirs(os.path.dirname(self.file_equi7_output_path), exist_ok=True)
#         os.makedirs(os.path.dirname(self.figure_output_path), exist_ok=True)

#         # Initialize data as None; will be populated in load_data method
#         self.data = None

#         logging.info("======================================================================\n >>  IDSProcessor initialized.")

#     def _set_up_logging(self):
#         """Set up logging to file with timestamp."""
#         logging.basicConfig(filename='log_ids_processor.log', level=logging.INFO, format='%(asctime)s - %(message)s')


#     def load_data(self):
#         logging.info(f"Loading CSV file for Region {self.region_id}...")
#         self.data = load_data(self.ids_region_file_path)
#         logging.info(f"Data loaded successfully. Records: {len(self.data)}")
#         return self.data

#     def exclude_include_overlapping_entries(self):
#         logging.info("Processing and cleaning disturbance data...")
#         self.data = process_and_merge_disturbances(self.data)
#         self.data = self.data[self.data['SURVEY_YEAR'] > 2009]
#         logging.info("Filtered data for survey years > 2009.")

#         logging.info("Removing temporal and spatial overlaps...")
#         gdf_no_overlap = remove_overlapping_entries(self.data, year_range=5)
#         logging.info(f"Number of records after removing overlaps: {len(gdf_no_overlap)}")

#         # Optional: Keep overlaps if needed for analysis
#         gdf_overlap = keep_overlapping_entries(self.data, year_range=2)
#         if gdf_overlap is not None:
#             logging.info(f"Number of records with detected overlaps: {len(gdf_overlap)}")
#             gdf_overlap_analyzed = analyze_overlaps(gdf_overlap)
#             self.data = pd.concat([gdf_no_overlap, gdf_overlap_analyzed], ignore_index=True)
#         else:
#             self.data = gdf_no_overlap
#         logging.info("Analyzing and enriching overlaps completed.")

#         self.data = analyze_and_enrich_overlaps(self.data)

#         # Rename columns to avoid issues with field name normalization
#         column_renames = {
#             'SURVEY_YEAR': 'SURVEY_Y',
#             'DA_Code_USDA': 'DA_C_USDA'
#         }
#         self.data = self.data.rename(columns=column_renames)
#         logging.info("Renamed columns for compatibility.")
#         return self.data

#     def filter_data(self):
#         logging.info("Filtering disturbance data...")
#         excluded_dca_types = ['other', 'multi_damage', 'other_abiotic', 'other_biotic']
#         self.data = filter_disturbance_data(self.data, excluded_dca_types, start_year=2016, end_year=2020)
#         logging.info(f"Data filtered successfully. Remaining records: {len(self.data)}")
#         return self.data

#     def print_status(self):
#         # Output summary for current data
#         total_elements = len(self.data)
#         unique_events = len(self.data['ID_E'].unique())
#         overlapping_events = total_elements - unique_events

#         logging.info(f"Number of elements: {total_elements}")
#         logging.info(f"Unique events | Total events: {unique_events} | {total_elements}")
#         logging.info(f"Overlapping events: {overlapping_events}")

#         print(f"Number of elements: {total_elements}")
#         print(f"Unique events | Total events: {unique_events} | {total_elements}")
#         print(f"Overlapping events: {overlapping_events}")

#     def save_and_plot(self):
#         logging.info("Saving data in original CRS...")
#         try:
#             # Save original CRS shapefile
#             save_shapefile(self.data, self.file_output_path)
#             logging.info(f"Data saved successfully to {self.file_output_path}")
#         except Exception as e:
#             logging.error(f"Error saving original CRS data: {e}")

#         logging.info("Reprojecting data to target CRS and saving...")
#         try:
#             # Reproject to target CRS
#             data_transformed = self.data.to_crs(self.target_crs)
#             save_shapefile(data_transformed, self.file_equi7_output_path)
#             logging.info(f"Reprojected data saved successfully to {self.file_equi7_output_path}")
#         except Exception as e:
#             logging.error(f"Error reprojecting and saving data: {e}")

#         logging.info("Plotting regions disturbances...")
#         try:
#             plot_regions_disturbances(
#                 self.data,
#                 self.region_shape_path,
#                 output_file=self.figure_output_path,
#                 custom_colors=parse_custom_colors(self.custom_colors_json),
#                 region_nr=self.region_id
#             )
#             logging.info(f"Plot saved to {self.figure_output_path}")
#         except Exception as e:
#             logging.error(f"Error during plotting: {e}")

import os
import logging
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from pathlib import Path
from func_plots import plot_regions_disturbances
from func_file_io import load_data, save_shapefile
from func_helper import parse_custom_colors
from func_data_preprocessing import (
    rename_columns_and_process_data, merge_and_iterate, filter_disturbance_data,
    remove_temporal_overlaps, keep_and_analyze_overlaps, filter_and_enrich_overlaps
)

class IDSProcessor:
    def __init__(self, region_id, env_path):
        self._set_up_logging()
        load_dotenv(dotenv_path=env_path)  # Load environment variables

        # Store region info
        self.region_id = str(region_id).zfill(2)
        self.target_crs = os.getenv('EQUI7_NA_EPSG')

        # Load paths from .env
        self.region_shape_path = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.ids_region_file_path = os.path.join(os.getenv('IDS_REGIONS_DIR'), os.getenv('IDS_SHAPE_FILE').format(region_id=self.region_id))
        self.file_output_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        self.file_equi7_output_path = self.file_output_path.replace('.shp', '_espg_27705.shp')
        self.figure_output_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_STUDY_AREA').format(region_id=self.region_id))  # Buffer is a placeholder

        self.custom_colors_json = os.getenv('COLORS')

        # Ensure required directories exist
        for path in [self.region_shape_path, self.ids_region_file_path, self.file_output_path, self.file_equi7_output_path, self.figure_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Initialize data
        self.data = None

        logging.info(f">> IDSProcessor initialized for Region {self.region_id}")

    def _set_up_logging(self):
        """Set up logging to file with timestamp."""
        logging.basicConfig(filename='ids_processor.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def loading(self):
        """Loads CSV disturbance data for the given region."""
        logging.info(f"Loading CSV file for Region {self.region_id}...")
        self.data = load_data(self.ids_region_file_path)
        logging.info(f"Data loaded successfully. Records: {len(self.data)}")
        return self.data

    def exclude_include_overlapping_entries(self):
        """Filters data, removes spatial & temporal overlaps, and processes disturbances."""
        logging.info("Processing and cleaning disturbance data...")

        # Apply the functions step by step
        ids = rename_columns_and_process_data(self.data)
        merged_ids_p = merge_and_iterate(ids)
        merged_ids_p["ID_E"] = merged_ids_p.index  # Add ID_E column based on the index

        gdf_no_overlap = remove_temporal_overlaps(merged_ids_p, year_range=5)
        logging.info(f"Remove temproal overlaps -  number of records: {len(gdf_no_overlap)}")
        gdf_overlap = keep_and_analyze_overlaps(merged_ids_p, year_range=2)

        logging.info(f"Keep temproal overlaps -  number of records: {len(gdf_overlap)}")
        data = pd.concat([gdf_no_overlap, gdf_overlap], ignore_index=True)
        
        data = filter_and_enrich_overlaps(data)

        # Rename columns for compatibility
        data.rename(columns={'SURVEY_YEA': 'SURVEY_Y', 'DA_Code_USDA': 'DA_C_USDA'}, inplace=True)

        logging.info(f"Column renaming complete. Final number of records: {len(data)}")
        self.data = data
        return self.data


    def filter_data(self, start_year, end_year, excluded_dca_types):
        """Filters disturbance data based on user-specified years and types."""
        logging.info("Filtering disturbance data...")
        self.data = filter_disturbance_data(self.data, excluded_dca_types, start_year, end_year)
        logging.info(f"Filtered dataset. Remaining records: {len(self.data)}")
        return self.data

    def print_status(self):
        """Prints and logs summary of dataset."""
        total_elements = len(self.data)
        unique_events = len(self.data['ID_E'].unique())
        overlapping_events = total_elements - unique_events

        logging.info(f"Total Elements: {total_elements}")
        logging.info(f"Unique Events: {unique_events} | Overlapping Events: {overlapping_events}")

        print(f"Total Elements: {total_elements}")
        print(f"Unique Events: {unique_events} | Overlapping Events: {overlapping_events}")

    def save_and_plot(self):
        """Saves dataset and generates disturbance plots."""
        logging.info("Saving data in original CRS...")
        try:
            save_shapefile(self.data, self.file_output_path)
            logging.info(f"Data saved to {self.file_output_path}")
        except Exception as e:
            logging.error(f"Error saving original CRS data: {e}")

        logging.info("Reprojecting data and saving...")
        try:
            data_transformed = self.data.to_crs(self.target_crs)
            save_shapefile(data_transformed, self.file_equi7_output_path)
            logging.info(f"Reprojected data saved to {self.file_equi7_output_path}")
        except Exception as e:
            logging.error(f"Error saving transformed data: {e}")

        logging.info("Generating disturbance plots...")
        try:
            plot_regions_disturbances(
                self.data,
                self.region_shape_path,
                output_file=self.figure_output_path,
                custom_colors=parse_custom_colors(self.custom_colors_json),
                region_nr=self.region_id
            )
            logging.info(f"Plot saved to {self.figure_output_path}")
        except Exception as e:
            logging.error(f"Error generating plots: {e}")
