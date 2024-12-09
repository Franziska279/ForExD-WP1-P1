# ids_processor.py
import os
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from func_plots import plot_regions_disturbances
from func_file_io import load_data, save_shapefile
from func_helper import parse_custom_colors
from func_data_preprocessing import (process_and_merge_disturbances,remove_overlapping_entries,filter_disturbance_data, 
                                     keep_overlapping_entries, analyze_overlaps, analyze_and_enrich_overlaps)

class IDSProcessor:
    def __init__(self, env_path):
        # Load environment variables and set up paths and parameters
        load_dotenv(dotenv_path=env_path)
        self.region = os.getenv('REGION')
        self.region_id = str(self.region).zfill(2)
        self.target_crs = "EPSG:27705"
        
        # File paths and configurations based on environment variables
        self.region_shape_path = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
        self.ids_region_file_path = f"{os.getenv('IDS_REGIONS')}CONUS_Region{self.region}_dissolved.csv"
        self.file_output_path = f"{os.getenv('RESULTS')}/region_{self.region_id}_dca_filtered_ids_usda_polygons.shp"
        self.file_equi7_output_path = f"{os.getenv('RESULTS')}/region{self.region_id}_dca_filtered_ids_usda_polygons_espg_27705.shp"
        self.figure_output_path = f"{os.getenv('FIGURES')}/p1_f1_disturbances_region_{self.region_id}.png"
        self.custom_colors_json = os.getenv('COLORS')

        # Initialize data as None; will be populated in load_data method
        self.data = None

    def load_data(self):
        print(f"Loading CSV file for Region {self.region_id}...")
        self.data = load_data(self.ids_region_file_path)
        return self.data

    def exclude_include_overlapping_entries(self):
        print("Processing and cleaning disturbance data...")
        self.data = process_and_merge_disturbances(self.data)
        self.data = self.data[self.data['SURVEY_YEAR'] > 2009]

        print("Removing temporal and spatial overlaps...")
        gdf_no_overlap = remove_overlapping_entries(self.data, year_range=5)
        print(f"> Number of records after removing overlaps: {len(gdf_no_overlap)}")

        # Optional: Keep overlaps if needed for analysis
        gdf_overlap = keep_overlapping_entries(self.data, year_range=2)
        if gdf_overlap is not None:
            print(f"> Number of records with detected overlaps: {len(gdf_overlap)}")
            gdf_overlap_analyzed = analyze_overlaps(gdf_overlap)
            self.data = pd.concat([gdf_no_overlap, gdf_overlap_analyzed], ignore_index=True)
        else:
            self.data = gdf_no_overlap

        self.data = analyze_and_enrich_overlaps(self.data)

        # Rename columns to avoid issues with field name normalization
        column_renames = {
            'SURVEY_YEAR': 'SURVEY_Y',
            'DA_Code_USDA': 'DA_C_USDA'
        }
        self.data = self.data.rename(columns=column_renames)
        return self.data

    def filter_data(self):
        excluded_dca_types = ['other', 'multi_damage', 'other_abiotic', 'other_biotic']
        self.data = filter_disturbance_data(self.data, excluded_dca_types, start_year=2015, end_year=2020)
        return self.data

    def print_status(self):
        # Output summary for current data
        # print("Current data length:", len(self.data))  # Debugging line
        total_elements = len(self.data)
        unique_events = len(self.data['ID_E'].unique())
        overlapping_events = total_elements - unique_events

        print(f"Number of elements: {total_elements}")
        print(f"Unique events | Total events: {unique_events} | {total_elements}")
        print(f"Overlapping events: {overlapping_events}")


    def save_and_plot(self):

        # Debugging: check data before saving
        print("Saving data in original CRS...")

        try:
            # Save original CRS shapefile
            save_shapefile(self.data, self.file_output_path)
            print(f"Data saved successfully to {self.file_output_path}")
        except Exception as e:
            print(f"Error saving original CRS data: {e}")

        print("Reprojecting data to target CRS and saving...")

        try:
            # Reproject to target CRS
            data_transformed = self.data.to_crs(self.target_crs)
            save_shapefile(data_transformed, self.file_equi7_output_path)
            #data_transformed.to_file(self.file_equi7_output_path, index=False)
            #print(f"Reprojected data saved successfully to {self.file_equi7_output_path}")
        except Exception as e:
            print(f"Error reprojecting and saving data: {e}")

        # Plot
        plot_regions_disturbances(
            self.data,
            self.region_shape_path,
            output_file=self.figure_output_path,
            custom_colors=parse_custom_colors(self.custom_colors_json),
            region_nr=self.region_id
        )
