"""
IDSProcessor — USDA Insect and Disease Survey Data Preparation
==============================================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Loads and prepares the USDA IDS (Insect and Disease Survey) forest disturbance
dataset for a given USFS region. The processor runs four steps in sequence:

  1. load()                         -- read the raw IDS shapefile from disk
  2. resolve_polygon_overlaps() -- merge fragmented polygons, remove
                                          temporal/spatial duplicates, and flag
                                          overlapping multi-year events
  3. filter_by_study_criteria()        -- restrict to a year range and disturbance
                                          types of interest, drop large polygons (>15 km²)
  4. save_and_plot()                   -- write the cleaned dataset as shapefiles
                                          (original CRS + Equi7 reprojection) and
                                          generate the study-area overview figure

All file paths are read from the .env file (see environment/.env.example).
Logging is configured centrally in main.py — do not call basicConfig here.
"""

import os
import logging
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from pathlib import Path
from func_file_io import load_data, save_shapefile
from func_helper import parse_color_map
from func_data_preprocessing import (
    clean_raw_ids_data, merge_fragmented_polygons, apply_study_filters,
    drop_compound_survey_events, extract_recurring_disturbances, build_overlap_pair_records
)


class IDSProcessor:
    def __init__(self, region_id, env_path):
        load_dotenv(dotenv_path=env_path)

        self.region_id = str(region_id).zfill(2)
        # Equi7 North America CRS — used for the reprojected output shapefile
        self.target_crs = os.getenv('EQUI7_NA_EPSG')

        # Assemble file paths from .env directory variables and file-name templates
        self.region_shape_path = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.ids_region_file_path = os.path.join(os.getenv('IDS_REGIONS_DIR'), os.getenv('IDS_SHAPE_FILE').format(region_id=self.region_id))
        self.file_output_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        # Reprojected output path derived from the standard output path
        self.file_equi7_output_path = self.file_output_path.replace('.shp', '_espg_27705.shp')
        self.figure_output_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_STUDY_AREA').format(region_id=self.region_id))
        self.custom_colors_json = os.getenv('COLORS')

        # Make sure all output directories exist before writing
        for path in [self.region_shape_path, self.ids_region_file_path,
                     self.file_output_path, self.file_equi7_output_path, self.figure_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.ids = None
        logging.info(f"IDSProcessor initialised for Region {self.region_id}")

    # ------------------------------------------------------------------
    # Step 1 — Load raw IDS data
    # ------------------------------------------------------------------
    def load(self):
        """Load the raw IDS shapefile for this region from disk."""
        logging.info(f"Loading IDS file for Region {self.region_id}: {self.ids_region_file_path}")
        self.ids = load_data(self.ids_region_file_path)
        logging.info(f"Loaded {len(self.ids)} records.")

    # ------------------------------------------------------------------
    # Step 2 — Resolve overlapping and fragmented polygons
    # ------------------------------------------------------------------
    def resolve_polygon_overlaps(self):
        """
        Clean and restructure the raw IDS data in four sub-steps:
          a) Rename columns (shapefile 10-char limit), severity filter, year filter (>2009)
          b) Iteratively merge spatially fragmented polygons that share DCA_ID + year
          c) Remove entries that spatially and temporally overlap within ±5 years
             (these are likely duplicates across survey years)
          d) Keep entries with overlaps within ±2 years for separate analysis —
             these represent the same disturbance recorded multiple times and carry
             useful year-lag information
        The non-overlapping and overlapping subsets are then concatenated and enriched.
        """
        logging.info("Cleaning and restructuring IDS data...")

        # a) Column cleanup and basic filtering
        ids = clean_raw_ids_data(self.ids)
        logging.info(f"After column cleanup: {len(ids)} records (removed {len(self.ids) - len(ids)})")

        # b) Merge spatially fragmented polygons (same disturbance, same year, intersecting)
        ids = merge_fragmented_polygons(ids)
        ids["ID_E"] = ids.index  # Assign a unique event ID based on the index
        logging.info(f"After geometry merging: {len(ids)} records")

        # c) Remove compound events — polygons overlapping spatially within ±5 survey years
        unique_surveys = drop_compound_survey_events(ids, year_range=5)
        logging.info(f"After temporal overlap removal (±5 yrs): {len(unique_surveys)} records remaining")

        # d) Separately keep entries with overlaps within ±2 years for year-lag analysis
        recurring_surveys = extract_recurring_disturbances(ids, year_range=2)

        if recurring_surveys is not None:
            logging.info(f"Overlapping records kept for year-lag analysis (±2 yrs): {len(recurring_surveys)}")
            ids = pd.concat([unique_surveys, recurring_surveys], ignore_index=True)
        else:
            logging.info("No overlapping records detected within ±2 years.")
            ids = unique_surveys

        logging.info(f"After concatenation: {len(ids)} records")

        # Enrich the overlapping entries with year difference and partner DCA_ID
        ids = build_overlap_pair_records(ids)
        logging.info(f"After enrichment: {len(ids)} records")

        # Rename columns to stay within the 10-character shapefile field name limit
        ids.rename(columns={'SURVEY_YEA': 'SURVEY_Y', 'DA_Code_USDA': 'DA_C_USDA'}, inplace=True)

        self.ids = ids
        logging.info(f"Step 2 complete. Final record count: {len(self.ids)}")

    # ------------------------------------------------------------------
    # Step 3 — Filter to study period and disturbance types
    # ------------------------------------------------------------------
    def filter_by_study_criteria(self, start_year, end_year, excluded_dca_types):
        """
        Restrict the dataset to the study year range and relevant disturbance types,
        and drop polygons larger than 15 km² (likely mapping artefacts).
        """
        logging.info(f"Filtering to {start_year}–{end_year}, excluding: {excluded_dca_types}")
        logging.info(f"Records before filter: {len(self.ids)}")
        self.ids = apply_study_filters(self.ids, excluded_dca_types, start_year, end_year)
        logging.info(f"Records after filter: {len(self.ids)}")

    # ------------------------------------------------------------------
    # Utility — print a short dataset summary
    # ------------------------------------------------------------------
    def log_summary(self):
        """Log and print a summary of the current dataset state."""
        total = len(self.ids)
        unique = len(self.ids['ID_E'].unique())
        logging.info(f"Total records: {total} | Unique events: {unique} | Overlapping: {total - unique}")
        print(f"Total records: {total} | Unique events: {unique} | Overlapping: {total - unique}")

    # ------------------------------------------------------------------
    # Step 4 — Save outputs and generate figure
    # ------------------------------------------------------------------
    def save_and_plot(self):
        """
        Write two shapefiles (original CRS and Equi7 reprojection) and generate
        the study-area overview figure showing all disturbance polygons.
        """
        # Save in the original CRS (WGS84 / EPSG:4326)
        try:
            save_shapefile(self.ids, self.file_output_path)
            logging.info(f"Saved (original CRS) → {self.file_output_path}")
        except Exception as e:
            logging.error(f"Error saving original CRS shapefile: {e}")

        # Reproject to Equi7 North America and save
        try:
            ids_equi7 = self.ids.to_crs(self.target_crs)
            save_shapefile(ids_equi7, self.file_equi7_output_path)
            logging.info(f"Saved (Equi7 CRS) → {self.file_equi7_output_path}")
        except Exception as e:
            logging.error(f"Error saving Equi7 shapefile: {e}")
