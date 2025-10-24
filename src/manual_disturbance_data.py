#!/usr/bin/env python3
"""
manual_disturbance_data.py

This script loads manual disturbance records from an ODS file, normalizes years, aggregates
disturbance years, attaches polygons from GeoJSON files per DCA_ID, and enriches the dataset
with survey years and S1 years from shapefiles. The final dataset is saved as a GeoPackage.
"""

import os
import re
import pandas as pd
import geopandas as gpd
import numpy as np
from dateutil import parser


# ----------------------------- CONFIGURATION ----------------------------- #
DATA_PATH = "/net/projects/forexd/WP1/Data/random_manual_sample_files/"
CSV_FILE = "Manual_Times_Final_15.ods"

# Paths to additional shapefiles
S1DM_FILE = "/net/projects/forexd/WP1/03_LearningDisturbances/Data/radar_enhanced_forest_disturbance_mapping_region_08_buffer_500_s1dm.shp"
IDS_FILE = "/net/projects/forexd/WP1/03_LearningDisturbances/Data/region_08_dca_filtered_ids_usda_polygons.shp"

# Map sheet/source names to subfolders
SOURCE_MAP = {
    "fire": "fire",
    "wind": "wind",
    "defoliators": "defoliators",
    "bark beetle": "bark_beetle"
}

# Output filename
OUTPUT_FILE = os.path.join(DATA_PATH, "final_disturbance_data.shp")


# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def extract_years(row):
    """
    Extracts all unique years from the row columns: 'Year', 'Clearcut', 'aditional disturbance years'.
    Supports integers, floats, strings with multiple years or full dates.

    Parameters
    ----------
    row : pandas.Series
        A row from the disturbance DataFrame.

    Returns
    -------
    list or np.nan
        Sorted list of unique years found, or np.nan if none.
    """
    years = []
    for col in ["Year", "Clearcut", "aditional disturbance years"]:
        val = row.get(col, None)
        if pd.isna(val):
            continue
        if isinstance(val, (int, float)):
            y = int(val)
            if 1900 <= y <= 2100:
                years.append(y)
            continue
        if isinstance(val, str):
            # Regex to find 4-digit years
            found = re.findall(r"(19\d{2}|20\d{2})", val)
            years.extend(int(y) for y in found)
            if not found:
                # Try parsing as a date
                try:
                    dt = parser.parse(val, fuzzy=True)
                    if 1900 <= dt.year <= 2100:
                        years.append(dt.year)
                except Exception:
                    pass
    return sorted(set(years)) if years else np.nan


def normalize_dca_id(dca_id):
    """
    Normalize DCA_ID: strip spaces and lowercase the first character.

    Parameters
    ----------
    dca_id : str
        Original DCA_ID string.

    Returns
    -------
    str
        Normalized DCA_ID.
    """
    dca_id = str(dca_id).strip()
    return dca_id[0].lower() + dca_id[1:] if dca_id else dca_id


# ----------------------------- MAIN SCRIPT ----------------------------- #
def main():
    # Load all sheets from the ODS file
    file_path = os.path.join(DATA_PATH, CSV_FILE)
    sheets = pd.read_excel(file_path, sheet_name=None, engine="odf")

    processed = []

    # Process each sheet
    for sheet_name, df in sheets.items():
        keep_cols = ["DCA_ID", "Year", "Month", "Day", "Clearcut", "aditional disturbance years"]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        # Normalize DCA_ID
        if "DCA_ID" in df.columns:
            df["DCA_ID"] = df["DCA_ID"].apply(normalize_dca_id)

        # Convert numeric columns to Int64
        for col in ["Year", "Day", "Clearcut"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Extract disturbance years
        df["disturbance_years"] = df.apply(extract_years, axis=1)

        # Add source sheet info
        df["Source"] = sheet_name
        processed.append(df)

    # Combine all sheets
    final_df = pd.concat(processed, ignore_index=True)

    # ----------------------------- Attach polygons ----------------------------- #
    geoms = []
    for idx, row in final_df.iterrows():
        dca_id = row["DCA_ID"]
        source_key = row["Source"].lower().strip()

        if source_key not in SOURCE_MAP:
            geoms.append(None)
            continue

        folder = os.path.join(DATA_PATH, SOURCE_MAP[source_key], dca_id)
        geom = None
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if "merged" in f.lower() and f.endswith(".geojson")]
            if files:
                gdf = gpd.read_file(os.path.join(folder, files[0]))
                geom = gdf.unary_union  # dissolve into a single geometry
        geoms.append(geom)

    # Create GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final_df, geometry=geoms, crs="EPSG:4326")

    # ----------------------------- Attach SURVEY_Y and S1_YEAR ----------------------------- #
    gdf_s1dm = gpd.read_file(S1DM_FILE)
    gdf_ids = gpd.read_file(IDS_FILE)

    survey_y_all = []
    s1_year_all = []

    for idx, row in final_gdf.iterrows():
        dca_id = row["DCA_ID"].lower().strip()
        ids = gdf_ids[gdf_ids["IDX_D"].astype(str).str.strip().str.lower() == dca_id]
        s1dm = gdf_s1dm[gdf_s1dm["IDX_D"].astype(str).str.strip().str.lower() == dca_id]

        survey_vals = []
        s1_vals = []

        if not ids.empty and "SURVEY_Y" in ids.columns:
            survey_vals.extend(ids["SURVEY_Y"].dropna().unique().tolist())
        if not s1dm.empty and "S1_YEAR" in s1dm.columns:
            s1_vals.extend(s1dm["S1_YEAR"].dropna().unique().tolist())

        survey_y_all.append(sorted(set(survey_vals)) if survey_vals else np.nan)
        s1_year_all.append(sorted(set(s1_vals)) if s1_vals else np.nan)

    final_gdf["SURVEY_Y"] = survey_y_all
    final_gdf["S1_YEAR"] = s1_year_all

    # ----------------------------- Save final GeoDataFrame ----------------------------- #
    final_gdf.to_file(OUTPUT_FILE)
    print(f"Final GeoDataFrame saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
