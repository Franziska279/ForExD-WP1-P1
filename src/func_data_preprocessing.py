"""
func_data_preprocessing.py — IDS Data Preprocessing Functions
==============================================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Preprocessing functions for the USDA IDS (Insect and Disease Survey) dataset,
called by IDSProcessor in ids_processor.py. The pipeline runs in this order:

  clean_raw_ids_data()             -- column cleanup, severity + year filter
  merge_fragmented_polygons()      -- merge fragmented polygons iteratively
  drop_compound_survey_events()    -- drop compound survey entries (±5 yrs)
  extract_recurring_disturbances() -- extract multi-year overlap records (±2 yrs)
  build_overlap_pair_records()     -- explode and annotate overlap pairs
  apply_study_filters()            -- final year/type/size filter

Also contains add_area_km2() and get_tile_basename() which are imported
by func_s1cd_preprocessing.py.
"""

import hashlib
import logging
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm


# ==============================================================
# IDS Preprocessing Pipeline
# ==============================================================

def clean_raw_ids_data(gdf):
    """
    First cleaning pass on the raw IDS GeoDataFrame:
      - Truncates all column names to 10 characters (ESRI shapefile limit)
      - Keeps only 'Severe' damage entries (or rows with no severity info)
      - Drops columns not needed downstream
      - Removes pre-2010 survey records
      - Fixes any invalid geometries using a zero-width buffer

    Returns the cleaned GeoDataFrame.
    """
    n_start = len(gdf)
    logging.info(f"clean_raw_ids_data: starting with {n_start} records")

    # ESRI shapefiles truncate field names to 10 characters — enforce this now
    # so column references are consistent throughout the pipeline
    gdf = gdf.rename(columns={col: col[:10] for col in gdf.columns})

    # Keep only records with severe damage or no severity recorded.
    # Light/moderate damage entries are excluded from the analysis.
    n_before = len(gdf)
    gdf = gdf[gdf['PERCENT_AF'].isna() | gdf['PERCENT_AF'].str.contains("Severe", case=False, na=False)]
    logging.info(f"Severity filter: {len(gdf)} remaining (removed {n_before - len(gdf)})")

    # Drop columns that carry no information for our analysis
    cols_to_drop = ['PERCENT_AF', 'HOST', 'HOST_CODE', 'DCA_CODE', 'DAMAGE_TYP', 'DAMAGE_T_1', 'cluster_id']
    gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])

    # Restrict to post-2009 surveys; earlier records have inconsistent spatial quality
    n_before = len(gdf)
    gdf = gdf[gdf['SURVEY_YEA'] > 2009]
    logging.info(f"Year filter (>2009): {len(gdf)} remaining (removed {n_before - len(gdf)})")

    # Fix invalid geometries that would cause spatial operations to fail
    n_invalid = (~gdf.geometry.is_valid).sum()
    if n_invalid > 0:
        logging.info(f"Fixing {n_invalid} invalid geometries with buffer(0)")
        gdf['geometry'] = gdf.geometry.apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

    logging.info(f"clean_raw_ids_data: finished with {len(gdf)} records (removed {n_start - len(gdf)} total)")
    return gdf


def merge_fragmented_polygons(gdf, max_iterations=10):
    """
    Iteratively merges spatially intersecting polygons that share the same
    DCA_ID and SURVEY_YEA. A single disturbance event is often mapped as
    multiple non-contiguous polygons — this step collapses them into one.

    Repeats until no further merges occur or max_iterations is reached.
    """
    n_prev = len(gdf)
    for iteration in range(max_iterations):
        logging.info(f"merge_fragmented_polygons: iteration {iteration + 1}, {n_prev} records")
        gdf = _dissolve_same_event_polygons(gdf)
        if len(gdf) == n_prev:
            logging.info("merge_fragmented_polygons: no further changes, stopping.")
            break
        n_prev = len(gdf)
    logging.info(f"merge_fragmented_polygons: finished with {len(gdf)} records")
    return gdf


def _dissolve_same_event_polygons(gdf):
    """
    Single-pass merge: finds all polygon pairs that intersect AND share the
    same DCA_ID and SURVEY_YEA, assigns them to a group, then dissolves each
    group into one polygon (keeping the first value for all attribute columns).
    """
    gdf = gdf.reset_index(drop=True)

    # Spatial self-join to find all intersecting polygon pairs
    intersections = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects",
                              lsuffix="left", rsuffix="right")

    # Keep only pairs that represent the same disturbance event (same type and year)
    intersections = intersections[
        (intersections["SURVEY_YEA_left"] == intersections["SURVEY_YEA_right"]) &
        (intersections["DCA_ID_left"]     == intersections["DCA_ID_right"])
    ]

    # Drop the "_right" duplicate columns and clean up suffixes
    intersections = intersections.loc[:, ~intersections.columns.str.endswith('_right')]
    intersections.columns = intersections.columns.str.replace('_left', '', regex=False)

    # Assign each polygon to a group; polygons in the same group will be dissolved
    gdf["group"] = -1
    group_id = 0
    for idx, row in intersections.iterrows():
        if gdf.at[row.name, "group"] == -1:
            gdf.at[row.name, "group"] = group_id
        gdf.loc[intersections.index[intersections.index == row.name], "group"] = gdf.at[row.name, "group"]
        group_id += 1

    return gdf.dissolve(by=["group"], aggfunc="first").reset_index(drop=True)


def drop_compound_survey_events(gdf, year_col='SURVEY_YEA', geom_col='geometry', year_range=5):
    """
    Removes entries that both:
      - spatially intersect another polygon, AND
      - fall within ±year_range survey years of that polygon

    These are compound events — the same disturbance location surveyed across
    multiple years — and are removed to avoid counting the same event multiple times.

    Returns the GeoDataFrame with compound entries dropped.
    """
    indices_to_drop = set()
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Dropping compound events (±{year_range} yrs)"):
        in_time_range = gdf[year_col].between(row[year_col] - year_range, row[year_col] + year_range)
        overlaps = gdf[in_time_range & gdf[geom_col].intersects(row[geom_col])]
        if len(overlaps) > 1:  # >1 because every polygon intersects itself
            indices_to_drop.update(overlaps.index)
    return gdf.drop(index=indices_to_drop)


def extract_recurring_disturbances(gdf, id_col='ID_E', year_col='SURVEY_YEA',
                                   geom_col='geometry', year_range=2):
    """
    Identifies polygons that spatially overlap another polygon within ±year_range years.
    These are disturbances recorded in consecutive survey years — kept separately
    for the year-lag analysis (Figure 4).

    For each overlapping pair, records the partner event IDs ('ID_O') and computes
    summary statistics (overlap duration, partner DCA_IDs).

    Returns a GeoDataFrame of recurring records, or None if none are found.
    """
    gdf = gdf.copy()
    gdf['ID_O'] = None  # Will hold the list of overlapping partner IDs

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Finding recurring disturbances (±{year_range} yrs)"):
        in_time_range = gdf[year_col].between(row[year_col] - year_range, row[year_col] + year_range)
        partner_events = gdf.loc[
            in_time_range &
            gdf[geom_col].intersects(row[geom_col]) &
            (gdf[id_col] != row[id_col])  # exclude self
        ]
        if not partner_events.empty:
            gdf.at[idx, 'ID_O'] = partner_events[id_col].tolist()

    # Keep only rows that have at least one recurring partner
    recurring_gdf = gdf.dropna(subset=['ID_O'])
    if recurring_gdf.empty:
        return None

    # Compute summary statistics for each recurring record
    recurring_gdf = recurring_gdf.copy()
    recurring_gdf['Longest_Duration'] = None
    recurring_gdf['DCA_ID_Count']     = None
    recurring_gdf['DCA_ID_List']      = None

    for idx, row in tqdm(recurring_gdf.iterrows(), total=len(recurring_gdf), desc="Analysing recurring events"):
        partner_events = recurring_gdf[recurring_gdf[id_col].isin(row['ID_O'])]
        recurring_gdf.at[idx, 'Longest_Duration'] = partner_events[year_col].max() - partner_events[year_col].min() + 1
        partner_dca_ids = partner_events['DCA_ID'].tolist()
        recurring_gdf.at[idx, 'DCA_ID_Count'] = len(partner_dca_ids)
        recurring_gdf.at[idx, 'DCA_ID_List']  = partner_dca_ids

    # Keep only records where the partner has the same disturbance type
    return recurring_gdf[recurring_gdf.apply(lambda r: r['DCA_ID_List'] == [r['DCA_ID']], axis=1)]


def build_overlap_pair_records(gdf, year_col='SURVEY_YEA', id_col='ID_E', dca_col='DCA_ID'):
    """
    Prepares the recurring-events subset for the year-lag analysis:
      - Explodes the 'ID_O' list so each overlapping pair becomes a separate row
      - Looks up the partner's survey year ('O_Year') and disturbance type ('O_DCA_ID')
      - Computes the year difference between the two observations ('O_Y_diff')

    The resulting table drives the year-lag figures showing how many years apart
    the same disturbance was detected by S1 vs. recorded in IDS.
    """
    # One row per overlapping pair (explode the partner-ID list)
    pairs = gdf.explode('ID_O').drop(columns=['Longest_Duration', 'DCA_ID_Count', 'DCA_ID_List'])

    # Build lookup tables from the original dataframe to annotate each pair
    survey_year_by_id = gdf.set_index(id_col)[year_col].to_dict()
    dca_type_by_id    = gdf.set_index(id_col)[dca_col].to_dict()

    pairs['O_Year']   = pairs['ID_O'].map(survey_year_by_id)
    pairs['O_DCA_ID'] = pairs['ID_O'].map(dca_type_by_id)
    pairs['O_Y_diff'] = pairs['O_Year'] - pairs[year_col]  # positive = partner detected later

    return pairs


def apply_study_filters(gdf, excluded_dca_types, start_year=2015, end_year=2021):
    """
    Final filter step applied to the combined (recurring + unique) dataset:
      1. Splits into unique-survey and recurring-survey subsets
      2. Applies year range and disturbance-type filters to both
      3. Drops polygons larger than 15 km² (likely mapping artefacts)
      4. For recurring pairs: removes pairs where the disturbance type differs
         between the two observations (unreliable pairing)
      5. Merges both subsets and assigns a unique IDX_D identifier to each record

    Parameters
    ----------
    gdf                : GeoDataFrame  combined IDS dataset with 'ID_O' column
    excluded_dca_types : list of str   DCA_ID values to drop (e.g. 'other')
    start_year         : int           first year to include (exclusive: > start_year)
    end_year           : int           last year to include (inclusive: <= end_year)
    """
    logging.info(f"apply_study_filters: starting with {len(gdf)} records")

    # --- Split into unique and recurring surveys ---
    unique_surveys    = gdf[gdf['ID_O'].isnull()].copy()
    recurring_surveys = gdf[gdf['ID_O'].notnull()].copy()
    logging.info(f"  Unique surveys: {len(unique_surveys)}, Recurring: {len(recurring_surveys)}")

    def _year_type_filter(df):
        return df[
            (df['SURVEY_Y'] > start_year) &
            (df['SURVEY_Y'] <= end_year) &
            (~df['DCA_ID'].isin(excluded_dca_types))
        ].copy()

    # --- Filter and size-cap unique surveys ---
    unique_filtered = _year_type_filter(unique_surveys)
    logging.info(f"  Unique after year/type filter: {len(unique_filtered)}")
    unique_filtered = _drop_oversized_polygons(unique_filtered)
    logging.info(f"  Unique after area filter (≤15 km²): {len(unique_filtered)}")

    # --- Filter and size-cap recurring surveys ---
    recurring_filtered = _year_type_filter(recurring_surveys)
    logging.info(f"  Recurring after year/type filter: {len(recurring_filtered)}")
    recurring_filtered = _drop_oversized_polygons(recurring_filtered)
    logging.info(f"  Recurring after area filter (≤15 km²): {len(recurring_filtered)}")

    # Ensure IDs are integers; remove self-pairing rows
    recurring_filtered['ID_E'] = recurring_filtered['ID_E'].astype(int)
    recurring_filtered['ID_O'] = recurring_filtered['ID_O'].astype(int)
    recurring_filtered = recurring_filtered[recurring_filtered['ID_E'] != recurring_filtered['ID_O']]

    # Remove pairs where the two observations disagree on disturbance type
    dca_mismatch    = recurring_filtered[recurring_filtered['DCA_ID'] != recurring_filtered['O_DCA_ID']]
    invalid_event_ids = set(dca_mismatch['ID_E']).union(set(dca_mismatch['ID_O']))
    clean_recurring = recurring_filtered[
        ~recurring_filtered['ID_E'].isin(invalid_event_ids) &
        ~recurring_filtered['ID_O'].isin(invalid_event_ids)
    ]
    logging.info(f"  Recurring after DCA mismatch removal: {len(clean_recurring)}")

    # --- Combine and assign unique identifiers ---
    ids = pd.concat([unique_filtered, clean_recurring], ignore_index=True)

    # IDX_D uniquely identifies each record: type_year_<geomHash>
    # Uses first 8 hex chars of SHA-1(WKB) so the ID is stable across
    # re-runs with different filter parameters (unlike a row index).
    ids['IDX_D'] = ids.apply(
        lambda row: (
            f"{row['DCA_ID']}_{row['SURVEY_Y']}_"
            f"{hashlib.sha1(row.geometry.wkb).hexdigest()[:8]}"
        ),
        axis=1,
    )

    result = gpd.GeoDataFrame(ids, geometry='geometry')
    logging.info(f"apply_study_filters: finished with {len(result)} records")
    return result


def _drop_oversized_polygons(gdf, max_km2=15):
    """Add area column and keep only polygons ≤ max_km2 square kilometres."""
    gdf = add_area_km2(gdf)
    return gdf[gdf['area_km2'] <= max_km2]


# ==============================================================
# Area Utilities
# ==============================================================

def add_area_km2(gdf, col_name='area_km2'):
    """
    Add a column with polygon area in square kilometres.

    Reprojects to EPSG:3857 (metres) for accurate area calculation, then
    attaches the result back to the original GeoDataFrame (preserving its CRS).

    Parameters
    ----------
    gdf      : GeoDataFrame
    col_name : str   name of the new area column (default 'area_km2';
                     S1CD code passes 'area' to match its expected column name)
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    projected = gdf.to_crs('EPSG:3857')
    gdf = gdf.copy()
    gdf[col_name] = projected.geometry.area / 1e6
    return gdf


# ==============================================================
# Utilities used by func_s1cd_preprocessing.py
# ==============================================================

def get_tile_basename(filename):
    """
    Extract the first 10 underscore-separated parts of a Sentinel-1 tile filename.
    Used to derive a consistent base name for output shapefiles.

    Example: 's1_cd_northamerica_year_2018_tile_EU010M_E040N027T3.nc'
             → 's1_cd_northamerica_year_2018_tile_EU010M_E040N027T3'
    """
    return '_'.join(filename.split('_')[:10])