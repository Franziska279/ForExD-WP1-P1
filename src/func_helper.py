"""
func_helper.py — General Helper Functions
==========================================
Author:  Franziska Müller (Uni Leipzig / MPI-BGC)
Project: ForExD-WP1-P1

Description
-----------
Miscellaneous utility functions used across the pipeline for:
  - Label formatting (plots)
  - Color palette parsing (from .env JSON string)
  - Region geometry extraction from the USFS administrative shapefile
  - Area and geometry calculations (convex hull, km², centroid shift)
  - GeoDataFrame manipulation (merge geometries, remove disturbance types)
  - Statistical helpers (overlap/Jaccard, paired t-test)
  - Signal-duration aggregation for the year-lag analysis
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from func_file_io import load_data


# ==============================================================
# Label and Colour Formatting
# ==============================================================

def format_label(label):
    """
    Convert a snake_case string to Title Case for use in plot labels.
    Example: 'bark_beetle' → 'Bark Beetle'
    """
    return ' '.join(word.capitalize() for word in label.split('_'))


def format_label_count(dca_id, count):
    """
    Format a disturbance type label with its event count for plot legends.
    Example: 'bark_beetle', 42 → 'Bark Beetle (42)'
    """
    return f'{dca_id.replace("_", " ").title()} ({count})'


def parse_color_map(colors_json):
    """
    Parse the COLORS JSON string from the .env file into a Python dictionary.

    The .env stores colours as a JSON string, e.g.:
      COLORS='{"wind": "#1f77b4", "bark_beetle": "#714709"}'

    Returns an empty dict if the string is empty, None, or invalid JSON.
    """
    if not colors_json:
        return {}
    try:
        return json.loads(colors_json)
    except json.JSONDecodeError:
        print("Warning: COLORS in .env is not valid JSON. Using empty colour map.")
        return {}


# ==============================================================
# Region Geometry Extraction
# ==============================================================

def load_region_boundary(path, region_id, crs=None):
    """
    Load the USFS administrative regions shapefile and return the largest
    polygon part for the given region_id.

    Multi-part regions (e.g. regions with island territories) are exploded and
    only the largest contiguous part is kept — this is the CONUS mainland area.

    Parameters
    ----------
    path      : str            path to the S_USA.AdministrativeRegion shapefile
    region_id : str or int     USFS region number (e.g. '08' or 8)
    crs       : str or None    if given, reproject the result to this CRS (e.g. 'EPSG:27705')

    Returns
    -------
    gpd.GeoDataFrame with a single row representing the region boundary.
    """
    usa_shape = gpd.read_file(path)
    region = usa_shape[usa_shape['REGION'] == region_id]
    region_conus = region.explode(index_parts=False).iloc[0:1]
    if crs:
        return region_conus.to_crs(crs)
    return region_conus


def load_mainland_regions(gdf_path, regions_to_clean=('05', '08')):
    """
    Load the full USFS regions shapefile and, for regions with offshore territories
    (e.g. 05 and 08), keep only the largest polygon (the mainland part).

    Region 10 (Alaska) is excluded entirely as it falls outside the study area.

    Parameters
    ----------
    gdf_path         : str   path to the shapefile
    regions_to_clean : list  region codes where only the largest polygon is kept

    Returns
    -------
    gpd.GeoDataFrame with cleaned region geometries.
    """
    gdf = gpd.read_file(gdf_path)
    gdf = gdf[gdf['REGION'] != '10']  # exclude Alaska

    cleaned = []
    for region_code in regions_to_clean:
        region_gdf = gdf[gdf['REGION'] == region_code]
        exploded = region_gdf.explode(index_parts=True)
        exploded['area'] = exploded.geometry.area
        # Keep only the largest polygon (mainland)
        largest = exploded.loc[exploded['area'].idxmax()]
        cleaned.append(largest.to_frame().T)

    remaining = gdf[~gdf['REGION'].isin(regions_to_clean)]
    updated = pd.concat([remaining, pd.concat(cleaned, ignore_index=True)], ignore_index=True)
    return gpd.GeoDataFrame(updated, geometry='geometry', crs=gdf.crs)


# ==============================================================
# Area and Geometry Calculations
# ==============================================================

def calculate_area_in_km2(gdf):
    """
    Add an 'area_km2' column with polygon area in square kilometres.
    Reprojects to EPSG:3857 (metres) for the calculation.
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    projected = gdf.to_crs('EPSG:3857')
    gdf = gdf.copy()
    gdf['area_km2'] = projected.geometry.area / 1e6
    return gdf


def add_convex_hull_area(gdf):
    """
    Replace each geometry with its convex hull, then compute area in km².
    Used to compare the bounding footprint of IDS vs. S1 detection polygons.
    """
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.convex_hull if geom else None)
    return calculate_area_in_km2(gdf)


def dissolve_to_event_level(gdf):
    """
    Dissolve polygons by 'IDX_D' and 'S1_YEAR' into a single MultiPolygon per group,
    keeping the first value for all non-geometry columns.

    Used to aggregate per-tile S1 detections back to the disturbance-event level
    before computing overlap metrics.
    """
    # Dissolve geometry per IDX_D/S1_YEAR group
    grouped = gdf.groupby(['IDX_D', 'S1_YEAR']).apply(
        lambda grp: grp.unary_union
    ).reset_index(name='geometry')

    # Carry over first value of all other columns
    for col in gdf.columns:
        if col not in ['IDX_D', 'S1_YEAR', 'geometry']:
            grouped[col] = gdf.groupby(['IDX_D', 'S1_YEAR'])[col].first().values

    grouped = gpd.GeoDataFrame(grouped, geometry='geometry')

    # Ensure geometries are MultiPolygon for consistent downstream handling
    grouped['geometry'] = grouped['geometry'].apply(
        lambda geom: MultiPolygon([geom]) if not isinstance(geom, MultiPolygon) else geom
    )
    grouped.set_crs(gdf.crs, allow_override=True, inplace=True)
    return grouped


# ==============================================================
# Filtering Helpers
# ==============================================================

def drop_drought(gdf):
    """Remove all rows with DCA_ID == 'drought' from the GeoDataFrame."""
    return gdf[gdf['DCA_ID'] != 'drought']


def drop_disturbance_types(gdf, ids_to_remove):
    """
    Remove rows where DCA_ID matches any value in ids_to_remove.

    Parameters
    ----------
    gdf           : gpd.GeoDataFrame
    ids_to_remove : str or list of str   DCA_ID value(s) to exclude
    """
    if isinstance(ids_to_remove, str):
        ids_to_remove = [ids_to_remove]
    return gdf[~gdf['DCA_ID'].isin(ids_to_remove)]


# ==============================================================
# Year-Lag / Signal Duration Helpers
# ==============================================================

def nearest_s1_detection_year(group):
    """
    For a group of S1 detection records sharing the same IDX_D, return the
    S1_YEAR that is closest to the IDS SURVEY_Y.

    Used when aggregating per-tile S1 results to a single representative year
    per disturbance event.
    """
    group['S1_YEAR']  = pd.to_numeric(group['S1_YEAR'],  errors='coerce')
    group['SURVEY_Y'] = pd.to_numeric(group['SURVEY_Y'], errors='coerce')
    group['S1_Y_diff'] = (group['S1_YEAR'] - group['SURVEY_Y']).abs()
    return group.loc[group['S1_Y_diff'].idxmin(), 'S1_YEAR']


def aggregate_detections_by_event(s1dm_gdf, crs=None):
    """
    Aggregate per-year S1 detection records to one row per disturbance event (IDX_D),
    adding a 'signal_duration' column that counts how many S1 years detected the event.

    For each IDX_D:
      - Geometries are merged into a single MultiPolygon (union)
      - The S1_YEAR closest to SURVEY_Y is kept as the representative detection year
      - All other columns take the first value within the group
      - 'signal_duration' = number of S1 years with a detection for that event

    Parameters
    ----------
    s1dm_gdf : gpd.GeoDataFrame   S1 detection results with 'IDX_D' column
    crs       : str or None         CRS to assign to the output (optional)

    Returns
    -------
    gpd.GeoDataFrame with one row per IDX_D and a 'signal_duration' column.
    """
    # Count how many S1 years detected each disturbance event
    counts_per_event = s1dm_gdf.groupby('IDX_D').size().reset_index(name='signal_duration')

    # Aggregate: union the geometry, keep first value for all other columns
    agg_funcs = {'geometry': unary_union}
    for col in s1dm_gdf.columns:
        if col not in ('IDX_D', 'geometry'):
            agg_funcs[col] = 'first'

    events_gdf = s1dm_gdf.groupby('IDX_D').agg(agg_funcs).reset_index()

    # Replace S1_YEAR with the year closest to SURVEY_Y for each event
    events_gdf['S1_YEAR'] = s1dm_gdf.groupby('IDX_D').apply(nearest_s1_detection_year).values

    events_gdf = gpd.GeoDataFrame(events_gdf, geometry='geometry')
    events_gdf = events_gdf.merge(counts_per_event, on='IDX_D', how='left')

    if crs:
        events_gdf.set_crs(crs, allow_override=True, inplace=True)

    return events_gdf


# ==============================================================
# Overlap and Statistical Helpers
# ==============================================================

def compute_spatial_overlap(ids, s1dm):
    """
    Calculate the spatial overlap percentage between IDS and S1DM polygons
    for each shared IDX_D:
      - percentage_ids  : fraction of the IDS polygon covered by S1DM
      - percentage_s1cd : fraction of the S1DM polygon covered by IDS

    Returns a DataFrame with columns [DCA_ID, percentage_ids, percentage_s1cd].
    """
    if ids.crs != s1dm.crs:
        s1dm = s1dm.to_crs(ids.crs)

    common = set(ids["IDX_D"]).intersection(s1dm["IDX_D"])
    ids_matched  = ids[ids["IDX_D"].isin(common)]
    s1dm_matched = s1dm[s1dm["IDX_D"].isin(common)]

    results = []
    for idx_d in common:
        ids_poly  = ids_matched[ids_matched["IDX_D"] == idx_d].geometry.union_all()
        s1dm_poly = s1dm_matched[s1dm_matched["IDX_D"] == idx_d].geometry.union_all()

        ids_area   = ids_poly.area
        s1dm_area  = s1dm_poly.area
        inter_area = ids_poly.intersection(s1dm_poly).area

        results.append({
            "DCA_ID":         ids_matched[ids_matched["IDX_D"] == idx_d]["DCA_ID"].iloc[0],
            "percentage_ids":  (inter_area / ids_area)  * 100 if ids_area  > 0 else 0,
            "percentage_s1cd": (inter_area / s1dm_area) * 100 if s1dm_area > 0 else 0,
        })

    return pd.DataFrame(results)


def compute_jaccard_overlap(geom_candidate, geom_manual):
    """
    Compute the spatial overlap (%) and Jaccard index between a candidate
    geometry (IDS or S1DM) and a reference manual geometry.

    Both geometries are validated before computation to avoid errors from
    self-intersecting or degenerate polygons.

    Returns
    -------
    overlap_pct : float   percent of manual area covered by candidate
    jaccard     : float   intersection / union (0–1)
    """
    geom_candidate = make_valid(geom_candidate)
    geom_manual    = make_valid(geom_manual)

    inter = geom_candidate.intersection(geom_manual)
    union = geom_candidate.union(geom_manual)

    overlap_pct = (inter.area / geom_manual.area) * 100 if geom_manual.area > 0 else 0
    jaccard     = (inter.area / union.area)              if union.area      > 0 else 0

    return overlap_pct, jaccard


def paired_ttest(x, y, alpha1=0.05, alpha2=0.1, alternative="greater"):
    """
    One-sided paired t-test to compare two sets of spatial accuracy metrics
    (e.g. S1DM Jaccard vs. IDS Jaccard for the same set of events).

    Parameters
    ----------
    x, y        : array-like   matched sample pairs (e.g. S1DM vs. IDS values)
    alpha1      : float        primary significance level (default 0.05)
    alpha2      : float        secondary significance level (default 0.10)
    alternative : str          'greater' or 'less'

    Returns
    -------
    pvalue     : float   p-value from the paired t-test
    sig_symbol : str     '*' if p < alpha1, '+' if p < alpha2, else ''
    """
    result = ttest_rel(x, y, alternative=alternative)
    if result.pvalue < alpha1:
        return result.pvalue, '*'
    elif result.pvalue < alpha2:
        return result.pvalue, '+'
    return result.pvalue, ''


def paired_wilcoxon_significance(x, y, alpha1=0.05, alpha2=0.1, alternative="greater"):
    """
    One-sided paired Wilcoxon signed-rank test to compare two sets of spatial
    accuracy metrics (e.g. S1DM Jaccard vs. IDS Jaccard for the same events).

    Parameters
    ----------
    x, y        : array-like   matched sample pairs (e.g. S1DM vs. IDS values)
    alpha1      : float        primary significance level (default 0.05)
    alpha2      : float        secondary significance level (default 0.10)
    alternative : str          'greater' or 'less'

    Returns
    -------
    pvalue     : float   p-value from the Wilcoxon signed-rank test
    sig_symbol : str     '*' if p < alpha1, '+' if p < alpha2, else ''
    """
    result = wilcoxon(x, y, alternative=alternative)
    if result.pvalue < alpha1:
        return result.pvalue, '*'
    elif result.pvalue < alpha2:
        return result.pvalue, '+'
    return result.pvalue, ''


def find_best_geometry_match(
    query_geom, candidates_gdf, id_col='IDX_D', min_overlap_pct=0.05
):
    """Find the candidate polygon with the greatest spatial overlap.

    For each polygon in *candidates_gdf*, computes what fraction of
    *query_geom*'s area is covered by that polygon (intersection / query
    area).  Returns the ``id_col`` value of the best match, or ``None``
    when no candidate meets *min_overlap_pct*.

    Used to connect manual label polygons to IDS/S1DM results when the
    folder-name IDX_D no longer matches the current shapefile IDX_D.

    Parameters
    ----------
    query_geom       : shapely.geometry   manual reference polygon
    candidates_gdf   : GeoDataFrame       IDS or S1DM polygons to search
    id_col           : str                column holding the identifier to return
    min_overlap_pct  : float              minimum intersection/query-area fraction

    Returns
    -------
    str or None   ``id_col`` of the best-overlapping candidate, or ``None``.
    """
    if candidates_gdf.empty or query_geom.is_empty:
        return None

    query_area = query_geom.area
    if query_area == 0:
        return None

    best_id, best_pct = None, 0.0
    for _, row in candidates_gdf.iterrows():
        try:
            pct = query_geom.intersection(row.geometry).area / query_area
        except Exception:
            continue
        if pct > best_pct:
            best_pct, best_id = pct, row[id_col]

    return best_id if best_pct >= min_overlap_pct else None
