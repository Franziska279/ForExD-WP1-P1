"""Visualisation functions for the ForExD-WP1-P1 pipeline.

Produces publication-quality figures comparing IDS forest-disturbance
survey labels against Sentinel-1-derived detections (S1DM) across
USFS Region 8 (south-eastern USA, 2016-2020).

Sections
--------
HELPERS                : Small formatting / statistical utilities.
ANALYSIS               : Pure-computation functions (no plotting).
STUDY AREA COMPONENTS  : Sub-routines that compose the study-area figure.
PLOTS                  : Top-level, figure-generating functions.
"""

# ── Standard library ──────────────────────────────────────────────────────
import logging
import os
import re

# ── Third-party ───────────────────────────────────────────────────────────
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde
from shapely.ops import unary_union
from shapely.validation import make_valid

# ── Local ─────────────────────────────────────────────────────────────────
from func_file_io import load_data, load_tcc_dataset
from func_helper import (
    compute_jaccard_overlap,
    compute_spatial_overlap,
    find_best_geometry_match,
    format_label,
    load_mainland_regions,
    load_region_boundary,
    paired_wilcoxon_significance,
)

# ── Module-level constants ────────────────────────────────────────────────

# Save resolution
SAVE_DPI_HIGH = 400
SAVE_DPI_MED = 300

# Coordinate system
CRS_WGS84 = "EPSG:4326"

# Disturbance types used throughout
DISTURBANCE_ORDER = ["wind", "bark_beetle", "defoliators"]
DIST_LABEL_MAP = {
    "wind": "Wind",
    "bark_beetle": "Bark Beetle",
    "defoliators": "Defoliators",
}

# Colours - event-count bar chart (plot_radar_reduction_potential)
COLOR_IDS = "#BCB6FF"
COLOR_S1DM_BAR = "#AF42AE"
COLOR_REDUCTION = "#FF3E41"

# Colours - validation boxplots (plot_manual_validation_boxplots)
PALETTE_VALIDATION = {"IDS": "#4C72B0", "S1DM": "#55A868"}


# ── Helpers ───────────────────────────────────────────────────────────────


def compute_median_p90(series):
    """Return the median and 90th percentile of a Series, safely.

    Drops NaN and infinite values before computing.  Returns (nan, nan)
    when the cleaned series is empty.

    Args:
        series (pd.Series): Input numeric series.

    Returns:
        tuple[float, float]: (median, 90th-percentile) or
            (nan, nan) if no valid values remain.
    """
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    return np.median(clean), np.percentile(clean, 90)


def format_ticks(x, pos):  # noqa: ARG001
    """Format an axis tick value to one decimal place.

    Designed for use with ``matplotlib.ticker.FuncFormatter``.

    Args:
        x (float): Tick value.
        pos: Tick position (required by the FuncFormatter protocol,
            not used here).

    Returns:
        str: Tick label formatted as ``'%.1f'``.
    """
    return f"{x:.1f}"


def log_array_stats(name, arr, category, label):
    """Log the median and 90th percentile of an array.

    Logs a warning instead when the array is empty.

    Args:
        name (str): Metric name (e.g. ``'area_km2'``).
        arr (array-like): Numeric array to summarise.
        category (str): Disturbance category (e.g. ``'bark_beetle'``).
        label (str): Dataset label (e.g. ``'IDS'`` or ``'S1DM'``).
    """
    median, p90 = compute_median_p90(pd.Series(arr))
    if np.isnan(median):
        logging.warning("[%s] %s %s: EMPTY", category, label, name)
        return None
    logging.info(
        "[%s] %s %s | median=%.4f | p90=%.4f",
        category, label, name, median, p90,
    )
    return {"dca_id": category, "dataset": label, "metric": name,
            "median": median, "p90": p90}


def set_two_positive_yticks(ax):
    """Reduce the y-axis of *ax* to at most two positive tick marks.

    Useful for KDE subplots where the default tick density is too high.

    Args:
        ax (matplotlib.axes.Axes): Target axes to modify in-place.
    """
    ymin, ymax = ax.get_ylim()
    ticks = [t for t in np.linspace(ymin, ymax, 4) if t > 0]
    if len(ticks) > 2:
        ticks = ticks[:2]
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))


# ── Analysis ──────────────────────────────────────────────────────────────


def map_manual_labels_by_geometry(
    manual_base_folder,
    ids_gdf,
    s1dm_gdf,
    disturbances=None,
    min_overlap_pct=0.05,
):
    """Build a mapping from manual-label folders to current IDS/S1DM IDX_D.

    Manual label folders are named with the IDX_D that was current when
    the labels were digitised.  When the IDS pipeline is re-run with
    different parameters the IDX_D strings change, breaking the link.
    This function reconnects them by **spatial overlap** instead:
    for each manual polygon, it finds the IDS polygon that overlaps it
    most (same DCA type), then uses that polygon's IDX_D to look up the
    corresponding S1DM detection.

    Can be called standalone to inspect or repair the mapping, and is
    used internally by ``compute_manual_validation_metrics`` as a
    geometry-based fallback whenever a direct IDX_D lookup fails.

    Args:
        manual_base_folder (str): Root folder containing one sub-folder
            per disturbance type, each with per-event sub-folders.
        ids_gdf (gpd.GeoDataFrame): Current IDS output with ``IDX_D``
            and ``DCA_ID`` columns.
        s1dm_gdf (gpd.GeoDataFrame): Current S1DM output with ``IDX_D``
            and ``DCA_ID`` columns.
        disturbances (list[str], optional): Disturbance types to scan.
            Defaults to ``DISTURBANCE_ORDER``.
        min_overlap_pct (float): Minimum fraction of the manual polygon
            that must be covered for a match to be accepted.

    Returns:
        pd.DataFrame: One row per manual event with columns
            ``manual_idx``, ``disturbance``, ``ids_idx``, ``s1dm_idx``.
            ``ids_idx`` / ``s1dm_idx`` are ``None`` when no geometry
            match was found.
    """
    disturbances = disturbances or DISTURBANCE_ORDER
    rows = []

    # Ensure both layers share the same CRS for distance comparisons
    if ids_gdf.crs != s1dm_gdf.crs:
        s1dm_gdf = s1dm_gdf.to_crs(ids_gdf.crs)

    for disturbance in disturbances:
        base_folder = os.path.join(manual_base_folder, disturbance)
        if not os.path.isdir(base_folder):
            logging.warning(
                "Manual folder not found, skipping: %s", base_folder
            )
            continue

        ids_subset = ids_gdf[ids_gdf["DCA_ID"] == disturbance]
        s1dm_subset = s1dm_gdf[s1dm_gdf["DCA_ID"] == disturbance]

        subfolders = [
            f for f in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, f))
        ]
        for manual_idx in subfolders:
            manual_file = os.path.join(
                base_folder,
                manual_idx,
                f"merged_union_multipolygon_{manual_idx}.geojson",
            )
            if not os.path.exists(manual_file):
                logging.warning("GeoJSON not found: %s", manual_file)
                rows.append({
                    "manual_idx": manual_idx,
                    "disturbance": disturbance,
                    "ids_idx": None,
                    "s1dm_idx": None,
                })
                continue

            gdf_manual = gpd.read_file(manual_file)
            if gdf_manual.empty:
                rows.append({
                    "manual_idx": manual_idx,
                    "disturbance": disturbance,
                    "ids_idx": None,
                    "s1dm_idx": None,
                })
                continue

            if gdf_manual.crs != ids_gdf.crs:
                gdf_manual = gdf_manual.to_crs(ids_gdf.crs)

            manual_geom = make_valid(gdf_manual.geometry.union_all())

            ids_idx = find_best_geometry_match(
                manual_geom, ids_subset,
                id_col="IDX_D",
                min_overlap_pct=min_overlap_pct,
            )
            # S1DM match: prefer the polygon that shares the matched IDS
            # IDX_D; fall back to independent geometry search if absent
            if ids_idx is not None:
                s1dm_direct = s1dm_subset[
                    s1dm_subset["IDX_D"] == ids_idx
                ]
                if not s1dm_direct.empty:
                    s1dm_idx = ids_idx
                else:
                    s1dm_idx = find_best_geometry_match(
                        manual_geom, s1dm_subset,
                        id_col="IDX_D",
                        min_overlap_pct=min_overlap_pct,
                    )
            else:
                s1dm_idx = find_best_geometry_match(
                    manual_geom, s1dm_subset,
                    id_col="IDX_D",
                    min_overlap_pct=min_overlap_pct,
                )

            rows.append({
                "manual_idx": manual_idx,
                "disturbance": disturbance,
                "ids_idx": ids_idx,
                "s1dm_idx": s1dm_idx,
            })

    df = pd.DataFrame(
        rows,
        columns=["manual_idx", "disturbance", "ids_idx", "s1dm_idx"],
    )
    n_matched = df["ids_idx"].notna().sum()
    logging.info(
        "map_manual_labels_by_geometry: %d / %d events matched",
        n_matched, len(df),
    )
    return df


def compute_event_count_reduction(s1dm_gdf, ids_gdf):
    """Compute per-type event counts and radar-survey reduction percentage.

    Filters out drought and fire events, counts IDS and S1DM events per
    disturbance type, then computes the percentage reduction in
    detections relative to IDS.

    Args:
        s1dm_gdf (gpd.GeoDataFrame): S1DM disturbances.
        ids_gdf (gpd.GeoDataFrame): IDS disturbances.

    Returns:
        pd.DataFrame: Columns ``DCA_ID``, ``IDS``, ``S1DM``,
            ``Reduction (%)``, sorted by ``DCA_ID``.
    """
    exclude = ["drought", "fire"]
    s1dm_gdf = s1dm_gdf[~s1dm_gdf["DCA_ID"].isin(exclude)]
    ids_gdf = ids_gdf[~ids_gdf["DCA_ID"].isin(exclude)]

    counts_df = pd.DataFrame({
        "IDS": ids_gdf["DCA_ID"].value_counts(),
        "S1DM": s1dm_gdf["DCA_ID"].value_counts(),
    }).fillna(0)
    counts_df = counts_df.reset_index()
    counts_df = counts_df.rename(columns={"index": "DCA_ID"})
    counts_df["Reduction (%)"] = (
        -100
        * (counts_df["IDS"] - counts_df["S1DM"])
        / counts_df["IDS"]
    )
    counts_df["DCA_ID"] = pd.Categorical(
        counts_df["DCA_ID"],
        categories=[
            "bark_beetle", "wind", "fire",
            "defoliators", "drought",
        ],
        ordered=True,
    )
    return counts_df.sort_values("DCA_ID")


def compute_manual_validation_metrics(
    ids_file, s1dm_file, manual_base_folder, idx_mapping=None
):
    """Compute spatial overlap and Jaccard metrics vs. manual references.

    For each event in the manual reference folders, computes overlap (%)
    and Jaccard index for both IDS and S1DM.  Runs a one-sided Wilcoxon
    signed-rank test per disturbance type to assess whether S1DM scores
    are significantly higher than IDS scores.

    When manual label folder names (old IDX_D) no longer match the
    current shapefile IDX_D, pass a pre-built *idx_mapping* from
    ``map_manual_labels_by_geometry``.  If *idx_mapping* is ``None``,
    the function first tries a direct IDX_D string match and falls back
    to geometry-based matching automatically.

    Args:
        ids_file (str): Path to the IDS shapefile.
        s1dm_file (str): Path to the S1DM shapefile.
        manual_base_folder (str): Root folder containing one sub-folder
            per disturbance type, each with per-event GeoJSON files
            named ``merged_union_multipolygon_<IDX_D>.geojson``.
        idx_mapping (pd.DataFrame, optional): Pre-built mapping from
            ``map_manual_labels_by_geometry`` with columns
            ``manual_idx``, ``ids_idx``, ``s1dm_idx``.  When provided,
            skips the automatic fallback lookup.

    Returns:
        tuple:
            - df_results (pd.DataFrame): Overlap and Jaccard scores per
              event with columns ``disturbance``, ``idx``,
              ``overlap_ids``, ``jaccard_ids``, ``overlap_s1dm``,
              ``jaccard_s1dm``, ``disturbance_label``.
            - sig_jaccard (dict[str, str]): Significance symbol per
              disturbance label for the Jaccard metric.
            - sig_overlap (dict[str, str]): Significance symbol per
              disturbance label for the overlap metric.
    """
    gdf_s1dm = gpd.read_file(s1dm_file)
    gdf_ids = gpd.read_file(ids_file)

    # Build a lookup dict from the pre-built mapping if provided
    mapping_dict = {}
    if idx_mapping is not None:
        for _, row in idx_mapping.iterrows():
            mapping_dict[row["manual_idx"]] = (
                row["ids_idx"], row["s1dm_idx"]
            )

    results = []
    skipped = {"no_file": [], "empty_manual": [], "no_ids_match": [], "no_s1dm_match": []}

    for disturbance in DISTURBANCE_ORDER:
        base_folder = os.path.join(manual_base_folder, disturbance)
        subfolders = [
            f for f in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, f))
        ]

        # Reproject IDS and S1DM subsets to WGS84 once per disturbance type
        ids_subset = gdf_ids[gdf_ids["DCA_ID"] == disturbance].to_crs(CRS_WGS84)
        s1dm_subset = gdf_s1dm[gdf_s1dm["DCA_ID"] == disturbance].to_crs(CRS_WGS84)
        matched_count = 0

        for idx_name in subfolders:
            manual_file = os.path.join(
                base_folder,
                idx_name,
                f"merged_union_multipolygon_{idx_name}.geojson",
            )
            if not os.path.exists(manual_file):
                skipped["no_file"].append(idx_name)
                continue

            gdf_manual = gpd.read_file(manual_file)
            if gdf_manual.empty:
                skipped["empty_manual"].append(idx_name)
                continue

            if gdf_manual.crs is None:
                gdf_manual = gdf_manual.set_crs(CRS_WGS84)
            gdf_manual = gdf_manual.to_crs(CRS_WGS84)
            geom_manual = make_valid(gdf_manual.geometry.union_all())

            # Resolve IDS and S1DM IDX_D for this event.
            # Priority: pre-built mapping → direct IDX_D match → geometry fallback.
            if idx_name in mapping_dict:
                ids_idx, s1dm_idx = mapping_dict[idx_name]
            else:
                # 1. Direct IDX_D match (folder names were renamed to match shapefiles)
                ids_idx = (
                    idx_name
                    if not ids_subset[ids_subset["IDX_D"] == idx_name].empty
                    else None
                )
                s1dm_idx = (
                    idx_name
                    if not s1dm_subset[s1dm_subset["IDX_D"] == idx_name].empty
                    else None
                )
                # 2. Geometry fallback when IDX_D string doesn't match
                if ids_idx is None:
                    ids_idx = find_best_geometry_match(
                        geom_manual, ids_subset, id_col="IDX_D"
                    )
                    if ids_idx is not None:
                        logging.info(
                            "[%s] %s — IDX_D not found, geometry fallback → ids_idx=%s",
                            disturbance, idx_name, ids_idx,
                        )
                if s1dm_idx is None and ids_idx is not None:
                    # Follow IDX_D from matched IDS to S1DM
                    s1dm_idx = (
                        ids_idx
                        if not s1dm_subset[s1dm_subset["IDX_D"] == ids_idx].empty
                        else find_best_geometry_match(
                            geom_manual, s1dm_subset, id_col="IDX_D"
                        )
                    )
                if ids_idx is None:
                    logging.warning(
                        "[%s] %s — no IDS match found, skipping",
                        disturbance, idx_name,
                    )
                    skipped["no_ids_match"].append(idx_name)
                    continue
                if s1dm_idx is None:
                    logging.warning(
                        "[%s] %s — IDS matched (IDX_D=%s) but no S1DM found",
                        disturbance, idx_name, ids_idx,
                    )
                    skipped["no_s1dm_match"].append(idx_name)
                    continue

            geom_ids = make_valid(
                ids_subset[ids_subset["IDX_D"] == ids_idx].geometry.union_all()
            )
            geom_s1dm = make_valid(
                s1dm_subset[s1dm_subset["IDX_D"] == s1dm_idx].geometry.union_all()
            )

            if geom_ids.is_empty or geom_s1dm.is_empty or geom_manual.is_empty:
                skipped["no_s1dm_match"].append(idx_name)
                continue

            overlap_ids, jaccard_ids = compute_jaccard_overlap(
                geom_ids, geom_manual
            )
            overlap_s1dm, jaccard_s1dm = compute_jaccard_overlap(
                geom_s1dm, geom_manual
            )
            results.append({
                "disturbance": disturbance,
                "idx": idx_name,
                "overlap_ids": overlap_ids,
                "jaccard_ids": jaccard_ids,
                "overlap_s1dm": overlap_s1dm,
                "jaccard_s1dm": jaccard_s1dm,
            })
            matched_count += 1

        matched_ids = [r["idx"] for r in results if r["disturbance"] == disturbance]
        logging.info(
            "[%s] matched %d / %d manual events: %s",
            disturbance, matched_count, len(subfolders), matched_ids,
        )

    df_results = pd.DataFrame(results)
    df_results["disturbance_label"] = (
        df_results["disturbance"].map(DIST_LABEL_MAP)
    )

    sig_jaccard, sig_overlap = {}, {}
    for d, group in df_results.groupby("disturbance_label"):
        if len(group) > 1:
            logging.info("Significance test: %s - Jaccard", d)
            _, sig_j = paired_wilcoxon_significance(
                group["jaccard_s1dm"], group["jaccard_ids"]
            )
            logging.info("Significance test: %s - Overlap", d)
            _, sig_o = paired_wilcoxon_significance(
                group["overlap_s1dm"], group["overlap_ids"]
            )
        else:
            sig_j, sig_o = "", ""
        sig_jaccard[d] = sig_j
        sig_overlap[d] = sig_o

    return df_results, sig_jaccard, sig_overlap


# ── Study area components ──────────────────────────────────────────────────


def add_disturbance_polygons(ax, s1dm_dissolved, custom_colors):
    """Overlay disturbance polygons on an existing axes, by type.

    Plots each disturbance type twice: first with a white edge (halo
    effect), then with the type colour as both fill and edge.

    Args:
        ax (matplotlib.axes.Axes): Target axes to draw on.
        s1dm_dissolved (gpd.GeoDataFrame): Disturbance polygons with a
            ``DCA_ID`` column.
        custom_colors (dict[str, str]): Mapping ``DCA_ID`` -> hex colour.
    """
    for disturbance, color in custom_colors.items():
        subset = s1dm_dissolved[s1dm_dissolved["DCA_ID"] == disturbance]
        if subset.empty:
            continue
        subset.plot(ax=ax, linewidth=3.5, color=color, edgecolor="white")
        subset.plot(ax=ax, linewidth=2.5, color=color, edgecolor=color)


def add_region_overview_inset(ax, usa_mainland, region_id):
    """Draw a CONUS overview map with the study region highlighted.

    Non-study regions are shown in grey; the study region is filled
    black.  Intended for use as an inset axes on a larger figure.

    Args:
        ax (matplotlib.axes.Axes): Target axes for the inset map.
        usa_mainland (gpd.GeoDataFrame): Full USFS regions shapefile
            (already clipped to mainland).
        region_id (str): USFS region identifier to highlight.
    """
    usa_mainland[usa_mainland["REGION"] != region_id].plot(
        ax=ax, color="grey", edgecolor="grey"
    )
    usa_mainland[usa_mainland["REGION"] == region_id].plot(
        ax=ax, color="black", edgecolor="black"
    )
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True)
    ax.axis("on")


def assemble_study_area_figure(
    cropped_forest,
    s1_tiles_boundary_path,
    usa_mainland,
    r8,
    gdf,
    region_id,
    custom_colors,
    save_path,
    logging,
):
    """Assemble and save the full study-area figure.

    Combines a TCC raster, S1 tile grid, disturbance polygons, region
    boundary, and a CONUS overview inset into a single figure.

    Args:
        cropped_forest: TCC xarray dataset (output of
            ``load_tcc_dataset``), clipped to the study region.
        s1_tiles_boundary_path (str): Path to the S1 tile boundaries
            shapefile.
        usa_mainland (gpd.GeoDataFrame): CONUS regions shapefile.
        r8 (gpd.GeoDataFrame): Region 8 boundary.
        gdf (gpd.GeoDataFrame): Disturbance polygons with ``DCA_ID``.
        region_id (str): USFS region identifier.
        custom_colors (dict[str, str]): Disturbance-type colour map.
        save_path (str): Output file path for the saved figure.
        logging: Logger instance passed from the caller.
    """
    # ── Data preparation ──────────────────────────────────────────
    cropped_forest = normalize_tcc(cropped_forest)
    s1_tiles = load_data(s1_tiles_boundary_path)
    logging.info("s1_tiles contains %d features", len(s1_tiles))

    target_crs = cropped_forest.rio.crs
    if s1_tiles.crs != target_crs:
        s1_tiles = s1_tiles.to_crs(target_crs)
    if r8.crs != target_crs:
        r8 = r8.to_crs(target_crs)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # ── Plot settings ─────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    ordered_types = [
        t for t in DISTURBANCE_ORDER if t in gdf["DCA_ID"].unique()
    ]

    # ── Plotting ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    sub_ax = fig.add_axes([0.04, 0.70, 0.25, 0.25])
    add_region_overview_inset(sub_ax, usa_mainland, region_id)

    custom_cmap = create_tcc_colormap()
    render_tcc_raster(ax, cropped_forest, custom_cmap)
    r8.boundary.plot(ax=ax, linewidth=1, color="#000000")
    add_disturbance_polygons(ax, gdf, custom_colors)
    s1_tiles.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

    # ── Plot details ──────────────────────────────────────────────
    ax.text(
        0.5, 0.92, "S1CD Tiles",
        fontsize=16, fontweight="normal", color="black",
        ha="center", va="center",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    ax.axis("off")
    legend_patches = [
        mpatches.Patch(
            color=custom_colors[dist], label=format_label(dist)
        )
        for dist in ordered_types
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=18,
        title="Disturbance Type",
        title_fontsize=20,
        loc="center left",
        facecolor="white",
        framealpha=1,
    )

    # ── Save ──────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()


def create_tcc_colormap():
    """Build a custom green colormap for TCC raster display.

    The first colour stop is forced to white so that zero-canopy-cover
    pixels appear white rather than the lightest green.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap
            named ``'CustomGreens'``.
    """
    cmap = plt.colormaps["Greens"]
    new_colors = cmap(np.linspace(0, 1, 100))
    new_colors[0, :] = [1, 1, 1, 1]
    return LinearSegmentedColormap.from_list("CustomGreens", new_colors)


def normalize_tcc(cropped_forest):
    """Scale TCC values to the range [0, 100].

    Divides by the maximum value then clips to ensure no values fall
    outside [0, 100].  Modifies the dataset in-place.

    Args:
        cropped_forest: xarray Dataset with a ``tcc`` data variable.

    Returns:
        The same dataset with normalised ``tcc`` values.
    """
    cropped_forest["tcc"] = (
        cropped_forest["tcc"] / cropped_forest["tcc"].max() * 100
    )
    cropped_forest["tcc"] = cropped_forest["tcc"].clip(min=0, max=100)
    return cropped_forest


def render_tcc_raster(ax, cropped_forest, custom_cmap):
    """Render a TCC raster onto *ax* and attach a horizontal colorbar.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        cropped_forest: xarray Dataset with a ``tcc`` data variable.
        custom_cmap (matplotlib.colors.Colormap): Colormap to apply.
    """
    plot = cropped_forest["tcc"].plot(
        ax=ax, cmap=custom_cmap,
        add_colorbar=False, add_labels=False,
    )
    cbar = plt.colorbar(
        plot, ax=ax,
        orientation="horizontal", pad=0.05,
        aspect=10, shrink=0.8,
    )
    cbar.mappable.set_cmap(custom_cmap)
    cbar.ax.set_position([0.37, 0.35, 0.35, 0.03])
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Tree Canopy Cover (%)", fontsize=16, labelpad=6)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.label.set_size(16)
    cbar.ax.xaxis.labelpad = 10


# ── Plots ─────────────────────────────────────────────────────────────────


def plot_annual_event_counts(
    s1dm_gdf,
    ids_path,
    exclude_types=None,
    ordered_types=None,
    custom_colors=None,
    output_file=None,
    year_col="SURVEY_Y",
    type_col="DCA_ID",
    figsize=(18, 6),
    log_scale=True,
):
    """Plot yearly disturbance event counts for IDS and S1DM.

    Loads data from disk, aggregates event counts per year and
    disturbance type, then plots side-by-side bar charts — one
    sub-panel per disturbance type.

    Args:
        s1dm_gdf (str): Path to the S1DM shapefile.
        ids_path (str): Path to the IDS shapefile.
        exclude_types (list[str], optional): Disturbance types to
            exclude (e.g. ``['fire', 'drought']``).
        ordered_types (list[str], optional): Desired plot order for
            disturbance types.
        custom_colors (dict[str, str], optional): Mapping type -> colour.
        output_file (str, optional): Path to save the figure.
        year_col (str): Column containing the survey year.
        type_col (str): Column containing the disturbance type.
        figsize (tuple[int, int]): Figure size passed to
            ``plt.subplots``.
        log_scale (bool): Use log scale on the y-axis.

    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]: Figure and
            flattened axes array.
    """
    # ── Data preparation ──────────────────────────────────────────
    exclude_types = exclude_types or []
    custom_colors = custom_colors or {}

    s1dm_gdf = gpd.read_file(s1dm_gdf)
    ids_gdf = gpd.read_file(ids_path)
    s1dm_gdf = s1dm_gdf.dissolve(by="IDX_D", as_index=False)
    s1dm_gdf = s1dm_gdf[~s1dm_gdf[type_col].isin(exclude_types)]
    ids_gdf = ids_gdf[~ids_gdf[type_col].isin(exclude_types)]
    s1dm_gdf = s1dm_gdf.rename(columns={year_col: "Year"})
    ids_gdf = ids_gdf.rename(columns={year_col: "Year"})

    s1dm_counts = (
        s1dm_gdf.groupby(["Year", type_col])
        .size().unstack(fill_value=0)
    )
    ids_counts = (
        ids_gdf.groupby(["Year", type_col])
        .size().unstack(fill_value=0)
    )
    all_types = sorted(
        set(s1dm_counts.columns).union(ids_counts.columns)
    )

    # ── Plot settings ─────────────────────────────────────────────
    default_colors = plt.cm.tab10.colors
    color_map = {
        dt: custom_colors.get(
            dt, default_colors[i % len(default_colors)]
        )
        for i, dt in enumerate(all_types)
    }
    all_types_ordered = (
        [t for t in ordered_types if t in all_types]
        if ordered_types else all_types
    )
    bar_width = 0.35
    n_cols = 3
    n_rows = int(np.ceil(len(all_types_ordered) / n_cols))

    # ── Plotting ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 7 * n_rows),
        sharex=True, sharey=True,
    )
    axes = axes.flatten()

    for i, dist_type in enumerate(all_types_ordered):
        ax = axes[i]
        years = sorted(
            set(s1dm_counts.index).union(ids_counts.index)
        )
        x = np.arange(len(years))

        base_rgb = to_rgb(color_map[dist_type])
        lighter_rgb = tuple(min(c + 0.4, 1.0) for c in base_rgb)
        bars_ids = ax.bar(
            x - bar_width / 2,
            ids_counts.get(
                dist_type, pd.Series(0, index=years)
            ).reindex(years, fill_value=0),
            width=bar_width, color=lighter_rgb, label="IDS",
        )
        bars_s1dm = ax.bar(
            x + bar_width / 2,
            s1dm_counts.get(
                dist_type, pd.Series(0, index=years)
            ).reindex(years, fill_value=0),
            width=bar_width,
            color=color_map[dist_type], label="S1DM",
        )

        # ── Plot details ──────────────────────────────────────────
        ax.set_title(
            dist_type.replace("_", " ").title(),
            pad=15, fontsize=22,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, fontsize=22)
        ax.tick_params(axis="y", labelsize=24)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(
                1,
                max(
                    s1dm_counts.max().max(),
                    ids_counts.max().max(),
                ) * 1.2,
            )
        ax.legend(
            loc="upper right", frameon=True,
            facecolor="white", edgecolor="lightgrey", fontsize=18,
        )
        for bars in [bars_ids, bars_s1dm]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height * 1.1,
                        f"{int(height)}",
                        ha="center", va="bottom", fontsize=14,
                    )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    axes[0].set_ylabel(
        r"${N_{\text{Disturbance Events}}}$",
        labelpad=20, fontsize=26,
    )
    axes[1].set_xlabel("Year", labelpad=15, fontsize=26)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # ── Save / return ─────────────────────────────────────────────
    if output_file:
        plt.savefig(output_file, dpi=SAVE_DPI_MED, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_detection_year_lag(gdf, custom_colors, save_path):
    """Plot detection year-lag counts as a line chart with directional annotations.

    The year lag is defined as ``SURVEY_Y - S1_YEAR``.  One line is
    drawn per disturbance type, ordered as ``DISTURBANCE_ORDER``.
    Arrows annotate which side of zero is before/after the IDS survey.

    Args:
        gdf (gpd.GeoDataFrame): Aggregated S1DM detections with columns
            ``SURVEY_Y``, ``S1_YEAR``, and ``DCA_ID``.
        custom_colors (dict[str, str]): Mapping ``DCA_ID`` -> hex colour.
        save_path (str): Output file path.
    """
    # ── Data preparation ──────────────────────────────────────────
    gdf = gdf.copy()
    gdf["SURVEY_Y"] = gdf["SURVEY_Y"].astype(int)
    gdf["S1_YEAR"] = gdf["S1_YEAR"].astype(int)
    gdf["diff_year"] = gdf["SURVEY_Y"] - gdf["S1_YEAR"]
    count_by_lag = (
        gdf.groupby(["diff_year", "DCA_ID"])
        .size()
        .reset_index(name="Count")
    )

    # ── Plotting ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for dca_id in DISTURBANCE_ORDER:
        if dca_id not in count_by_lag["DCA_ID"].unique():
            continue
        dca_data = count_by_lag[count_by_lag["DCA_ID"] == dca_id]
        sns.lineplot(
            data=dca_data,
            x="diff_year", y="Count",
            label=format_label(dca_id),
            color=custom_colors.get(dca_id, "#000000"),
            marker="o", markersize=10, linewidth=3,
            ax=ax,
        )

    ax.invert_xaxis()

    # ── Directional annotations ───────────────────────────────────
    max_count = count_by_lag["Count"].max()
    arrowprops = dict(
        arrowstyle="-|>",
        facecolor="gray",
        edgecolor="gray",
        linewidth=2,
        alpha=0.8,
    )
    ax.annotate("After IDS", xy=(-1.5, max_count * 0.9), xytext=(-1, max_count * 0.9),
                arrowprops=arrowprops, fontsize=16, color="dimgray", ha="right", va="center")
    ax.annotate("Before IDS", xy=(1.5, max_count * 0.9), xytext=(1, max_count * 0.9),
                arrowprops=arrowprops, fontsize=16, color="dimgray", ha="left", va="center")

    # ── Plot details ──────────────────────────────────────────────
    ax.set_xlabel(
        r"${Lag_{\text{(IDS-S1DM)}}}$", fontsize=22, labelpad=10
    )
    ax.set_ylabel(
        r"${N_{\text{Annual Signals}}}$", fontsize=22, labelpad=10
    )
    ax.set_xticks(range(-2, 3))
    ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=18, loc="center right", title_fontsize=1)
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()


def plot_manual_disturbance_examples(
    disturbance_files,
    folder_manual,
    ids_gdf,
    s1dm_gdf,
    figure_path=None,
    disturbances=None,
    layers=None,
    layer_colors=None,
    disturbance_colors=None,
    figsize=(22, 20),
    fontsize_title=22,
    fontsize_axis=16,
    fontsize_legend=25,
):
    """Plot PlanetScope examples overlaid with IDS, S1DM, and manual polygons.

    Creates a grid of panels (one row per disturbance, one column per
    layer) showing PlanetScope RGB composites with polygon overlays.
    An extra column on the right labels each disturbance row.

    Args:
        disturbance_files (dict): Keys are disturbance names; values are
            dicts with ``'file'`` (raster path), ``'idx'`` (int), and
            ``'date'`` (str).
        folder_manual (str): Folder containing
            ``planet_labels/merged_<disturbance>_manual.shp`` files.
        ids_gdf (gpd.GeoDataFrame): IDS polygon layer.
        s1dm_gdf (gpd.GeoDataFrame): S1DM polygon layer.
        figure_path (str, optional): Path to save the output figure.
        disturbances (list[str], optional): Ordered disturbance names
            to plot.  Defaults to ``DISTURBANCE_ORDER``.
        layers (list[str], optional): Layer names to overlay.
            Defaults to ``["Manual", "IDS", "S1DM"]``.
        layer_colors (dict[str, str], optional): Colour per layer.
        disturbance_colors (dict[str, str], optional): Colour per
            disturbance for the info column.
        figsize (tuple[int, int]): Figure size.
        fontsize_title (int): Title font size.
        fontsize_axis (int): Axis label font size.
        fontsize_legend (int): Legend font size.
    """
    # ── Data preparation ──────────────────────────────────────────
    disturbances = disturbances or DISTURBANCE_ORDER
    layers = layers or ["Manual", "IDS", "S1DM"]
    layer_colors = layer_colors or {
        "Manual": "#F03A47", "IDS": "#70c2fc", "S1DM": "white",
    }
    disturbance_colors = disturbance_colors or {
        "wind": "#1f77b4",
        "bark_beetle": "#714709",
        "defoliators": "#FF9505",
    }

    # ── Plot settings ─────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=layer_colors[k], label=k) for k in layers
    ]

    # ── Plotting ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(disturbances), len(layers) + 1,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1] * len(layers) + [0.3]},
    )

    def add_subplot_label(ax, label):
        ax.text(
            0.05, 0.95, label,
            transform=ax.transAxes, fontsize=30,
            fontweight="normal", va="top", ha="left", color="black",
            bbox=dict(
                facecolor="white", edgecolor="black",
                boxstyle="circle,pad=0.3",
            ),
        )

    for i, disturbance in enumerate(disturbances):
      try:
        ds = rioxarray.open_rasterio(
            disturbance_files[disturbance]["file"]
        )
        ds_4326 = ds.rio.reproject(CRS_WGS84)
        rgb = (
            ds_4326.transpose("y", "x", "band")
            .values.astype(np.float32)
        )

        mask = np.any(rgb > 0, axis=2)
        rows_nz = np.where(mask.any(axis=1))[0]
        cols_nz = np.where(mask.any(axis=0))[0]
        r_min, r_max = rows_nz[[0, -1]]
        c_min, c_max = cols_nz[[0, -1]]
        pad = min(5, (r_max - r_min) // 4, (c_max - c_min) // 4)
        r_min += pad; r_max -= pad; c_min += pad; c_max -= pad
        if disturbance == "wind":
            extra = min(110, (c_max - c_min) // 2)
            c_min += extra
        rgb_crop = rgb[r_min:r_max + 1, c_min:c_max + 1, :]
        rgb_crop = np.clip(
            rgb_crop / np.percentile(rgb_crop, 99), 0, 1
        )
        if rgb_crop.shape[2] == 3:
            alpha = np.any(rgb_crop > 0, axis=2).astype(float)
            rgb_crop = np.dstack((rgb_crop, alpha))
        elif rgb_crop.shape[2] > 4:
            rgb_crop = rgb_crop[:, :, :4]
        x = ds_4326.x.values[c_min:c_max + 1]
        y = ds_4326.y.values[r_min:r_max + 1]
        extent = (x.min(), x.max(), y.min(), y.max())
        if extent[0] == extent[1] or extent[2] == extent[3] or rgb_crop.size == 0:
            logging.error(
                "[%s] invalid scene extent %s (crop shape %s), skipping",
                disturbance, extent, rgb_crop.shape,
            )
            continue

        idx_d = disturbance_files[disturbance]["idx"]
        manual_file = os.path.join(
            folder_manual,
            disturbance,
            idx_d,
            f"merged_union_multipolygon_{idx_d}.geojson",
        )
        gdf_manual = gpd.read_file(manual_file)
        if gdf_manual.crs is None:
            gdf_manual = gdf_manual.set_crs(CRS_WGS84)
        manual_4326 = gdf_manual.to_crs(CRS_WGS84)

        # Find IDS that intersect the manual polygon; prefer the exact IDX_D
        # match when it is present to avoid picking up neighbouring events.
        manual_union = unary_union(manual_4326.geometry)
        ids_4326 = ids_gdf.to_crs(CRS_WGS84)
        s1dm_4326 = s1dm_gdf.to_crs(CRS_WGS84)
        ids_candidates = ids_4326[ids_4326.intersects(manual_union)]
        if idx_d in ids_candidates["IDX_D"].values:
            ids_4326 = ids_candidates[ids_candidates["IDX_D"] == idx_d]
        else:
            ids_4326 = ids_candidates
        matched_idx = ids_4326["IDX_D"].unique()
        s1dm_4326 = s1dm_4326[s1dm_4326["IDX_D"].isin(matched_idx)]
        logging.info(
            "Processing %s IDX_D=%s — %d IDS matched (IDX_D: %s), %d S1DM",
            disturbance, idx_d, len(ids_4326), list(matched_idx), len(s1dm_4326),
        )

        from shapely.geometry import box as shapely_box
        scene_geom = shapely_box(extent[0], extent[2], extent[1], extent[3])
        merged_s1dm = gpd.GeoSeries(
            [unary_union(s1dm_4326.geometry).intersection(scene_geom)],
            crs=s1dm_4326.crs,
        )

        for j, layer in enumerate(layers):
            ax = (
                axes[i, j]
                if len(disturbances) > 1 else axes[j]
            )
            ax.imshow(rgb_crop, extent=extent)
            if layer == "Manual" and not manual_4326.empty:
                manual_4326.boundary.plot(
                    ax=ax,
                    edgecolor=layer_colors["Manual"],
                    linewidth=3,
                )
            elif layer == "IDS" and not ids_4326.empty:
                ids_4326.boundary.plot(
                    ax=ax,
                    edgecolor=layer_colors["IDS"],
                    linewidth=3,
                )
            elif layer == "S1DM" and not s1dm_4326.empty:
                valid_s1dm = merged_s1dm[~merged_s1dm.is_empty]
                if not valid_s1dm.empty:
                    valid_s1dm.boundary.plot(
                        ax=ax,
                        edgecolor=layer_colors["S1DM"],
                        linewidth=2,
                    )
            # Lock view to scene extent — geopandas .plot() expands limits.
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.legend(
                handles=[legend_patches[layers.index(layer)]],
                fontsize=fontsize_legend,
            )
            add_subplot_label(ax, chr(97 + i * len(layers) + j))
            ax.add_patch(Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                linewidth=3, edgecolor="black", facecolor="none",
            ))
            ax.axis("off")

        # ── Plot details: info column ──────────────────────────────
        ax_info = axes[i, -1]
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        name = disturbance.replace("_", " ").title()
        date = disturbance_files[disturbance]["date"]
        color = disturbance_colors.get(disturbance, "black")
        heights = {0: 0.92, 1: 0.92, 2: 0.96}
        y_offsets = {0: 0.04, 1: 0.04, 2: 0.02}
        ax_info.add_patch(Rectangle(
            (0, y_offsets[i]), 1, heights[i],
            facecolor="white", edgecolor=color, linewidth=3,
            transform=ax_info.transAxes, clip_on=False,
        ))
        ax_info.text(
            0.2, 0.5, name,
            ha="center", va="center", fontsize=25,
            fontweight="bold", color=color,
            rotation=90, transform=ax_info.transAxes,
        )
        ax_info.text(
            0.6, 0.5,
            f"PlanetScope Image Composite\non {date}",
            ha="center", va="center", fontsize=20,
            color="black", rotation=90,
            transform=ax_info.transAxes,
        )
      except Exception as exc:
        logging.error("[%s] skipped due to error: %s", disturbance, exc, exc_info=True)

    # ── Save ──────────────────────────────────────────────────────
    plt.tight_layout()
    plt.subplots_adjust(top=0.99, hspace=0.05, wspace=0.05)
    if figure_path:
        plt.savefig(figure_path, dpi=SAVE_DPI_MED)
    plt.show()


def plot_manual_validation_boxplots(
    df_results, sig_jaccard, sig_overlap, save_path=None
):
    """Plot Jaccard IoU and overlap boxplots vs. manual references.

    Renders two side-by-side boxplots comparing IDS and S1DM accuracy
    against manually digitised reference polygons, with Wilcoxon
    significance symbols annotated above each group.

    Args:
        df_results (pd.DataFrame): Output of
            ``compute_manual_validation_metrics`` with columns
            ``disturbance_label``, ``jaccard_ids``, ``jaccard_s1dm``,
            ``overlap_ids``, ``overlap_s1dm``.
        sig_jaccard (dict[str, str]): Significance symbol per
            disturbance label for the Jaccard metric.
        sig_overlap (dict[str, str]): Significance symbol per
            disturbance label for the overlap metric.
        save_path (str, optional): Output file path.
    """
    # ── Data preparation ──────────────────────────────────────────
    dist_order = list(DIST_LABEL_MAP.values())

    df_jaccard = df_results.melt(
        id_vars=["disturbance_label", "idx"],
        value_vars=["jaccard_ids", "jaccard_s1dm"],
        var_name="method", value_name="jaccard",
    )
    df_jaccard["method"] = df_jaccard["method"].map(
        {"jaccard_ids": "IDS", "jaccard_s1dm": "S1DM"}
    )
    df_overlap = df_results.melt(
        id_vars=["disturbance_label", "idx"],
        value_vars=["overlap_ids", "overlap_s1dm"],
        var_name="method", value_name="overlap",
    )
    df_overlap["method"] = df_overlap["method"].map(
        {"overlap_ids": "IDS", "overlap_s1dm": "S1DM"}
    )

    # ── Plot settings ─────────────────────────────────────────────
    sns.set_theme(
        style="whitegrid", font="DejaVu Sans", font_scale=1.3
    )
    box_kw = dict(
        linewidth=2.0,
        fliersize=8,
        whiskerprops=dict(color="black", linewidth=3),
        capprops=dict(color="black", linewidth=3),
        medianprops=dict(color="black", linewidth=3),
        flierprops=dict(marker="o", markersize=6, alpha=0.7),
        boxprops=dict(edgecolor="black", linewidth=2.5),
    )

    # ── Plotting ──────────────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [2, 2], "wspace": 0.3},
    )
    sns.boxplot(
        data=df_jaccard, x="disturbance_label", y="jaccard",
        hue="method", order=dist_order,
        palette=PALETTE_VALIDATION, ax=ax0, **box_kw,
    )
    sns.boxplot(
        data=df_overlap, x="disturbance_label", y="overlap",
        hue="method", order=dist_order,
        palette=PALETTE_VALIDATION, ax=ax1, **box_kw,
    )

    # ── Plot details ──────────────────────────────────────────────
    ax0.set_ylabel(
        "Intersection over Union [0-1]", fontsize=20, labelpad=15
    )
    ax0.set_xlabel("Disturbance Type", fontsize=20, labelpad=15)
    ax0.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax0.tick_params(axis="both", labelsize=16)
    ax0.set_ylim(-0.01, 0.6)
    ax0.text(
        0.5, 1.02, "a)", transform=ax0.transAxes,
        ha="center", va="bottom", fontsize=22,
    )
    for i, dist in enumerate(dist_order):
        if sig_jaccard.get(dist):
            y_max = (
                df_jaccard[df_jaccard["disturbance_label"] == dist]
                ["jaccard"].max()
            )
            ax0.text(
                i, min(y_max + 0.03, 0.98), sig_jaccard[dist],
                ha="center", va="bottom", fontsize=28, color="black",
            )

    ax1.set_ylabel("Overlap [%]", fontsize=20, labelpad=15)
    ax1.set_xlabel("Disturbance Type", fontsize=20, labelpad=15)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax1.tick_params(axis="both", labelsize=16)
    ax1.set_ylim(-1, 100)
    ax1.text(
        0.5, 1.02, "b)", transform=ax1.transAxes,
        ha="center", va="bottom", fontsize=22,
    )
    for i, dist in enumerate(dist_order):
        if sig_overlap.get(dist):
            y_max = (
                df_overlap[df_overlap["disturbance_label"] == dist]
                ["overlap"].max()
            )
            ax1.text(
                i, min(y_max + 4, 98), sig_overlap[dist],
                ha="center", va="bottom", fontsize=36, color="black",
            )

    handles = [
        mpatches.Patch(
            color=PALETTE_VALIDATION["IDS"], label="IDS"
        ),
        mpatches.Patch(
            color=PALETTE_VALIDATION["S1DM"], label="S1DM"
        ),
        mlines.Line2D(
            [], [], color="black", marker=r"$*$",
            linestyle="None", markersize=16, label="P < 0.05",
        ),
        mlines.Line2D(
            [], [], color="black", marker=r"$+$",
            linestyle="None", markersize=16, label="P < 0.1",
        ),
    ]
    ax1.legend(
        handles=handles, loc="upper right",
        fontsize=14, frameon=True, facecolor="white",
    )
    ax0.get_legend().remove()
    sns.despine()
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()


def plot_radar_reduction_potential(
    s1dm_gdf, ids_gdf, save_path, plot_reduction=True
):
    """Plot IDS vs S1DM event counts and optional reduction percentage.

    The upper panel shows a grouped bar chart of event counts per
    disturbance type on a log scale.  When ``plot_reduction=True`` a
    second panel below shows the percentage reduction in detections
    relative to IDS.

    Args:
        s1dm_gdf (gpd.GeoDataFrame): S1DM disturbances (drought and
            fire are filtered out internally).
        ids_gdf (gpd.GeoDataFrame): IDS disturbances.
        save_path (str): Output file path.
        plot_reduction (bool): Whether to add the reduction-% panel.
    """
    # ── Data preparation ──────────────────────────────────────────
    counts = compute_event_count_reduction(s1dm_gdf, ids_gdf)
    dca_labels = [format_label(lbl) for lbl in counts["DCA_ID"]]

    # ── Plot settings ─────────────────────────────────────────────
    label_fs = 30
    legend_fs = 30
    legend_title_fs = 30
    tick_fs = 26
    annot_fs = 26
    bar_w = 0.35
    bar_offset = 0.2

    fig_h = 12 if plot_reduction else 7
    fig = plt.figure(figsize=(24, fig_h))
    gs = GridSpec(
        nrows=2 if plot_reduction else 1,
        ncols=1,
        height_ratios=[5, 2] if plot_reduction else [5],
    )

    # ── Plotting ──────────────────────────────────────────────────
    bar_pos = range(len(counts))
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(
        bar_pos, counts["IDS"],
        width=bar_w, color=COLOR_IDS, label="IDS",
    )
    ax1.bar(
        [p + bar_w for p in bar_pos], counts["S1DM"],
        width=bar_w, color=COLOR_S1DM_BAR, label="S1DM",
    )

    # ── Plot details ──────────────────────────────────────────────
    for i, (n_ids, n_s1dm) in enumerate(
        zip(counts["IDS"], counts["S1DM"])
    ):
        ax1.text(
            bar_pos[i], n_ids + 2, str(int(n_ids)),
            ha="center", va="bottom",
            color="black", fontsize=annot_fs,
        )
        ax1.text(
            bar_pos[i] + bar_w, n_s1dm + 2, str(int(n_s1dm)),
            ha="center", va="bottom",
            color="black", fontsize=annot_fs,
        )

    ax1.set_ylabel(
        r"${N_{\text{Disturbance Events}}}$",
        fontsize=label_fs, labelpad=20,
    )
    ax1.set_yscale("log")
    ax1.set_ylim(1, counts[["IDS", "S1DM"]].max().max() * 2)
    legend = ax1.legend(fontsize=legend_fs, title="Datasets")
    legend.get_title().set_fontsize(legend_title_fs)
    ax1.grid(False)
    plt.yticks(fontsize=tick_fs)

    group_w = bar_w * 2
    tick_pos = [p + group_w / 2 - bar_w / 2.5 for p in bar_pos]
    plt.xticks(tick_pos, dca_labels, fontsize=tick_fs, ha="center")
    ax1.tick_params(axis="x", which="major", pad=15)

    if plot_reduction:
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(
            [p + bar_offset for p in bar_pos],
            counts["Reduction (%)"],
            width=bar_w * 2, color=COLOR_REDUCTION,
            label="Reduction (%)",
        )
        for i, pct in enumerate(counts["Reduction (%)"]):
            ax2.text(
                bar_pos[i] + bar_offset, pct - 2,
                f"{pct:.2f}%",
                ha="center", va="top",
                color="black", fontsize=annot_fs,
            )
        ax2.set_xlabel("Disturbance Type", fontsize=label_fs)
        ax2.set_ylim(0, -110)
        ax2.invert_yaxis()
        ax2.set_ylabel(
            "Reduction\nPercentage (%)",
            fontsize=label_fs, labelpad=20,
        )
        ax2.set_yticks([0, -20, -40, -60, -80, -100])
        ax2.set_yticklabels(
            ["0", "-20", "-40", "-60", "-80", "-100"]
        )
        plt.xticks(
            [p + bar_offset for p in bar_pos],
            dca_labels, fontsize=tick_fs, ha="right",
        )
        ax2.set_xticklabels("", ha="right", fontsize=1)
        ax2.grid(False)
        tick_pos = [p + group_w / 2 - bar_w / 2.5 for p in bar_pos]
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(
            dca_labels, fontsize=tick_fs, ha="center"
        )
        plt.yticks(fontsize=tick_fs)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()


def plot_size_position_comparison(
    gdf, ids, s1dm_convex, ids_convex, custom_colors, save_path, stats_path=None
):
    """Compare IDS and S1DM polygon size and centroid position.

    Produces a three-column figure for each disturbance category:

    - Column a: KDE of disturbance polygon area (km2)
    - Column b: KDE of convex-hull area (km2)
    - Column c: Histogram + KDE of centroid shift (m)

    Args:
        gdf (gpd.GeoDataFrame): S1DM events with ``area_km2`` and
            ``centroid_shift_m`` columns.
        ids (gpd.GeoDataFrame): IDS events with ``area_km2``.
        s1dm_convex (gpd.GeoDataFrame): S1DM convex-hull areas.
        ids_convex (gpd.GeoDataFrame): IDS convex-hull areas.
        custom_colors (dict[str, str]): Mapping ``DCA_ID`` -> colour.
        save_path (str): Output file path.
    """
    # ── Data preparation ──────────────────────────────────────────
    stats_rows = []
    gdf = gdf.sort_values("DCA_ID").reset_index(drop=True)
    ids = ids.sort_values("DCA_ID").reset_index(drop=True)
    ids_convex = (
        ids_convex.sort_values("DCA_ID").reset_index(drop=True)
    )
    s1dm_convex = (
        s1dm_convex.sort_values("DCA_ID").reset_index(drop=True)
    )
    category_order = list(custom_colors.keys())

    def clean(x):
        return x.replace([np.inf, -np.inf], np.nan).dropna().values

    # ── Plot settings ─────────────────────────────────────────────
    fontsize_legend = 35
    fontsize_label_big = 55
    fontsize_label_med = 42
    fontsize_tick = 40
    padding_label = 18

    default_palette = sns.color_palette("tab10", n_colors=10)
    custom_palette = {
        lbl: custom_colors.get(
            lbl, default_palette[i % len(default_palette)]
        )
        for i, lbl in enumerate(category_order)
    }
    sns.set_theme(style="whitegrid")

    # ── Plotting ──────────────────────────────────────────────────
    fig, axs = plt.subplots(
        len(category_order), 3,
        figsize=(40, 8 * len(category_order)),
        gridspec_kw={"width_ratios": [3, 3, 2], "wspace": 0.3},
    )

    for i, category in enumerate(category_order):
        fmt = ".4f" if category == "bark_beetle" else ".2f"
        is_bark = category == "bark_beetle"

        # ── a) Disturbance area KDE ───────────────────────────────
        ax = axs[i, 0]
        vals_ids = clean(
            ids.loc[ids["DCA_ID"] == category, "area_km2"]
        )
        vals_s1dm = clean(
            gdf.loc[gdf["DCA_ID"] == category, "area_km2"]
        )
        stats_rows += filter(None, [
            log_array_stats("area_km2", vals_ids, category, "IDS"),
            log_array_stats("area_km2", vals_s1dm, category, "S1DM"),
        ])

        if len(vals_ids) < 2 or len(vals_s1dm) < 2:
            continue

        xmin = min(vals_ids.min(), vals_s1dm.min())
        xmax = max(vals_ids.max(), vals_s1dm.max())
        h1 = h2 = h3 = h4 = None

        if len(np.unique(vals_ids)) > 1:
            kde_ids = gaussian_kde(vals_ids)
            x = np.linspace(xmin, xmax, 1000)
            h1, = ax.plot(
                x, kde_ids(x), color="black", linewidth=4,
                label="IDS",
            )
            med_ids = np.median(vals_ids)
            h2, = ax.plot(
                [med_ids, med_ids],
                [0, float(kde_ids([med_ids]))],
                color="black", linestyle="--", linewidth=3,
                label=f"IDS Median: {med_ids:{fmt}}",
            )

        # S1DM
        if len(np.unique(vals_s1dm)) > 1:
            kde_r = gaussian_kde(vals_s1dm)
            x = np.linspace(xmin, xmax, 1000)
            h3, = ax.plot(
                x, kde_r(x),
                color=custom_palette[category], linewidth=4,
                label="S1DM",
            )
            med_r = np.median(vals_s1dm)
            h4, = ax.plot(
                [med_r, med_r],
                [0, float(kde_r([med_r]))],
                color=custom_palette[category],
                linestyle="--", linewidth=3,
                label=f"S1DM Median: {med_r:{fmt}}",
            )

        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=5, integer=True, prune=None)
        )
        set_two_positive_yticks(ax)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(format_ticks)
        )
        ax.legend(handles=[h1, h3, h2, h4], fontsize=fontsize_legend)
        ax.set_ylabel(
            "PDF" if category == "bark_beetle" else "",
            fontsize=fontsize_label_big, labelpad=padding_label,
        )
        ax.tick_params(axis="both", labelsize=fontsize_tick)
        ax.set_xlabel(
            r"$A_{\mathrm{D}} \; (\mathrm{km^2})$"
            if i == len(category_order) - 1 else "",
            fontsize=fontsize_label_med, labelpad=padding_label,
        )
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 0.5 if is_bark else 10)

        # ── b) Convex-hull area KDE ───────────────────────────────
        ax = axs[i, 1]
        h1 = h2 = h3 = h4 = None
        vals_ids = clean(
            ids_convex.loc[
                ids_convex["DCA_ID"] == category, "area_km2"
            ]
        )
        vals_s1dm = clean(
            s1dm_convex.loc[
                s1dm_convex["DCA_ID"] == category, "area_km2"
            ]
        )
        stats_rows += filter(None, [
            log_array_stats("convex_area_km2", vals_ids, category, "IDS"),
            log_array_stats("convex_area_km2", vals_s1dm, category, "S1DM"),
        ])

        if len(vals_ids) < 2 or len(vals_s1dm) < 2:
            continue

        xmin = min(vals_ids.min(), vals_s1dm.min())
        xmax = max(vals_ids.max(), vals_s1dm.max())

        if len(np.unique(vals_ids)) > 1:
            kde_ids = gaussian_kde(vals_ids)
            x = np.linspace(xmin, xmax, 1000)
            h1, = ax.plot(
                x, kde_ids(x), color="black", linewidth=4,
                label="IDS",
            )
            med_ids = np.median(vals_ids)
            h2, = ax.plot(
                [med_ids, med_ids],
                [0, float(kde_ids([med_ids]))],
                color="black", linestyle="--", linewidth=3,
                label=f"IDS Median: {med_ids:{fmt}}",
            )

        # S1DM
        if len(np.unique(vals_s1dm)) > 1:
            kde_r = gaussian_kde(vals_s1dm)
            x = np.linspace(xmin, xmax, 1000)
            h3, = ax.plot(
                x, kde_r(x),
                color=custom_palette[category], linewidth=4,
                label="S1DM",
            )
            med_r = np.median(vals_s1dm)
            h4, = ax.plot(
                [med_r, med_r],
                [0, float(kde_r([med_r]))],
                color=custom_palette[category],
                linestyle="--", linewidth=3,
                label=f"S1DM Median: {med_r:{fmt}}",
            )

        ax.set_ylabel(
            "PDF" if category == "bark_beetle" else "",
            fontsize=fontsize_label_big, labelpad=padding_label,
        )
        ax.legend(handles=[h1, h3, h2, h4], fontsize=fontsize_legend)
        ax.tick_params(axis="both", labelsize=fontsize_tick)
        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=5, integer=True, prune=None)
        )
        set_two_positive_yticks(ax)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(format_ticks)
        )
        ax.set_xlabel(
            r"$A_{\mathrm{CH}} \; (\mathrm{km^2})$"
            if i == len(category_order) - 1 else "",
            fontsize=fontsize_label_med, labelpad=padding_label,
        )
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 5 if is_bark else 15)

        # ── c) Centroid shift histogram + KDE ─────────────────────
        ax = axs[i, 2]
        vals = clean(
            gdf.loc[gdf["DCA_ID"] == category, "centroid_shift_m"]
        )
        row = log_array_stats("centroid_shift_m", vals, category, "SHIFT")
        if row:
            stats_rows.append(row)

        if len(vals) < 2:
            continue

        hist = ax.hist(
            vals, bins=30,
            color=custom_palette[category], alpha=0.6,
        )
        bin_w = hist[1][1] - hist[1][0]

        if len(np.unique(vals)) > 1:
            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 1000)
            ax.plot(
                x, kde(x) * len(vals) * bin_w,
                color=custom_palette[category], linewidth=4,
            )
            med = np.median(vals)
            kde_at_med = kde(med)[0]
            scaled_med = kde_at_med * len(vals) * bin_w
            ax.plot(
                [med, med], [0, scaled_med],
                linestyle="--", linewidth=4,
                color=custom_palette[category],
                label=f"M: {int(round(med))} m",
            )
            ax.plot(
                med, scaled_med, "o",
                color=custom_palette[category], markersize=16,
            )

        ax.set_ylabel(
            r"$N_{\mathrm{Events}}$"
            if category == "bark_beetle" else "",
            fontsize=fontsize_label_med, labelpad=padding_label,
        )
        ax.set_xlim(-0.01, 2400)
        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=3, integer=True, prune=None)
        )
        set_two_positive_yticks(ax)
        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%d")
        )
        ax.set_xlabel(
            r"$\Delta_{\mathrm{Centroid}} \; (\mathrm{m})$"
            if i == len(category_order) - 1 else "",
            fontsize=fontsize_label_med, labelpad=padding_label,
        )
        ax.tick_params(axis="both", labelsize=fontsize_tick)
        ax.legend(
            handles=[
                mpatches.Patch(
                    color=custom_palette[category],
                    label=category.replace("_", " ").title(),
                ),
                Line2D(
                    [0], [0], linestyle="--",
                    color=custom_palette[category],
                    label=f"M: {int(round(med))} m",
                ),
            ],
            fontsize=fontsize_legend, loc="upper right",
        )

    # ── Plot details ──────────────────────────────────────────────
    for j, title in enumerate(
        [r"$\mathbf{a)}$", r"$\mathbf{b)}$", r"$\mathbf{c)}$"]
    ):
        axs[0, j].set_title(
            title, fontsize=50, fontweight="bold",
            pad=30, loc="center",
        )

    # ── Save ──────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()

    if stats_path and stats_rows:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        logging.info("Saved size/shift stats to %s", stats_path)


def plot_spatial_overlap_histograms(
    ids_gdf, s1dm_gdf, custom_colors, save_path=None
):
    """Plot spatial overlap percentage histograms for IDS and S1DM.

    Two histograms side-by-side: the left shows what fraction of the
    IDS polygon is covered by S1DM; the right shows the reverse.
    Zero-overlap events land in the grey-shaded bin left of 0 so they
    are visually distinct from non-zero overlaps.

    Args:
        ids_gdf (gpd.GeoDataFrame): IDS polygon layer.
        s1dm_gdf (gpd.GeoDataFrame): S1DM polygon layer.
        custom_colors (dict[str, str]): Mapping ``DCA_ID`` -> colour.
        save_path (str, optional): Output file path.
    """
    # ── Data preparation ──────────────────────────────────────────
    results_df = compute_spatial_overlap(ids_gdf, s1dm_gdf)
    results_df = results_df[results_df["DCA_ID"] != "drought"]
    ordered_types = [
        t for t in DISTURBANCE_ORDER
        if t in results_df["DCA_ID"].unique()
    ]
    # Zero bin [-20, 1e-9) is 20 units wide — same as all other bins — so the
    # grey bar is visually the same width as the coloured bars.  The 1e-9
    # boundary (instead of exactly 0) ensures 0.0 values land in the grey bin.
    bins = [-20, 1e-9, 20, 40, 60, 80, 100]

    # ── Plotting ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].axvspan(-20, 0, color="grey", alpha=0.3)
    axes[1].axvspan(-20, 0, color="grey", alpha=0.3)

    hist_kw = dict(
        bins=bins, element="bars", kde=False,
        multiple="dodge", shrink=0.8, stat="probability",
        palette=custom_colors, legend=False,
    )
    sns.histplot(
        data=results_df, x="percentage_ids",
        hue="DCA_ID", hue_order=ordered_types,
        ax=axes[0], **hist_kw,
    )
    sns.histplot(
        data=results_df, x="percentage_s1cd",
        hue="DCA_ID", hue_order=ordered_types,
        ax=axes[1], **hist_kw,
    )

    # ── Legend with median values ─────────────────────────────────
    legend_ids, legend_s1dm = [], []
    for dca_id in ordered_types:
        color = custom_colors[dca_id]
        sub = results_df[results_df["DCA_ID"] == dca_id]
        med_ids = np.median(sub["percentage_ids"])
        med_s1dm = np.median(sub["percentage_s1cd"])
        med_ids = 0 if np.isclose(med_ids, 0, atol=1e-10) else med_ids
        med_s1dm = 0 if np.isclose(med_s1dm, 0, atol=1e-10) else med_s1dm
        lbl_ids = f"{med_ids:.2f}" if med_ids != 0 else "0"
        lbl_s1dm = f"{med_s1dm:.2f}" if med_s1dm != 0 else "0"
        legend_ids.append(Line2D(
            [0], [0], color=color, lw=3,
            label=f"{format_label(dca_id)} (Median: {lbl_ids})",
        ))
        legend_s1dm.append(Line2D(
            [0], [0], color=color, lw=3,
            label=f"{format_label(dca_id)} (Median: {lbl_s1dm})",
        ))

    # ── Axes formatting ───────────────────────────────────────────
    # All bins are 20 units wide; ticks centred on each bar (-10, 10, 30, 50, 70, 90).
    tick_positions = [-10, 10, 30, 50, 70, 90]
    tick_labels    = [0, 20, 40, 60, 80, 100]

    x_label_ids = (
        r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}"
        r"{A_{\text{IDS}}}$"
    )
    x_label_s1dm = (
        r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}"
        r"{A_{\text{S1DM}}}$"
    )
    axes[0].set_xlabel(x_label_ids, fontsize=26, labelpad=15)
    axes[0].set_ylabel("Probability", fontsize=18, labelpad=15)
    axes[0].tick_params(axis="both", which="major", labelsize=16)
    axes[0].grid(True, linestyle="--", alpha=0.8)
    axes[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0].set_xlim(left=-20, right=101)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    axes[0].legend(handles=legend_ids, loc="upper right", fontsize=16, title_fontsize=18)

    axes[1].set_xlabel(x_label_s1dm, fontsize=26, labelpad=15)
    axes[1].tick_params(axis="both", which="major", labelsize=16)
    axes[1].grid(True, linestyle="--", alpha=0.8)
    axes[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axes[1].set_xlim(left=-20, right=101)
    axes[1].set_ylabel("")
    axes[1].tick_params(left=False)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].legend(handles=legend_s1dm, loc="upper right", fontsize=16, title_fontsize=18)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()


def plot_study_area(
    area_path,
    region_id,
    tcc_nc_path,
    s1_tiles_boundary_path,
    ids,
    custom_colors,
    save_path,
    logging,
):
    """Load data and produce the study-area overview figure.

    Convenience entry point that loads the required datasets from disk
    then delegates all figure assembly to
    ``assemble_study_area_figure``.

    Args:
        area_path (str): Path to the USFS administrative regions
            shapefile.
        region_id (str): USFS region identifier (e.g. ``'08'``).
        tcc_nc_path (str): Path to the TCC NetCDF file.
        s1_tiles_boundary_path (str): Path to the S1 tile boundaries
            shapefile.
        ids (gpd.GeoDataFrame): IDS disturbance polygons to overlay.
        custom_colors (dict[str, str]): Disturbance-type colour map.
        save_path (str): Output file path for the saved figure.
        logging: Logger instance passed from the caller.
    """
    logging.info("Loading USA mainland and Region 8 boundary ...")
    mainland = load_mainland_regions(area_path)
    region_8 = load_region_boundary(area_path, region_id=region_id)

    logging.info("Loading TCC dataset for Region 8 ...")
    tcc_dataset = load_tcc_dataset(tcc_nc_path)

    logging.info("Assembling study-area figure ...")
    assemble_study_area_figure(
        tcc_dataset,
        s1_tiles_boundary_path,
        mainland,
        region_8,
        ids,
        region_id,
        custom_colors,
        save_path,
        logging,
    )


def plot_overlap_omission(ids_gdf, s1dm_gdf, figure_path, summary_path):
    """Plot per-disturbance overlap/omission percentages and save a summary CSV.

    Overlap is the fraction of IDS events captured by S1DM; omission is the
    complementary loss, shown as a negative bar.  The summary CSV contains
    event counts and area statistics per disturbance type.

    Args:
        ids_gdf (gpd.GeoDataFrame): Filtered IDS polygons with ``DCA_ID``.
        s1dm_gdf (gpd.GeoDataFrame): S1DM polygons with ``DCA_ID``, ``IDX_D``,
            ``S1_YEAR``, and ``area_km2``.
        figure_path (str): Output path for the PNG figure.
        summary_path (str): Output path for the CSV summary table.
    """
    # ── Data preparation ──────────────────────────────────────────
    s1dm = s1dm_gdf.dissolve(by="IDX_D", as_index=False)
    s1dm["area_km2"] = s1dm.to_crs("EPSG:27705").geometry.area / 1e6
    s1dm = s1dm[~s1dm["DCA_ID"].isin(["drought", "fire"])]
    ids = ids_gdf[~ids_gdf["DCA_ID"].isin(["drought", "fire"])]

    counts = pd.DataFrame({
        "IDS": ids["DCA_ID"].value_counts(),
        "S1DM": s1dm["DCA_ID"].value_counts(),
    }).fillna(0)
    order = ["bark_beetle", "wind", "defoliators"]
    counts = counts.reindex([d for d in order if d in counts.index])
    counts["remaining_pct"] = (counts["S1DM"] / counts["IDS"]) * 100
    counts["loss_pct"] = -((counts["IDS"] - counts["S1DM"]) / counts["IDS"]) * 100

    # ── Summary CSV ───────────────────────────────────────────────
    summary = s1dm.groupby("DCA_ID").agg(
        n_events=("IDX_D", "count"),
        year_min=("S1_YEAR", "min"),
        year_max=("S1_YEAR", "max"),
        median_area=("area_km2", "median"),
        mean_area=("area_km2", "mean"),
        total_area=("area_km2", "sum"),
        min_area=("area_km2", "min"),
        max_area=("area_km2", "max"),
        std_area=("area_km2", "std"),
    ).reset_index()
    summary["year_range"] = summary["year_min"].astype(str) + "-" + summary["year_max"].astype(str)
    summary = summary.drop(columns=["year_min", "year_max"])
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logging.info("Saved summary table to %s", summary_path)

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(counts))
    x_labels = [DIST_LABEL_MAP.get(d, d) for d in counts.index]

    overlap_bars = ax.bar(
        x, counts["remaining_pct"],
        color="#0F4980", label="Overlap",
        edgecolor="white", linewidth=1.5, zorder=5,
    )
    omission_bars = ax.bar(
        x, counts["loss_pct"],
        color="#BF0E0E", label="Omission",
        edgecolor="white", linewidth=1.5, zorder=5,
    )

    ax.axhline(0, color="grey", linewidth=1, zorder=6)

    for bar in overlap_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", zorder=10)
    for bar in omission_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h - 2,
                f"{abs(h):.1f}%", ha="center", va="top",
                fontsize=16, fontweight="bold", zorder=10)

    # ── Axes formatting ───────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=18)
    ax.set_xlabel("Disturbance Type", fontsize=18, labelpad=10)
    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_ylim(-100, 100)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", left=False, labelleft=True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("dimgray")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    ax.legend(loc="lower left", fontsize=18)
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=SAVE_DPI_HIGH, bbox_inches="tight")
    plt.show()
    logging.info("Saved overlap/omission figure to %s", figure_path)


def _ids_uses_hash_format(ids_file):
    """Return True if the IDS shapefile uses hash-based IDX_D (new format).

    New format: ``bark_beetle_2018_e808498a`` — last segment is 8 hex chars.
    Old format: ``bark_beetle_2018_1136``     — last segment is all digits.
    """
    gdf = gpd.read_file(ids_file)
    if "IDX_D" not in gdf.columns or gdf.empty:
        return False
    sample = gdf["IDX_D"].dropna().iloc[0]
    suffix = sample.rsplit("_", 1)[-1]
    return bool(re.fullmatch(r"[0-9a-f]{8}", suffix))


def run_manual_validation(
    ids_file,
    s1dm_file,
    manual_base_folder,
    result_path=None,
    save_path=None,
):
    """Compute manual validation metrics and render the summary figure.

    Thin wrapper that calls ``compute_manual_validation_metrics`` then
    passes the results to ``plot_manual_validation_boxplots``.

    When the IDS file uses hash-based IDX_D (new format), automatically
    loads the lookup table ``manual_labels_idx_lookup.csv`` from the
    parent directory of *manual_base_folder* to map old folder names to
    current IDX_D values.  Falls back to the existing direct-match /
    geometry-fallback logic when the old sequential IDX_D format is
    detected.

    Args:
        ids_file (str): Path to the IDS shapefile.
        s1dm_file (str): Path to the S1DM shapefile.
        manual_base_folder (str): Root folder with per-disturbance
            manual reference sub-folders.
        result_path (str, optional): Reserved for future use.
        save_path (str, optional): Path to save the figure.
    """
    idx_mapping = None

    if _ids_uses_hash_format(ids_file):
        lookup_csv = os.path.join(
            os.path.dirname(os.path.normpath(manual_base_folder)),
            "manual_labels_idx_lookup.csv",
        )
        if os.path.exists(lookup_csv):
            raw = pd.read_csv(lookup_csv)
            idx_mapping = pd.DataFrame({
                "manual_idx": raw["manual_folder"],
                "ids_idx":    raw["new_IDX_D"],
                "s1dm_idx":   raw["new_IDX_D"],
            })
            logging.info(
                "run_manual_validation: hash IDX_D detected — "
                "loaded lookup table (%d entries) from %s",
                len(idx_mapping), lookup_csv,
            )
        else:
            logging.warning(
                "run_manual_validation: hash IDX_D detected but lookup "
                "table not found at %s — falling back to geometry matching",
                lookup_csv,
            )

    df_results, sig_jaccard, sig_overlap = (
        compute_manual_validation_metrics(
            ids_file, s1dm_file, manual_base_folder,
            idx_mapping=idx_mapping,
        )
    )
    plot_manual_validation_boxplots(
        df_results, sig_jaccard, sig_overlap,
        save_path=save_path,
    )
