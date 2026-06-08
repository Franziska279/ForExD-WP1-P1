#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual Disturbance Analysis
===========================

This script computes overlap, Jaccard similarity, and Hausdorff distance 
between manually mapped disturbance polygons and S1DM/IDS polygons. 
It also performs one-sided paired t-tests with two significance thresholds
and creates publication-ready plots with significance annotations.

Author: Franziska Müller
Date: 2025-09-23
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.validation import make_valid
from scipy.stats import ttest_rel

# -------------------
# Helper Functions
# -------------------

def paired_ttest(x, y, alpha1=0.05, alpha2=0.1, alternative="greater"):
    """
    Perform a one-sided paired t-test and return p-value and significance symbol.

    Parameters
    ----------
    x : array-like
        First sample (e.g., S1DM values)
    y : array-like
        Second sample (e.g., IDS values)
    alpha1 : float
        Primary significance threshold (default 0.05)
    alpha2 : float
        Secondary significance threshold (default 0.1)
    alternative : str
        One-sided alternative hypothesis ("greater" or "less")

    Returns
    -------
    pvalue : float
        The p-value of the t-test
    sig_symbol : str
        Significance symbol: "*" if p < alpha1, "†" if p < alpha2, else ""
    """
    t_res = ttest_rel(x, y, alternative=alternative)
    if t_res.pvalue < alpha1:
        sig = "*"
    elif t_res.pvalue < alpha2:
        sig = "†"
    else:
        sig = ""
    return t_res.pvalue, sig


def compute_hausdorff_distance(geom_candidate, geom_manual):
    """Compute Hausdorff distance between two valid geometries."""
    geom_candidate = make_valid(geom_candidate)
    geom_manual = make_valid(geom_manual)
    if geom_candidate.is_empty or geom_manual.is_empty:
        return None
    return geom_candidate.hausdorff_distance(geom_manual)


def compute_jaccard_overlap_hausdorff(geom_candidate, geom_manual):
    """
    Compute overlap percentage, Jaccard similarity, and Hausdorff distance
    between candidate and manual geometries.

    Returns
    -------
    overlap_pct : float
        Percent overlap relative to manual polygon
    jaccard : float
        Jaccard similarity (intersection / union)
    hausdorff : float
        Hausdorff distance
    """
    geom_candidate = make_valid(geom_candidate)
    geom_manual = make_valid(geom_manual)

    inter = geom_candidate.intersection(geom_manual)
    union = geom_candidate.union(geom_manual)

    area_inter = inter.area
    area_union = union.area
    area_manual = geom_manual.area

    overlap_pct = (area_inter / area_manual) * 100 if area_manual > 0 else 0
    jaccard = (area_inter / area_union) if area_union > 0 else 0
    hausdorff = compute_hausdorff_distance(geom_candidate, geom_manual)

    return overlap_pct, jaccard, hausdorff


def load_shapefiles(s1dm_path, ids_path):
    """
    Load S1DM and IDS shapefiles as GeoDataFrames.

    Returns
    -------
    gdf_s1dm, gdf_ids : GeoDataFrame
    """
    gdf_s1dm = gpd.read_file(s1dm_path)
    gdf_ids = gpd.read_file(ids_path)
    return gdf_s1dm, gdf_ids


def compute_metrics_per_disturbance(base_folder, disturbances, gdf_s1dm, gdf_ids):
    """
    Iterate over disturbances and compute metrics for each sample.
    
    Returns
    -------
    df_results : DataFrame
    """
    results = []

    for disturbance in disturbances:
        folder_base = os.path.join(base_folder, disturbance)
        subfolders = [f for f in os.listdir(folder_base) if os.path.isdir(os.path.join(folder_base, f))]

        for idx in subfolders:
            folder = os.path.join(folder_base, idx)
            manual_file = os.path.join(folder, f"merged_union_multipolygon_{idx}.geojson")
            if not os.path.exists(manual_file):
                continue

            gdf_manual = gpd.read_file(manual_file)
            if gdf_manual.empty:
                continue
            geom_manual = make_valid(gdf_manual.geometry.union_all())

            geom_ids = make_valid(gdf_ids[gdf_ids["IDX_D"] == idx].geometry.union_all())
            geom_s1dm = make_valid(gdf_s1dm[gdf_s1dm["IDX_D"] == idx].geometry.union_all())

            if geom_ids.is_empty or geom_s1dm.is_empty or geom_manual.is_empty:
                continue

            overlap_ids, jaccard_ids, hausdorff_ids = compute_overlap_jaccard_hausdorff(geom_ids, geom_manual)
            overlap_s1dm, jaccard_s1dm, hausdorff_s1dm = compute_overlap_jaccard_hausdorff(geom_s1dm, geom_manual)

            results.append({
                "disturbance": disturbance,
                "idx": idx,
                "overlap_ids": overlap_ids,
                "jaccard_ids": jaccard_ids,
                "hausdorff_ids": hausdorff_ids,
                "overlap_s1dm": overlap_s1dm,
                "jaccard_s1dm": jaccard_s1dm,
                "hausdorff_s1dm": hausdorff_s1dm,
                "better_overlap": "S1DM" if overlap_s1dm > overlap_ids else "IDS",
                "better_jaccard": "S1DM" if jaccard_s1dm > jaccard_ids else "IDS",
                "better_hausdorff": "S1DM" if hausdorff_s1dm < hausdorff_ids else "IDS"
            })

    df_results = pd.DataFrame(results)
    return df_results


def plot_metrics(df_results, out_folder, dist_order):
    """
    Create boxplots and barplots for overlap, Jaccard, Hausdorff with
    significance symbols.
    """
    # Map disturbance labels
    dist_labels = {d.lower(): d.title() for d in dist_order}
    df_results["disturbance_label"] = df_results["disturbance"].map(dist_labels)
    df_results["disturbance_label"] = pd.Categorical(df_results["disturbance_label"], categories=dist_order, ordered=True)

    # Melt for plots
    df_melt = df_results.melt(id_vars=["disturbance_label", "idx"],
                               value_vars=["jaccard_ids", "jaccard_s1dm"],
                               var_name="method", value_name="jaccard")
    df_melt["method"] = df_melt["method"].map({"jaccard_ids": "IDS", "jaccard_s1dm": "S1DM"})

    df_hausdorff = df_results.melt(id_vars=["disturbance_label", "idx"],
                                   value_vars=["hausdorff_ids", "hausdorff_s1dm"],
                                   var_name="method", value_name="hausdorff")
    df_hausdorff["method"] = df_hausdorff["method"].map({"hausdorff_ids": "IDS", "hausdorff_s1dm": "S1DM"})

    mean_overlap = df_results.groupby("disturbance_label")[["overlap_ids", "overlap_s1dm"]].mean().reset_index()
    mean_overlap_melt = mean_overlap.melt(id_vars="disturbance_label",
                                         value_vars=["overlap_ids", "overlap_s1dm"],
                                         var_name="method", value_name="overlap")
    mean_overlap_melt["method"] = mean_overlap_melt["method"].map({"overlap_ids":"IDS","overlap_s1dm":"S1DM"})

    # Compute significance per disturbance
    sig_jaccard, sig_hausdorff, sig_overlap = {}, {}, {}
    for d, group in df_results.groupby("disturbance_label"):
        if len(group) > 1:
            _, sig_j = paired_ttest(group["jaccard_s1dm"], group["jaccard_ids"])
            _, sig_h = paired_ttest(group["hausdorff_s1dm"], group["hausdorff_ids"])
            _, sig_o = paired_ttest(group["overlap_s1dm"], group["overlap_ids"])
            sig_jaccard[d] = sig_j
            sig_hausdorff[d] = sig_h
            sig_overlap[d] = sig_o
        else:
            sig_jaccard[d] = ""
            sig_hausdorff[d] = ""
            sig_overlap[d] = ""

    # Plotting style
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.3)
    palette = sns.color_palette("tab10")

    # Jaccard boxplot
    plt.figure(figsize=(11,6))
    sns.boxplot(data=df_melt, x="disturbance_label", y="jaccard", hue="method",
                order=dist_order, palette=palette, fliersize=3, linewidth=1.5)
    for i, dist in enumerate(dist_order):
        if sig_jaccard[dist]:
            y_max = df_melt[df_melt["disturbance_label"] == dist]["jaccard"].max()
            color = "red" if sig_jaccard[dist] == "*" else "black"
            plt.text(i, y_max + 0.02, sig_jaccard[dist], ha="center", va="bottom", fontsize=18, color=color)
    plt.title("Jaccard Similarity with Manual Polygons", weight="bold")
    plt.ylabel("Jaccard Similarity")
    plt.xlabel("Disturbance Type")
    plt.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "boxplot_jaccard_sig.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Hausdorff boxplot
    plt.figure(figsize=(11,6))
    sns.boxplot(data=df_hausdorff, x="disturbance_label", y="hausdorff", hue="method",
                order=dist_order, palette=palette, fliersize=3, linewidth=1.5)
    for i, dist in enumerate(dist_order):
        if sig_hausdorff[dist]:
            y_max = df_hausdorff[df_hausdorff["disturbance_label"] == dist]["hausdorff"].max()
            color = "red" if sig_hausdorff[dist] == "*" else "black"
            plt.text(i, y_max + 0.02, sig_hausdorff[dist], ha="center", va="bottom", fontsize=18, color=color)
    plt.title("Hausdorff Distance with Manual Polygons", weight="bold")
    plt.ylabel("Hausdorff Distance")
    plt.xlabel("Disturbance Type")
    plt.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "boxplot_hausdorff_sig.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Overlap barplot
    plt.figure(figsize=(11,6))
    sns.barplot(data=mean_overlap_melt, x="disturbance_label", y="overlap", hue="method",
                order=dist_order, palette=palette, edgecolor="black", linewidth=1.2)
    for i, dist in enumerate(dist_order):
        if sig_overlap[dist]:
            y_max = mean_overlap_melt[mean_overlap_melt["disturbance_label"] == dist]["overlap"].max()
            color = "red" if sig_overlap[dist] == "*" else "black"
            plt.text(i, y_max + 0.5, sig_overlap[dist], ha="center", va="bottom", fontsize=18, color=color)
    plt.title("Mean % Overlap with Manual Polygons", weight="bold")
    plt.ylabel("Overlap (% of Manual Area)")
    plt.xlabel("Disturbance Type")
    plt.legend(title="Method", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "bar_mean_overlap_sig.png"), dpi=300, bbox_inches="tight")
    plt.show()


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    # Paths
    s1dm_file = "/net/projects/forexd/WP1/03_LearningDisturbances/Data/radar_enhanced_forest_disturbance_mapping_region_08_buffer_500_s1dm.shp"
    ids_file = "/net/projects/forexd/WP1/03_LearningDisturbances/Data/region_08_dca_filtered_ids_usda_polygons.shp"
    base_folder = "/net/projects/forexd/WP1/Data/random_manual_sample_files/"
    out_folder = "/net/projects/forexd/WP1/Figures/manual_disturbances/"
    os.makedirs(out_folder, exist_ok=True)
    
    disturbances = ["defoliators", "bark_beetle", "wind"]
    dist_order = [ "Wind", "Bark Beetle","Defoliators"]

    # Load shapefiles
    gdf_s1dm, gdf_ids = load_shapefiles(s1dm_file, ids_file)

    # Compute metrics
    df_results = compute_metrics_per_disturbance(base_folder, disturbances, gdf_s1dm, gdf_ids)

    # Save results
    df_results.to_csv(os.path.join(out_folder, "summary_overlap_jaccard_hausdorff.csv"), index=False)

    # Plot results
    plot_metrics(df_results, out_folder, dist_order)

    print("Analysis complete. Results saved to:", out_folder)