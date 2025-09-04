from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import os

def calculate_jaccard_similarity(g1, g2):
    """
    Calculate the Jaccard similarity between two geometries.

    The Jaccard similarity is defined as the area of the intersection divided by
    the area of the union of the two geometries. This function returns a float
    between 0 and 1, where 0 means no overlap and 1 means identical geometries.

    Args:
        g1 (shapely.geometry.base.BaseGeometry): First geometry (Polygon or MultiPolygon).
        g2 (shapely.geometry.base.BaseGeometry): Second geometry (Polygon or MultiPolygon).

    Returns:
        float: Jaccard similarity coefficient.
    """
    inter = g1.intersection(g2).area
    union = g1.union(g2).area
    return inter / union if union > 0 else 0


def compute_geometry_similarity(df_s1dm, df_ids):
    """
    Merge two GeoDataFrames on 'IDX_D' and compute the Jaccard similarity between
    their geometries.

    This function performs a merge of `df_s1dm` and `df_ids` on the column 'IDX_D',
    then computes the Jaccard similarity between the geometries from each DataFrame
    and stores the result in a new column called 'jaccard'.

    Args:
        df_s1dm (geopandas.GeoDataFrame): GeoDataFrame containing geometries with suffix '_s1dm'.
        df_ids (geopandas.GeoDataFrame): GeoDataFrame containing geometries with suffix '_ids'.

    Returns:
        geopandas.GeoDataFrame: Merged GeoDataFrame with an additional 'jaccard' column.
    """
    df_merged = df_s1dm.merge(
        df_ids[['IDX_D', 'geometry']],
        on='IDX_D',
        suffixes=('_s1dm', '_ids')
    )

    df_merged['jaccard'] = df_merged.apply(
        lambda row: calculate_jaccard_similarity(row['geometry_s1dm'], row['geometry_ids']),
        axis=1
    )

    return df_merged


def split_and_sort_by_dca(df):
    """
    Split the merged GeoDataFrame into multiple DataFrames by disturbance type and
    sort each by descending Jaccard similarity.

    The disturbance types are identified by the 'DCA_ID' column and include:
    'bark_beetle', 'wind', 'defoliators', and 'fire'.

    Args:
        df_merged (geopandas.GeoDataFrame): Merged GeoDataFrame containing 'DCA_ID' and 'jaccard'.

    Returns:
        dict: Dictionary mapping each disturbance type (str) to its sorted GeoDataFrame.
    """
    disturbances = ['bark_beetle', 'wind', 'defoliators', 'fire']

    sorted_dfs = {
        label: df[df['DCA_ID'] == label].sort_values('jaccard', ascending=False)
        for label in disturbances
    }

    return sorted_dfs


def plot_geometries_overlap(row, idx, out_dir="./", geom1='geometry_s1dm', geom2='geometry_ids'):
    """
    Plot overlap of two geometries from a given row of a GeoDataFrame.

    Args:
        row (pandas.Series): A row containing geometry columns and metadata.
        out_dir (str): Output directory.
        geom1 (str): First geometry column (e.g., 'geometry_s1dm').
        geom2 (str): Second geometry column (e.g., 'geometry_ids').
    """
    id_value = row.get('IDX_D', 'unknown')
    dca_id = row.get('DCA_ID', 'unknown')

    fig, ax = plt.subplots(figsize=(8, 8))

    g1 = row[geom1]
    if g1:
        gpd.GeoSeries([g1]).plot(ax=ax, color='red', alpha=0.5, edgecolor='black')

    g2 = row[geom2]
    if g2:
        gpd.GeoSeries([g2]).plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')

    red_patch = mpatches.Patch(color='red', alpha=0.5, label='S1DM')
    blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='IDS')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')

    ax.set_title(f"IDX_D = {id_value}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()

    output_path = os.path.join(out_dir, str(dca_id))
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{idx}_top_{id_value}.png"))
    plt.close()

def save_top_50_per_disturbance(dfs, out_dir="./", figures_dir="/net/projects/forexd/WP1/03_LearningDisturbances/Figures/"):
    """
    Save the top 50 rows of each disturbance type GeoDataFrame as ESRI Shapefiles.

    Each disturbance type is saved as a separate file named
    'top50_<disturbance>.shp' inside the specified output directory.

    Args:
        dfs (dict): Dictionary with disturbance type keys and GeoDataFrame values.
        out_dir (str, optional): Output directory path where shapefiles will be saved.
                                 Defaults to the current directory "./".
    """

    for label, df in dfs.items():
        top50 = df.head(50).copy()
       
        # apply plot function for each disturbance row and save the plot
        for idx, (_, row) in enumerate(top50.iterrows(), start=1):
            plot_geometries_overlap(row, idx=idx, out_dir=figures_dir)

        # Set active geometry column explicitly
        active_geom = 'geometry_s1dm'  
        other_geom = 'geometry_ids' if active_geom == 'geometry_s1dm' else 'geometry_s1dm'

        top50.set_geometry(active_geom, inplace=True)

        # Konvertiere andere Geometriespalte zu WKT (String)
        top50[f'{other_geom}_wkt'] = top50[other_geom].apply(lambda g: g.wkt if g else None)

        # Lösche die andere Geometry-Spalte, da sonst Fehler bei to_file
        top50.drop(columns=[other_geom], inplace=True)

        path = f"{out_dir}/top50_{label}.shp"
        top50.to_file(path, driver="ESRI Shapefile")


def main():

    df_ids = gpd.read_file('/net/projects/forexd/WP1/03_LearningDisturbances/Data/region_08_dca_filtered_ids_usda_polygons.shp')
    df_s1dm = gpd.read_file('/net/projects/forexd/WP1/03_LearningDisturbances/Data/radar_enhanced_forest_disturbance_mapping_region_08_buffer_500_s1dm.shp')

    df_merged = compute_geometry_similarity(df_s1dm, df_ids)
    df_sorted = split_and_sort_by_dca(df_merged)
    df_barkbeetle, df_wind, df_defoliators, df_fire = (
        df_sorted[label] for label in ['bark_beetle', 'wind', 'defoliators', 'fire']
    )
    figures_dir="/net/projects/forexd/WP1/02_ImprovedLabels/Figures/top50_jaccard_similarity/"
    out_dir='/net/projects/forexd/WP1/02_ImprovedLabels/Data/top50_jaccard_similarity/'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    save_top_50_per_disturbance(df_sorted, out_dir=out_dir, figures_dir=figures_dir)



if __name__ == "__main__":
    main()