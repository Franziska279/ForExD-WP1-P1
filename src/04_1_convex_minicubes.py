import geopandas as gpd
from shapely.geometry import box
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import pandas as pd
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon
from tqdm import tqdm



def main():
    print("Step 1: Starting the loading ...")
    minicubes_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/s2_minicube_bounderies_all.shp"
    convex_hulls_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/convex_hulls_refdm.shp"
    intersected_output_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/convex_minicubes_combination.shp"

    mini_gdf = gpd.read_file(minicubes_path)
    convex_gdf = gpd.read_file(convex_hulls_path)

    # Create a new column to store the overlap area for each mini_gdf row
    convex_gdf['max_overlap_area'] = 0
    convex_gdf['best_cube_fn'] = None

    # Iterate over each row in mini_gdf
    for idx, convex_row in convex_gdf.iterrows():
        print(idx)
        convex_geometry = convex_gdf['geometry']
        max_overlap_area = 0
        best_cube_fn = None
        
        # Iterate over each row in convex_gdf
        for _, mini_row in mini_gdf.iterrows():
            mini_geometry = convex_row['geometry']
            overlap = mini_geometry.intersection(mini_geometry)
            
            if not overlap.is_empty:
                overlap_area = overlap.area
                if overlap_area > max_overlap_area:
                    max_overlap_area = overlap_area
                    best_cube_fn = mini_row['USDA_IDX']
        
        # Update the mini_gdf with the best overlapping cube and its overlap area
        convex_gdf.at[idx, 'max_overlap_area'] = max_overlap_area
        convex_gdf.at[idx, 'best_cube_fn'] = best_cube_fn

    # Optionally, filter mini_gdf to see only rows with non-zero overlap
    convex_gdf_with_overlap = convex_gdf[convex_gdf['max_overlap_area'] > 0]

    # Print the updated mini_gdf
    convex_gdf_with_overlap.to_file(intersected_output_path)



if __name__ == "__main__":
    main()
