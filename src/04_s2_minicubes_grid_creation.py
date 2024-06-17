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


def get_tiff_bounds(tiff_path):
    # Open the TIFF file
    with rasterio.open(tiff_path) as dataset:
        # Get the bounds of the TIFF file
        bounds = dataset.bounds
        print(f"Bounds of {tiff_path}:")
        print(f"Left: {bounds.left}, Bottom: {bounds.bottom}, Right: {bounds.right}, Top: {bounds.top}")
        return bounds

def create_grid(bounds, cell_size):
   
    # Define the extent of the USA (example bounding box)
    extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Calculate the side length of the square based on the area (0.034 square units)
    side_length = np.sqrt(cell_size)

    # Generate grid of polygons covering the USA extent
    grid_polygons = []
    minx, miny, maxx, maxy = extent
    x_steps = int(np.ceil((maxx - minx) / side_length))
    y_steps = int(np.ceil((maxy - miny) / side_length))

    for x in range(x_steps):
        for y in range(y_steps):
            polygon = Polygon([(minx + x * side_length, miny + y * side_length),
                            (minx + (x + 1) * side_length, miny + y * side_length),
                            (minx + (x + 1) * side_length, miny + (y + 1) * side_length),
                            (minx + x * side_length, miny + (y + 1) * side_length)])
            grid_polygons.append(polygon)

    # Create a GeoDataFrame from the list of polygons
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs='EPSG:4326')
    
    return grid_gdf

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon
from tqdm import tqdm

def create_convex_hulls(refdm_path, ids_path, grid_gdf):
    """
    Create convex hulls from the reference disturbance map (REFDM) and intersecting USDA polygons,
    then find the grid cells that intersect with these convex hulls.

    Parameters:
    refdm_path (str): Path to the reference disturbance map shapefile.
    ids_path (str): Path to the USDA polygons CSV file.
    grid_gdf (GeoDataFrame): GeoDataFrame containing the grid cells.

    Returns:
    intersected_gdf (GeoDataFrame): GeoDataFrame of grid cells that intersect with convex hulls.
    convex_hulls_gdf (GeoDataFrame): GeoDataFrame of the convex hulls created from merged geometries.
    """

    # Step 1: Load the REFDM shapefile and USDA polygons CSV file
    print("Step 5.1: Load the REFDM shapefile and USDA polygons CSV file")
    refdm_gdf = gpd.read_file(refdm_path)
    ids_usda = pd.read_csv(ids_path)
    ids_usda['geometry'] = ids_usda['geometry'].apply(wkt.loads)
    ids_gdf = gpd.GeoDataFrame(ids_usda, geometry='geometry')

    # Ensure CRS match
    if refdm_gdf.crs is None:
        print("Warning: REFDM GeoDataFrame has no CRS. Assuming WGS 84.")
        refdm_gdf = refdm_gdf.set_crs("EPSG:4326")

    if ids_gdf.crs is None:
        print("Warning: USDA GeoDataFrame has no CRS. Assuming WGS 84.")
        ids_gdf = ids_gdf.set_crs("EPSG:4326")

    if refdm_gdf.crs != ids_gdf.crs:
        print("CRS mismatch detected. Reprojecting USDA polygons to match REFDM CRS.")
        ids_gdf = ids_gdf.to_crs(refdm_gdf.crs)

    # Step 2: Dissolve REFDM polygons by 'USDA_IDX'
    print("Step 5.2: Dissolving the REFDM polygons by 'USDA_IDX'")
    dissolved_refdm = refdm_gdf.dissolve(by='USDA_IDX').reset_index()

    # Step 3: Spatial join to intersect dissolved REFDM with USDA polygons
    print("Step 5.3: Intersecting USDA polygons with dissolved REFDM")
    merged_gdf = gpd.sjoin(dissolved_refdm, ids_gdf, how='left', predicate='intersects')

    # Keep only 'geometry' and 'USDA_IDX' columns
    merged_gdf = merged_gdf[['geometry', 'USDA_IDX']]

    # Step 4: Create convex hulls from the dissolved geometries
    print("Step 5.4: Creating convex hulls from dissolved geometries")
    dissolved_merged_gdf = merged_gdf.dissolve(by='USDA_IDX')
    
    convex_hulls = []
    for geom in tqdm(dissolved_merged_gdf['geometry'], desc="Creating convex hulls", unit="hull"):
        convex_hulls.append(MultiPolygon([geom.convex_hull]))

    convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=refdm_gdf.crs).reset_index()

    # Ensure CRS match before spatial join
    if grid_gdf.crs != convex_hulls_gdf.crs:
        print("CRS mismatch detected between grid and convex hulls. Reprojecting grid to match convex hulls CRS.")
        grid_gdf = grid_gdf.to_crs(convex_hulls_gdf.crs)

    # Step 5: Find grid cells intersecting with convex hulls
    print("Step 5.5: Finding grid cells intersecting with convex hulls")
    intersected_gdf = gpd.sjoin(grid_gdf, convex_hulls_gdf, how='inner', predicate='intersects')
    intersected_gdf = intersected_gdf.drop(columns=['index_right','index']).drop_duplicates().reset_index(drop=True)

    # Output the number of intersected grids
    print(f"Amount of Grids: {len(intersected_gdf)}")

    return intersected_gdf, convex_hulls_gdf


def plot_combined(intersected_gdf, convex_hulls_gdf):
    # Step 2: Plot the intersected grids and convex hulls
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot intersected grids in the first subplot
    intersected_gdf.boundary.plot(ax=axs[0], edgecolor='blue', linewidth=1)
    axs[0].set_title(f"{len(intersected_gdf)} grid cells")
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')

    # Plot intersecting convex hulls in the second subplot
    convex_hulls_gdf.boundary.plot(ax=axs[1], edgecolor='red', linewidth=1)
    intersected_gdf.boundary.plot(ax=axs[1], edgecolor='blue', linewidth=1)
    axs[1].set_title(f"{len(intersected_gdf)} grids with corresponding disturbance events")
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')

    plt.tight_layout()
    
    # Save the figure to results.png
    plt.savefig('/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/gird_disturbance_events.png', dpi=300)  # Adjust dpi as needed for quality
    
    plt.show()



# Define necessary functions here
# e.g., get_tiff_bounds, create_grid, create_convex_hulls, plot_combined
# Make sure these functions are defined elsewhere in your script or imported

def main():
    print("Step 1: Starting the grid creation process")

    # Paths to input datasets
    TCC_path_2017 = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"
    refdm_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping.shp"
    ids_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"

    # Step 2: Get the rough boundary of Region 8
    print("Step 2: Obtaining the rough boundary of Region 8")
    bounds = get_tiff_bounds(TCC_path_2017)

    # Step 3: Create a grid based on these bounds with a cell size of 0.043 degrees
    print("Step 3: Creating a grid with a cell size of 0.043 degrees")
    cell_size = 0.043
    grid_gdf = create_grid(bounds, cell_size)

    # Step 4: Save the grid as a shapefile
    output_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/grid.shp"
    print(f"Step 4: Saving the grid to {output_path}")
    grid_gdf.to_file(output_path)

    # Step 5: Create convex hulls and find the intersecting grid cells
    print("Step 5: Creating convex hulls and identifying intersecting grid cells")
    intersected, convex_hulls = create_convex_hulls(refdm_path, ids_path, grid_gdf)

    # Step 6: Plot the grids
    print("Step 6: Plotting the grids")
    plot_combined(intersected, convex_hulls)

    # Save the intersected grid cells
    intersected_output_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/intersected_grid.shp"
    intersected.to_file(intersected_output_path)
    print(f"Step 7: Saved intersected grids to {intersected_output_path}")
    
    # Final output
    print(f"Step 8: Completed the process. Total number of grids created: {len(grid_gdf)}")

if __name__ == "__main__":
    main()
