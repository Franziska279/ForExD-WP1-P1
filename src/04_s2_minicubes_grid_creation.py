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


def create_convex_hulls(refdm_path, ids_path, grid_gdf):

    refdm_gdf = gpd.read_file(refdm_path)
    ids_usda = pd.read_csv(ids_path)
    ids_usda['geometry'] = ids_usda['geometry'].apply(wkt.loads)
    ids_gdf = gpd.GeoDataFrame(ids_usda, geometry='geometry')

    dissolved_refdm = refdm_gdf[['USDA_IDX', 'geometry']].dissolve(by='USDA_IDX')
    dissolved_refdm = dissolved_refdm.reset_index()
    print(len(dissolved_refdm))
    
    merged_gdf = gpd.sjoin(dissolved_refdm, ids_gdf, how='left', op='intersects')

    merged_geometries = merged_gdf.groupby('USDA_IDX')['geometry'].apply(lambda x: x.unary_union)

    convex_hulls = merged_geometries.apply(lambda geom: MultiPolygon([geom.convex_hull]))

    convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=refdm_gdf.crs)

    convex_hulls_gdf = convex_hulls_gdf.reset_index()
    
    
    # Spatial join to find grids that intersect with convex hulls
    intersected_gdf = gpd.sjoin(grid_gdf, convex_hulls_gdf, how='inner', op='intersects')

    # Display the resulting GeoDataFrame with intersected grids
    print("Intersected Grids:", len(intersected_gdf))

    intersected_gdf = intersected_gdf.drop(columns=['index_right', 'USDA_IDX'])
    intersected_gdf = intersected_gdf.drop_duplicates()
    # Reset the index to ensure the GeoDataFrame retains its original structure
    intersected_gdf.reset_index(inplace=True)
    print(f"Amount of Grids: {len(intersected_gdf)}")

    return intersected_gdf , convex_hulls_gdf
    
import matplotlib.pyplot as plt

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




def main():

    TCC_path_2017 = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"
    # Load refdm dataset
    refdm_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping.shp"
    # Load ids dataset
    ids_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"
    bounds = get_tiff_bounds(TCC_path_2017)

    # Step 2: Create a grid based on these bounds with a cell size of 0.42 degrees
    cell_size = 0.043
    grid_gdf = create_grid(bounds, cell_size)

    # Step 3: Save the grid as a shapefile
    output_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/grid.shp"

    intersected, convex_hulls = create_convex_hulls(refdm_path, ids_path, grid_gdf)

    plot_combined(intersected, convex_hulls)

    intersected.to_file(output_path)
    print(f"Saved grid to {output_path}")
    print(f"Amount of grids {len(grid_gdf)}")




if __name__ == "__main__":

    main()

