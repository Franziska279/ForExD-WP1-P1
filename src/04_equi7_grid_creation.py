import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
from shapely import wkt
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from equi7grid_lite import Equi7Grid
from dotenv import load_dotenv
import os
from pathlib import Path
import os
import time
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm  # For progress bars
from dotenv import load_dotenv

# Set font sizes for various components
plt.rcParams.update({
    'font.size': 14,           # Global font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X and Y label font size
    'xtick.labelsize': 14,     # X tick label font size
    'ytick.labelsize': 14,     # Y tick label font size
})


def get_region_shape(path, region_id):
    
    usa = gpd.read_file(path)
    country = usa[usa.REGION == region_id]
    
    region = country.explode()[0:1] 

    return region
    

def generate_equi7_grid(usa_filepath, resolution, pixel_size, region_id, output_shapefile):
    size = resolution * pixel_size
    print(f"Resolution: {resolution}, Pixel Size: {pixel_size}, Minimum Grid Size: {size}")
    
    # Initialize the Equi7 grid system
    grid_system = Equi7Grid(min_grid_size=size)
    
    # Load and filter the USA shapefile
    usa = gpd.read_file(usa_filepath)
    country = usa[usa['REGION'] == region_id]
    region = country.explode().reset_index(drop=True)
    
    if len(region) == 0:
        raise ValueError(f"No region found with ID {region_id}")
    
    # Ensure 'region' is a GeoDataFrame
    if not isinstance(region, gpd.GeoDataFrame):
        region = gpd.GeoDataFrame(region, geometry='geometry')
    
    # Generate the Equi7 grid
    print("Generating grid...")
    grid = grid_system.create_grid(
        level=0,
        zone="NA",
        mask=region  # Ensure 'region' is a GeoDataFrame with CRS
    )
    
    # Plot and save the grid boundaries
    ax = grid.boundary.plot()
    plt.title(f'Equi7 Grid Boundaries Region {region_id}')
    plt.savefig(f"{output_shapefile.replace('.shp', '.png')}")
    plt.show()
    
    # Save the grid to shapefile
    grid.to_file(output_shapefile)
    print(f"Grid saved to {output_shapefile}")

    return grid


def plot_intersection_batches(intersection, region_shape_path, region_id, grid_figure_output_path, target_crs):
    # Load and reproject region shape
    r8 = get_region_shape(region_shape_path, region_id)
    region_shape = r8.to_crs(target_crs)  # Reproject to EPSG:27705

    # Step 1: Plot the intersection data
    fig, ax = plt.subplots(figsize=(12, 12))  # Increased figure size for better readability

    # Plot the region boundary first to be in the background
    region_shape.boundary.plot(ax=ax, edgecolor='#264653', linewidth=2, linestyle='--', label='Region Boundary')

    # Plot the intersection data on top
    intersection.boundary.plot(ax=ax, edgecolor='#E76F51', linewidth=1, label='Intersection Data')

    # Step 2: Add convex hull around specific batches (0-100, 100-200, etc.)
    def add_convex_hull(ax, geometries, batch_idx, **kwargs):
        """Add a convex hull around the geometries and label it with the batch index."""
        # Combine geometries into one using the union_all method
        combined_geom = geometries.geometry.unary_union
        if isinstance(combined_geom, MultiPolygon):
            combined_geom = combined_geom.convex_hull
        else:
            combined_geom = combined_geom.convex_hull
        
        # Convert convex hull to polygon and add to plot
        if combined_geom.is_empty:
            return

        x, y = combined_geom.exterior.xy
        ax.plot(x, y, **kwargs)
        # Add label in the centroid of the geometry
        centroid = combined_geom.centroid
        ax.text(centroid.x, centroid.y, f'{batch_idx}', fontsize=20, color='#264653', ha='center')

    # Define batch size (100)
    batch_size = 100

    # Loop over the data in batches of 100
    for i in range(0, len(intersection), batch_size):
        batch_geometries = intersection.iloc[i:i+batch_size]
        add_convex_hull(ax, batch_geometries, batch_idx=i//batch_size, color='#2A9D8F', lw=2)  # Draw the convex hull and label

    # Enhance plot aesthetics
    ax.set_aspect('equal')
    ax.set_title(f'Intersected Grid for Region {region_id} with {len(intersection)} Grid Cells\n (Marking the Grids in Batches of 100)', fontsize=16, fontweight='bold', loc='center')
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines

    # Add legend
    ax.legend(loc='best', fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(grid_figure_output_path, dpi=300)  # Save with high resolution

    # Show the figure
    plt.show()


def create_convex_hulls(refdm, ids, output_path):
    """
    Create and save convex hulls from REFDM data and USDA polygon IDs by 
    spatially joining and generating convex hulls for each USDA_IDX.

    Parameters:
    - refdm_path: Path to the REFDM shapefile.
    - ids_path: Path to the USDA polygon IDs shapefile.
    - output_path: Path to save the resulting convex hulls shapefile.

    Returns:
    - convex_hulls_gdf: A GeoDataFrame containing the convex hulls.
    """
    

    # Step 2: Dissolve REFDM geometries by 'USDA_IDX'
    dissolved_refdm = refdm[['IDX_D', 'geometry']].dissolve(by='IDX_D').reset_index()
    print(f"Number of dissolved geometries: {len(dissolved_refdm)}")
    
    # Step 3: Spatial join between dissolved REFDM geometries and USDA IDs
    merged_gdf = gpd.sjoin(dissolved_refdm, ids, how='left', predicate='intersects', on_attribute=['IDX_D'])
    
    # Step 4: Group merged geometries by 'USDA_IDX' and combine them using unary union
    merged_geometries = merged_gdf.groupby('IDX_D')['geometry'].apply(lambda x: x.unary_union)
    
    # Step 5: Create convex hulls for each grouped geometry
    convex_hulls = merged_geometries.apply(lambda geom: MultiPolygon([geom.convex_hull]))
    
    # Step 6: Create a new GeoDataFrame for the convex hulls with the same CRS as REFDM
    convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=refdm.crs).reset_index()
    
    # Step 7: Save convex hulls to the specified shapefile path
    convex_hulls_gdf.to_file(output_path)
    print(f"Convex hulls saved to {output_path}")
    
    return convex_hulls_gdf


def intersect_grid(convex_hulls_gdf, grid_gdf, output_shapefile_path, output_figure_path, target_crs, region=None, region_id=None):
    """
    Perform spatial join between convex hulls and grid, plot, and save the results.

    Parameters:
    - convex_hulls_gdf: GeoDataFrame of convex hulls.
    - grid_gdf: GeoDataFrame of grid.
    - output_shapefile_path: File path to save the intersected grid as a shapefile.
    - output_figure_path: File path to save the figure as an image.
    - target_crs: Target Coordinate Reference System (CRS) to project geometries.
    - region: Optional GeoDataFrame for a specific region boundary (optional).
    - region_id: ID of the region to be displayed in the plot title (optional).

    Returns:
    - intersected_gdf_equi7: GeoDataFrame of intersected grids with convex hulls.
    """
    # Perform spatial join between the grid and convex hulls
    intersected_gdf_equi7 = gpd.sjoin(grid_gdf, convex_hulls_gdf, how='inner', predicate='intersects')
    
    # Clean the intersected data
    intersected_gdf_equi7 = (
        intersected_gdf_equi7
        .drop(columns=['index_right', 'IDX_D', 'level', 'land', 'zone'])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    # Save the intersected GeoDataFrame as a shapefile
    intersected_gdf_equi7.to_file(output_shapefile_path)
    print(f"Intersected grids saved to: {output_shapefile_path}")
    
    # Plot intersected geometries and region boundary (if provided)
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
    
    # Plot the intersected grids
    intersected_gdf_equi7.boundary.plot(
        ax=ax, color='black', linewidth=0.6, linestyle='-', zorder=2,
        label=f'Intersected Grids ({len(intersected_gdf_equi7)})'
    )
    
    # If region is provided, plot its boundary
    if region is not None:
        region = region.to_crs(target_crs)
        region.boundary.plot(
            ax=ax, color='red', linewidth=0.5, linestyle='--', zorder=3,
            label=f'Region {region_id} Boundary'
        )
    
    # Customize plot appearance
    ax.set_title(f'Intersected Grids with EQUI7 NA Grid and Region {region_id} Boundary', fontsize=18, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')  # Keep aspect ratio
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12)
    
    # Save the plot as an image
    plt.savefig(output_figure_path, dpi=300)
    print(f"Plot saved to: {output_figure_path}")
    
    # Display the plot
    plt.show()
    
    return intersected_gdf_equi7

def add_minicube_index(intersected_grid, refdm, output, equi7_crs):

    reprojected_refdm = refdm.to_crs(equi7_crs)

    # Ensure both GeoDataFrames are in the same CRS for proper spatial operations
    if reprojected_refdm.crs != intersected_grid.crs:
        intersected_grid = intersected_grid.to_crs(reprojected_refdm.crs)

    # Function to get intersecting indices
    def get_intersecting_indices(geometry, grid):
        intersecting_indices = grid[grid.intersects(geometry)].index.tolist()
        return intersecting_indices

    # Apply the function to each row in reprojected_refdm
    reprojected_refdm['minicube_index'] = reprojected_refdm['geometry'].apply(lambda geom: get_intersecting_indices(geom, intersected_grid))

    # Add the 'cube_amount' column by counting the length of each list in 'minicube_index'
    reprojected_refdm['cube_amount'] = reprojected_refdm['minicube_index'].apply(len)

    reprojected_refdm.to_file(output)


def main():

    # Load environment variables from the .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    # Retrieve environment variables
    s2_minicubes_folder = os.getenv('EQUI7_GRIDS')
    print(f"Equi7 grids folder: {s2_minicubes_folder}")

    # Retrieve the CRS (Coordinate Reference System) for Equi7 NA
    equi7_crs = os.getenv('EQUI7_NA_EPSG')

    # Ensure the 'REGION' environment variable is set
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("The 'REGION' environment variable is not set. Please ensure it is defined in the .env file.")

    # Format region ID as a two-digit string
    region_id = str(region).zfill(2)

    # Parameters for the grid
    resolution = 10
    pixel_size = 512

    # Define file paths for shapefiles and output locations
    usa_filepath = f"{os.getenv('REGION_SHAPE')}/S_USA.AdministrativeRegion.shp"
    ids_path = f"{os.getenv('RESULTS')}/region{region_id}_dca_filtered_ids_usda_polygons.shp"
    refdm_path = f"{os.getenv('RESULTS')}/radar_results/radar_enhanced_forest_disturbance_mapping_region_{region_id}.shp"
    output_path_refdm = f"{os.getenv('RESULTS')}/radar_results/radar_enhanced_forest_disturbance_mapping_region_{region_id}_epsg_27705.shp"
    output_path_grid = f"{s2_minicubes_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}.shp"
    output_path_conves = f"{os.getenv('RESULTS')}/radar_results/convex_hulls_refdm_region_{region_id}_epsg_4326.shp"
    output_path_intersetion = f"{s2_minicubes_folder}/grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersetion.shp"
    output_figure_intersection = f"{os.getenv('FIGURES')}/p1_f4_grid_equi7_{resolution}_{pixel_size}_region_{region_id}.png"
    grid_figure_output_path = f"{os.getenv('FIGURES')}/p1_f5_grid_equi7_{resolution}_{pixel_size}_region_{region_id}_intersecting_cells_batches.png"

    # Function to load region shape from USA boundary shapefile
    region_shape = get_region_shape(usa_filepath, region_id=region_id)

    # Step 1: Read the REFDM and USDA IDs shapefiles
    print(f"Loading REFDM shapefile from: {refdm_path}")
    refdm_gdf = gpd.read_file(refdm_path)

    print(f"Loading USDA IDs shapefile from: {ids_path}")
    ids_gdf = gpd.read_file(ids_path)

    # Step 2: Generate the Equi7 grid and save it as a shapefile
    print("\nGenerating the Equi7 grid...")
    start_time_grid = time.time()  # Start timer for grid creation

    grid = generate_equi7_grid(
        usa_filepath=usa_filepath,
        resolution=resolution,
        pixel_size=pixel_size,
        region_id=region_id,
        output_shapefile=output_path_grid
    )

    # Print the time taken to generate the grid
    print(f"Grid generation completed in {time.time() - start_time_grid:.2f} seconds.")
    print(f"Grid saved to: {output_path_grid}\n")

    # Step 3: Create convex hulls from the REFDM and USDA shapefiles
    print("Creating convex hulls from REFDM and USDA polygons...")

    # Use tqdm to display a progress bar during convex hull creation
    start_time_convex = time.time()  # Start timer for convex hull creation

    convex_hulls = create_convex_hulls(refdm_gdf, ids_gdf, output_path_conves)

    # Print the time taken for convex hull creation
    print(f"Convex hull creation completed in {time.time() - start_time_convex:.2f} seconds.")
    print(f"Convex hulls saved to: {output_path_conves}\n")

    # Step 4: Reproject the convex hulls to Equi7 CRS
    print(f"Reprojecting convex hulls to CRS: {equi7_crs}")
    reprojected_convex_hulls = convex_hulls.to_crs(equi7_crs)
    print(f"Reprojection completed.\n")

    # Plot the grid (optional, can be removed if not needed)
    grid.plot()

    # Step 5: Intersect the reprojected convex hulls with the grid and save the result
    print("Performing spatial intersection between convex hulls and grid...")
    start_time_intersection = time.time()

    intersected = intersect_grid(
        reprojected_convex_hulls,
        grid,
        output_path_intersetion,
        output_figure_intersection,
        equi7_crs,
        region=region_shape,
        region_id=region_id
    )

    print("Add minicube indecies to correponing REFDM ...")
    add_minicube_index(intersected, refdm_gdf, output_path_refdm, equi7_crs)

    print("Plot ...")
    plot_intersection_batches(intersected, usa_filepath, region_id, grid_figure_output_path, equi7_crs)

    
    # Print the time taken for intersection
    print(f"Intersection completed in {time.time() - start_time_intersection:.2f} seconds.")
    print(f"Intersected grid saved to: {output_path_intersetion}")
    print(f"Reprojected REFDM saved to: {output_path_refdm}")
    print(f"Intersection plot saved to: {output_figure_intersection}\n")
    print(f"Intersection plot with batches saved to: {grid_figure_output_path}\n")

    # Print a message indicating the script has finished
    print("Process completed successfully!\n")



if __name__ == "__main__":
    main()