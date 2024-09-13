import geopandas as gpd
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from pyproj import CRS, Transformer
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def reproject_to_crs(input_file, output_file, crs):
    """
    Reproject the input file to a specified CRS.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the reprojected file.
        crs (str or dict): CRS in PROJ string or dictionary format.
    """
    command = f"gdalwarp -t_srs '{crs}' {input_file} {output_file}"
    subprocess.run(command, shell=True, check=True)
    print("Reprojection completed.")

def change_resolution(input_file, output_file, resolution):
    """
    Change the resolution of the input file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the output file with changed resolution.
        resolution (tuple): Resolution in meters (e.g., (20, 20)).
    """
    command = f"gdalwarp -tr {resolution[0]} {resolution[1]} {input_file} {output_file}"
    subprocess.run(command, shell=True, check=True)
    print("Resolution change completed.")

def crop_to_bounds(input_file, output_file, minx, miny, maxx, maxy):
    """
    Crop the input file based on bounding coordinates.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the cropped output file.
        minx (float): Minimum x coordinate (easting).
        miny (float): Minimum y coordinate (northing).
        maxx (float): Maximum x coordinate (easting).
        maxy (float): Maximum y coordinate (northing).
    """
    command = f"gdalwarp -te {minx} {miny} {maxx} {maxy} {input_file} {output_file}"
    subprocess.run(command, shell=True, check=True)
    print("Cropping completed.")

def get_region_shape_bounds(filepath, region_nr, output_path):
    # Load the shapefile
    usa = gpd.read_file(filepath)
    
    # Filter for the specified region
    country = usa[usa['REGION'] == region_nr]
    
    # Explode multipolygons (if any) and get the first part
    region = country.explode(index_parts=False)[0:1]
    
    # Get the bounding box of the region
    bounds = region.total_bounds  # Returns (x_min, y_min, x_max, y_max)
    
    # Extract values from the bounding box
    x_min, y_min, x_max, y_max = bounds
    
    # Plot the region shape
    fig, ax = plt.subplots(figsize=(8, 8))
    region.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=1)
    
    # Plot the bounding box as a rectangle
    bbox = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         edgecolor='brown', facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(bbox)
    
    # Plot the bounding box corners with red dots
    ax.scatter([x_min, x_max, x_min, x_max], [y_min, y_min, y_max, y_max], color='red', zorder=5)
    
    # Set axis labels and title
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_title(f'Region {region_nr} with Bounding Box')
    
    # Save the figure to the specified output path
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    # Return the bounding box coordinates
    return x_min, y_min, x_max, y_max

def load_and_plot_tif_with_shape(tif_filepath, shapefile_gdf, output_path, region_id):
    """
    Plot the raster and the shapefile boundary.
    
    Args:
        tif_filepath (str): Path to the TIFF file.
        shapefile_gdf (GeoDataFrame): GeoDataFrame of the shapefile.
        output_path (str): Path to save the figure.
    """
    # Load the raster using rasterio
    with rasterio.open(tif_filepath) as src:
        # Plot the raster data using rasterio's show function
        fig, ax = plt.subplots(figsize=(10, 10))
        show(src, ax=ax, cmap='viridis', title=f"TCC Raster and Region {region_id} Boundary")

        # Plot the shapefile boundary on top
        shapefile_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label=f'Region {region_id} Boundary')
        
        # Improve aspect ratio and styling
        ax.set_aspect('equal', 'box')
        ax.legend(handles=[Patch(color='red', label=f'Region {region_id} Boundary')], loc='upper right')

        # Save the plot to the specified output path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.plot()
        plt.close(fig)

def main():
    print("Set up environment variables ...")
    # Load environment variables from a .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)
    
    # Get the region from environment variables and pad it with leading zero if necessary
    region = os.getenv('REGION')
    if region is None:
        raise ValueError("REGION environment variable is not set")

    print(f"Working on USDA Region {region} ...")
    region_id = str(region).zfill(2)

    # Define file paths
    output_dir = os.getenv('TCC_PATH')  # Fetch the TCC_PATH from environment variables
    if output_dir is None:
        raise ValueError("TCC_PATH environment variable is not set")
    output_dir = output_dir.rstrip('/') + '/'

    input_raster_file = output_dir + "nlcd_tcc_conus_2017_v2021-4.tif"
    output_file_resampled = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m.tif"

    output_file_epsg4326 = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326.tif"
    output_file_epsg27705 = output_dir + "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_27705.tif"

    output_file_cropped_epsg4326 = output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_{region_id}.tif"
    output_file_cropped_epsg27705 = output_dir + f"wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_27705_cropped_region_{region_id}.tif"

    region_shape_path_epsg4326 = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion.shp"
    region_shape_path_epsg27705 = f"{os.getenv('REGION_SHAPE')}S_USA.AdministrativeRegion_EPSG27705.shp"

    output_path_figure_epsg4326 = output_dir + f"bounds_epsg4326_region_{region_id}.png"
    output_path_final_figure_epsg4326 = output_dir + f"output_epsg4326_region_{region_id}.png"

    output_path_figure_epsg27705 = output_dir + f"bounds_epsg27705_region_{region_id}.png"
    output_path_final_figure_epsg27705 = output_dir + f"output_epsg27705_region_{region_id}.png"

    # CRS definitions
    crs_na = '+proj=aeqd +lat_0=52 +lon_0=-97.5 +x_0=8264722.17686 +y_0=4867518.35323 +datum=WGS84 +units=m +no_defs'
    crs_epsg27705 = 'EPSG:27705'
    crs_epsg4326 = 'EPSG:4326'
    
    
    # # Step 1: Change the resolution to 20x20 meters
    # print("Step 1: Changing the resolution to 20x20 meters...")
    # change_resolution(input_raster_file, output_file_resampled, (20, 20))



    print("Step 2: Handeling EPSG:4326 ...")
   
    # print("Step 2.1: Reprojecting the file to EPSG:4326...")
    # reproject_to_crs(output_file_resampled, output_file_epsg4326, crs_epsg4326)

    print(f"Step 2.2: Extracting bounds for Region {region_id}  with EPSG:4326 ...")
    minx, miny, maxx, maxy = get_region_shape_bounds(region_shape_path_epsg4326, region_id, output_path_figure_epsg4326)
    print(f"  Bounds: {minx}, {miny}, {maxx}, {maxy}")
    
    print("Step 2.3: Cropping the raster based on the shapefile bounds with EPSG:4326...")
    crop_to_bounds(output_file_epsg4326, output_file_cropped_epsg4326, minx, miny, maxx, maxy)
    
    print("Step 2.4: Plotting the final result with EPSG:4326...")
    usa = gpd.read_file(region_shape_path_epsg4326)
    country = usa[usa['REGION'] == region_id]
    region = country.explode(index_parts=False)[0:1]
    load_and_plot_tif_with_shape(output_file_cropped_epsg4326, region, output_path_final_figure_epsg4326, region_id)


    print("Step 3: Handeling EPSG:27705 ...")

    # print("Step 3.1: Reprojecting the file to EPSG:27705...")
    # reproject_to_crs(output_file_resampled, output_file_epsg27705, crs_epsg27705)

    print(f"Step 3.2: Extracting bounds for Region {region_id}  with EPSG:27705 ...")
    minx, miny, maxx, maxy = get_region_shape_bounds(region_shape_path_epsg27705, region_id, output_path_figure_epsg27705)
    print(f"  Bounds: {minx}, {miny}, {maxx}, {maxy}")
    
    print("Step 3.3: Cropping the raster based on the shapefile bounds with EPSG:27705...")
    crop_to_bounds(output_file_epsg27705, output_file_cropped_epsg27705, minx, miny, maxx, maxy)

    print("Step 3.4: Plotting the final result with EPSG:27705...")
    usa = gpd.read_file(region_shape_path_epsg27705)
    country = usa[usa['REGION'] == region_id]
    region = country.explode(index_parts=False)[0:1]
    load_and_plot_tif_with_shape(output_file_cropped_epsg27705, region, output_path_final_figure_epsg27705, region_id)

    print("Preprocessing completed.")

if __name__ == "__main__":
    main()
