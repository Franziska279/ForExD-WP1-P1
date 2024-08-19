
"""
Script for creating and visualizing a downsampled Tree Canopy Cover (TCC) map for Region 8 
of the United States. This script performs the following steps:
1. Load USA mainland and Region 8 shape data.
2. Load forest disturbance data.
3. Create a downsampled TCC map for Region 8.
4. Load the downsampled TCC map.
5. Plot and save the TCC map with disturbances and region outlines.

Functions:
- normalize_tcc: Normalize the 'tcc' values in the cropped forest data to range between 0 and 100.
- plot_mainland_map: Plot the entire USA mainland with Region 8 highlighted.
- create_custom_colormap: Create a custom colormap for the TCC plot.
- plot_tcc_map: Plot the TCC map within Region 8 boundaries.
- plot_disturbance_types: Plot disturbance types within Region 8 with corresponding colors and white edges.
- plot_figure_1_2: Plot the TCC map with disturbance types and save the figure.
- main: Main function to orchestrate loading data, creating the TCC map, and plotting the results.
"""

import sys
import os
sys.path.insert(1, '../Tools/')
import argparse 
from shapely.geometry import box
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import xarray as xr
import rioxarray
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from shapely import wkt


custom_colors = {
    'wind': '#1f77b4',      # tab:blue
    'fire': '#d62728',      # tab:red
    'defoliators': '#2ca02c',  # tab:green
    'drought': '#FFBA08', # tab:yellow
    'bark_beetle': '#714709'  # tab:brown
}


def get_mainland(gdf_path):
    """
    Cleans the mainland parts of specified regions from the given GeoDataFrame.
    
    Parameters:
        gdf_path (str): Path to the input GeoDataFrame file.
    
    Returns:
        GeoDataFrame: Cleaned GeoDataFrame with mainland parts of specified regions.
    """
    gdf = gpd.read_file(gdf_path)
    regions_to_clean = ['05', '10']
    cleaned_parts = []

    for region in regions_to_clean:
        region_gdf = gdf[gdf['REGION'] == region]
        exploded = region_gdf.explode(index_parts=True)
        exploded['area'] = exploded.area
        # Get the mainland part with the maximum area
        mainland_part = exploded.loc[exploded['area'].idxmax()]
        cleaned_region = exploded[exploded['area'] == mainland_part['area']]
        cleaned_parts.append(cleaned_region)

    cleaned_parts_gdf = gpd.GeoDataFrame(pd.concat(cleaned_parts, ignore_index=True))

    # Combine cleaned parts with the rest of the GeoDataFrame
    #usa_mainland = gdf[~gdf['REGION'].isin(regions_to_clean)].append(cleaned_parts_gdf, ignore_index=True)

    # Update mainland by removing small parts and adding cleaned parts
    usa_mainland = pd.concat([gdf[~gdf['REGION'].isin(regions_to_clean)], cleaned_parts_gdf], ignore_index=True)

    return usa_mainland


def get_region_8(path):
    """
    Extracts the first part of REGION 08 from the given GeoDataFrame.
    
    Parameters:
        path (str): Path to the input GeoDataFrame file.
    
    Returns:
        GeoDataFrame: GeoDataFrame with the first part of REGION 08.
    """
    usa = gpd.read_file(path)
    region_8 = usa[usa['REGION'] == '08']
    # Explode the geometries and reset the index to get the first part
    region_8_exploded = region_8.explode(index_parts=True).reset_index(drop=True)
    return region_8_exploded.iloc[[0]]

def load_refdm_dataset(refdm_path):
    """
    Load and process the REFDM dataset by dissolving it based on the USDA_IDX column.

    Parameters:
        refdm_path (str): Path to the REFDM shapefile.
    
    Returns:
        GeoDataFrame: Processed REFDM GeoDataFrame with unique events.
    """
    # Load the shapefile using geopandas
    refdm_dataset = gpd.read_file(refdm_path)

    # Print CRS and dataset size
    print("CRS:", refdm_dataset.crs)
    print(f"Size of refdm_dataset: {len(refdm_dataset)}")

    # Dissolve the dataset by the USDA_IDX column
    refdm_dissolved = refdm_dataset.dissolve(by='USDA_IDX')
    print(f"Size of unique refdm_dataset events: {len(refdm_dissolved)}")

    # Reset the index
    refdm_dissolved.reset_index(inplace=True)
    
    return refdm_dissolved


def load_ids_dataset(path):
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf_ids = gdf.rename(columns={'index_usda': 'USDA_IDX'})
    return gdf_ids


def load_tcc_dataset(tcc_nc_path):
    """
    Load and process the REFDM dataset by dissolving it based on the USDA_IDX column.

    Parameters:
        refdm_path (str): Path to the REFDM shapefile.
    
    Returns:
        GeoDataFrame: Processed REFDM GeoDataFrame with unique events.
    """
    # Load the shapefile using geopandas
    tcc_dataset = xr.open_dataset(tcc_nc_path)
    return tcc_dataset


def create_downsampled_tcc_map(forest_map_path, area_path, forest_map_downsampled_path, forest_map_downsampled_path_final):

    try:
        print("Step 1: Get Region 8 geometry ...")
        # Get Region 8 geometry
        r8_geometry = get_region_8(area_path)
        r8_union = r8_geometry.unary_union

        print("Step 2: Load the forest map TIFF file ...")
        # Load the forest map TIFF file
        forest_map = rioxarray.open_rasterio(forest_map_path, masked=True).squeeze()

        print("Step 3: Ensure the CRS is EPSG:4326")
        # Ensure the CRS is EPSG:4326
        forest_map = forest_map.rio.write_crs("EPSG:4326")

        print("Step 4: Coarsen the data to reduce memory usage ...")
        factor = 100  # Adjust this factor as needed to reduce memory usage
        forest_map = forest_map.coarsen(x=factor, y=factor, boundary='trim').mean()

        print("Step 5: Crop the forest map to Region 8 ...")
        # Crop the forest map to Region 8
        forest_map_downsampled_cropped = forest_map.rio.clip([r8_union], forest_map.rio.crs, drop=True, from_disk=True)

        # Path to the cropped NetCDF file
        forest_map_downsampled_cropped.to_netcdf(forest_map_downsampled_path)
        print(f"Step 6: Cropped NetCDF file saved to {forest_map_downsampled_path}")

        print("Step 7: Load the cropped NetCDF file using xarray ...")
        loaded_tcc_region_8 = xr.open_dataset(forest_map_downsampled_path)

        print("Step 8: Restructure the data")
        # Rename the variable from __xarray_dataarray_variable__ to tcc
        loaded_tcc_region_8 = loaded_tcc_region_8.rename({'__xarray_dataarray_variable__': 'tcc'})
        # Remove the spatial_ref variable
        loaded_tcc_region_8 = loaded_tcc_region_8.drop_vars('spatial_ref')

        print("Step 9: Save the final NetCDF file ...")
        loaded_tcc_region_8.to_netcdf(forest_map_downsampled_path_final, mode='w')
        print(f"Step 10: Saved final NetCDF file to {forest_map_downsampled_path_final}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        # Delete intermediate file
        if os.path.exists(forest_map_downsampled_path):
            os.remove(forest_map_downsampled_path)
            print(f"Step 11: Deleted intermediate file: {forest_map_downsampled_path}")


def normalize_tcc(cropped_forest):
    """
    Normalize the 'tcc' values in the cropped forest data to range between 0 and 100.
    """
    cropped_forest['tcc'] = (cropped_forest['tcc'] / cropped_forest['tcc'].max()) * 100
    cropped_forest['tcc'] = cropped_forest['tcc'].clip(min=0, max=100)
    return cropped_forest

def plot_mainland_map(ax, usa_mainland):
    """
    Plot the entire USA mainland with Region 8 highlighted.
    """
    usa_mainland[usa_mainland['REGION'] != '08'].plot(ax=ax, color='grey', edgecolor='grey')
    usa_mainland[usa_mainland['REGION'] == '08'].plot(ax=ax, color='black', edgecolor='black')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.axis('on')

def create_custom_colormap():
    """
    Create a custom colormap for the TCC plot.
    """
    cmap = plt.colormaps['Greens']
    new_colors = cmap(np.linspace(0, 1, 100))
    new_colors[0, :] = [1, 1, 1, 1]  # Set the first color (corresponding to 0) to white
    return LinearSegmentedColormap.from_list('CustomGreens', new_colors)

def plot_tcc_map(ax, cropped_forest, custom_cmap):
    """
    Plot the TCC map within Region 8 boundaries.
    """
    plot = cropped_forest['tcc'].plot(ax=ax, cmap=custom_cmap, add_colorbar=False)
    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.05, aspect=10, shrink=0.8)
    #cbar.ax.set_position([0.15, 0.2, 0., 0.03])  # [left, bottom, width, height]
    cbar.ax.set_position([0.35, 0.31, 0.35, 0.03])  # [left, bottom, width, height]
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0', '25', '50', '75', '100'])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Tree Canopy Cover (%)', fontsize=16, labelpad=6)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.label.set_size(16)
    cbar.ax.xaxis.labelpad = 10

def plot_disturbance_types(ax, refdm_dissolved, custom_colors):
    """
    Plot disturbance types within Region 8 with corresponding colors and white edges.
    """
    for disturbance, color in custom_colors.items():
        # Plot with white edge first
        refdm_dissolved[refdm_dissolved['DCA_ID'] == disturbance].plot(
            ax=ax, linewidth=3.5, color=color, edgecolor='white'
        )
        # Then plot with actual color and thinner edge
        refdm_dissolved[refdm_dissolved['DCA_ID'] == disturbance].plot(
            ax=ax, linewidth=2.5, color=color, edgecolor=color
        )

def plot_figure_1(cropped_forest, usa_mainland, r8, refdm_dissolved, custom_colors, save_dir):
    """
    Plot the TCC map with disturbance types and save the figure.
    """
    # Normalize the TCC values
    cropped_forest = normalize_tcc(cropped_forest)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot the entire USA in grey in the upper left corner
    sub_ax = fig.add_axes([0.005, 0.70, 0.25, 0.25])  # [left, bottom, width, height]
    plot_mainland_map(sub_ax, usa_mainland)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Plot the TCC map within Region 8 boundaries
    plot_tcc_map(ax, cropped_forest, custom_cmap)
    
    # Plot the region outline
    r8.boundary.plot(ax=ax, linewidth=2, color='#297045')
    
    # Plot disturbance types
    plot_disturbance_types(ax, refdm_dissolved, custom_colors)
    
    # Customize the plot
    ax.axis('off')  # Remove axis and frame
    
    # Create legend for disturbance types
    legend_patches = [mpatches.Patch(color=color, label=disturbance.capitalize()) for disturbance, color in custom_colors.items()]
    ax.legend(handles=legend_patches, fontsize=18, title="Disturbance Type", title_fontsize=20, loc='center left', facecolor='white', framealpha=1)
    
    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'p1_f1_ids.png'), bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to orchestrate loading data, creating the TCC map, and plotting the results.
    """
    forest_map_path = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/wp1_nlcd_tcc_conus_2017_v2021_4_20m_4326_cropped_region_08.tif"
    forest_map_downsampled_path = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/intermediate_tcc_map_region_8.nc"
    area_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/data/S_USA.AdministrativeRegion/S_USA.AdministrativeRegion.shp"
    refdm_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping.shp"
    ids_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"
    tcc_map_region_8 = "/Net/Groups/BGI/work_2/ForExD/WP1/Data/nlcd_tcc_CONUS_2017_v2021-4/tcc_map_region_8.nc"
    figure_dir = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/figures/"

    print("Load the USA Mainland and Region 8 Shape ...")
    mainland = get_mainland(area_path)
    region_8 = get_region_8(area_path)

    print("Load the Forest Disturbances ...")
    #refdm_dissolved = load_refdm_dataset(refdm_path)
    ids =load_ids_dataset(ids_path)

    # Uncomment the following line to create the downsampled TCC map
    # create_downsampled_tcc_map(forest_map_path, area_path, forest_map_downsampled_path, tcc_map_region_8)
    
    print("Load the TCC Region 8 Map ...")
    tcc_dataset = load_tcc_dataset(tcc_map_region_8)
    
    print("Plot Study area figure ...")
    plot_figure_1(tcc_dataset, mainland, region_8, ids, custom_colors, save_dir=figure_dir)


if __name__ == "__main__":
    main()
