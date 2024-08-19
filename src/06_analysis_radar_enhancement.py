
"""
Script for creating and visualizing the analysis on how or if the radar enhanced dataset improves the disturbances dataset:

This includes the creation of a map of the study area the Region 8 of the United States, with the following steps:
1. Load USA mainland and Region 8 shape data.
2. Load forest disturbance data.
3. Create a downsampled TCC map for Region 8.
4. Load the downsampled TCC map.
5. Plot and save the TCC map with disturbances and region outlines.

Secondly an analysis to deduce what disturbance type is well detected by radar and wich ones are not.

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
from matplotlib.gridspec import GridSpec
from shapely import wkt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches  # Import for custom legend
from shapely.geometry import Point
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches  # Import for custom legend
import os
import pandas as pd
import numpy as np

# Function to adjust the color brightness
def adjust_color_brightness(color, amount=0.5):
    c = mcolors.ColorConverter().to_rgb(color)
    return mcolors.to_hex([min(1, max(0, c[i] * amount)) for i in range(3)])



custom_colors = {
    'wind': '#1f77b4',      # tab:blue
    'fire': '#d62728',      # tab:red
    'defoliators': '#BF40BF',  # tab:green
    'drought': '#FFBA08', # tab:yellow
    'bark_beetle': '#714709'  # tab:brown
}

def format_label(label):
    return ' '.join(word.capitalize() for word in label.split('_'))


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

def load_dissolved_refdm_dataset(refdm_path):
    """
    Load and process the REFDM dataset by dissolving it based on the USDA_IDX column.

    Parameters:
        refdm_path (str): Path to the REFDM shapefile.
    
    Returns:
        GeoDataFrame: Processed REFDM GeoDataFrame with unique events.
    """
    # Load the shapefile using geopandas
    refdm_dataset = gpd.read_file(refdm_path)

    # Dissolve the dataset by the USDA_IDX column
    refdm_dissolved = refdm_dataset.dissolve(by='USDA_IDX')
    print(f"Size of unique refdm_dataset events: {len(refdm_dissolved)}")

    # Reset the index
    refdm_dissolved.reset_index(inplace=True)
    
    return refdm_dataset, refdm_dissolved


def load_ids_dataset(path):
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf_ids = gdf.rename(columns={'index_usda': 'USDA_IDX'})
    gdf_ids['centroid_shift_m'] = 0
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
            ax=ax, linewidth=2.5, color=color, edgecolor='white'
        )
        # Then plot with actual color and thinner edge
        refdm_dissolved[refdm_dissolved['DCA_ID'] == disturbance].plot(
            ax=ax, linewidth=1.5, color=color, edgecolor=color
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
    r8.boundary.plot(ax=ax, linewidth=1, color='black')
    
    # Plot disturbance types
    plot_disturbance_types(ax, refdm_dissolved, custom_colors)
    
    # Customize the plot
    ax.axis('off')  # Remove axis and frame
    
    legend_patches = [mpatches.Patch(color=color, label=format_label(disturbance)) for disturbance, color in custom_colors.items()]
    ax.legend(handles=legend_patches, fontsize=18, title="Disturbance Type", title_fontsize=20, loc='center left', facecolor='white', framealpha=1)

    
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()


def create_study_area_map(area_path, ids, tcc_map_region_8, path):
    """
    Function to orchestrate loading data, creating the TCC map, and plotting the study area map.
    """

    print("Load the USA Mainland and Region 8 Shape ...")
    mainland = get_mainland(area_path)
    region_8 = get_region_8(area_path)


    # Uncomment the following line to create the downsampled TCC map
    # create_downsampled_tcc_map(forest_map_path, area_path, forest_map_downsampled_path, tcc_map_region_8)
    
    print("Load the TCC Region 8 Map ...")
    tcc_dataset = load_tcc_dataset(tcc_map_region_8)
    
    print("Plot Study area figure ...")
    plot_figure_1(tcc_dataset, mainland, region_8, ids, custom_colors, save_dir=path)



def plot_radar_reduction_potential(refdm_gdf, ids_gdf, path):
    # Define font sizes and bar parameters
    title_fontsize = 34
    legend_title_fontsize = 30
    label_fontsize = 28
    legend_fontsize = 28
    tick_fontsize = 26
    annotation_fontsize = 26
    bar_width = 0.35  # Width of the bars
    double_bar_width = bar_width * 2  # Make the bottom bars as wide as the combined width of the top two bars
    bar_offset = 0.2  # Offset to move the lower bars to the right

    dca_counts_refdm = refdm_gdf['DCA_ID'].value_counts()
    dca_counts_ids = ids_gdf['DCA_ID'].value_counts()

    # Combine the counts into a single DataFrame
    counts_df = pd.DataFrame({
        'IDS': dca_counts_ids,
        'REFDM': dca_counts_refdm
    }).fillna(0)  # Fill NaN with 0 for counts that are missing in either dataset

    # Reset index to turn DCA_ID into a column
    counts_df.reset_index(inplace=True)
    counts_df.rename(columns={'index': 'DCA_ID'}, inplace=True)
    # Calculate reduction percentage
    counts_df['Reduction (%)'] = -100 * (counts_df['IDS'] - counts_df['REFDM']) / counts_df['IDS']

    # Ensure the DCA_ID is in the specified order
    counts_df['DCA_ID'] = pd.Categorical(counts_df['DCA_ID'], categories=['bark_beetle', 'wind', 'fire', 'defoliators', 'drought'], ordered=True)
    counts_df_sorted = counts_df.sort_values('DCA_ID')

    # Capitalize DCA_ID labels
    dca_labels = [format_label(label) for label in counts_df_sorted['DCA_ID']]

    # Create a figure with 2 rows and 1 column for the two subplots
    fig = plt.figure(figsize=(24, 12))  # Increase the figure height
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[5, 2])  # Adjust the height ratios

    # Plot Counts in the first subplot
    ax1 = fig.add_subplot(gs[0])
    bar_positions = range(len(counts_df_sorted))  # X positions for bars

    ax1.bar(bar_positions, counts_df_sorted['IDS'], width=bar_width, color="#BCB6FF", label='IDS')
    ax1.bar([pos + bar_width for pos in bar_positions], counts_df_sorted['REFDM'], width=bar_width, color="#AF42AE", label='REFDM')

    # Add annotations for counts above bars
    for i, (count_ids, count_refdm) in enumerate(zip(counts_df_sorted['IDS'], counts_df_sorted['REFDM'])):
        ax1.text(bar_positions[i], count_ids + 2, str(int(count_ids)), ha='center', va='bottom', color='black', fontsize=annotation_fontsize)
        ax1.text(bar_positions[i] + bar_width, count_refdm + 2, str(int(count_refdm)), ha='center', va='bottom', color='black', fontsize=annotation_fontsize)

    # Set labels and title for the first subplot
    ax1.set_ylabel('Amount of Disturbance Events', fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding
    ax1.set_title('Radar Detection Potential per Disturbance Type', fontsize=title_fontsize, pad=20)  # Increase font size for title
    ax1.set_yscale('log')
    ax1.set_ylim(1, counts_df_sorted[['IDS', 'REFDM']].max().max() * 2)  # Set y-limit for log scale
    legend = ax1.legend(fontsize=legend_fontsize, title='Datasets')  # Increase font size for legend and add title
    legend.get_title().set_fontsize(legend_title_fontsize)
    ax1.grid(False)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(bar_positions, dca_labels, fontsize=tick_fontsize)
    ax1.tick_params(axis='x', which='major', pad=15)
    
    # Plot Reduction (%) in the second subplot with negative y-axis
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar([pos + bar_offset for pos in bar_positions], counts_df_sorted['Reduction (%)'], width=double_bar_width, color='#FF3E41', label='Reduction (%)')

    # Add annotations for reduction below bars
    for i, reduction_percentage in enumerate(counts_df_sorted['Reduction (%)']):
        ax2.text(bar_positions[i] + bar_offset, reduction_percentage - 2, f'{reduction_percentage:.2f}%', ha='center', va='top', color='black', fontsize=annotation_fontsize)

    # Set labels and title for the second subplot
    ax2.set_xlabel('Disturbance Type', fontsize=label_fontsize)  # Increase font size for xlabel
    ax2.set_ylim(0, -75)
    ax2.invert_yaxis()
    ax2.set_ylabel('Reduction \nPercentage (%)', fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding

    # Set y-axis ticks to only show 4 ticks
    ax2.set_yticks([0, -20, -40, -60])
    ax2.set_yticklabels(['0', '-20', '-40', '-60'])

    plt.xticks([pos + bar_offset for pos in bar_positions], dca_labels, fontsize=tick_fontsize, ha='right')  # Adjust the rotation and alignment

    # Rotate x-axis labels for ax2
    ax2.set_xticklabels(dca_labels, ha='right', fontsize=0)  # Rotate and increase font size
    ax2.grid(False)

    # Adjust x-axis ticks and labels
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()  # Ensures labels, titles, and legends do not overlap
    
    plt.savefig(path, bbox_inches='tight')


def calculate_area_and_centroid_meters(gdf):
    """Calculate area in square meters and square kilometers for a GeoDataFrame."""
    # Set the coordinate reference system (CRS) for the GeoDataFrame to WGS84
    gdf.crs = 'EPSG:4326'
    
    # Calculate the area in degrees (not used for final calculation, but keeping it here if needed for other purposes)
    #gdf['area_degrees'] = gdf.geometry.area
    
    # Define the target projection system (e.g., UTM Zone 18N)
    target_crs = 'EPSG:32618'
    
    # Reproject the GeoDataFrame to the target projection system
    gdf_projected = gdf.to_crs(target_crs)
    
    # Calculate the area in square meters
    gdf_projected['area_meters'] = gdf_projected.geometry.area
    gdf_projected['centroid_meters'] = gdf_projected.geometry.centroid
    
    # Assign the calculated area back to the original GeoDataFrame
    gdf['area_meters'] = gdf_projected['area_meters']
    # Convert square meters to square kilometers
    gdf['square_km'] = gdf['area_meters'] / 1e6
    
    gdf['centroid_meters'] = gdf_projected['centroid_meters']
    
    # Return the GeoDataFrame with the new area columns
    return gdf


def calculate_size_shift_difference(ids_gdf, refdm_gdf_dissolved):
    """
    Calculate centroid shift and size difference between polygons in two GeoDataFrames based on USDA_IDX.

    Parameters:
    - ids_gdf (GeoDataFrame): GeoDataFrame containing polygons with USDA_IDX and geometry.
    - refdm_gdf_dissolved (GeoDataFrame): GeoDataFrame containing reference polygons with USDA_IDX and geometry.

    Returns:
    - result_gdf (GeoDataFrame): GeoDataFrame containing USDA_IDX, centroid_shift, size_difference, and geometry.
    """

    # Calculate centroids for each polygon in both GeoDataFrames
    ids = calculate_area_and_centroid_meters(ids_gdf)
    refdm = calculate_area_and_centroid_meters(refdm_gdf_dissolved)

    # Initialize lists to hold the results
    usda_idx_list = []
    centroid_shift_list = []
    size_difference_list = []

    # Iterate through unique USDA_IDX values in ids_gdf
    unique_usda_idx = ids['USDA_IDX'].unique()

    for usda_idx in unique_usda_idx:
        # Get the corresponding rows in both GeoDataFrames for the current USDA_IDX
        ids_row = ids[ids['USDA_IDX'] == usda_idx]
        refdm_row = refdm[refdm['USDA_IDX'] == usda_idx]
        
        # Check if both rows exist (they should exist if data is correctly structured)
        if not ids_row.empty and not refdm_row.empty:
            # Calculate the centroid shift between centroids of the two polygons
            centroid_shift = ids_row.iloc[0]['centroid_meters'].distance(refdm_row.iloc[0]['centroid_meters'])
            
            # Calculate the size difference between areas of the two polygons
            size_difference = refdm_row.iloc[0]['square_km'] - ids_row.iloc[0]['square_km']
            
            # Append the results to the lists
            usda_idx_list.append(usda_idx)
            centroid_shift_list.append(centroid_shift)
            size_difference_list.append(size_difference)

    # Create a DataFrame from the results
    results_df = pd.DataFrame({
        'USDA_IDX': usda_idx_list,
        'centroid_shift_m': centroid_shift_list,
        'size_difference_km2': size_difference_list
    })

    ids_gdf = ids_gdf.drop(columns=['centroid_shift_m'])
    # Merge with original gdf to get geometry
    result_gdf = ids_gdf.merge(results_df, on='USDA_IDX')

    # Convert to GeoDataFrame with original geometry and CRS
    result_gdf = gpd.GeoDataFrame(result_gdf, geometry='geometry', crs=ids_gdf.crs)

    return result_gdf


def plot_combined_areasize_shift_per_disturbances(refdm, ids, path):
    # Define the order of the categories
    category_order = ['bark_beetle','drought', 'wind', 'defoliators', 'fire']


    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = {label: custom_colors.get(label, default_colors.pop(0)) for label in category_order}

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a grid with 5 rows and 4 columns for the plots
    fig, axs = plt.subplots(5, 4, figsize=(35, 40), gridspec_kw={'width_ratios': [4, 0.005, 3, 0.5]})

    # Define font sizes
    fontsize_supertitle = 50
    fontsize_legend = 44
    fontsize_title = 44
    fontsize_label = 44
    fontsize_tick = 35
    padding_label = 40
    padding_title = 35

    for i, category in enumerate(category_order):
        
        # Create combined data for violin plots
        combined_data = pd.concat([
            ids[ids['DCA_ID'] == category][['DCA_ID', 'area_km2', 'centroid_shift_m']].assign(Source='IDS'),
            refdm[refdm['DCA_ID'] == category][['DCA_ID', 'area_km2', 'centroid_shift_m']].assign(Source='REFDM')
        ])

        # Violin plot for size differences (GDF and IDS)
        ax = axs[i, 0]

        # Filter data for the current category (REFDM and IDS)
        data_refdm = combined_data[(combined_data['Source'] == 'REFDM') & (combined_data['DCA_ID'] == category)]
        data_ids = combined_data[(combined_data['Source'] == 'IDS') & (combined_data['DCA_ID'] == category)]

        median_ids = data_ids['area_km2'].median()
        median_refdm = data_refdm['area_km2'].median()
        

        # Define colors for IDS and REFDM
        color_refdm = adjust_color_brightness(custom_palette[category], amount=0.4)
        color_ids = adjust_color_brightness(custom_palette[category], amount=0.8)  # Slightly darker

        # Plot KDE for IDS
        sns.kdeplot(
            data=combined_data[combined_data['Source'] == 'IDS']['area_km2'],
            ax=ax,
            color=color_ids,
            linestyle='-',
            linewidth=5,
            label='IDS'
        )

        # Plot KDE for REFDM
        sns.kdeplot(
            data=combined_data[combined_data['Source'] == 'REFDM']['area_km2'],
            ax=ax,
            color=color_refdm,
            linestyle='-',
            linewidth=5,
            label='REFDM'
        )

        ax.axvline(x=median_ids, color=color_ids, linestyle='--', linewidth=5, label='IDS Median')
        ax.axvline(x=median_refdm, color=color_refdm, linestyle='--', linewidth=5, label='REFDM Median')

        ax.legend(fontsize=fontsize_tick)
        ax.set_ylim(0)
        ax.set_xlim(0)

        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel('km²', fontsize=fontsize_label, labelpad=padding_label)  # Set x-axis label
        else:
            ax.set_xlabel('', labelpad=padding_label)

        if i == 2:  # Only set y-label for the first column
            ax.set_ylabel('Density', fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)
        
        if i == 0:  # Only set y-label for the first column
            ax.set_title('Size of disturbance area', fontsize=fontsize_title, pad=padding_title)

       
        # Distribution plot for centroid shifts
        ax = axs[i, 2]
        sns.histplot(
            data=refdm[refdm['DCA_ID'] == category],
            x='centroid_shift_m',
            kde=True,
            line_kws={'linewidth': 5},  # Adjust the line width of the KDE curve
            color=custom_palette[category],
            ax=ax
        )
        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)

        ax.set_ylim(0)
        ax.set_xlim(0, 1600)

        if i == 2:  # Only set y-label for the first column
            ax.set_ylabel('Amount of Events', fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)
        
        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel('m', fontsize=fontsize_label, labelpad=padding_label)  # Set x-axis label
        else:
            ax.set_xlabel('')

        if i == 0:  # Only set y-label for the first column
            ax.set_title('Shift of Disturbance Location', fontsize=fontsize_title, pad=padding_title)
        
        # Custom legend in fourth column
        ax = axs[i, 3]
        ax.legend(handles=[mpatches.Patch(color=custom_palette[category], label=format_label(category))], loc='center', fontsize=fontsize_legend)
        ax.axis('off')

        axs[i, 1].axis('off')

    # Adjust the layout to place shared labels
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    plt.savefig(path, bbox_inches='tight')




def plot_combined_areasize_shift_per_disturbances_quantiles(refdm, ids, path):
    # Define the order of the categories
    category_order = ['bark_beetle','drought', 'wind', 'defoliators', 'fire']


    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = {label: custom_colors.get(label, default_colors.pop(0)) for label in category_order}

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a grid with 5 rows and 4 columns for the plots
    fig, axs = plt.subplots(5, 4, figsize=(35, 40), gridspec_kw={'width_ratios': [4, 0.005, 3, 0.5]})

    # Define font sizes
    fontsize_supertitle = 50
    fontsize_legend = 44
    fontsize_title = 44
    fontsize_label = 44
    fontsize_tick = 35
    padding_label = 40
    padding_title = 35

    for i, category in enumerate(category_order):
        
        # Create combined data for violin plots
        combined_data = pd.concat([
            ids[ids['DCA_ID'] == category][['DCA_ID', 'area_km2', 'centroid_shift_m']].assign(Source='IDS'),
            refdm[refdm['DCA_ID'] == category][['DCA_ID', 'area_km2', 'centroid_shift_m']].assign(Source='REFDM')
        ])

        # Violin plot for size differences (GDF and IDS)
        ax = axs[i, 0]

        # Filter data for the current category (REFDM and IDS)
        data_refdm = combined_data[(combined_data['Source'] == 'REFDM') & (combined_data['DCA_ID'] == category)]
        data_ids = combined_data[(combined_data['Source'] == 'IDS') & (combined_data['DCA_ID'] == category)]

        median_ids = data_ids['area_km2'].median()
        median_refdm = data_refdm['area_km2'].median()
       
        # Calculate the 10th and 90th percentiles for the x-axis limits
        lower_percentile = np.percentile(combined_data['area_km2'], 2)
        upper_percentile = np.percentile(combined_data['area_km2'], 98)

        # Plot KDE plot for IDS
        sns.kdeplot(
            data=data_ids['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=5,  # Adjust line width
            label='IDS',
            alpha=0.8,  # Adjust transparency for IDS plot
            linestyle='--',  # Dashed line style for IDS
        )

        ax.axvline(x=median_ids, color=custom_palette[category], linestyle='--', linewidth=5, alpha=0.8, label='IDS Median')

        # Plot KDE plot for REFDM
        sns.kdeplot(
            data=data_refdm['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=5,  # Adjust line width
            label='REFDM',
            alpha=0.4,  # Adjust transparency for REFDM plot
            linestyle='-',  # Solid line style for REFDM
        )

        ax.axvline(x=median_refdm, color=custom_palette[category], linestyle='-', linewidth=5, alpha=0.4, label='REFDM Median')
        ax.legend(fontsize=fontsize_tick)

        # Ensure y-axis starts from 0 and ends at 1
        ax.set_ylim(0)
        ax.set_xlim(lower_percentile, upper_percentile)  # Set x-axis limit based on percentiles
        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)

        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel('km²', fontsize=fontsize_label, labelpad=padding_label)  # Set x-axis label
        else:
            ax.set_xlabel('', labelpad=padding_label)

        if i == 2:  # Only set y-label for the first column
            ax.set_ylabel('Density', fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)
        
        if i == 0:  # Only set y-label for the first column
            ax.set_title('Size of disturbance area', fontsize=fontsize_title, pad=padding_title)

       
        # Distribution plot for centroid shifts
        ax = axs[i, 2]
        sns.histplot(
            data=refdm[refdm['DCA_ID'] == category],
            x='centroid_shift_m',
            kde=True,
            line_kws={'linewidth': 5},  # Adjust the line width of the KDE curve
            color=custom_palette[category],
            ax=ax
        )
        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)

        ax.set_ylim(0)
        ax.set_xlim(0, 1600)

        if i == 2:  # Only set y-label for the first column
            ax.set_ylabel('Amount of Events', fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)
        
        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel('m', fontsize=fontsize_label, labelpad=padding_label)  # Set x-axis label
        else:
            ax.set_xlabel('')

        if i == 0:  # Only set y-label for the first column
            ax.set_title('Shift of Disturbance Location', fontsize=fontsize_title, pad=padding_title)
        
        # Custom legend in fourth column
        ax = axs[i, 3]
        ax.legend(handles=[mpatches.Patch(color=custom_palette[category], label=format_label(category))], loc='center', fontsize=fontsize_legend)
        ax.axis('off')

        axs[i, 1].axis('off')

    # Adjust the layout to place shared labels
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    plt.savefig(path, bbox_inches='tight')


def plot_size_shift_comparison_errorbars(gdf, path):

    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = [custom_colors.get(label, default_colors.pop(0)) for label in gdf['DCA_ID'].unique()]

    # Calculate medians and quantiles for each disturbance type
    grouped = gdf.groupby('DCA_ID').agg({
        'centroid_shift_m': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'size_difference_km2': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
    }).reset_index()

    # Flatten the column names after aggregation
    grouped.columns = ['DCA_ID', 'centroid_shift_median', 'centroid_shift_q25', 'centroid_shift_q75', 'size_diff_median', 'size_diff_q25', 'size_diff_q75']

    # Set Seaborn style
    sns.set(style="whitegrid")

    plt.figure(figsize=(14,10))

    # Show quadrant lines
    plt.axhline(0, color='black', linewidth=2.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=2.5, linestyle='--')


    # Plot the median values and IQR with shaded regions
    for i, row in grouped.iterrows():
        plt.scatter(
            row['centroid_shift_median'],
            row['size_diff_median'],
            color=custom_colors.get(row['DCA_ID'], 'grey'),
            s=500,  # Larger scatter points
            edgecolor='w',
            alpha=0.7,
            label=row['DCA_ID']
        )
        plt.errorbar(
            row['centroid_shift_median'],
            row['size_diff_median'],
            xerr=[[row['centroid_shift_median'] - row['centroid_shift_q25']], [row['centroid_shift_q75'] - row['centroid_shift_median']]],
            yerr=[[row['size_diff_median'] - row['size_diff_q25']], [row['size_diff_q75'] - row['size_diff_median']]],
            fmt='o',
            color=custom_colors.get(row['DCA_ID'], 'grey'),
            alpha=0.8,
            capsize=10,  # Larger error bar caps
            capthick=5.5,
            linewidth=5.5  # Thicker error bar lines
        )

    # Customize the plot
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel('Position Shift (m)', fontsize=20)
    plt.ylabel('Size Difference (km²)', fontsize=20)
    plt.title('Area and Position Changes per Disturbance Types', fontsize=26, pad=20)

    # Set legend
    # Get current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create a dictionary with formatted labels
    by_label = dict(zip(labels, handles))
    # Format the labels
    formatted_labels = [format_label(label) for label in by_label.keys()]
    # Create the legend with formatted labels
    legend = plt.legend(by_label.values(), formatted_labels, title='Disturbance Types', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    legend.set_title('Disturbance Types', prop={'size': 18})  # Set legend title size

    # Set decimal tick labels
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Increase tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def plot_length_of_disturbance_events(refdm_dataset, path):

    dca_ids = refdm_dataset['DCA_ID'].unique()

    # Define the width of each bar
    bar_width = 0.8

    # Create subplots for each DCA_ID with increased height
    fig, axs = plt.subplots(1, len(dca_ids), figsize=(20, 6), sharex=True)

    # Iterate over each DCA_ID
    for i, dca_id in enumerate(dca_ids):
        # Filter refdm_dataset for the current DCA_ID
        dca_data = refdm_dataset[refdm_dataset['DCA_ID'] == dca_id]
        
        # Group the data by USDA_IDX and count occurrences
        usda_idx_counts = dca_data.groupby('USDA_IDX').size()
        
        # Count occurrences of USDA_IDX with 1, 2, or 3 instances
        usda_idx_1_instance = sum(usda_idx_counts == 1)
        usda_idx_2_instances = sum(usda_idx_counts == 2)
        usda_idx_3_instances = sum(usda_idx_counts == 3)
        
        # Calculate the x positions for the bars
        x = np.arange(3)
        
        # Get the base color for the current DCA_ID
        color = custom_colors.get(dca_id, '#000000')
        
        # Plot the bars for each year without offset
        bars = axs[i].bar(x, [usda_idx_1_instance, usda_idx_2_instances, usda_idx_3_instances], width=bar_width, color=color)
        
        # Set plot title with larger font size
        axs[i].set_title(f'{format_label(dca_id)}', fontsize=20)

        # Set y-axis label only for the first subplot
        if i == 0:
            axs[i].set_ylabel('Number of Events', fontsize=20)
        
        # Set x-axis tick labels and rotation
        axs[i].set_xticks(x)
        axs[i].tick_params(axis='x', labelsize=18, rotation=0)
        axs[i].tick_params(axis='y', labelsize=18)

        # Set x-tick labels
        axs[i].set_xticklabels(['1', '2', '3'], fontdict={'fontsize': 18})

    # Set common x-axis label with larger font size
    fig.text(0.5, -0.05, 'Number of Appearances (years)', ha='center', fontsize=20)
    fig.tight_layout()

    plt.savefig(path, bbox_inches='tight')



def main():

    """
    Main function to orchestrate loading data, creating the TCC map, and plotting the results.
    """
    data_dir = "/Net/Groups/BGI/work_2/ForExD/WP1/Data"
    result_dir = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/"
    save_dir = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/figures/"

    area_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/data/S_USA.AdministrativeRegion/S_USA.AdministrativeRegion.shp"
    tcc_map_region_8 = os.path.join(data_dir, "nlcd_tcc_CONUS_2017_v2021-4/tcc_map_region_8.nc")
    refdm_path = os.path.join(result_dir, "radar_enhanced_forest_disturbance_mapping.shp")
    ids_path = os.path.join(result_dir, "region8_dca_filtered_ids_usda_polygons.csv")

    figure_1 = "p1_f1_ids.png"
    figure_reduction_potential = "p1_f2_radar_potential_analysis.png"
    figure_size_shift = "p1_f3_size_shift_analysis.png"
    figure_size_shift_quantiles = "p1_f4_size_shift_errorbars_analysis_quantiles.png"
    figure_size_shift_error = "p1_f5_size_shift_errorbars_analysis.png"
    figure_disturbance_duration = "p1_f6_disturbance_duration_analysis.png"

     # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    print("Script to analyse the radar enhanced forest disturbance dataset:\n")

    try:
        # Step 1: Loading CSV file and converting to GeoDataFrame
        print("Step 1: Loading data ...")
        refdm, refdm_dissolved = load_dissolved_refdm_dataset(refdm_path)
        ids = load_ids_dataset(ids_path)


        print("Step 2: Plotting study area ...")
        create_study_area_map(area_path, ids, tcc_map_region_8, path=os.path.join(save_dir, figure_1))

        print("Step 3: Plotting Radar potential ...")
        plot_radar_reduction_potential(refdm_dissolved, ids, path=os.path.join(save_dir, figure_reduction_potential))

        print("Step 4: Plotting Size and Location Difference and Density ...")
        result_gdf = calculate_size_shift_difference(ids, refdm_dissolved)
        plot_combined_areasize_shift_per_disturbances(result_gdf, ids, path=os.path.join(save_dir, figure_size_shift))
        plot_combined_areasize_shift_per_disturbances_quantiles(result_gdf, ids, path=os.path.join(save_dir, figure_size_shift_quantiles))
        plot_size_shift_comparison_errorbars(result_gdf, path=os.path.join(save_dir, figure_size_shift_error))

        print("Step 5: Plotting Duration of Disturbance Events ...")
        plot_length_of_disturbance_events(refdm, path=os.path.join(save_dir, figure_disturbance_duration))

        print("Main process completed successfully.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Log the error or take appropriate action


if __name__ == "__main__":
    main()