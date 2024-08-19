import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib import cm
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import xarray as xr
from shapely.geometry import mapping, shape, MultiPolygon, box, Point
from affine import Affine
import rasterio
from shapely import wkt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches  # Import for custom legend



custom_colors = {
    'wind': '#1f77b4',      # tab:blue
    'fire': '#d62728',      # tab:red
    'defoliators': '#2ca02c',  # tab:green
    'drought': '#FFBA08', # tab:yellow
    'bark_beetle': '#714709'  # tab:brown
}

def plot_radar_reduction_potential(refdm_gdf, ids_gdf, save_dir):
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
    dca_labels = [label.capitalize() for label in counts_df_sorted['DCA_ID']]

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
    
    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'ids_refdm_radar_reduction_potential.png'), bbox_inches='tight')

    plt.show()


def calculate_area(gdf):
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
    print("     Calculating area in square meters...")
    gdf_projected['area_meters'] = gdf_projected.geometry.area
    
    # Assign the calculated area back to the original GeoDataFrame
    gdf['area_meters'] = gdf_projected['area_meters']
    # Convert square meters to square kilometers
    gdf['square_km'] = gdf['area_meters'] / 1e6
    # Print step information
    print("     Converting area from square meters to square kilometers...")
    
    # Return the GeoDataFrame with the new area columns
    return gdf


def plot_length_of_disturbance_events(refdm_dataset, save_dir):

    dca_ids = refdm_dataset['DCA_ID'].unique()

    # Function to create shades of a color
    def create_shades(color, n):
        base_color = np.array(mcolors.to_rgb(color))
        return [mcolors.to_hex(base_color * (1 - 0.2 * i)) for i in range(n)]

    # Define the width of each bar
    bar_width = 0.2

    # Extract unique DCA_IDs from the dataset
    dca_ids = refdm_dataset['DCA_ID'].unique()

    # Create subplots for each DCA_ID
    fig, axs = plt.subplots(1, len(dca_ids), figsize=(20, 4), sharex=True)

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
        
        # Get the color shades for the current DCA_ID
        shades = create_shades(custom_colors.get(dca_id, '#000000'), 3)
        
        # Plot the bars for each year
        axs[i].bar(x - bar_width, [usda_idx_1_instance, 0, 0], width=bar_width, label='1 Year', color=shades[0])
        axs[i].bar(x, [0, usda_idx_2_instances, 0], width=bar_width, label='2 Years', color=shades[1])
        axs[i].bar(x + bar_width, [0, 0, usda_idx_3_instances], width=bar_width, label='3 Years', color=shades[2])
        
        # Set plot title and y-axis label with larger font sizes
        axs[i].set_title(f'{dca_id.capitalize()}', fontsize=16)
        axs[i].set_ylabel('Number of Events', fontsize=16)
        axs[i].set_xticks(x)
        axs[i].tick_params(axis='x', labelsize=16, rotation=45)
        axs[i].tick_params(axis='y', labelsize=16)
        axs[i].legend(fontsize=14)
        
        # Set x-tick labels separately to avoid FixedFormatter warning
        axs[i].set_xticklabels(['1 Year', '2 Years', '3 Years'])

    # Set common x-axis label with larger font size
    fig.text(0.5, -0.05, 'Number of Appearances', ha='center', fontsize=18)
    fig.tight_layout()

    # Show the plot
    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'refdm_length_of_disturbance_events.png'), bbox_inches='tight')

    plt.show()


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
    ids_gdf['centroid'] = ids_gdf.geometry.centroid
    refdm_gdf_dissolved['centroid'] = refdm_gdf_dissolved.geometry.centroid

    # Initialize lists to hold the results
    usda_idx_list = []
    centroid_shift_list = []
    size_difference_list = []

    # Iterate through unique USDA_IDX values in ids_gdf
    unique_usda_idx = ids_gdf['USDA_IDX'].unique()

    for usda_idx in unique_usda_idx:
        # Get the corresponding rows in both GeoDataFrames for the current USDA_IDX
        gdf_row = ids_gdf[ids_gdf['USDA_IDX'] == usda_idx]
        refdm_row = refdm_gdf_dissolved[refdm_gdf_dissolved['USDA_IDX'] == usda_idx]
        
        # Check if both rows exist (they should exist if data is correctly structured)
        if not gdf_row.empty and not refdm_row.empty:
            # Calculate the centroid shift between centroids of the two polygons
            centroid_shift = gdf_row.iloc[0]['centroid'].distance(refdm_row.iloc[0]['centroid'])
            
            # Calculate the size difference between areas of the two polygons
            size_difference = gdf_row.iloc[0]['geometry'].area - refdm_row.iloc[0]['geometry'].area
            
            # Append the results to the lists
            usda_idx_list.append(usda_idx)
            centroid_shift_list.append(centroid_shift)
            size_difference_list.append(size_difference)

    # Create a DataFrame from the results
    results_df = pd.DataFrame({
        'USDA_IDX': usda_idx_list,
        'centroid_shift': centroid_shift_list,
        'size_difference': size_difference_list
    })

    print(f"Merge with original gdf to get geometry")
    # Merge with original gdf to get geometry
    result_gdf = ids_gdf.merge(results_df, on='USDA_IDX')

    # Convert to GeoDataFrame with original geometry and CRS
    result_gdf = gpd.GeoDataFrame(result_gdf, geometry='geometry', crs=ids_gdf.crs)

    return result_gdf

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




def main():
    # Define paths
    refdm_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/radar_enhanced_forest_disturbance_mapping.shp"
    usa_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/data/S_USA.AdministrativeRegion/S_USA.AdministrativeRegion.shp"
    ids_path = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/region8_dca_filtered_ids_usda_polygons.csv"
    save_dir = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/figures/"

    figure_1 = "p1_f1_ids.png"
    figure_reduction_potential = "p1_f2_radar_potential_analysis.png"
    figure_size_shift = "p1_f3_size_shift_analysis.png"
    figure_disturbance_duration = "p1_f4_disturbance_duration_analysis.png"

    try:
        # Step 1: Loading CSV file and converting to GeoDataFrame
        print("Step 1: Loading and converting CSV file to GeoDataFrame...")
        df = pd.read_csv(ids_path)
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.rename(columns={'index_usda': 'USDA_IDX'})

        # Step 2: Loading and processing shapefile
        print("Step 2: Loading and processing shapefile...")
        refdm_dataset = gpd.read_file(refdm_path)
        refdm_dissolved = refdm_dataset.dissolve(by='USDA_IDX').reset_index()

        print("Step 3: Calculating size shift difference and plotting scatterplots...")
        result_gdf = calculate_size_shift_difference(gdf, refdm_dissolved)
       
        # Step 3: Plotting functions
        print("Step 4: Plotting ...")
        print("4.1. Plot Figure 1 - Study area")
        plot_disturbance_events(refdm_dissolved, get_region_8(usa_path), custom_colors, save_dir)
        
        print("Step 5: Plotting radar reduction potential ...")
        plot_radar_reduction_potential(refdm_dissolved, gdf, save_dir)
        
        print("Step 6: Plotting length of disturbance events ...")
        plot_length_of_disturbance_events(refdm_dataset, save_dir)
    
        
        print("Step 8: Plotting size shift difference errorbar...")
        plot_size_shift_comparison_errorbars(result_gdf, save_dir)
        
        print("Main process completed successfully.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Log the error or take appropriate action


if __name__ == "__main__":
    main()
