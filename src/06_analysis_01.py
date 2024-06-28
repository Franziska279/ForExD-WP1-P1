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
    counts_df['Reduction (%)'] =  - 100 * (counts_df['IDS'] - counts_df['REFDM']) / counts_df['IDS']

    # Ensure the DCA_ID is in the specified order
    counts_df['DCA_ID'] = pd.Categorical(counts_df['DCA_ID'], categories=['bark_beetle', 'wind', 'fire', 'defoliators', 'drought'], ordered=True)
    counts_df_sorted = counts_df.sort_values('DCA_ID')

    # Create a figure with 2 rows and 1 column for the two subplots
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])  # Define height ratios

    # Plot Counts in the first subplot
    ax1 = fig.add_subplot(gs[0])
    bar_width = 0.35  # Width of the bars
    bar_positions = range(len(counts_df_sorted))  # X positions for bars

    ax1.bar(bar_positions, counts_df_sorted['IDS'], width=bar_width, color="#BCB6FF", label='IDS')
    ax1.bar([pos + bar_width for pos in bar_positions], counts_df_sorted['REFDM'], width=bar_width, color="#AF42AE", label='REFDM')

    # Add annotations for counts above bars
    for i, (count_ids, count_refdm) in enumerate(zip(counts_df_sorted['IDS'], counts_df_sorted['REFDM'])):
        ax1.text(bar_positions[i], count_ids + 2, str(int(count_ids)), ha='center', va='bottom', color='black', fontsize=16)
        ax1.text(bar_positions[i] + bar_width, count_refdm + 2, str(int(count_refdm)), ha='center', va='bottom', color='black', fontsize=16)

    # Set labels and title for the first subplot
    ax1.set_ylabel('Amount of Disturbance Events', fontsize=20)  # Increase font size for ylabel
    ax1.set_title('Radar Detection Potential per Disturbance Type', fontsize=24)  # Increase font size for title
    ax1.set_yscale('log')
    #ax1.legend(fontsize=20, title='Datasets')
    legend = ax1.legend(fontsize=20, title='Datasets')  # Increase font size for legend and add title
    legend.get_title().set_fontsize('24')
    ax1.grid(False)
    plt.yticks(fontsize=16)
    plt.xticks(bar_positions, counts_df_sorted['DCA_ID'], fontsize=18)

    # Plot Reduction (%) in the second subplot with negative y-axis

    double_bar_width = bar_width * 2  # Make the bottom bars as wide as the combined width of the top two bars
    bar_offset = 0.2  # Offset to move the lower bars to the right
    # Rotate x-axis labels for ax2 and apply a small offset
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar([pos + bar_offset for pos in bar_positions], counts_df_sorted['Reduction (%)'], width=double_bar_width, color='#FF3E41', label='Reduction (%)')


    # Add annotations for reduction below bars
    for i, reduction_percentage in enumerate(counts_df_sorted['Reduction (%)']):
        ax2.text(bar_positions[i] + bar_offset, reduction_percentage - 2, f'{reduction_percentage:.2f}%', ha='center', va='top', color='black', fontsize=16)

    # Set labels and title for the second subplot
    ax2.set_xlabel('Disturbance Type', fontsize=20)  # Increase font size for xlabel
    ax2.set_ylim(0, -75)
    ax2.invert_yaxis()
    ax2.set_ylabel('Reduction \nPercentage \n(%)', fontsize=20)  # Increase font size for ylabel
    plt.xticks([pos + 0.2 for pos in bar_positions], counts_df_sorted['DCA_ID'], fontsize=18, ha='right')  # Adjust the rotation and alignment

    # Rotate x-axis labels for ax2
    ax2.set_xticklabels(counts_df_sorted['DCA_ID'], ha='right', fontsize=0)  # Rotate and increase font size
    ax2.grid(False)

    # Adjust x-axis ticks and labels
    plt.yticks(fontsize=16)
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


def plot_disturbance_events(data, region, color_dict, save_dir):
    # Set up Seaborn's plotting aesthetics
    sns.set(style="whitegrid", context="talk")

    # Map colors to DCA_IDs
    data['color'] = data['DCA_ID'].map(color_dict)

    # Plot the GeoDataFrame
    fig, ax = plt.subplots(figsize=(15, 15))  # Adjust the figsize as needed

    # Plot the region outline
    region.boundary.plot(ax=ax, linewidth=2, color='black')

    # Plot each disturbance type with its corresponding color
    for disturbance, color in color_dict.items():
        data[data['DCA_ID'] == disturbance].plot(
            ax=ax, linewidth=1.5, color=color, edgecolor=color  # Adjust linewidth and edgecolor as needed
        )

    # Customize the plot
    ax.set_title('Radar Enhanced Forest Disturbance Mapping Events', fontsize=20)
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)

    # Create custom legend on the left side
    legend_patches = [mpatches.Patch(color=color, label=disturbance) for disturbance, color in color_dict.items()]
    ax.legend(handles=legend_patches, fontsize=14, title="Disturbance Type", title_fontsize=16, loc='upper left')

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'refdm_disturbance_events_region_8.png'), bbox_inches='tight')
    plt.show()


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


def plot_scatterplot_size_shift_per_disturbances(gdf, save_dir):

    # Define the order of the categories
    category_order = ['bark_beetle', 'wind', 'fire', 'drought', 'defoliators']

    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = {label: custom_colors.get(label, default_colors.pop(0)) for label in category_order}

    # Multiply 'size_difference' by 100
    gdf['size_difference_scaled'] = gdf['size_difference'] * 100
    gdf['centroid_shift_scaled'] = gdf['centroid_shift'] * 10

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a grid with 3 columns and 2 rows for the plots and legend
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), gridspec_kw={'width_ratios': [1, 1, 1]})

    # Flatten the axes array for easier indexing
    axs = axs.flatten()

    # Plot the scatterplots
    for i, category in enumerate(category_order):
        ax = axs[i]
        sns.scatterplot(
            data=gdf[gdf['DCA_ID'] == category],
            x='centroid_shift_scaled',
            y='size_difference_scaled',
            hue='DCA_ID',
            palette=custom_palette,
            s=200,
            edgecolor='w',
            alpha=0.7,
            ax=ax,
            legend=False  # Disable legend in each subplot
        )
        ax.axhline(0, color='grey', linestyle='--', linewidth=2)
        ax.axvline(0, color='grey', linestyle='--', linewidth=2)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlim(gdf['centroid_shift_scaled'].min() - 0.05, gdf['centroid_shift_scaled'].max() + 0.05)
        ax.set_ylim(gdf['size_difference_scaled'].min() - 0.05, gdf['size_difference_scaled'].max() + 0.05)
        #ax.set_title(category.capitalize(), fontsize=24)
        
        ax.set_ylabel('')

        ax.set_xlabel('')

    # Remove the unused last subplot (bottom right corner)
    fig.delaxes(axs[-1])

    # Add a custom legend to the last subplot
    legend_patches = [mpatches.Patch(color=color, label=label.capitalize()) for label, color in custom_palette.items()]
    legend_ax = fig.add_subplot(2, 3, 6)
    legend_ax.legend(handles=legend_patches, title='Disturbance Types', loc='center', fontsize=24, title_fontsize=26)
    legend_ax.axis('off')

    # Set shared x and y labels
    fig.text(0.5, 0.004, 'Centroid Shift (distance)\nSmaller Shift: High Overlap | Larger Shift: Low Overlap', ha='center', fontsize=24)
    fig.text(0.01, 0.5, 'Size Difference (area)\nNegative: IDS Smaller | Positive: IDS Larger', va='center', rotation='vertical', fontsize=24)

    # Set the overall title and adjust layout
    plt.subplots_adjust(top=0.4)
    fig.suptitle('Centroid Shift vs. Size Difference for Disturbance Types', fontsize=28)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'size_shift_comparison_scatterplots.png'), bbox_inches='tight')
    plt.show()


def plot_size_shift_comparison_errorbars(gdf, save_dir):

    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = [custom_colors.get(label, default_colors.pop(0)) for label in gdf['DCA_ID'].unique()]

    # Multiply 'size_difference' by 100
    gdf['size_difference_scaled'] = gdf['size_difference'] * 1000
    # Multiply 'centroid_shift' by 100
    gdf['centroid_shift_scaled'] = gdf['centroid_shift'] * 100

    # Calculate medians and quantiles for each disturbance type
    grouped = gdf.groupby('DCA_ID').agg({
        'centroid_shift_scaled': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'size_difference_scaled': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
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
    plt.xlabel('Centroid Shift (distance)\nSmaller Shift: High Overlap | Larger Shift: Low Overlap', fontsize=18)
    plt.ylabel('Size Difference (area\nNegative: IDS Smaller | Positive: IDS Larger', fontsize=18)
    plt.title('Centroid Shift vs. Size Difference for Disturbance Types', fontsize=20, pad=20)

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(by_label.values(), by_label.keys(), title='Disturbance Types', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    legend.set_title('Disturbance Types', prop={'size': 18})  # Set legend title size

    # Set decimal tick labels
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Increase tick label size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'size_shift_comparison_errorbars.png'), bbox_inches='tight')
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
        print("Step 4: Plotting disturbance events...")
        plot_disturbance_events(refdm_dissolved, get_region_8(usa_path), custom_colors, save_dir)
        
        print("Step 5: Plotting radar reduction potential...")
        plot_radar_reduction_potential(refdm_dissolved, gdf, save_dir)
        
        print("Step 6: Plotting length of disturbance events...")
        plot_length_of_disturbance_events(refdm_dataset, save_dir)
        
        print("Step 7: Plotting size shift difference scatterplots...")
        plot_scatterplot_size_shift_per_disturbances(result_gdf, save_dir)
        
        print("Step 8: Plotting size shift difference errorbar...")
        plot_size_shift_comparison_errorbars(result_gdf, save_dir)
        
        print("Main process completed successfully.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Log the error or take appropriate action


if __name__ == "__main__":
    main()
