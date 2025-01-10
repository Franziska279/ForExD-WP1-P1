import matplotlib.pyplot as plt
import pandas as pd  # Assuming you're using pandas for timestamps
import geopandas as gpd
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from func_helper import format_label
from matplotlib.gridspec import GridSpec

## IDS Preprocessing 
def plot_regions_disturbances(disturbance_gdf, filepath, output_file, custom_colors, region_nr):
    """
    Plots disturbances within a specified region with custom colors for each disturbance type.

    Parameters:
    - disturbance_gdf: GeoDataFrame containing disturbance geometries with a 'DCA_ID' column.
    - filepath: Path to the shapefile containing regional boundaries.
    - output_file: Path for saving the output plot. If None, the plot is only displayed.
    - custom_colors: Dictionary mapping 'DCA_ID' values to specific colors for plotting.
    - region_nr: The region number to filter from the regional boundary shapefile.
    """
    # Load regional boundaries and filter for the specified region
    usa = gpd.read_file(filepath)
    country_region = usa[usa['REGION'] == region_nr]

    # Check if the region exists in the shapefile
    if country_region.empty:
        print(f"No data found for Region {region_nr}.")
        return

    # Explode geometries for multipart handling, but keep as a GeoDataFrame
    country_region = country_region.explode(index_parts=True)
    
    # Extract the boundary as a GeoSeries
    region_boundary = country_region['geometry']

    # Initialize plot with specified figure size
    fig, ax = plt.subplots(figsize=(12, 12))

    # List to collect legend entries for disturbances and region boundary
    legend_handles = []

    # Plot each disturbance type with a custom color
    for dca_id, color in custom_colors.items():
        # Filter disturbances by 'DCA_ID' and only plot valid geometries
        disturbance_subset = disturbance_gdf[(disturbance_gdf['DCA_ID'] == dca_id) & (disturbance_gdf.is_valid)]
        
        if not disturbance_subset.empty:
            # Plot disturbance geometries
            disturbance_subset.plot(ax=ax, color=color, edgecolor=color, linewidth=0.5, zorder=2)
            # Add entry to legend with disturbance type and count
            legend_handles.append(Patch(color=color, label=f"{format_label(dca_id)} ({len(disturbance_subset)})"))

    # Plot region boundary with a dashed line for each geometry in the boundary series
    region_boundary.boundary.plot(ax=ax, color='black', linewidth=0.5, linestyle='--', zorder=3)
    # Add region boundary to legend
    legend_handles.append(Patch(facecolor='none', edgecolor='black', linestyle='--', linewidth=1,
                                label=f'Region {region_nr} Boundary'))

    # Customize plot appearance
    ax.set_title(f'Disturbances within Region {region_nr}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)  # Add grid for reference
    ax.set_aspect('equal')  # Ensure equal scaling on both axes

    # Add legend with custom handles for disturbances and region boundary
    ax.legend(handles=legend_handles, loc='best', fontsize=12)

    # Save plot if output path is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()


## TCC PLots
def plot_region_bounds(region, x_min, y_min, x_max, y_max, region_nr, output_path):
        
        # Plotting the region shape and saving the figure
        fig, ax = plt.subplots(figsize=(8, 8))
        region.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=1)
        bbox = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             edgecolor='brown', facecolor='none', linewidth=2, linestyle='--')
        ax.add_patch(bbox)
        ax.scatter([x_min, x_max, x_min, x_max], [y_min, y_min, y_max, y_max], color='red', zorder=5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Region {region_nr} with Bounding Box')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)



def plot_tcc_region_bounds(tif, region, region_id, output_path, cmap="viridis", downsample_factor=4):
    """
    Plot the TCC raster with region boundaries overlaid and include a colorbar.
    Optimized to reduce memory load by downsampling.

    Parameters:
    - tif (xarray.DataArray or rasterio object): TCC raster data.
    - region (geopandas.GeoDataFrame): Region geometry to overlay.
    - region_id (str): Identifier for the region boundary.
    - output_path (str): Path to save the output image.
    - cmap (str): Colormap for the raster.
    - downsample_factor (int): Factor by which to reduce resolution.
    """
    # Squeeze the single band dimension if it exists and downsample the data for lighter plotting
    data_to_plot = tif.squeeze().coarsen(x=downsample_factor, y=downsample_factor, boundary="trim").mean()

    # Set up the plot with a manageable figure size
    fig, ax = plt.subplots(figsize=(10, 8))
    data_to_plot.plot(ax=ax, cmap=cmap, add_colorbar=True)

    # Plot the region boundary in red
    region.boundary.plot(ax=ax, color='red', linewidth=2)

    # Add title and axis labels
    ax.set_title(f"TCC Raster and Region {region_id} Boundary")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a legend for the boundary
    ax.legend(handles=[Patch(color='red', label=f'Region {region_id} Boundary')], loc='upper right')

    # Save and close the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Lower DPI if necessary
    plt.close(fig)

# Timeseries Analyze Plots
def create_plots(combined_dataset, refdm_filtered, ids_filtered, grid, unique_minicubes, dca, ID, custom_colors, mean_ids, mean_refdm, year, var='ndvi'):
    """
    Create two subplots: one for NDVI data and another for time series data.

    Parameters:
    - combined_dataset: Dataset containing NDVI data
    - refdm_filtered: GeoDataFrame for REFDM boundaries
    - ids_filtered: GeoDataFrame for IDS boundaries
    - grid: GeoDataFrame for grid boundaries
    - unique_minicubes: Unique minicube IDs
    - dca: Data category for coloring
    - ID: Event ID
    - custom_colors: Dictionary of custom colors
    - mean_ids: Mean values for IDS
    - mean_refdm: Mean values for REFDM
    - var: Variable to plot (default is 'ndvi')
    """
    
    time_index = 240  # Set the time index you want to plot
    ndvi_data = combined_dataset[var].isel(time=time_index)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))  # 1 row, 2 columns
    
    # --- First subplot: NDVI Data ---
    # Plot the boundaries of the geometries for refdm_filtered
    refdm_filtered.boundary.plot(ax=ax1, color='magenta', linewidth=2, label='S1DM')
    
    # Plot the boundaries of the geometries for ids_filtered
    ids_filtered.boundary.plot(ax=ax1, color='black', linewidth=3, linestyle='-', label='IDS')
    
    # Optional: Uncomment if you have a grid to plot
    grid.boundary.plot(ax=ax1, color='white', linewidth=3, linestyle=':')
    
    # Plot the NDVI data with a colormap
    ndvi_data.plot(ax=ax1, cmap='Greens', add_colorbar=True, cbar_kwargs={'shrink': 0.8})  # Shrink colorbar
    
    # Set axis labels
    ax1.set_xlabel('Longitude', fontsize=18)
    ax1.set_ylabel('Latitude', fontsize=18)
    
    # Set a title for the NDVI plot
    ax1.set_title(f' ', fontsize=16)  # Empty title
    
    # Add a legend (remove Minicube ID from legend)
    ax1.legend(loc='upper right', fontsize=10)  # Smaller legend
    
    # --- Second subplot: Time Series Data ---
    # Set the color for REFDM based on the DCA category using custom_colors
    refdm_color = custom_colors.get(dca, 'gray')  # Default to gray if dca not found
    
    # Add a red dotted line at y=0
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)

     # Create a date range for the entire year
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    
    # Highlight the entire year with a gray box
    ax2.axvspan(start_date, end_date, color='gray', alpha=0.5, label='Survey Year')

    
    # Extract the time coordinates for the x-axis
    time = mean_ids['time'].values
     # Highlight the year with a gray box
    # Plot the median for IDS (always black)
    ax2.plot(time, mean_ids[var], color='black', label='IDS ', linewidth=2)
    
    # Plot the median for REFDM (use the custom color)
    ax2.plot(time, mean_refdm[var], color=refdm_color, label='S1DM ', linewidth=2)
    
    # Set plot title and labels
    ax2.set_xlabel("Disturbance Year", fontsize=18)
    ax2.set_ylabel(f"{var.upper()}", fontsize=18)
    
    # Add a legend
    ax2.legend(loc='lower right', fontsize=10)  # Smaller legend
    
    # Super title for both plots (centered)
    fig.suptitle(
        f"{dca.capitalize()} Event with ID_E={ID} on the Cubes {unique_minicubes}",
        ha='center', fontsize=28
    )
    
    # Adjust layout to ensure super title is centered
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to give space for the super title
    
    # Show the plots
    plt.show()



# Analysis plots

def plot_radar_reduction_potential(refdm_gdf, ids_gdf, save_path, plot_reduction=True):
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
        'S1DM': dca_counts_refdm
    }).fillna(0)  # Fill NaN with 0 for counts that are missing in either dataset

    # Reset index to turn DCA_ID into a column
    counts_df.reset_index(inplace=True)
    counts_df.rename(columns={'index': 'DCA_ID'}, inplace=True)
    # Calculate reduction percentage
    counts_df['Reduction (%)'] = -100 * (counts_df['IDS'] - counts_df['S1DM']) / counts_df['IDS']

    # Ensure the DCA_ID is in the specified order
    counts_df['DCA_ID'] = pd.Categorical(counts_df['DCA_ID'], categories=['bark_beetle', 'wind', 'fire', 'defoliators', 'drought'], ordered=True)
    counts_df_sorted = counts_df.sort_values('DCA_ID')

    # Capitalize DCA_ID labels
    dca_labels = [format_label(label) for label in counts_df_sorted['DCA_ID']]

    # Create a figure with 2 rows and 1 column for the two subplots, conditionally plotting the second subplot
    fig = plt.figure(figsize=(24, 12))  # Increase the figure height
    gs = GridSpec(nrows=2 if plot_reduction else 1, ncols=1, height_ratios=[5, 2] if plot_reduction else [5])

    # Plot Counts in the first subplot
    ax1 = fig.add_subplot(gs[0])
    bar_positions = range(len(counts_df_sorted))  # X positions for bars

    ax1.bar(bar_positions, counts_df_sorted['IDS'], width=bar_width, color="#BCB6FF", label='IDS')
    ax1.bar([pos + bar_width for pos in bar_positions], counts_df_sorted['S1DM'], width=bar_width, color="#AF42AE", label='S1DM')

    # Add annotations for counts above bars
    for i, (count_ids, count_refdm) in enumerate(zip(counts_df_sorted['IDS'], counts_df_sorted['S1DM'])):
        ax1.text(bar_positions[i], count_ids + 2, str(int(count_ids)), ha='center', va='bottom', color='black', fontsize=annotation_fontsize)
        ax1.text(bar_positions[i] + bar_width, count_refdm + 2, str(int(count_refdm)), ha='center', va='bottom', color='black', fontsize=annotation_fontsize)

    # Set labels and title for the first subplot
    ax1.set_ylabel('Number of Disturbance Events', fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding
    ax1.set_yscale('log')
    ax1.set_ylim(1, counts_df_sorted[['IDS', 'S1DM']].max().max() * 2)  # Set y-limit for log scale
    legend = ax1.legend(fontsize=legend_fontsize, title='Datasets')  # Increase font size for legend and add title
    legend.get_title().set_fontsize(legend_title_fontsize)
    ax1.grid(False)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(bar_positions, dca_labels, fontsize=tick_fontsize)
    ax1.tick_params(axis='x', which='major', pad=15)

    # Plot Reduction (%) in the second subplot if plot_reduction is True
    if plot_reduction:
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar([pos + bar_offset for pos in bar_positions], counts_df_sorted['Reduction (%)'], width=double_bar_width, color='#FF3E41', label='Reduction (%)')

        # Add annotations for reduction below bars
        for i, reduction_percentage in enumerate(counts_df_sorted['Reduction (%)']):
            ax2.text(bar_positions[i] + bar_offset, reduction_percentage - 2, f'{reduction_percentage:.2f}%', ha='center', va='top', color='black', fontsize=annotation_fontsize)

        # Set labels and title for the second subplot
        ax2.set_xlabel('Disturbance Type', fontsize=label_fontsize)  # Increase font size for xlabel
        ax2.set_ylim(0, -110)
        ax2.invert_yaxis()
        ax2.set_ylabel('Reduction \nPercentage (%)', fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding

        # Set y-axis ticks to only show 4 ticks
        ax2.set_yticks([0, -20, -40, -60, -80, -100])
        ax2.set_yticklabels(['0', '-20', '-40', '-60', '-80', '-100'])

        plt.xticks([pos + bar_offset for pos in bar_positions], dca_labels, fontsize=tick_fontsize, ha='right')  # Adjust the rotation and alignment

        # Rotate x-axis labels for ax2
        ax2.set_xticklabels(dca_labels, ha='right', fontsize=0)  # Rotate and increase font size
        ax2.grid(False)

        # Adjust x-axis ticks and labels
        plt.yticks(fontsize=tick_fontsize)

    # Adjust x-axis ticks and labels
    plt.tight_layout()  # Ensures labels, titles, and legends do not overlap

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
