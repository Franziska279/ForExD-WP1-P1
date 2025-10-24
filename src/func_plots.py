import logging
import matplotlib.pyplot as plt
import pandas as pd  # Assuming you're using pandas for timestamps
import geopandas as gpd
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from func_helper import format_label
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from shapely import wkt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm 
import os
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from shapely.validation import make_valid
from shapely.ops import unary_union
import os
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.ops import unary_union
from matplotlib.patches import Rectangle
import os
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from shapely.validation import make_valid
from scipy.stats import ttest_rel

from func_helper import parse_custom_colors, format_label, calculate_minimum_outerline_area, get_mainland, load_and_extract_region, calculate_overlap_percentages, paired_ttest_significance, compute_overlap_jaccard
from func_file_io import load_tcc_nc_dataset, load_data

def format_ticks(x, pos):
    """Format the ticks to always have one decimal place."""
    return f'{x:.1f}'
    
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


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

def plot_stacked_bar_chart(ids, custom_colors, save_to=None):
    """
    Plot a stacked bar chart of event counts and destroyed area by survey year and DCA_ID.
    
    Parameters:
    - ids: DataFrame containing the data with 'SURVEY_Y', 'DCA_ID', 'IDX_D', and 'area_km2'.
    - custom_colors: Dictionary mapping DCA_IDs to colors.
    - save_to: Optional; if provided, the plot will be saved to this filename (including extension).
    
    """
    # Aggregate data for counts by SURVEY_Y and DCA_ID
    count_by_year_and_dca = ids.groupby(['SURVEY_Y', 'DCA_ID'])['IDX_D'].count().unstack(fill_value=0)

    # Convert the counts to integers
    count_by_year_and_dca = count_by_year_and_dca.astype(int)

    # Aggregate data for area by SURVEY_Y and DCA_ID
    area_by_year_and_dca = ids.groupby(['SURVEY_Y', 'DCA_ID'])['area_km2'].sum().unstack(fill_value=0)

    # Convert the areas to integers
    area_by_year_and_dca = area_by_year_and_dca.astype(int)

    # Create positions for grouped bars
    x = np.arange(len(count_by_year_and_dca.index))  # X-axis positions
    bar_width = 0.4  # Width of the grouped bars

    # Retrieve DCA_IDs and their custom colors
    dca_ids = count_by_year_and_dca.columns
    colors = [custom_colors.get(str(dca_id), 'gray') for dca_id in dca_ids]  # Default to gray if color is missing

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Remove background color for the plot (set it to default white)
    ax1.set_facecolor('white')

    # Plot stacked bars for Event Count with transparency (alpha=0.5)
    for i, dca_id in enumerate(dca_ids):
        bottom_count = count_by_year_and_dca.iloc[:, :i].sum(axis=1)  # Stack previous bars
        ax1.bar(
            x - bar_width / 2,  # Position for count bars
            count_by_year_and_dca[dca_id],
            width=bar_width,
            bottom=bottom_count,
            color=colors[i],  # Use custom color for DCA_ID
            alpha=0.5,  # Make count bars transparent
        )

    # Plot stacked bars for Destroyed Area
    for i, dca_id in enumerate(dca_ids):
        bottom_area = area_by_year_and_dca.iloc[:, :i].sum(axis=1)  # Stack previous bars
        ax1.bar(
            x + bar_width / 2,  # Position for area bars
            area_by_year_and_dca[dca_id],
            width=bar_width,
            bottom=bottom_area,
            color=colors[i],  # Use custom color for DCA_ID
        )

    # Add edge outlines around the total bar for counts (white)
    for pos in x:
        total_height = count_by_year_and_dca.sum(axis=1).iloc[pos]
        ax1.bar(
            pos - bar_width / 2,  # Same position as count bars
            total_height,
            width=bar_width,
            color='none',  # Transparent to keep the existing stacks
            edgecolor='gray',
            alpha=0.5,
            linewidth=1.5,
        )

    # Add edge outlines around the total bar for areas (black)
    for pos in x:
        total_height = area_by_year_and_dca.sum(axis=1).iloc[pos]
        ax1.bar(
            pos + bar_width / 2,  # Same position as area bars
            total_height,
            width=bar_width,
            color='none',  # Transparent to keep the existing stacks
            edgecolor='black',
            linewidth=1.5,
        )

    # Set larger font size for labels, ticks, and legend
    ax1.set_xlabel('Survey Year', fontsize=24, labelpad=20)  # Add padding to the x-axis label
    ax1.set_ylabel('Counts / Area (km²)', fontsize=24, labelpad=20)  # Add padding to the y-axis label
    ax1.set_xticks(x)
    ax1.set_xticklabels(count_by_year_and_dca.index, rotation=0, fontsize=20)  # 90 degree rotation for x labels

    # Define a FuncFormatter to remove the ".0" on y-axis labels
    def format_y_ticks(x, pos):
        return f'{int(x)}'  # Convert to integer without decimals

    # Apply the FuncFormatter to the y-axis
    ax1.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

    # Apply padding for ticks (buffers between ticks and labels)
    ax1.tick_params(axis='y', labelsize=20, pad=15)  # Add padding to the y-axis ticks
    ax1.tick_params(axis='x', labelsize=20, pad=15)  # Add padding to the x-axis ticks

    # Set the y-axis ticks locator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks on the y-axis

    # Add a grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

    # Create the legend for Count, Area, and DCA_IDs
    count_legend = plt.Line2D([0], [0], color='gray', lw=4, alpha=0.5)  # Transparent for count
    area_legend = plt.Line2D([0], [0], color='black', lw=4)  # Full color for area

    # Add the DCA_ID legends as colored boxes (rectangles)
    dca_legend = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(dca_ids))]

    # Add the combined legend (Count, Area, and DCA_IDs)
    legend = ax1.legend(
        [count_legend, area_legend] + dca_legend,  # Count, Area, and DCA_IDs
        ['Count', 'Area'] + [f'{format_label(dca)}' for dca in dca_ids],  # Labels for the legend
        loc='upper left',  # Position the legend in the upper left inside the plot
        fontsize=20,  # Larger font size for legend
        frameon=True,
    )

    # Tight layout and show plot
    fig.tight_layout()

    # Save the plot if a filename is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')  # Save at high resolution
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

# Analysis plots
def plot_radar_reduction_potential(refdm_gdf, ids_gdf, save_path, plot_reduction=True):
    # Define font sizes and bar parameters
    title_fontsize = 34
    legend_title_fontsize = 30
    label_fontsize = 30
    legend_fontsize = 30
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

    # Adjust the figure height based on whether the reduction plot is included
    if plot_reduction:
        figure_height = 12 
    else: 
        figure_height = 7   # Adjust figure height

    # Create a figure with adjusted layout
    fig = plt.figure(figsize=(24, figure_height))
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
    ax1.set_ylabel(r"${N_{\text{Disturbance Events}}}$", fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding
    ax1.set_yscale('log')
    ax1.set_ylim(1, counts_df_sorted[['IDS', 'S1DM']].max().max() * 2)  # Set y-limit for log scale
    legend = ax1.legend(fontsize=legend_fontsize, title='Datasets')  # Increase font size for legend and add title
    legend.get_title().set_fontsize(legend_title_fontsize)
    ax1.grid(False)
    plt.yticks(fontsize=tick_fontsize)

    # Calculate the midpoint of the grouped bars and adjust to move slightly left
    group_width = bar_width * 2  # Total width of a group (2 bars per group)
    tick_positions = [pos + group_width / 2 - (bar_width / 2.5) for pos in bar_positions]  # Slightly move left
    plt.xticks(tick_positions, dca_labels, fontsize=tick_fontsize, ha='center')  # Center the x-ticks


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
        ax2.set_xticklabels('', ha='right', fontsize=1)  # Rotate and increase font size
        ax2.grid(False)

        # Adjust x-axis ticks and labels
        plt.yticks(fontsize=tick_fontsize)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb

def plot_disturbance_counts(refdm_gdf, ids_path, exclude_types=None, ordered_types=None,
                            custom_colors=None, output_file=None, year_col='SURVEY_Y',
                            type_col='DCA_ID',  figsize=(18,6),
                            log_scale=True):
    """
    Plot the yearly counts of disturbances for IDS and S1DM datasets as side-by-side barplots.

    Parameters
    ----------
    refdm_gdf : GeoDataFrame
        S1DM reference disturbances GeoDataFrame.
    ids_gdf : GeoDataFrame
        IDS disturbances GeoDataFrame.
    year_col: str
        Column name in ids_gdf containing the year.
    type_col : str, default 'DCA_ID'
        Column representing disturbance type.
    exclude_types : list of str, optional
        Disturbance types to exclude from the plot (e.g., ['fire', 'drought']).
    ordered_types : list of str, optional
        Desired order of disturbance types.
    custom_colors : dict, optional
        Mapping disturbance type -> hex color. Missing types will use default tab10 colors.
    output_file : str, optional
        Path to save the figure.
    figsize : tuple, default (18,12)
        Figure size.
    log_scale : bool, default True
        Whether to use logarithmic scale for y-axis.

    Returns
    -------
    fig, axes : matplotlib.Figure, numpy.ndarray
        The figure and axes array.
    """
    # Default arguments
    exclude_types = exclude_types or []
    custom_colors = custom_colors or {}

    # ---- Load data ----
    refdm_gdf = gpd.read_file(refdm_gdf)
    ids_gdf = gpd.read_file(ids_path)
    
    # Dissolve S1DM polygons by IDX_D
    refdm_gdf = refdm_gdf.dissolve(by='IDX_D', as_index=False)
    
    # Filter out unwanted types
    refdm_gdf = refdm_gdf[~refdm_gdf[type_col].isin(exclude_types)]
    ids_gdf = ids_gdf[~ids_gdf[type_col].isin(exclude_types)]
    
    # Rename year columns
    refdm_gdf = refdm_gdf.rename(columns={year_col: 'Year'})
    ids_gdf = ids_gdf.rename(columns={year_col: 'Year'})
    
    # Aggregate counts per year and type
    refdm_counts = refdm_gdf.groupby(['Year', type_col]).size().unstack(fill_value=0)
    ids_counts = ids_gdf.groupby(['Year', type_col]).size().unstack(fill_value=0)
    
    # All disturbance types
    all_types = sorted(list(set(refdm_counts.columns).union(set(ids_counts.columns))))
    
    # Colors
    default_colors = plt.cm.tab10.colors
    color_map = {dist_type: custom_colors.get(dist_type, default_colors[i % len(default_colors)])
                 for i, dist_type in enumerate(all_types)}
    
    # Apply order if provided
    if ordered_types:
        all_types_ordered = [t for t in ordered_types if t in all_types]
    else:
        all_types_ordered = all_types
    
    # Prepare subplots
    n_cols = 3
    n_rows = int(np.ceil(len(all_types_ordered)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 7*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    bar_width = 0.35
    
    for i, dist_type in enumerate(all_types_ordered):
        ax = axes[i]
        years = sorted(set(refdm_counts.index).union(ids_counts.index))
        x = np.arange(len(years))
        
        # IDS (lighter shade)
        base_rgb = to_rgb(color_map[dist_type])
        lighter_rgb = tuple(min(c + 0.4, 1.0) for c in base_rgb)
        bars_ids = ax.bar(x - bar_width/2,
                          ids_counts.get(dist_type, pd.Series(0, index=years)).reindex(years, fill_value=0),
                          width=bar_width, color=lighter_rgb, label='IDS')
        
        # S1DM
        bars_refdm = ax.bar(x + bar_width/2,
                            refdm_counts.get(dist_type, pd.Series(0, index=years)).reindex(years, fill_value=0),
                            width=bar_width, color=color_map[dist_type], label='S1DM')
        
        # Title
        ax.set_title(dist_type.replace('_',' ').title(), pad=15, fontsize=22)
        
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, fontsize=22)
        ax.tick_params(axis='y', labelsize=24)
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(1, max(refdm_counts.max().max(), ids_counts.max().max())*1.2)
        
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgrey', fontsize=18)
        
        # Numbers on top of bars
        for bars in [bars_ids, bars_refdm]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height*1.1, f'{int(height)}',
                            ha='center', va='bottom', fontsize=14)
        
        # Despine top and right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Axis labels
    axes[0].set_ylabel(r"${N_{\text{Disturbance Events}}}$", labelpad=20, fontsize=26)
    axes[1].set_xlabel('Year', labelpad=15, fontsize=26)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Save figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, axes


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import ticker

from matplotlib.lines import Line2D


def safe_stats(series):
    """Return median and 90th percentile safely, or NaN if no data."""
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    return np.median(clean), np.percentile(clean, 90)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import logging

def format_ticks(x, pos):
    """Format the ticks to always have one decimal place."""
    return f'{x:.1f}'

def plot_d_area_ch_area_centroid_disturbances_test(
        gdf, ids, s1dm_convex, ids_convex, custom_colors, save_path
    ):
    """
    Plot comparison between IDS and REFDM disturbance areas and centroid shifts,
    with legend inside the centroid shift plot. Includes safety checks and 
    recalculation of centroids to avoid NaNs.
    """

    # Determine the unique DCA_ID values from the filtered dataframe
    unique_dca_ids = gdf['DCA_ID'].unique()

    # Sort the categories based on the custom order
    category_order = sorted(unique_dca_ids, key=lambda x: custom_colors.get(x, x))  
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = {label: custom_colors.get(label, default_colors.pop(0)) for label in category_order}

    sns.set(style="whitegrid")

    n_categories = len(category_order)
    n_cols = 3  # Fixed number of columns
    fig, axs = plt.subplots(
        n_categories, 
        n_cols, 
        figsize=(40, 8 * n_categories), 
        gridspec_kw={'width_ratios': [3, 3, 2], 'wspace': 0.3} # Reduce the width ratios for empty columns
    )

    
    fontsize_supertitle = 44
    fontsize_legend = 40
    fontsize_title = 46
    fontsize_label = 50
    fontsize_tick = 35
    padding_label = 25
    padding_title = 25

    for i, category in enumerate(category_order):
        # Create combined data for violin plots using .loc[] to avoid SettingWithCopyWarning
        combined_data = pd.concat([
            #ids.loc[ids['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='IDS'),
            ids.loc[ids['DCA_ID'] == category, ['DCA_ID', 'area_km2']].copy().assign(Source='IDS'),
            #gdf.loc[gdf['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
            gdf.loc[gdf['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
        ])

        combined_convex = pd.concat([
            ids_convex.loc[ids_convex['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='IDS'),
            s1dm_convex.loc[s1dm_convex['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
        ])

        ax = axs[i, 0]
        data_refdm = combined_data.loc[(combined_data['Source'] == 'REFDM') & (combined_data['DCA_ID'] == category)]
        data_ids = combined_data.loc[(combined_data['Source'] == 'IDS') & (combined_data['DCA_ID'] == category)]

        median_refdm = np.median(data_refdm['area_km2'])
        median_ids = np.median(data_ids['area_km2'])

        if np.isnan(median_ids) or np.isnan(median_refdm):
            logging.warning(f"No valid data for category {category}, skipping KDE.")
            continue  # Skip this category completely

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Plot KDE for IDS
        sns.kdeplot(
            data=data_ids['area_km2'],
            color='black',
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='IDS',
            alpha=0.8,
            linestyle='-'
        )

        # Plot KDE for REFDM
        sns.kdeplot(
            data=data_refdm['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='S1DM',
            alpha=1,
            linestyle='-'
        )

        # Calculate the 90th percentile for disturbance area
        percentile_90_refdm = np.percentile(data_refdm['area_km2'], 90)
        percentile_90_ids = np.percentile(data_ids['area_km2'], 90)


        print(f"Median IDS for category {category}: {median_ids} km²")
        print(f"Median S1DM for category {category}: {median_refdm} km²")
        print(f"90th Percentile IDS for category {category}: {percentile_90_ids} km²")
        print(f"90th Percentile S1DM for category {category}: {percentile_90_refdm} km²")

        # # Add vertical lines at the 90th percentile
        # ax.axvline(x=percentile_90_ids, color='black', linestyle=':', linewidth=3, alpha=0.8, label=f'IDS 90th: {percentile_90_ids:.2f}', marker='o', markersize=16)
        # ax.axvline(x=percentile_90_refdm, color=custom_palette[category], linestyle=':', linewidth=3, alpha=1, label=f'S1DM 90th: {percentile_90_refdm:.2f}', marker='s', markersize=16)

        # Adjust median formatting for Bark Beetle DCA_ID
        if category == 'bark_beetle':
            median_refdm_label = f'S1DM Median: {median_refdm:.4f}' if median_refdm > 0 else 'S1DM Median: 0.0'
            median_ids_label = f'IDS Median: {median_ids:.4f}' if median_ids > 0 else 'IDS Median: 0.0'
        else:
            median_refdm_label = f'S1DM Median: {median_refdm:.2f}'
            median_ids_label = f'IDS Median: {median_ids:.2f}'

        from scipy.stats import gaussian_kde

        # Calculate KDE for IDS
        kde_ids = gaussian_kde(data_ids['area_km2'])
        x_vals_ids = np.linspace(min(data_ids['area_km2']), max(data_ids['area_km2']), 1000)
        kde_vals_ids = kde_ids(x_vals_ids)
        kde_at_median_ids = kde_ids(median_ids)[0]  # KDE value at median

        # Draw the median line for IDS
        ax.plot([median_ids, median_ids], [0, kde_at_median_ids], color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label)
        ax.plot(median_ids, kde_at_median_ids, 'o', color='black', markersize=10)  # Dot at the median peak

        # Calculate KDE for REFDM
        kde_refdm = gaussian_kde(data_refdm['area_km2'])
        x_vals_refdm = np.linspace(min(data_refdm['area_km2']), max(data_refdm['area_km2']), 1000)
        kde_vals_refdm = kde_refdm(x_vals_refdm)
        kde_at_median_refdm = kde_refdm(median_refdm)[0]  # KDE value at median

        # Draw the median line for REFDM
        ax.plot([median_refdm, median_refdm], [0, kde_at_median_refdm], color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label)
        ax.plot(median_refdm, kde_at_median_refdm, 'o', color=custom_palette[category], markersize=10)  # Dot at the median peak

        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.tick_params(axis='x', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel(r"${A_{\text{D}}\text{ (km²)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel('', labelpad=padding_label)

        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks and limit to 4 while ensuring not to include 0
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        #ax.set_xlim(-0.01, 10)
        if category == 'bark_beetle':
            ax.set_xlim(0, 0.5)  # or 1 depending on how much space you want
        else:
             ax.set_xlim(-0.01, 10)
        ax.set_ylim(0)
        ax.legend(fontsize=fontsize_tick)

        ax = axs[i, 1]
        data_refdm = combined_convex.loc[(combined_convex['Source'] == 'REFDM') & (combined_convex['DCA_ID'] == category)]
        data_ids = combined_convex.loc[(combined_convex['Source'] == 'IDS') & (combined_convex['DCA_ID'] == category)]

        median_refdm = np.median(data_refdm['area_km2'])
        median_ids = np.median(data_ids['area_km2'])
        

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Plot KDE for IDS
        sns.kdeplot(
            data=data_ids['area_km2'],
            color='black',
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='IDS',
            alpha=0.8,
            linestyle='-'
        )

        # Plot KDE for REFDM
        sns.kdeplot(
            data=data_refdm['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='S1DM',
            alpha=1,
            linestyle='-'
        )

        #  # Calculate the 90th percentile for disturbance area
        percentile_90_refdm = np.percentile(data_refdm['area_km2'], 90)
        percentile_90_ids = np.percentile(data_ids['area_km2'], 90)


        print(f"Median IDS for category {category}: {median_ids} km²")
        print(f"Median S1DM for category {category}: {median_refdm} km²")
        print(f"90th Percentile IDS for category {category}: {percentile_90_ids} km²")
        print(f"90th Percentile S1DM for category {category}: {percentile_90_refdm} km²")

        # # Add vertical lines at the 90th percentile
        # ax.axvline(x=percentile_90_ids, color='black', linestyle=':', linewidth=3, alpha=0.8, label=f'IDS 90th: {percentile_90_ids:.2f}', marker='o', markersize=16)
        # ax.axvline(x=percentile_90_refdm, color=custom_palette[category], linestyle=':', linewidth=3, alpha=1, label=f'S1DM 90th: {percentile_90_refdm:.2f}', marker='s', markersize=16)

        # Adjust median formatting for Bark Beetle DCA_ID
        if category == 'bark_beetle':
            median_refdm_label = f'S1DM Median: {median_refdm:.4f}' if median_refdm > 0 else 'S1DM Median: 0.0'
            median_ids_label = f'IDS Median: {median_ids:.4f}' if median_ids > 0 else 'IDS Median: 0.0'
        else:
            median_refdm_label = f'S1DM Median: {median_refdm:.2f}'
            median_ids_label = f'IDS Median: {median_ids:.2f}'

        # ax.axvline(x=median_ids, color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label, marker='o', markersize=16)
        # ax.axvline(x=median_refdm, color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label, marker='s', markersize=16)
        
        from scipy.stats import gaussian_kde

        # Calculate KDE for IDS
        kde_ids = gaussian_kde(data_ids['area_km2'])
        x_vals_ids = np.linspace(min(data_ids['area_km2']), max(data_ids['area_km2']), 1000)
        kde_vals_ids = kde_ids(x_vals_ids)
        kde_at_median_ids = kde_ids(median_ids)[0]  # KDE value at median

        # Draw the median line for IDS
        ax.plot([median_ids, median_ids], [0, kde_at_median_ids], color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label)
        ax.plot(median_ids, kde_at_median_ids, 'o', color='black', markersize=10)  # Dot at the median peak

        # Calculate KDE for REFDM
        kde_refdm = gaussian_kde(data_refdm['area_km2'])
        x_vals_refdm = np.linspace(min(data_refdm['area_km2']), max(data_refdm['area_km2']), 1000)
        kde_vals_refdm = kde_refdm(x_vals_refdm)
        kde_at_median_refdm = kde_refdm(median_refdm)[0]  # KDE value at median

        # Draw the median line for REFDM
        ax.plot([median_refdm, median_refdm], [0, kde_at_median_refdm], color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label)
        ax.plot(median_refdm, kde_at_median_refdm, 'o', color=custom_palette[category], markersize=10)  # Dot at the median peak


        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.tick_params(axis='x', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel(r"${A_{\text{CH}}\text{ (km²)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel('', labelpad=padding_label)

        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks and limit to 4 while ensuring not to include 0
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        #ax.set_xlim(-0.01, 15)
        if category == 'bark_beetle':
            ax.set_xlim(0, 5)  # or 1 depending on how much space you want
        else:
            ax.set_xlim(-0.01, 15)

        ax.set_ylim(0)
        ax.legend(fontsize=fontsize_tick)


        # -----------------------------
        # Centroid shift histogram
        ax = axs[i, 2]
        sns.histplot(
            data=gdf.loc[gdf['DCA_ID'] == category],  # Use .loc[] to filter rows by DCA_ID
            x='centroid_shift_m',
            kde=True,
            line_kws={'linewidth': 4},
            color=custom_palette[category],
            ax=ax,
            stat='count'  # This ensures the histogram is using 'count' for the y-axis
        )

        # Calculate the median for the current category
        median_value = gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'].median()

        # Calculate the 90th percentile for the current category
        percentile_90_value = gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'].quantile(0.9)

        # Print the median and 90th percentile values
        print(f"Median for category {category}: {median_value} m")
        print(f"90th Percentile for category {category}: {percentile_90_value} m")
        # Format the median value (e.g., round to nearest integer)
        formatted_median = f"{int(round(median_value))}"  # Rounds to nearest integer, removes decimals

        # Get the histogram count and bin edges
        hist_data = ax.patches  # This gives us all the histogram bars
        bin_edges = [patch.get_x() for patch in hist_data] + [hist_data[-1].get_x() + hist_data[-1].get_width()]
        bin_width = bin_edges[1] - bin_edges[0]  # Assuming uniform bin widths

        # Calculate the KDE for the current category
        kde = gaussian_kde(gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'])
        kde_at_median = kde(median_value)[0]  # KDE value at median

        # Scale the KDE to the histogram's count scale
        n_total = len(gdf.loc[gdf['DCA_ID'] == category])  # Total number of data points
        hist_max = max([patch.get_height() for patch in hist_data])  # Maximum histogram height (count)

        # Adjust the KDE value based on histogram's maximum count and total data points
        scaled_kde = kde_at_median * n_total * bin_width

        # Plot the median line and adjust to the histogram's count scale
        ax.plot(
            [median_value, median_value], [0, scaled_kde],  # Extend only up to the scaled KDE value
            color=custom_palette[category],
            linestyle='--',
            linewidth=4,
            label=f"M: {formatted_median} m"
        )

        # Add a dot at the upper limit of the median line (on the scaled KDE curve)
        ax.plot(median_value, scaled_kde, 'o', color=custom_palette[category], markersize=20)


        if i == len(category_order) - 1:
            ax.set_xlabel(r"${\Delta_{\text{Centroid}} \text{ (m)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.set_xlim(-0.01, 2400)
        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks for centroid shifts
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower')) 

        # Add the custom legend
        ax.legend(handles=[
                        mpatches.Patch(color=custom_palette[category], label=format_label(category)),  # Main category label
                        Line2D([0], [0], color=custom_palette[category], linestyle='--', linewidth=2, label=f"M: {formatted_median} m")  # Median line legend
                    ],
                  loc='upper right', fontsize=fontsize_legend, frameon=True, fancybox=True, facecolor='white', edgecolor='black', 
                    handlelength=1,   
                    handleheight=0.5)

    # -----------------------------
    # Add common y-axis labels
    fig.text(0.07, 0.5, r"${PDF}$", va='center', rotation='vertical', fontsize=fontsize_label)
    fig.text(0.39, 0.5, r"${PDF}$", va='center', rotation='vertical', fontsize=fontsize_label)
    fig.text(0.69, 0.5, r"${N_{\text{Events}}}$", va='center', rotation='vertical', fontsize=fontsize_label)

    # Add column labels above each subplot
    column_labels = ['a)', 'b)', 'c)']
    x_positions = [0.25, 0.55, 0.82]  # Adjust positions based on your figure layout

    for x_pos, label in zip(x_positions, column_labels):
        fig.text(x_pos, 0.92, label, fontsize=fontsize_title, fontweight='bold', ha='center', va='center')


    fig.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def plot_d_area_ch_area_centroid_disturbances(gdf, ids, s1dm_convex, ids_convex, custom_colors, save_path):
    """
    Plot the comparison between IDS and REFDM disturbance areas and centroid shifts,
    with the legend placed inside the centroid shift plot.
    """
    # Determine the unique DCA_ID values from the filtered dataframe
    unique_dca_ids = gdf['DCA_ID'].unique()

    # Sort the categories based on the custom order
    category_order = sorted(unique_dca_ids, key=lambda x: custom_colors.get(x, x))  
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = {label: custom_colors.get(label, default_colors.pop(0)) for label in category_order}

    sns.set(style="whitegrid")

    n_categories = len(category_order)
    n_cols = 3  # Fixed number of columns
    fig, axs = plt.subplots(
        n_categories, 
        n_cols, 
        figsize=(40, 8 * n_categories), 
        gridspec_kw={'width_ratios': [3, 3, 2], 'wspace': 0.3} # Reduce the width ratios for empty columns
    )

    
    fontsize_supertitle = 44
    fontsize_legend = 40
    fontsize_title = 46
    fontsize_label = 50
    fontsize_tick = 35
    padding_label = 25
    padding_title = 25

    for i, category in enumerate(category_order):
        # Create combined data for violin plots using .loc[] to avoid SettingWithCopyWarning
        combined_data = pd.concat([
            #ids.loc[ids['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='IDS'),
            ids.loc[ids['DCA_ID'] == category, ['DCA_ID', 'area_km2']].copy().assign(Source='IDS'),
            #gdf.loc[gdf['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
            gdf.loc[gdf['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
        ])

        combined_convex = pd.concat([
            ids_convex.loc[ids_convex['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='IDS'),
            s1dm_convex.loc[s1dm_convex['DCA_ID'] == category, ['DCA_ID', 'area_km2']].assign(Source='REFDM')
        ])

        # Add a new column for convex diameter
        combined_convex = combined_convex.assign(
            convex_diameter_km=np.sqrt(4 * combined_convex['area_km2'] / np.pi)
        )

        ax = axs[i, 0]
        data_refdm = combined_data.loc[(combined_data['Source'] == 'REFDM') & (combined_data['DCA_ID'] == category)]
        data_ids = combined_data.loc[(combined_data['Source'] == 'IDS') & (combined_data['DCA_ID'] == category)]

        median_refdm = np.median(data_refdm['area_km2'])
        median_ids = np.median(data_ids['area_km2'])

        if np.isnan(median_ids) or np.isnan(median_refdm):
            logging.warning(f"No valid data for category {category}, skipping KDE.")
            continue  # Skip this category completely

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Plot KDE for IDS
        sns.kdeplot(
            data=data_ids['area_km2'],
            color='black',
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='IDS',
            alpha=0.8,
            linestyle='-'
        )

        # Plot KDE for REFDM
        sns.kdeplot(
            data=data_refdm['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='S1DM',
            alpha=1,
            linestyle='-'
        )

        # Calculate the 90th percentile for disturbance area
        percentile_90_refdm = np.percentile(data_refdm['area_km2'], 90)
        percentile_90_ids = np.percentile(data_ids['area_km2'], 90)


        print(f"Median IDS for category {category}: {median_ids} km²")
        print(f"Median S1DM for category {category}: {median_refdm} km²")
        print(f"90th Percentile IDS for category {category}: {percentile_90_ids} km²")
        print(f"90th Percentile S1DM for category {category}: {percentile_90_refdm} km²")

        # # Add vertical lines at the 90th percentile
        # ax.axvline(x=percentile_90_ids, color='black', linestyle=':', linewidth=3, alpha=0.8, label=f'IDS 90th: {percentile_90_ids:.2f}', marker='o', markersize=16)
        # ax.axvline(x=percentile_90_refdm, color=custom_palette[category], linestyle=':', linewidth=3, alpha=1, label=f'S1DM 90th: {percentile_90_refdm:.2f}', marker='s', markersize=16)

        # Adjust median formatting for Bark Beetle DCA_ID
        if category == 'bark_beetle':
            median_refdm_label = f'S1DM Median: {median_refdm:.4f}' if median_refdm > 0 else 'S1DM Median: 0.0'
            median_ids_label = f'IDS Median: {median_ids:.4f}' if median_ids > 0 else 'IDS Median: 0.0'
        else:
            median_refdm_label = f'S1DM Median: {median_refdm:.2f}'
            median_ids_label = f'IDS Median: {median_ids:.2f}'

        from scipy.stats import gaussian_kde

        # Calculate KDE for IDS
        kde_ids = gaussian_kde(data_ids['area_km2'])
        x_vals_ids = np.linspace(min(data_ids['area_km2']), max(data_ids['area_km2']), 1000)
        kde_vals_ids = kde_ids(x_vals_ids)
        kde_at_median_ids = kde_ids(median_ids)[0]  # KDE value at median

        # Draw the median line for IDS
        ax.plot([median_ids, median_ids], [0, kde_at_median_ids], color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label)
        ax.plot(median_ids, kde_at_median_ids, 'o', color='black', markersize=10)  # Dot at the median peak

        # Calculate KDE for REFDM
        kde_refdm = gaussian_kde(data_refdm['area_km2'])
        x_vals_refdm = np.linspace(min(data_refdm['area_km2']), max(data_refdm['area_km2']), 1000)
        kde_vals_refdm = kde_refdm(x_vals_refdm)
        kde_at_median_refdm = kde_refdm(median_refdm)[0]  # KDE value at median

        # Draw the median line for REFDM
        ax.plot([median_refdm, median_refdm], [0, kde_at_median_refdm], color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label)
        ax.plot(median_refdm, kde_at_median_refdm, 'o', color=custom_palette[category], markersize=10)  # Dot at the median peak

        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.tick_params(axis='x', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel(r"${A_{\text{D}}\text{ (km²)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel('', labelpad=padding_label)

        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks and limit to 4 while ensuring not to include 0
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        ax.set_xlim(-0.01, 10)
        ax.set_ylim(0)
        ax.legend(fontsize=fontsize_tick)

        ax = axs[i, 1]
        data_refdm = combined_convex.loc[(combined_convex['Source'] == 'REFDM') & (combined_convex['DCA_ID'] == category)]
        data_ids = combined_convex.loc[(combined_convex['Source'] == 'IDS') & (combined_convex['DCA_ID'] == category)]

        # median_refdm = np.median(data_refdm['area_km2'])
        # median_ids = np.median(data_ids['area_km2'])
        median_refdm = np.median(data_refdm['convex_diameter_km'])
        median_ids = np.median(data_ids['convex_diameter_km'])
        

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Plot KDE for IDS
        sns.kdeplot(
            data=data_ids['area_km2'],
            color='black',
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='IDS',
            alpha=0.8,
            linestyle='-'
        )

        # Plot KDE for REFDM
        sns.kdeplot(
            data=data_refdm['area_km2'],
            color=custom_palette[category],
            ax=ax,
            common_norm=True,
            linewidth=4,
            label='S1DM',
            alpha=1,
            linestyle='-'
        )

        #  # Calculate the 90th percentile for disturbance area
        percentile_90_refdm = np.percentile(data_refdm['area_km2'], 90)
        percentile_90_ids = np.percentile(data_ids['area_km2'], 90)


        print(f"Median IDS for category {category}: {median_ids} km²")
        print(f"Median S1DM for category {category}: {median_refdm} km²")
        print(f"90th Percentile IDS for category {category}: {percentile_90_ids} km²")
        print(f"90th Percentile S1DM for category {category}: {percentile_90_refdm} km²")

        # # Add vertical lines at the 90th percentile
        # ax.axvline(x=percentile_90_ids, color='black', linestyle=':', linewidth=3, alpha=0.8, label=f'IDS 90th: {percentile_90_ids:.2f}', marker='o', markersize=16)
        # ax.axvline(x=percentile_90_refdm, color=custom_palette[category], linestyle=':', linewidth=3, alpha=1, label=f'S1DM 90th: {percentile_90_refdm:.2f}', marker='s', markersize=16)

        # Adjust median formatting for Bark Beetle DCA_ID
        if category == 'bark_beetle':
            median_refdm_label = f'S1DM Median: {median_refdm:.4f}' if median_refdm > 0 else 'S1DM Median: 0.0'
            median_ids_label = f'IDS Median: {median_ids:.4f}' if median_ids > 0 else 'IDS Median: 0.0'
        else:
            median_refdm_label = f'S1DM Median: {median_refdm:.2f}'
            median_ids_label = f'IDS Median: {median_ids:.2f}'

        # ax.axvline(x=median_ids, color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label, marker='o', markersize=16)
        # ax.axvline(x=median_refdm, color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label, marker='s', markersize=16)
        
        from scipy.stats import gaussian_kde

        # Calculate KDE for IDS
        kde_ids = gaussian_kde(data_ids['area_km2'])
        x_vals_ids = np.linspace(min(data_ids['area_km2']), max(data_ids['area_km2']), 1000)
        kde_vals_ids = kde_ids(x_vals_ids)
        kde_at_median_ids = kde_ids(median_ids)[0]  # KDE value at median

        # Draw the median line for IDS
        ax.plot([median_ids, median_ids], [0, kde_at_median_ids], color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label)
        ax.plot(median_ids, kde_at_median_ids, 'o', color='black', markersize=10)  # Dot at the median peak

        # Calculate KDE for REFDM
        kde_refdm = gaussian_kde(data_refdm['area_km2'])
        x_vals_refdm = np.linspace(min(data_refdm['area_km2']), max(data_refdm['area_km2']), 1000)
        kde_vals_refdm = kde_refdm(x_vals_refdm)
        kde_at_median_refdm = kde_refdm(median_refdm)[0]  # KDE value at median

        # Draw the median line for REFDM
        ax.plot([median_refdm, median_refdm], [0, kde_at_median_refdm], color=custom_palette[category], linestyle='--', linewidth=3, alpha=1, label=median_refdm_label)
        ax.plot(median_refdm, kde_at_median_refdm, 'o', color=custom_palette[category], markersize=10)  # Dot at the median peak


        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.tick_params(axis='x', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel(r"${A_{\text{CH}}\text{ (km²)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel('', labelpad=padding_label)

        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks and limit to 4 while ensuring not to include 0
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        ax.set_xlim(-0.01, 15)
        ax.set_ylim(0)
        ax.legend(fontsize=fontsize_tick)


        
        ax = axs[i, 2]
        sns.histplot(
            data=gdf.loc[gdf['DCA_ID'] == category],  # Use .loc[] to filter rows by DCA_ID
            x='centroid_shift_m',
            kde=True,
            line_kws={'linewidth': 4},
            color=custom_palette[category],
            ax=ax,
            stat='count'  # This ensures the histogram is using 'count' for the y-axis
        )

        # Calculate the median for the current category
        median_value = gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'].median()

        # Calculate the 90th percentile for the current category
        percentile_90_value = gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'].quantile(0.9)

        # Print the median and 90th percentile values
        print(f"Median for category {category}: {median_value} m")
        print(f"90th Percentile for category {category}: {percentile_90_value} m")
        # Format the median value (e.g., round to nearest integer)
        formatted_median = f"{int(round(median_value))}"  # Rounds to nearest integer, removes decimals

        # Get the histogram count and bin edges
        hist_data = ax.patches  # This gives us all the histogram bars
        bin_edges = [patch.get_x() for patch in hist_data] + [hist_data[-1].get_x() + hist_data[-1].get_width()]
        bin_width = bin_edges[1] - bin_edges[0]  # Assuming uniform bin widths

        # Calculate the KDE for the current category
        kde = gaussian_kde(gdf.loc[gdf['DCA_ID'] == category, 'centroid_shift_m'])
        kde_at_median = kde(median_value)[0]  # KDE value at median

        # Scale the KDE to the histogram's count scale
        n_total = len(gdf.loc[gdf['DCA_ID'] == category])  # Total number of data points
        hist_max = max([patch.get_height() for patch in hist_data])  # Maximum histogram height (count)

        # Adjust the KDE value based on histogram's maximum count and total data points
        scaled_kde = kde_at_median * n_total * bin_width

        # Plot the median line and adjust to the histogram's count scale
        ax.plot(
            [median_value, median_value], [0, scaled_kde],  # Extend only up to the scaled KDE value
            color=custom_palette[category],
            linestyle='--',
            linewidth=4,
            label=f"M: {formatted_median} m"
        )

        # Add a dot at the upper limit of the median line (on the scaled KDE curve)
        ax.plot(median_value, scaled_kde, 'o', color=custom_palette[category], markersize=20)


        if i == len(category_order) - 1:
            ax.set_xlabel(r"${\Delta_{\text{Centroid}} \text{ (m)} }$", fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.set_xlim(-0.01, 2400)
        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks for centroid shifts
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower')) 

        # Add the custom legend
        ax.legend(handles=[
                        mpatches.Patch(color=custom_palette[category], label=format_label(category)),  # Main category label
                        Line2D([0], [0], color=custom_palette[category], linestyle='--', linewidth=2, label=f"M: {formatted_median} m")  # Median line legend
                    ],
                  loc='upper right', fontsize=fontsize_legend, frameon=True, fancybox=True, facecolor='white', edgecolor='black', 
                    handlelength=1,   
                    handleheight=0.5)

 

    # Add common y-axis labels for the plots
    fig.text(0.07, 0.5, r"${PDF}$", va='center', rotation='vertical', fontsize=fontsize_label)
    fig.text(0.39, 0.5, r"${PDF}$", va='center', rotation='vertical', fontsize=fontsize_label)
    fig.text(0.69, 0.5, r"${N_{\text{Events}}}$", va='center', rotation='vertical', fontsize=fontsize_label)

    fig.text(0.24, 0.9, "a)", va='center', rotation='horizontal', fontsize=fontsize_label)
    fig.text(0.55, 0.9, "b)", va='center', rotation='horizontal', fontsize=fontsize_label)
    fig.text(0.8, 0.9, "c)", va='center', rotation='horizontal', fontsize=fontsize_label)

    # Adjust layout for better alignment
    #plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust margins (left, bottom, right, top)

    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def plot_disturbance_signal_duration(refdm_dissolved, save_path):
    """
    Plot a grouped bar chart showing the count of events by signal_duration for each DCA_ID.

    Parameters:
    - refdm_dissolved (GeoDataFrame): GeoDataFrame containing 'DCA_ID' and 'signal_duration' columns.
    - save_path (str): Path to save the generated plot.
    """
    # Count how often each signal_duration occurs for each DCA_ID
    duration_counts = refdm_dissolved.groupby(['DCA_ID', 'signal_duration']).size().reset_index(name='Count')

    # Pivot the table for easy plotting
    pivot_table = duration_counts.pivot(index='DCA_ID', columns='signal_duration', values='Count').fillna(0)

    # Create a custom colormap: Darker shades of red
    colors = ['#FFAF6E', '#F28353', '#E65837', '#D92C1C', '#CC0000']  # Light to dark red
    cmap = mcolors.ListedColormap(colors)

    # Plot the grouped bar plot
    fig, ax = plt.subplots(figsize=(15, 5))  # Adjusted size: wider and taller

    # Set font sizes for various components
    plt.rcParams.update({
        'font.size': 16,           # Global font size
        'axes.titlesize': 18,      # Title font size
        'axes.labelsize': 20,      # X and Y label font size
        'xtick.labelsize': 18,     # X tick label font size
        'ytick.labelsize': 18,     # Y tick label font size
    })

    # Plot the pivot table with the custom colormap
    pivot_table.plot(kind='bar', ax=ax, cmap=cmap, width=0.8, edgecolor='None', legend=False)  # Reduced bar width

    # Add labels to each bar with a buffer
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                format(height, '.0f'),
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                xytext=(0, 8),  # 8 points vertical offset (to move it away from the top of the bars)
                textcoords='offset points',
                fontsize=13  # Set the font size smaller here
            )

    # Set the y-axis limit to the next multiple of 100 above the max count
    max_count = pivot_table.values.max()
    ax.set_ylim(0, np.ceil(max_count / 100) * 100 + 50)  # Added extra space at the top

    # Format x-axis labels
    ax.set_xticklabels([format_label(label.get_text()) for label in ax.get_xticklabels()])
    
    # Set labels and title with additional buffer
    ax.set_xlabel('Disturbance Type', labelpad=20, fontsize=20)
    ax.set_ylabel('Number of Events', labelpad=20, fontsize=20)

    plt.xticks(rotation=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    # Explicitly set the font size for the x-tick labels
    ax.tick_params(axis='x', labelsize=18)  # Set font size for x-tick labels
    ax.tick_params(axis='y', labelsize=18) 

    # Create a colorbar and position it closer to the plot
    norm = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5], ncolors=len(colors), clip=True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', pad=0.04)
    cbar.set_label(f'S1 Signal Persistance', fontsize=20, labelpad=20)  # Added clearer label
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(['1 Year', '2 Years', '3 Years', '4 Years', '5 Years'])
    # Remove the top and right spines (keeping the left and bottom spines)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(False) # linestyle='--', color='lightgray', alpha=0.5)  # Adjust alpha to your preference
    plt.tight_layout()

    # Save the figure with a specific DPI to fit an A4 page
    plt.savefig(save_path, dpi=400, bbox_inches='tight')

    plt.show()

def plot_signal_counts_by_diff_year(gdf, custom_colors, save_path):
    """
    Plot the counts of disturbances for each difference in years (SURVEY_Y - S1_YEAR),
    grouped by DCA_ID, with custom colors.
    Disturbances are ordered as wind, bark_beetle, defoliators.
    """

    # Calculate counts by difference in years
    gdf = gdf.copy()
    gdf['SURVEY_Y'] = gdf['SURVEY_Y'].astype(int)
    gdf['S1_YEAR'] = gdf['S1_YEAR'].astype(int)
    gdf['diff_year'] = gdf['SURVEY_Y'] - gdf['S1_YEAR']
    count_by_diff_year_dca = gdf.groupby(['diff_year', 'DCA_ID']).size().reset_index(name='Count')

    ordered_types = ['wind', 'bark_beetle', 'defoliators']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot each disturbance type
    for dca_id in ordered_types:
        if dca_id in count_by_diff_year_dca['DCA_ID'].unique():
            dca_data = count_by_diff_year_dca[count_by_diff_year_dca['DCA_ID'] == dca_id]
            sns.lineplot(
                data=dca_data,
                x='diff_year',
                y='Count',
                label=format_label(dca_id),
                color=custom_colors.get(dca_id, '#000000'),
                marker='o',
                markersize=10,
                linewidth=4,
                ax=ax
            )

    # Format plot
    ax.set_title("", fontsize=18, pad=10)
    ax.set_xlabel(r"${Lag_{\text{(IDS-S1DM)}}}$", fontsize=24, labelpad=10)
    ax.set_ylabel(r"${N_{\text{Annual Signals}}}$", fontsize=24, labelpad=10)
    ax.set_xticks(range(-2, 3))
    ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=18)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=22, loc='upper right', title="", title_fontsize=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def plot_signal_counts_by_diff_year_combined(gdf1, gdf2, custom_colors, save_path):
    """
    Create two side-by-side line plots showing the counts of disturbances for each difference in years (SURVEY_Y - S1_YEAR),
    grouped by DCA_ID, with custom colors.
    Disturbances are ordered as wind, bark_beetle, defoliators.
    """

    def preprocess_and_count(gdf):
        gdf.loc[:, 'SURVEY_Y'] = gdf['SURVEY_Y'].astype(int)
        gdf.loc[:, 'S1_YEAR'] = gdf['S1_YEAR'].astype(int)
        gdf.loc[:, 'diff_year'] = gdf['SURVEY_Y'] - gdf['S1_YEAR']
        return gdf.groupby(['diff_year', 'DCA_ID']).size().reset_index(name='Count')

    count_by_diff_year_dca1 = preprocess_and_count(gdf1)
    count_by_diff_year_dca2 = preprocess_and_count(gdf2)

    # Desired order
    ordered_types = ['wind', 'bark_beetle', 'defoliators']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    def plot_single_dataset(ax, data, title, show_legend=False):
        for dca_id in ordered_types:
            if dca_id in data['DCA_ID'].unique():
                dca_data = data.loc[data['DCA_ID'] == dca_id]
                sns.lineplot(
                    data=dca_data,
                    x="diff_year",
                    y="Count",
                    label=format_label(dca_id) if show_legend else None,
                    color=custom_colors.get(dca_id, '#000000'),
                    marker="o",
                    markersize=10,
                    linewidth=4,
                    ax=ax,
                )

        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xlabel(r"${Lag_{\text{(IDS-S1DM)}}}$", fontsize=20, labelpad=10)
        ax.set_ylabel(r"${N_{\text{Annual Signals}}}$", fontsize=20, labelpad=10)
        ax.set_xticks(range(-2, 3))
        ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=14)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5)

        # Despine top and right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=18, loc='upper right', title="", title_fontsize=1)

    plot_single_dataset(axes[0], count_by_diff_year_dca1, "", show_legend=False)
    plot_single_dataset(axes[1], count_by_diff_year_dca2, "", show_legend=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


def calculate_plot_overlap_percentage_kde(ids, s1dm,custom_colors, save_path):

    # Ensure both datasets have the same coordinate reference system (CRS)
    if ids.crs != s1dm.crs:
        s1dm = s1dm.to_crs(ids.crs)

    # Filter for common IDX_D values
    common_idx_d = set(ids["IDX_D"]).intersection(s1dm["IDX_D"])

    # Filter both datasets for common IDX_D values
    ids_common = ids[ids["IDX_D"].isin(common_idx_d)]
    s1cd_common = s1dm[s1dm["IDX_D"].isin(common_idx_d)]

    # Initialize a result list to store the calculations
    results = []

    # Loop through common IDX_Ds
    for idx_d in common_idx_d:
        # Get the corresponding polygons from both datasets
        ids_poly = ids_common[ids_common["IDX_D"] == idx_d].geometry.union_all()
        s1cd_poly = s1cd_common[s1cd_common["IDX_D"] == idx_d].geometry.union_all()

        # Calculate the areas
        ids_area = ids_poly.area
        intersection_area = ids_poly.intersection(s1cd_poly).area

        # Calculate percentage coverage
        percentage = (intersection_area / ids_area) * 100 if ids_area > 0 else 0

        # Calculate the area of the S1CD polygon
        s1cd_area = s1cd_poly.area
        # Calculate percentage coverage of S1CD
        percentage_s1cd = (intersection_area / s1cd_area) * 100 if s1cd_area > 0 else 0

        # Get DCA_ID for this IDX_D (assuming it exists in both datasets)
        dca_id = ids_common[ids_common["IDX_D"] == idx_d]["DCA_ID"].iloc[0]

        # Append results
        results.append({"IDX_D": idx_d, "DCA_ID": dca_id, "percentage_ids": percentage, "percentage_s1cd": percentage_s1cd})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: S1CD coverage of IDS polygons
    for dca_id in results_df['DCA_ID'].unique():
        if dca_id in custom_colors:
            dca_id_df = results_df[results_df['DCA_ID'] == dca_id]
            
            # Calculate the KDE for 'percentage_ids'
            kde = gaussian_kde(dca_id_df['percentage_ids'])
            
            # Generate a range of values for x (from 0 to 100)
            x_vals = np.linspace(0, 100, 1000)
            
            # Evaluate the KDE on the x_vals range
            kde_vals = kde(x_vals)
            
            # Find the median (50th percentile) by integrating the CDF
            cdf_vals = np.cumsum(kde_vals) / np.sum(kde_vals)
            median_idx = np.where(cdf_vals >= 0.5)[0][0]
            median = x_vals[median_idx]
            
            # Find the y-value (KDE value) at the median
            kde_at_median = kde(median)
            
            # Plot the KDE for the current DCA_ID
            #sns.kdeplot(dca_id_df['percentage_ids'], fill=False, label=f"{format_label(dca_id)} ({median:.1f}%)", color=custom_colors[dca_id], ax=axes[0])
            sns.histplot(dca_id_df['percentage_ids'], bins=10, cumulative=False, multiple="dodge", stat="density", 
                    element="bars", color=custom_colors[dca_id], label=f"{format_label(dca_id)}", 
                    linewidth=2, alpha=0.7, ax=axes[0])

            # Get y-axis limits
            y_min, y_max = axes[0].get_ylim()
            
            # Plot the median line from the curve to y=0 using y_min and y_max
            axes[0].plot([median, median], [0, kde_at_median[0]], color=custom_colors[dca_id], linestyle='--')
            
            # Add the median label (formatted to 2 decimal places)
            #axes[0].text(median, kde_at_median[0] + (y_max - y_min) * 0.01, f'{dca_id} Median: {median:.2f}', color=custom_colors[dca_id], ha='left', va='bottom')
            
            # Add a dot at the upper limit of the median line (on the KDE curve)
            axes[0].plot(median, kde_at_median[0], 'o', color=custom_colors[dca_id], markersize=6)

    # Plot 2: IDS coverage of S1CD polygons
    for dca_id in results_df['DCA_ID'].unique():
        if dca_id in custom_colors:
            dca_id_df = results_df[results_df['DCA_ID'] == dca_id]
            
            # Calculate the KDE for 'percentage_s1cd'
            kde = gaussian_kde(dca_id_df['percentage_s1cd'])
            
            # Generate a range of values for x (from 0 to 100)
            x_vals = np.linspace(0, 100, 1000)
            
            # Evaluate the KDE on the x_vals range
            kde_vals = kde(x_vals)
            
            # Find the median (50th percentile) by integrating the CDF
            cdf_vals = np.cumsum(kde_vals) / np.sum(kde_vals)
            median_idx = np.where(cdf_vals >= 0.5)[0][0]
            median = x_vals[median_idx]
            
            # Find the y-value (KDE value) at the median
            kde_at_median = kde(median)
            
            # Plot the KDE for the current DCA_ID
            # sns.kdeplot(dca_id_df['percentage_s1cd'], fill=False, label=f"{format_label(dca_id)} ({median:.1f}%)", color=custom_colors[dca_id], ax=axes[1])
            
            # Plot cumulative histogram
            sns.histplot(dca_id_df['percentage_s1cd'], bins=10, cumulative=False, multiple="dodge", stat="density", 
                    element="bars", color=custom_colors[dca_id], label=f"{format_label(dca_id)}", 
                    linewidth=2, alpha=0.7, ax=axes[1])

            # Get y-axis limits
            y_min, y_max = axes[1].get_ylim()
            
            # Plot the median line from the curve to y=0 using y_min and y_max
            axes[1].plot([median, median], [0, kde_at_median[0]], color=custom_colors[dca_id], linestyle='--')
            
            # Add a dot at the upper limit of the median line (on the KDE curve)
            axes[1].plot(median, kde_at_median[0], 'o', color=custom_colors[dca_id], markersize=6)

    # Customize both plots
    # Customize both plots
    axes[0].set_xlabel(r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}{A_{\text{IDS}}}$", fontsize=24, labelpad=10)  # Added labelpad
    axes[0].set_ylabel('KDE', fontsize=20, labelpad=10)  # Added labelpad
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].set_xlim(0, 100)  # Cut off the x-axis at 100 and avoid negative values
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].legend(loc='upper right', fontsize=18)

    axes[1].set_xlabel(r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}{A_{\text{S1DM}}}$", fontsize=24, labelpad=10)  # Added labelpad
    axes[1].set_ylabel('KDE', fontsize=20, labelpad=10)  # Added labelpad
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].set_xlim(0, 100)  # Cut off the x-axis at 100 and avoid negative values
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].legend(loc='upper right', fontsize=18)

    # Format y-axis tick labels to 2 decimal places
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        # Remove the upper and right spines
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # Set the grid color to light gray
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with a specific DPI to fit an A4 page
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def calculate_plot_overlap_percentage(ids, s1dm, custom_colors, save_path):
    # Ensure CRS matches
    if ids.crs != s1dm.crs:
        s1dm = s1dm.to_crs(ids.crs)

    # Filter for common IDX_D values
    common_idx_d = set(ids["IDX_D"]).intersection(s1dm["IDX_D"])
    ids_common = ids[ids["IDX_D"].isin(common_idx_d)]
    s1cd_common = s1dm[s1dm["IDX_D"].isin(common_idx_d)]

    # Store results
    results = []
    for idx_d in common_idx_d:
        ids_poly = ids_common[ids_common["IDX_D"] == idx_d].geometry.unary_union
        s1cd_poly = s1cd_common[s1cd_common["IDX_D"] == idx_d].geometry.unary_union
        
        ids_area = ids_poly.area
        s1cd_area = s1cd_poly.area
        intersection_area = ids_poly.intersection(s1cd_poly).area
        
        percentage_ids = (intersection_area / ids_area) * 100 if ids_area > 0 else 0
        percentage_s1cd = (intersection_area / s1cd_area) * 100 if s1cd_area > 0 else 0
        
        dca_id = ids_common[ids_common["IDX_D"] == idx_d]["DCA_ID"].iloc[0]
        results.append({"DCA_ID": dca_id, "percentage_ids": percentage_ids, "percentage_s1cd": percentage_s1cd})

    results_df = pd.DataFrame(results)
    
    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot IDS coverage with custom colors
    sns.kdeplot(
        data=results_df, x="percentage_ids", hue="DCA_ID", hue_order=["bark_beetle", "wind", "fire", "defoliators"],
        log_scale=True, element="step", fill=False,
        cumulative=True, stat="density", common_norm=False, ax=axes[0], palette=custom_colors
    )

    # Plot S1CD coverage with custom colors
    sns.kdeplot(
        data=results_df, x="percentage_s1cd", hue="DCA_ID", hue_order=["bark_beetle", "wind", "fire", "defoliators"],
        log_scale=True, element="step", fill=False,
        cumulative=True, stat="density", common_norm=False, ax=axes[1], palette=custom_colors
    )
    
    # Customize plots
    axes[0].set_xlabel("IDS Coverage (%)", fontsize=16)
    axes[1].set_xlabel("S1CD Coverage (%)", fontsize=16)
    
    for ax in axes:
        ax.set_ylabel("Density", fontsize=16)  # Change to "Density"
        ax.set_xlim(0, 100)  # Ensure percentages are within 0 to 100
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(title="DCA_ID", fontsize=12)
    
    # Tight layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_percentages_histograms(ids_gdf, s1dm_gdf, custom_colors, save_path=None):
    """
    Plots stacked histograms for percentage_ids and percentage_s1cd with customized bins, formatting, and legends.
    Disturbances are ordered as wind, bark_beetle, defoliators.
    """
    results_df = calculate_overlap_percentages(ids_gdf, s1dm_gdf)

    # Set bin edges: first bin is from -10 to 0, then 1-10, 11-20, etc.
    bins = [-20] + list(np.arange(1, 101, 20))  

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Larger figure

    # Remove drought disturbances
    results_df = results_df[results_df["DCA_ID"] != "drought"]

    # Desired order
    ordered_types = ['wind', 'bark_beetle', 'defoliators']
    ordered_types = [t for t in ordered_types if t in results_df["DCA_ID"].unique()]

    # Highlight the area around 0
    axes[0].axvspan(-20, 0, color='grey', alpha=0.3)
    axes[1].axvspan(-20, 0, color='grey', alpha=0.3)

    # Plot stacked histogram for percentage_ids
    sns.histplot(
        data=results_df, 
        x="percentage_ids", 
        bins=bins, 
        binwidth=5,
        element='bars',
        kde=False, 
        ax=axes[0], 
        hue="DCA_ID",  
        hue_order=ordered_types,
        multiple="dodge", 
        shrink=.8, 
        stat="probability",
        palette=custom_colors,  
        legend=False  
    )

    # Plot stacked histogram for percentage_s1cd
    sns.histplot(
        data=results_df, 
        x="percentage_s1cd", 
        bins=bins, 
        binwidth=5,
        element='bars',
        kde=False, 
        ax=axes[1], 
        hue="DCA_ID",  
        hue_order=ordered_types,
        multiple="dodge", 
        shrink=.8, 
        stat="probability",
        palette=custom_colors,  
        legend=False  
    )

    # Add median values to the legend in the desired order
    legend_handles = []
    legend_handles_s1dm = []
    for dca_id in ordered_types:
        if dca_id in results_df["DCA_ID"].unique():
            color = custom_colors[dca_id]
            dca_df = results_df[results_df["DCA_ID"] == dca_id]

            median_value = np.median(dca_df["percentage_ids"])
            median_value_s1dm = np.median(dca_df["percentage_s1cd"])

            # Show more decimals for bark_beetle
            if dca_id == "bark_beetle":
                median_label = f"{median_value:.10f}" if median_value != 0 else "0"
                median_label_s1dm = f"{median_value_s1dm:.10f}" if median_value_s1dm != 0 else "0"
            else:
                median_label = f"{median_value:.2f}" if median_value != 0 else "0"
                median_label_s1dm = f"{median_value_s1dm:.2f}" if median_value_s1dm != 0 else "0"

            legend_handles.append(Line2D([0], [0], color=color, lw=4, 
                                        label=f"{format_label(dca_id)} (Median: {median_label})"))
            legend_handles_s1dm.append(Line2D([0], [0], color=color, lw=4, 
                                            label=f"{format_label(dca_id)} (Median: {median_label_s1dm})"))

    # Formatting for the first subplot (percentage_ids)
    axes[0].set_xlabel(r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}{A_{\text{IDS}}}$", fontsize=28, labelpad=15)
    axes[0].set_ylabel("Probability", fontsize=22, labelpad=15)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(True, linestyle="--", alpha=0.8)
    axes[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0].set_xlim(left=-20, right=101)
    axes[0].set_xticks(np.arange(0, 101, 20))

    # Formatting for the second subplot (percentage_s1cd)
    axes[1].set_xlabel(r"$\frac{A_{\text{IDS} \bigcap \text{S1DM}}}{A_{\text{S1DM}}}$", fontsize=28, labelpad=15)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].grid(True, linestyle="--", alpha=0.8)
    axes[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axes[1].set_xlim(left=-20, right=101)
    axes[1].set_ylabel('')
    axes[1].tick_params(left=False)
    axes[1].set_xticks(np.arange(0, 101, 20))

    # Add larger legends
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=20, title_fontsize=22)
    axes[1].legend(handles=legend_handles_s1dm, loc='upper right', fontsize=20, title_fontsize=22)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    plt.show()

def plot_study_area(area_path, region_id, tcc_nc_path, s1_tiles_boundary_path, ids, custom_colors, save_path, logging):
    logging.info("Load the USA Mainland and Region 8 Shape ...")
    mainland = get_mainland(area_path)
    region_8 = load_and_extract_region(area_path, region_id=region_id)

    logging.info("Load the TCC Region 8 Map ...")
    tcc_dataset = load_tcc_nc_dataset(tcc_nc_path)
    logging.info("Plot Study area figure ...")
    combine_study_area_plots(tcc_dataset, s1_tiles_boundary_path, mainland, region_8, ids, region_id, custom_colors, save_path, logging)

def combine_study_area_plots(cropped_forest, s1_tiles_boundary_path, usa_mainland, r8, gdf, region_id, custom_colors, save_path, logging):
    """
    Plot the TCC map with disturbance types and save the figure.
    """
    # Normalize the TCC values
    cropped_forest = normalize_tcc(cropped_forest)
    s1_tiles = load_data(s1_tiles_boundary_path)
    logging.info(f"s1_tiles contains {len(s1_tiles)} features")
    if s1_tiles.crs != cropped_forest.rio.crs:
        s1_tiles = s1_tiles.to_crs('EPSG:4326')

    # Set Seaborn style
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Ensure s1_tiles, r8, and disturbances use the raster CRS
    target_crs = cropped_forest.rio.crs

    if s1_tiles.crs != target_crs:
        s1_tiles = s1_tiles.to_crs(target_crs)

    if r8.crs != target_crs:
        r8 = r8.to_crs(target_crs)

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    
    # Plot the entire USA in grey in the upper left corner
    #sub_ax = fig.add_axes([-0.05, 0.78, 0.25, 0.25])  # [left, bottom, width, height]
    #sub_ax = fig.add_axes([0.02, 0.78, 0.20, 0.20])  
    sub_ax = fig.add_axes([0.04, 0.70, 0.25, 0.25])  # [left, bottom, width, height]
    plot_mainland_map(sub_ax, usa_mainland, region_id)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Plot the TCC map within Region 8 boundaries
    plot_tcc_map(ax, cropped_forest, custom_cmap)
    
    # Plot the region outline
    r8.boundary.plot(ax=ax, linewidth=1, color='#000000')
    
     # Add a label in the center of Region 8
    ax.text(
        0.5, 0.92, "S1CD Tiles",  # Centered in x, slightly lower than before
        fontsize=16, fontweight='normal', color="black",  # Not bold
        ha='center', va='center',
        transform=ax.transAxes,  # Position relative to axis
        bbox=dict(facecolor='white', edgecolor='black')
    )

    # Plot disturbance types
    plot_disturbance_types(ax, gdf, custom_colors)
    
    # Customize the plot
    ax.axis('off')  # Remove axis and frame
    
    # # Create legend for disturbance types
    # legend_patches = [mpatches.Patch(color=color, label=format_label(disturbance)) for disturbance, color in custom_colors.items()]
    # ax.legend(handles=legend_patches, fontsize=18, title="Disturbance Type", title_fontsize=20, loc='center left', facecolor='white', framealpha=1)
    # Create legend with fixed order: wind → bark_beetle → defoliators
    ordered_types = ['wind', 'bark_beetle', 'defoliators']
    legend_patches = [
        mpatches.Patch(color=custom_colors[dist], label=format_label(dist))
        for dist in ordered_types if dist in gdf['DCA_ID'].unique()
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=18,
        title="Disturbance Type",
        title_fontsize=20,
        loc='center left',
        facecolor='white',
        framealpha=1
    )

    s1_tiles.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()

def normalize_tcc(cropped_forest):
    """
    Normalize the 'tcc' values in the cropped forest data to range between 0 and 100.
    """
    cropped_forest['tcc'] = (cropped_forest['tcc'] / cropped_forest['tcc'].max()) * 100
    cropped_forest['tcc'] = cropped_forest['tcc'].clip(min=0, max=100)
    return cropped_forest

def plot_mainland_map(ax, usa_mainland, region_id):
    """
    Plot the entire USA mainland with Region 8 highlighted.
    """
    usa_mainland[usa_mainland['REGION'] != region_id].plot(ax=ax, color='grey', edgecolor='grey')
    usa_mainland[usa_mainland['REGION'] == region_id].plot(ax=ax, color='black', edgecolor='black')
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
    plot = cropped_forest['tcc'].plot(ax=ax, cmap=custom_cmap, add_colorbar=False, add_labels=False)
    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.05, aspect=10, shrink=0.8)
    cbar.mappable.set_cmap(custom_cmap)
    #cbar.ax.set_position([0.15, 0.2, 0., 0.03])  # [left, bottom, width, height]
    cbar.ax.set_position([0.37, 0.35, 0.35, 0.03])  # [left, bottom, width, height]
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
        subset = refdm_dissolved[refdm_dissolved['DCA_ID'] == disturbance]
        if subset.empty:
            continue  # Skip if no polygons of this type

        # Plot with white edge first
        subset.plot(ax=ax, linewidth=3.5, color=color, edgecolor='white')

        # Then plot with actual color and thinner edge
        subset.plot(ax=ax, linewidth=2.5, color=color, edgecolor=color)


def plot_disturbance_layers(
    disturbance_files,
    folder_manual,
    ids_gdf,
    s1dm_gdf,
    figure_path=None,
    disturbances=["wind", "bark_beetle", "defoliators"],
    layers=["Manual", "IDS", "S1DM"],
    layer_colors={"Manual": "#F03A47", "IDS": "#70c2fc", "S1DM": "white"},
    disturbance_colors={"wind": "#1f77b4", "bark_beetle": "#714709", "defoliators": "#FF9505"},
    figsize=(22, 20),
    fontsize_title=22,
    fontsize_axis=16,
    fontsize_legend=25
):
    """
    Create a publication-ready figure showing PlanetScope images of forest disturbances
    overlaid with manual, IDS, and S1DM polygons.

    Parameters
    ----------
    disturbance_files : dict
        Dictionary with disturbance names as keys and each value a dict containing:
        'file' (str, raster path), 'idx' (int), 'date' (str).
    folder_manual : path
        Dictionary containing 'manual', 'input', and 'figure' folder paths.
    ids_gdf : geopandas.GeoDataFrame
        IDS polygon layer.
    s1dm_gdf : geopandas.GeoDataFrame
        S1DM polygon layer.
    disturbances : list, optional
        Ordered list of disturbances to plot (default: ["wind", "bark_beetle", "defoliators"]).
    layers : list, optional
        List of layers to overlay (default: ["manual", "IDS", "S1DM"]).
    layer_colors : dict, optional
        Dictionary mapping layer names to colors (default provided).
    disturbance_colors : dict, optional
        Dictionary mapping disturbances to colors (default provided).
    figure_path : str, optional
        If provided, the figure will be saved to this path.
    figsize : tuple, optional
        Figure size.
    fontsize_title, fontsize_axis, fontsize_legend : int, optional
        Font sizes for figure elements.
    """
    legend_patches = [mpatches.Patch(color=layer_colors[key], label=key) for key in layers]

    fig, axes = plt.subplots(
        len(disturbances), len(layers) + 1,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1]*len(layers) + [0.3]}
    )

    def add_subplot_label(ax, label):
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=30,
                fontweight='normal', va='top', ha='left', color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))

    for i, disturbance in enumerate(disturbances):
        # Load PlanetScope RGB raster
        ds = rioxarray.open_rasterio(disturbance_files[disturbance]["file"])
        ds_4326 = ds.rio.reproject("EPSG:4326")
        rgb = ds_4326.transpose("y", "x", "band").values.astype(np.float32)

        # Mask and crop non-empty pixels
        mask = np.any(rgb > 0, axis=2)
        rows_nonzero = np.where(mask.any(axis=1))[0]
        cols_nonzero = np.where(mask.any(axis=0))[0]
        row_min, row_max = rows_nonzero[[0, -1]]
        col_min, col_max = cols_nonzero[[0, -1]]
        row_min += 5; row_max -= 5; col_min += 5; col_max -= 5
        if disturbance == "wind":
            col_min += 110
        rgb_cropped = rgb[row_min:row_max+1, col_min:col_max+1, :]
        rgb_cropped = np.clip(rgb_cropped / np.percentile(rgb_cropped, 99), 0, 1)
        if rgb_cropped.shape[2] == 3:
            alpha = np.any(rgb_cropped > 0, axis=2).astype(float)
            rgb_cropped = np.dstack((rgb_cropped, alpha))
        elif rgb_cropped.shape[2] > 4:
            rgb_cropped = rgb_cropped[:, :, :4]
        x = ds_4326.x.values[col_min:col_max+1]
        y = ds_4326.y.values[row_min:row_max+1]
        extent_cropped = (x.min(), x.max(), y.min(), y.max())

        # Load manual geometry
        manual_file = os.path.join(folder_manual, f"planet_labels/merged_{disturbance}_manual.shp")
        manual = gpd.read_file(manual_file)
        idx_row = disturbance_files[disturbance]["idx"]
        row_data = manual.iloc[idx_row]
        idx_d = row_data["IDX_D"]
        print(f"Processing {disturbance} (IDX_D={idx_d})")
        ids_match = ids_gdf[ids_gdf["IDX_D"] == idx_d]
        s1dm_match = s1dm_gdf[s1dm_gdf["IDX_D"] == idx_d]
        manual_geom = gpd.GeoSeries([row_data.get("geom_manua", row_data.geometry)])
        manual_geom = manual_geom.set_crs("EPSG:4326") if manual_geom.crs is None else manual_geom

        ids_match_4326 = ids_match.to_crs("EPSG:4326") if ids_match.crs != "EPSG:4326" else ids_match.copy()
        s1dm_match_4326 = s1dm_match.to_crs("EPSG:4326") if s1dm_match.crs != "EPSG:4326" else s1dm_match.copy()

        merged_s1dm_geom = unary_union(s1dm_match_4326.geometry[0:1] if disturbance=="bark_beetle" else s1dm_match_4326.geometry)
        merged_s1dm = gpd.GeoSeries([merged_s1dm_geom], crs=s1dm_match_4326.crs)
        manual_geom_4326 = manual_geom.to_crs("EPSG:4326")

        # Plot layers
        for j, layer in enumerate(layers):
            ax = axes[i, j] if len(disturbances) > 1 else axes[j]
            ax.imshow(rgb_cropped, extent=extent_cropped)
            if layer == "manual" and not manual_geom_4326.empty:
                manual_geom_4326.boundary.plot(ax=ax, edgecolor=layer_colors["Manual"], linewidth=3)
            elif layer == "IDS" and not ids_match_4326.empty:
                ids_match_4326.boundary.plot(ax=ax, edgecolor=layer_colors["IDS"], linewidth=3)
            elif layer == "S1DM" and not s1dm_match_4326.empty:
                merged_s1dm.boundary.plot(ax=ax, edgecolor=layer_colors["S1DM"], linewidth=2)
            ax.legend(handles=[legend_patches[layers.index(layer)]], fontsize=fontsize_legend)
            add_subplot_label(ax, chr(97 + i*len(layers) + j))
            ax.add_patch(Rectangle((extent_cropped[0], extent_cropped[2]),
                                   extent_cropped[1]-extent_cropped[0],
                                   extent_cropped[3]-extent_cropped[2],
                                   linewidth=3, edgecolor='black', facecolor='none'))
            ax.axis("off")

        # Extra info column
        ax_info = axes[i, -1]
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1); ax_info.axis("off")
        disturbance_name = disturbance.replace("_", " ").title()
        date = disturbance_files[disturbance]["date"]
        color = disturbance_colors.get(disturbance, "black")
        heights = {0: 0.92, 1: 0.92, 2: 0.96}; y_offsets = {0: 0.04, 1: 0.04, 2: 0.02}
        ax_info.add_patch(Rectangle((0, y_offsets[i]), 1, heights[i],
                                    facecolor="white", edgecolor=color, linewidth=3,
                                    transform=ax_info.transAxes, clip_on=False))
        ax_info.text(0.2, 0.5, disturbance_name,
                     ha="center", va="center", fontsize=25,
                     fontweight="bold", color=color,
                     rotation=90, transform=ax_info.transAxes)
        ax_info.text(0.6, 0.5, f"PlanetScope Image Composite\non {date}",
                     ha="center", va="center", fontsize=20,
                     color="black", rotation=90, transform=ax_info.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.99, hspace=0.05, wspace=0.05)
    if figure_path:
        plt.savefig(figure_path, dpi=300)
    plt.show()



# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def compute_overlap_jaccard(geom_candidate, geom_manual):
    """Compute Overlap (%) and Jaccard index between candidate and manual geometries."""
    geom_candidate = make_valid(geom_candidate)
    geom_manual = make_valid(geom_manual)

    inter = geom_candidate.intersection(geom_manual)
    union = geom_candidate.union(geom_manual)

    area_inter = inter.area
    area_union = union.area
    area_manual = geom_manual.area

    overlap_pct = (area_inter / area_manual) * 100 if area_manual > 0 else 0
    jaccard = (area_inter / area_union) if area_union > 0 else 0
    return overlap_pct, jaccard


def paired_ttest_significance(x, y, alpha1=0.05, alpha2=0.1, alternative="greater"):
    """One-sided paired t-test; returns significance symbol."""
    t_res = ttest_rel(x, y, alternative=alternative)
    if t_res.pvalue < alpha1:
        return t_res.pvalue, "*"
    elif t_res.pvalue < alpha2:
        return t_res.pvalue, "+"
    return t_res.pvalue, ""


# ==============================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================

def analyze_and_plot_manual_significance(ids_file, s1dm_file, manual_base_folder, result_path=None, save_path=None):
    """
    Compute overlap and Jaccard metrics for IDS vs S1DM against manual references.
    Uses actual folder/file names (already renamed to new IDX_D).
    """
    gdf_s1dm = gpd.read_file(s1dm_file)
    gdf_ids = gpd.read_file(ids_file)

    disturbances = ['wind', 'bark_beetle', 'defoliators']
    results = []
    skipped_entries = {"no_file": [], "empty_manual": [], "empty_geoms": []}

    # ---- Loop over disturbances ----
    for disturbance in disturbances:
        base_folder = os.path.join(manual_base_folder, disturbance)
        subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

        for idx_name in subfolders:
            #print(idx_name)
            manual_file = os.path.join(base_folder, idx_name, f"merged_union_multipolygon_{idx_name}.geojson")

            if not os.path.exists(manual_file):
                skipped_entries["no_file"].append(idx_name)
                continue

            gdf_manual = gpd.read_file(manual_file)
            if gdf_manual.empty:
                skipped_entries["empty_manual"].append(idx_name)
                continue

            geom_manual = make_valid(gdf_manual.geometry.union_all())
            geom_ids = make_valid(gdf_ids[gdf_ids["IDX_D"] == idx_name].geometry.union_all())
            geom_s1dm = make_valid(gdf_s1dm[gdf_s1dm["IDX_D"] == idx_name].geometry.union_all())

            if geom_ids.is_empty or geom_s1dm.is_empty or geom_manual.is_empty:
                skipped_entries["empty_geoms"].append(idx_name)
                continue

            overlap_ids, jaccard_ids = compute_overlap_jaccard(geom_ids, geom_manual)
            overlap_s1dm, jaccard_s1dm = compute_overlap_jaccard(geom_s1dm, geom_manual)

            results.append({
                "disturbance": disturbance,
                "idx": idx_name,
                "overlap_ids": overlap_ids,
                "jaccard_ids": jaccard_ids,
                "overlap_s1dm": overlap_s1dm,
                "jaccard_s1dm": jaccard_s1dm,
            })

    # ---- Convert to DataFrame ----
    df_results = pd.DataFrame(results)
    dist_labels = {"wind": "Wind", "bark_beetle": "Bark Beetle", "defoliators": "Defoliators"}
    df_results["disturbance_label"] = df_results["disturbance"].map(dist_labels)

    # ---- Compute significance per disturbance ----
    sig_jaccard, sig_overlap = {}, {}
    for d, group in df_results.groupby("disturbance_label"):
        if len(group) > 1:
            _, sig_j = paired_ttest_significance(group["jaccard_s1dm"], group["jaccard_ids"])
            _, sig_o = paired_ttest_significance(group["overlap_s1dm"], group["overlap_ids"])
        else:
            sig_j, sig_o = "", ""
        sig_jaccard[d], sig_overlap[d] = sig_j, sig_o

    # ==============================================================
    # PLOTTING
    # ==============================================================

    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.3)
    palette_custom = {"IDS": "#BCB6FF", "S1DM": "#AF42AE"}
    dist_order = ["Wind", "Bark Beetle", "Defoliators"]

    # --- Jaccard plot ---
    df_jaccard = df_results.melt(
        id_vars=["disturbance_label", "idx"],
        value_vars=["jaccard_ids", "jaccard_s1dm"],
        var_name="method", value_name="jaccard"
    )
    df_jaccard["method"] = df_jaccard["method"].map({"jaccard_ids": "IDS", "jaccard_s1dm": "S1DM"})

    # --- Overlap plot ---
    df_overlap = df_results.melt(
        id_vars=["disturbance_label", "idx"],
        value_vars=["overlap_ids", "overlap_s1dm"],
        var_name="method", value_name="overlap"
    )
    df_overlap["method"] = df_overlap["method"].map({"overlap_ids": "IDS", "overlap_s1dm": "S1DM"})

    # --- Create figure ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax0, ax1 = axes

    # --- a) Jaccard boxplot ---
    sns.boxplot(data=df_jaccard, x="disturbance_label", y="jaccard", hue="method",
                order=dist_order, palette=palette_custom, fliersize=5, linewidth=2.0, ax=ax0)
    ax0.set_ylabel("Jaccard Similarity", fontsize=22)
    ax0.set_xlabel("Disturbance Type", fontsize=22)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax0.tick_params(axis='both', labelsize=20)
    ax0.set_ylim(-0.01, 0.55)
    ax0.text(-0.05, 1.05, "a)", transform=ax0.transAxes, fontsize=22, fontweight='bold')

    for i, dist in enumerate(dist_order):
        if sig_jaccard.get(dist):
            y_max = df_jaccard[df_jaccard["disturbance_label"] == dist]["jaccard"].max()
            ax0.text(i, min(y_max + 0.035, 0.48), sig_jaccard[dist],
                     ha="center", va="bottom", fontsize=30, color="black", fontweight="bold")

    # --- b) Overlap boxplot ---
    sns.boxplot(data=df_overlap, x="disturbance_label", y="overlap", hue="method",
                order=dist_order, palette=palette_custom, fliersize=5, linewidth=2.0, ax=ax1)
    ax1.set_ylabel("Overlap (% of Manual Area)", fontsize=22)
    ax1.set_xlabel("Disturbance Type", fontsize=22)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_ylim(-1, 105)
    ax1.text(-0.05, 1.05, "b)", transform=ax1.transAxes, fontsize=22, fontweight='bold')

    for i, dist in enumerate(dist_order):
        if sig_overlap.get(dist):
            y_max = df_overlap[df_overlap["disturbance_label"] == dist]["overlap"].max()
            ax1.text(i, min(y_max + 5.0, 90), sig_overlap[dist],
                     ha="center", va="bottom", fontsize=30, color="black", fontweight="bold")

    # --- Legend ---
    handles = [
        mpatches.Patch(color=palette_custom['IDS'], label='IDS'),
        mpatches.Patch(color=palette_custom['S1DM'], label='S1DM'),
        mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=16, label='p < 0.05'),
        mlines.Line2D([], [], color='black', marker='+', linestyle='None', markersize=16, label='p < 0.1')
    ]
    ax1.legend(handles=handles, loc='upper right', fontsize=16, frameon=True, facecolor='white')
    ax0.get_legend().remove()

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()
