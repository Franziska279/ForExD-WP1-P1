
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
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm  # Import tqdm for the progress bar
import numpy as np
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap
import rioxarray
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd



# Set font sizes for various components
plt.rcParams.update({
    'font.size': 14,           # Global font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X and Y label font size
    'xtick.labelsize': 14,     # X tick label font size
    'ytick.labelsize': 14,     # Y tick label font size
})


def format_label(label):
    """
    Format a label by capitalizing each word and replacing underscores with spaces.

    Parameters:
    label (str): The input label with underscores.

    Returns:
    str: The formatted label with each word capitalized and underscores replaced by spaces.
    """
    return ' '.join(word.capitalize() for word in label.split('_'))


def format_label_count(dca_id, count):
    """
    Format the label to include the count of events.

    Parameters:
    dca_id (str): The DCA_ID to be formatted.
    count (int): The count of events to be included in the label.

    Returns:
    str: The formatted label including the count of events in parentheses.
    """
    return f'{dca_id.replace("_", " ").title()} ({count})'

def calculate_s1cd_outline(s1cd_folder, save_path):

    results = []
    # Iterate over files in the directory with a progress bar
    for file_name in tqdm(os.listdir(s1cd_folder), desc="Processing S1 Tiles"):
        file_path = os.path.join(s1cd_folder, file_name)
        
        # Skip non-file entries (e.g., directories)
        if not os.path.isfile(file_path):
            continue

        # Check for NetCDF files based on file extension
        if not file_name.endswith('.nc'):
            continue

        # Load the NetCDF file with xarray
        try:
            ds = xr.open_dataset(file_path)

            # Try to extract x_bnds and y_bnds, or fall back to x and y if they don't exist
            if 'x_bnds' in ds and 'y_bnds' in ds:
                x_bnds = ds['x_bnds'].values
                y_bnds = ds['y_bnds'].values

                # Get the min/max x and y bounds
                x_min = np.min(x_bnds)
                x_max = np.max(x_bnds)
                y_min = np.min(y_bnds)
                y_max = np.max(y_bnds)

            elif 'x' in ds and 'y' in ds:
                x_vals = ds['x'].values
                y_vals = ds['y'].values

                # Use min and max of x and y as bounds
                x_min = np.min(x_vals)
                x_max = np.max(x_vals)
                y_min = np.min(y_vals)
                y_max = np.max(y_vals)

            else:
                raise ValueError(f"No 'x_bnds' or 'x' found in file {file_name}")

            # Create a polygon from the bounding box
            bounding_box = Polygon([
                (x_min, y_min),  # Bottom-left
                (x_max, y_min),  # Bottom-right
                (x_max, y_max),  # Top-right
                (x_min, y_max),  # Top-left
                (x_min, y_min)   # Close the polygon
            ])

            # Append the filename and bounding box to results
            results.append({
                'filename': file_name,
                'geometry': bounding_box
            })

            print(f"Processed file: {file_name}")

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Convert DataFrame to GeoDataFrame with the original CRS (assumed EQUI7 here)
    results_gdf = gpd.GeoDataFrame(results_df, geometry='geometry', crs='EPSG:27705')  
    
    # Reproject to EPSG:4326 (WGS 84)
    results_gdf = results_gdf.to_crs(epsg=4326)
    results_gdf.to_file(save_path) # Replace with the correct CRS if different

    return results_gdf

def get_mainland(gdf_path):
    """
    Cleans the mainland parts of specified regions from the given GeoDataFrame.
    
    Parameters:
        gdf_path (str): Path to the input GeoDataFrame file.
    
    Returns:
        GeoDataFrame: Cleaned GeoDataFrame with mainland parts of specified regions.
    """
    gdf = gpd.read_file(gdf_path)
    regions_to_clean = ['05', '08', '10']
    cleaned_parts = []

    for region in regions_to_clean:
        region_gdf = gdf[gdf['REGION'] == region]
        exploded = region_gdf.explode(index_parts=True)
        exploded['area'] = exploded.area
        # Get the mainland part with the maximum area
        mainland_part = exploded.loc[exploded['area'].idxmax()]
        cleaned_region = exploded[exploded['area'] == mainland_part['area']]
        if region != "10":
            cleaned_parts.append(cleaned_region)

    cleaned_parts_gdf = gpd.GeoDataFrame(pd.concat(cleaned_parts, ignore_index=True))

    # Combine cleaned parts with the rest of the GeoDataFrame
    #usa_mainland = gdf[~gdf['REGION'].isin(regions_to_clean)].append(cleaned_parts_gdf, ignore_index=True)

    # Update mainland by removing small parts and adding cleaned parts
    usa_mainland = pd.concat([gdf[~gdf['REGION'].isin(regions_to_clean)], cleaned_parts_gdf], ignore_index=True)

    return usa_mainland



def get_region_shape(path, region_id):
    
    usa = gpd.read_file(path)
    country = usa[usa.REGION == region_id]
    
    region = country.explode()[0:1] 

    return region
    

import geopandas as gpd
import pandas as pd

def load_dissolved_refdm(refdm_path):
    """
    Load and process a GeoDataFrame from the given path. The processing includes converting
    necessary columns to numeric, dissolving geometries by ID_E and S1_YEAR, calculating 
    the duration for each ID_E, and returning the processed DataFrame.

    Parameters:
    refdm_path (str): Path to the GeoDataFrame file.

    Returns:
    GeoDataFrame: Processed GeoDataFrame with a 'Duration' column indicating the number 
                  of unique years for each ID_E.
    """
    # Load the GeoDataFrame
    refdm = gpd.read_file(refdm_path)
    

    print("CRS:", refdm.crs)
    print(f"Size of refdm_dataset: {len(refdm)}")

    # Convert columns to numeric, if not already
    refdm['SURVEY_Y'] = pd.to_numeric(refdm['SURVEY_Y'], errors='coerce')
    refdm['S1_YEAR'] = pd.to_numeric(refdm['S1_YEAR'], errors='coerce')
    
    # Dissolve geometries by ID_E and S1_YEAR
    dissolved_refdm = refdm.dissolve(by=['ID_E', 'S1_YEAR']).reset_index()
    
    # Group by ID_E and aggregate unique years
    unique_years_per_id = dissolved_refdm.groupby('ID_E')['S1_YEAR'].unique().reset_index()
    
    # Calculate the duration (number of unique years) for each ID_E
    unique_years_per_id['Duration'] = unique_years_per_id['S1_YEAR'].apply(len)
    
    # Merge the calculated duration with the main DataFrame
    dissolved_df = dissolved_refdm.merge(unique_years_per_id[['ID_E', 'Duration']], on='ID_E')
    
    # Dissolve geometries again by ID_E to ensure aggregation and reset index
    dissolved_df = dissolved_df.dissolve(by=['ID_E']).reset_index()
    print(f"Size of unique refdm_dataset events: {len(dissolved_df)}")
    return dissolved_df


def load_ids_dataset(path):
    gdf_ids = gpd.read_file(path)
    #gdf_ids = gdf.rename(columns={'index_usda': 'USDA_IDX'})
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


def create_downsampled_tcc_map(forest_map_path, area_path, region_id, forest_map_downsampled_path, forest_map_downsampled_path_final):

    try:
        print("Step 1: Get Region 8 geometry ...")
        # Get Region 8 geometry
        r8_geometry = get_region_shape(area_path, region_id)
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

def parse_custom_colors(colors_json):
    """
    Parse a JSON string to extract custom color mappings.

    Parameters:
    colors_json (str): A JSON string containing color mappings where keys are color names
                       and values are color codes.

    Returns:
    dict: A dictionary containing color mappings extracted from the JSON string. 
          Returns an empty dictionary if the input JSON string is empty or None.
    """
    # Check if the JSON string is provided
    if colors_json:
        try:
            # Attempt to parse the JSON string into a Python dictionary
            custom_colors = json.loads(colors_json)
        except json.JSONDecodeError:
            # Handle JSON decoding errors (e.g., invalid JSON format)
            print("Error: Invalid JSON format.")
            custom_colors = {}
    else:
        # Default to an empty dictionary if the JSON string is empty or None
        custom_colors = {}

    return custom_colors

def plot_figure_1(cropped_forest, usa_mainland, r8, data, s1cd, custom_colors, save_figure_path):
    """
    Plot the TCC map with disturbance types and save the figure.
    """
    # Normalize the TCC values
    cropped_forest = normalize_tcc(cropped_forest)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot the entire USA in grey in the upper left corner
    sub_ax = fig.add_axes([-0.05, 0.75, 0.25, 0.25])  # [left, bottom, width, height]
    plot_mainland_map(sub_ax, usa_mainland)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Plot the TCC map within Region 8 boundaries
    plot_tcc_map(ax, cropped_forest, custom_cmap)
    
    # Plot the region outline with grids
    r8.boundary.plot(ax=ax, linewidth=2, color='black')
    
    # Plot S1CD boundaries with grid lines
    s1cd.boundary.plot(ax=ax, linewidth=2, color='#150442')
    # Add gridlines for s1cd
    s1cd.boundary.plot(ax=ax, linestyle='--', color='#150442', linewidth=0.5)
    
    # Plot disturbance types
    plot_disturbance_types(ax, data, custom_colors)
    
    # Customize the plot
    ax.axis('off')  # Remove axis and frame
    ax.set_title(' ')
    # Create legend for disturbance types
    legend_patches = [mpatches.Patch(color=color, label=format_label(disturbance)) for disturbance, color in custom_colors.items()]
    ax.legend(handles=legend_patches, fontsize=18, title="Disturbance Type", title_fontsize=20, loc='center left', facecolor='white', framealpha=1)
    
    # Add a label indicating the grid lines for s1cd
    ax.text(0.45, 0.95, 'S1CD Tiles', transform=ax.transAxes, fontsize=20, color='#150442', bbox=dict(facecolor='white', alpha=1, edgecolor='#150442'))

    
    plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_radar_reduction_potential(refdm_gdf, ids_gdf, save_path):
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

    # Create a figure with 2 rows and 1 column for the two subplots
    fig = plt.figure(figsize=(24, 12))  # Increase the figure height
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[5, 2])  # Adjust the height ratios

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
    #ax1.set_title('Radar Detection Potential per Disturbance Type', fontsize=title_fontsize, pad=20)  # Increase font size for title
    ax1.set_yscale('log')
    ax1.set_ylim(1, counts_df_sorted[['IDS', 'S1DM']].max().max() * 2)  # Set y-limit for log scale
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
    ax2.set_xlabel('Disturbance Type', fontsize= label_fontsize)  # Increase font size for xlabel
    ax2.set_ylim(0, -110)
    ax2.invert_yaxis()
    ax2.set_ylabel('Reduction \nPercentage (%)', fontsize=label_fontsize, labelpad=20)  # Increase font size for ylabel and add padding

    # Set y-axis ticks to only show 4 ticks
    ax2.set_yticks([0, -20, -40, -60 , -80, -100])
    ax2.set_yticklabels(['0', '-20', '-40', '-60', '-80', '-100'])

    plt.xticks([pos + bar_offset for pos in bar_positions], dca_labels, fontsize=tick_fontsize, ha='right')  # Adjust the rotation and alignment

    # Rotate x-axis labels for ax2
    ax2.set_xticklabels(dca_labels, ha='right', fontsize=0)  # Rotate and increase font size
    ax2.grid(False)

    # Adjust x-axis ticks and labels
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()  # Ensures labels, titles, and legends do not overlap
    
    plt.savefig(save_path , dpi=300, bbox_inches='tight')

    plt.show()

def plot_disturbance_duration(refdm_dissolved, save_path):
    # Count how often each duration occurs for each DCA_ID
    duration_counts = refdm_dissolved.groupby(['DCA_ID', 'Duration']).size().reset_index(name='Count')

    # Pivot the table for easy plotting
    pivot_table = duration_counts.pivot(index='DCA_ID', columns='Duration', values='Count').fillna(0)

    # Create a custom colormap: Darker shades of red
    colors = ['#FFAF6E', '#F28353', '#E65837', '#D92C1C', '#CC0000']  # Light to dark red
    cmap = mcolors.ListedColormap(colors)

    # Plot the grouped bar plot
    fig, ax = plt.subplots(figsize=(15, 6))  # Adjusted size: wider and shorter

    # Set font sizes for various components
    plt.rcParams.update({
        'font.size': 16,           # Global font size
        'axes.titlesize': 18,      # Title font size
        'axes.labelsize': 20,      # X and Y label font size
        'xtick.labelsize': 18,     # X tick label font size
        'ytick.labelsize': 18,     # Y tick label font size
    })

    # Plot the pivot table with the custom colormap
    pivot_table.plot(kind='bar', ax=ax, cmap=cmap, width=0.8, edgecolor='None', legend=False)

    # Add labels to each bar with a buffer
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                format(height, '.0f'),
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                xytext=(0, 4),  # 4 points vertical offset
                textcoords='offset points',
                fontsize=13  # Set the font size smaller here
            )

    # Set the y-axis limit to the next multiple of 100 above the max count
    max_count = pivot_table.values.max()
    ax.set_ylim(0, np.ceil(max_count / 100) * 100)

    # Format x-axis labels
    ax.set_xticklabels([format_label(label.get_text()) for label in ax.get_xticklabels()])

    # Set labels and title with additional buffer
    ax.set_xlabel('Disturbance Type', labelpad=20)
    ax.set_ylabel('Number of Events', labelpad=20)
    ax.set_title(' ', fontsize=20)

    plt.xticks(rotation=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    # Create a colorbar and position it closer to the plot
    norm = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5], ncolors=len(colors), clip=True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Duration', fontsize=20, labelpad=10)  # Add a bit of space to the label
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(['1 Year', '2 Years', '3 Years', '4 Years', '5 Years'])

    plt.tight_layout()

    # Save the figure with a specific DPI to fit an A4 page
    plt.savefig(save_path, dpi=400, bbox_inches='tight')

    plt.show()



def calculate_area_and_centroid_meters(gdf):
    """Calculate area in square meters, square kilometers, and convex hull area for a GeoDataFrame."""
    # Set the coordinate reference system (CRS) for the GeoDataFrame to WGS84
    gdf.crs = 'EPSG:4326'
    
    # Define the target projection system (e.g., UTM Zone 18N)
    target_crs = 'EPSG:27705'
    
    # Reproject the GeoDataFrame to the target projection system
    gdf_projected = gdf.to_crs(target_crs)
    
    # Calculate the area of the geometry in square meters
    gdf_projected['area_meters'] = gdf_projected.geometry.area
    gdf_projected['square_km'] = gdf_projected['area_meters'] / 1e6
    # Calculate the convex hull of each geometry
    gdf_projected['convex_hull'] = gdf_projected.geometry.convex_hull
    
    # Calculate the area of the convex hull in square meters
    gdf_projected['area_convex_meters'] = gdf_projected['convex_hull'].area
    
    # Convert the area to square kilometers
    gdf_projected['area_convex_km'] = gdf_projected['area_convex_meters'] / 1e6
    
    # Calculate the centroid of the geometry in projected coordinates (meters)
    gdf_projected['centroid_meters'] = gdf_projected.geometry.centroid
    
    # Assign the calculated fields back to the original GeoDataFrame (which is still in WGS84)
    gdf['area_meters'] = gdf_projected['area_meters']
    gdf['square_km'] = gdf_projected['square_km']
    gdf['area_convex_meters'] = gdf_projected['area_convex_meters']
    gdf['area_convex_km'] = gdf_projected['area_convex_km']
    gdf['centroid_meters'] = gdf_projected['centroid_meters']
    
    # Return the GeoDataFrame with the new area and convex hull columns
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
    size_difference_list_ref = []

    convex_size_difference_list = []
    convex_size_difference_list_ref = []


    # Iterate through unique USDA_IDX values in ids_gdf
    unique_usda_idx = ids['ID_E'].unique()

    for usda_idx in unique_usda_idx:
        # Get the corresponding rows in both GeoDataFrames for the current ID_E
        ids_row = ids[ids['ID_E'] == usda_idx]
        refdm_row = refdm[refdm['ID_E'] == usda_idx]
        
        # Check if both rows exist (they should exist if data is correctly structured)
        if not ids_row.empty and not refdm_row.empty:
            # Calculate the centroid shift between centroids of the two polygons
            centroid_shift = ids_row.iloc[0]['centroid_meters'].distance(refdm_row.iloc[0]['centroid_meters'])
            
            # Calculate the size difference between areas of the two polygons
            size_difference = ids_row.iloc[0]['square_km'] - refdm_row.iloc[0]['square_km']

            size_difference_redfm = refdm_row.iloc[0]['square_km'] - ids_row.iloc[0]['square_km']

            # Calculate the size difference between areas of the two polygons
            convex_size_difference = ids_row.iloc[0]['area_convex_km'] - refdm_row.iloc[0]['area_convex_km']

            convex_size_difference_redfm = refdm_row.iloc[0]['area_convex_km'] - ids_row.iloc[0]['area_convex_km']
            
            # Append the results to the lists
            usda_idx_list.append(usda_idx)
            centroid_shift_list.append(centroid_shift)
            size_difference_list.append(size_difference)
            size_difference_list_ref.append(size_difference_redfm)
            convex_size_difference_list.append(convex_size_difference)
            convex_size_difference_list_ref.append(convex_size_difference_redfm)

    # Create a DataFrame from the results
    results_df = pd.DataFrame({
        'ID_E': usda_idx_list,
        'centroid_shift_m': centroid_shift_list,
        'size_difference_km2': size_difference_list,
        'size_difference_ref_km2': size_difference_list_ref
        # 'convex_size_difference_km2': convex_size_difference_list,
        # 'convex_size_difference_ref_km2': convex_size_difference_list_ref
    })

    #ids_gdf = ids_gdf.drop(columns=['centroid_shift_m'])
    # Merge with original gdf to get geometry
    result_gdf = ids_gdf.merge(results_df, on='ID_E')

    # Convert to GeoDataFrame with original geometry and CRS
    result_gdf = gpd.GeoDataFrame(result_gdf, geometry='geometry', crs=ids_gdf.crs)

    return result_gdf


def plot_size_shift_comparison_errorbars(gdf, custom_colors, save_path):

    # Get the default colors from 'tab10' palette for the rest of the disturbance types
    default_palette = sns.color_palette('tab10', n_colors=10)
    default_colors = [color for color in default_palette if color not in custom_colors.values()]

    # Combine the custom colors with the default colors
    custom_palette = [custom_colors.get(label, default_colors.pop(0)) for label in gdf['DCA_ID'].unique()]

    # Calculate medians and quantiles for each disturbance type
    grouped = gdf.groupby('DCA_ID').agg({
        'centroid_shift_m': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'size_difference_ref_km2': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
    }).reset_index()

    # Flatten the column names after aggregation
    grouped.columns = ['DCA_ID', 'centroid_shift_median', 'centroid_shift_q25', 'centroid_shift_q75', 'size_diff_median', 'size_diff_q25', 'size_diff_q75']

    # Set Seaborn style
    sns.set(style="whitegrid")

    plt.figure(figsize=(10,6))

    # Show quadrant lines
    plt.axhline(0, color='black', linewidth=2.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=2.5, linestyle='--')


    # Plot the median values and IQR with shaded regions
    for i, row in grouped.iterrows():
        plt.scatter(
            row['centroid_shift_median'],
            row['size_diff_median'],
            color=custom_colors.get(row['DCA_ID'], 'grey'),
            s=300,  # Larger scatter points
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
            capthick=3.5,
            linewidth=3.5  # Thicker error bar lines
        )

    # Customize the plot
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel('Position Shift (m)', fontsize=20)
    plt.ylabel('Size Difference (km²)', fontsize=20)
    plt.title('Area and Position Changes per Disturbance Types', fontsize=26, pad=20)

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(by_label.values(), by_label.keys(), title='Disturbance Types', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    legend.set_title('Disturbance Types', prop={'size': 18})  # Set legend title size

    # Set decimal tick labels
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Increase tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
   
    plt.savefig(save_path, bbox_inches='tight')


def format_ticks(x, pos):
    """Format the ticks to always have one decimal place."""
    return f'{x:.1f}'

def plot_area_size_shift_per_disturbances(gdf, ids, custom_colors, save_path):
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
    n_cols = 3  
    
    fig, axs = plt.subplots(n_categories, n_cols, figsize=(35, 40), gridspec_kw={'width_ratios': [3, 0.02, 2]})
    
    fontsize_supertitle = 44
    fontsize_legend = 46
    fontsize_title = 46
    fontsize_label = 44
    fontsize_tick = 40
    padding_label = 25
    padding_title = 25

    for i, category in enumerate(category_order):
        # Create combined data for violin plots
        combined_data = pd.concat([
            ids[ids['DCA_ID'] == category][['DCA_ID', 'area_km2']].assign(Source='IDS'),
            gdf[gdf['DCA_ID'] == category][['DCA_ID', 'area_km2']].assign(Source='REFDM')
        ])

        ax = axs[i, 0]
        data_refdm = combined_data[(combined_data['Source'] == 'REFDM') & (combined_data['DCA_ID'] == category)]
        data_ids = combined_data[(combined_data['Source'] == 'IDS') & (combined_data['DCA_ID'] == category)]

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
            linestyle='--'
        )

        #ax.axvline(x=median_ids, color='black', linestyle='--', linewidth=3, alpha=0.8, label=f'IDS Median: {median_ids:.2f}', marker='o', markersize=16)#

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

        #ax.axvline(x=median_refdm, color=custom_palette[category], linestyle='-', linewidth=3, alpha=1, label=f'S1DM Median: {median_refdm:.2f}', marker='s', markersize=16)#

        # Adjust median formatting for Bark Beetle DCA_ID
        if category == 'bark_beetle':
            median_refdm_label = f'S1DM Median: {median_refdm:.4f}' if median_refdm > 0 else 'S1DM Median: 0.0'
            median_ids_label = f'IDS Median: {median_ids:.4f}' if median_ids > 0 else 'IDS Median: 0.0'
        # elif category == 'wind':
        #     median_refdm_label = f'S1DM Median: {median_refdm:.3f}' if median_refdm > 0 else 'S1DM Median: 0.0'
        #     median_ids_label = f'IDS Median: {median_ids:.3f}' if median_ids > 0 else 'IDS Median: 0.0'
        else:
            median_refdm_label = f'S1DM Median: {median_refdm:.2f}'
            median_ids_label = f'IDS Median: {median_ids:.2f}'

        ax.axvline(x=median_ids, color='black', linestyle='--', linewidth=3, alpha=0.8, label=median_ids_label, marker='o', markersize=16)

        ax.axvline(x=median_refdm, color=custom_palette[category], linestyle='-', linewidth=3, alpha=1, label=median_refdm_label, marker='s', markersize=16)

        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.tick_params(axis='x', labelsize=fontsize_tick)

        if i == len(category_order) - 1:  # Only set x-label for the bottom row
            ax.set_xlabel('Disturbance Area (km²)', fontsize=fontsize_label, labelpad=padding_label)
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

        # Distribution plot for centroid shifts
        ax = axs[i, 2]
        sns.histplot(
            data=gdf[gdf['DCA_ID'] == category],
            x='centroid_shift_m',
            kde=True,
            line_kws={'linewidth': 4},
            color=custom_palette[category],
            ax=ax
        )

        if i == len(category_order) - 1:
            ax.set_xlabel('Centroid Shift (m)', fontsize=fontsize_label, labelpad=padding_label)
        else:
            ax.set_xlabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.set_xlim(0, 2600)

        ax.set_ylabel(' ', fontsize=fontsize_label, labelpad=padding_label)

        # Set the x-ticks for centroid shifts
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))  # Prune limits for edge ticks
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower')) 
        
        # Add the custom legend
        ax.legend(handles=[mpatches.Patch(color=custom_palette[category], label=format_label(category))],
                  loc='upper right', fontsize=fontsize_legend, frameon=True, fancybox=True, facecolor='white', edgecolor='black')

        axs[i, 1].axis('off')

    # Add common y-axis labels for the plots
    fig.text(0.05, 0.5, 'PDF', va='center', rotation='vertical', fontsize=fontsize_label)
    fig.text(0.63, 0.5, 'Number of Events', va='center', rotation='vertical', fontsize=fontsize_label)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()



def main():

    """
    Main function to orchestrate loading data, creating the TCC map, and plotting the results.
    """
   # Load environment variables from the .env file
    env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
    load_dotenv(dotenv_path=env_path)

    # Retrieve and parse custom color settings from environment variables
    custom_colors_json = os.getenv('COLORS')
    custom_colors = parse_custom_colors(custom_colors_json)

    # Retrieve environment variables
    s2_minicubes_folder = os.getenv('EQUI7_GRIDS')
    print(f"Equi7 grids folder: {s2_minicubes_folder}")

    # Check if the folder exists
    if not os.path.isdir(s2_minicubes_folder):
        raise FileNotFoundError(f"The folder {s2_minicubes_folder} does not exist.")

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
    s1_tiles_boundary_path = f"{os.getenv('RESULTS')}/radar_results/s1cd_tiles_bounds_region_{region_id}.shp"

    # TCC Paths
    forest_map_path = f"{os.getenv('TCC_PATH')}/wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_region_08.tif"
    forest_map_downsampled_path = f"{os.getenv('TCC_PATH')}/intermediate_tcc_map_region_8.nc"
    tcc_map_region_8 = f"{os.getenv('TCC_PATH')}/tcc_map_region_8.nc"

    figure_output_path = f"{os.getenv('FIGURES')}"
    if not os.path.exists(figure_output_path):
            os.makedirs(figure_output_path)
            
    # Retrieve environment variables
    s1_tiles_folder = os.getenv('SENTINEL1_TILES')
    print(f"S1 Tiles folder: {s1_tiles_folder}")
    if not os.path.isdir(s1_tiles_folder):
        raise FileNotFoundError(f"The folder {s1_tiles_folder} does not exist.")

    figure_study_area_path = figure_output_path + "p1_f6_refdm_study_area.png"
    figure_radar_reduction_potential_path = figure_output_path + "p1_f7_ids_refdm_radar_reduction_potential.png"
    figure_disturbance_duration_path = figure_output_path + "p1_f8_disturbance_duration.png"
    figure_size_position_change_errrorbar_path = figure_output_path + "p1_f9_size_position_change_errrorbar.png"
    figure_size_position_change_path = figure_output_path + "p1_f10_size_position_change.png"
    figure_size_position_quantiles_change_path = figure_output_path + "p1_f11_size_position_quantiles_change.png"


    print("Script to analyse the radar enhanced forest disturbance dataset:\n")

    try:
        print("Loading datsets:\n")

        print("Load the Forest Disturbances...")
        refdm = load_dissolved_refdm(refdm_path)
        ids_gdf = load_ids_dataset(ids_path)
        
        print("Load the S1CD Outlines  ...")
        tiles_bounds = calculate_s1cd_outline(s1_tiles_folder, s1_tiles_boundary_path)

        print("Load the USA Mainland and Region 8 Shape ...")
        mainland = get_mainland(usa_filepath)
        region_8 = get_region_shape(usa_filepath, region_id=region_id)

        #print("Create the downsampled TCC Map for Region 8 Shape ...")
        #create_downsampled_tcc_map(forest_map_path, usa_filepath, region_id, forest_map_downsampled_path, tcc_map_region_8)

        print("Load the TCC Region 8 Map ...")
        tcc_dataset = load_tcc_dataset(tcc_map_region_8)

        print("Plotting data\n")
        print("Plot 1: Study area ...")
        plot_figure_1(tcc_dataset, mainland, region_8, refdm, tiles_bounds, custom_colors, save_figure_path=figure_study_area_path)


        print("\nPlot 2: Radar reduction potential ...")
        plot_radar_reduction_potential(refdm, ids_gdf, figure_radar_reduction_potential_path )
        print(f"Remove drougth from data...")
        # Filter out rows where 'DCA_ID' equals 'drought'
        slimmed_refdm = refdm[refdm['DCA_ID'] != 'drought']

        print("\nPlot 3: Disturbance duration ...")
        plot_disturbance_duration(slimmed_refdm, figure_disturbance_duration_path)

        print("\nStep 4: Plotting Size and Location Difference and Density ...")
        gdf = calculate_size_shift_difference(ids_gdf, slimmed_refdm)
        plot_area_size_shift_per_disturbances(gdf, ids_gdf, custom_colors, figure_size_position_change_path)
        plot_size_shift_comparison_errorbars(gdf, custom_colors, figure_size_position_change_errrorbar_path)
        print("Main process completed successfully.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Log the error or take appropriate action


if __name__ == "__main__":
    main()