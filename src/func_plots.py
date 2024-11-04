import matplotlib.pyplot as plt
import pandas as pd  # Assuming you're using pandas for timestamps

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