import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import warnings
from shapely.geometry import mapping
import rioxarray
from shapely import wkt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import sys
import pandas as pd
import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import warnings
from shapely.geometry import mapping
import rioxarray
from shapely import wkt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import sys
import pandas as pd
import logging

import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rioxarray.exceptions import NoDataInBounds


def calculate_trend_and_distance(data, var):
    """
    Calculate the trend and distance for a given variable.
    
    Parameters:
    - data: xarray.Dataset or pandas.DataFrame containing the time series data.
    - var: str, the variable name to analyze.
    
    Returns:
    - slope: The slope of the trend line.
    - distance: The distance between the lowest and highest points in the time series.
    """
    # Convert xarray.Dataset to pandas.DataFrame if necessary
    if isinstance(data, xr.Dataset):
        df = data.to_dataframe().reset_index()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Unsupported data type. Provide xarray.Dataset or pandas.DataFrame.")
    
    # Ensure the variable exists in the data
    if var not in df.columns:
        raise ValueError(f"Variable '{var}' not found in the data.")
    
    # Drop rows with NaN values in the specified variable column
    df = df.dropna(subset=[var])
    
    # Ensure there are enough data points after dropping NaNs
    if len(df) > 1:
        # Calculate the trend using linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df.index, df[var])

        # Calculate the distance between the lowest and highest points
        min_value = df[var].min()
        max_value = df[var].max()
        distance = abs(max_value - min_value)

        return slope, distance
    else:
        return np.nan, np.nan

def extract_info(index):
    # Remove the '.nc' extension
    parts = index.split('_')
    
    if len(parts) < 3:
        return None, None, None

    idx = parts[0]
    year = parts[1]
    dist_type = '_'.join(parts[2:])
    
    return idx, year, dist_type

def process_row(row, data_ids):
    usda_idx = row['USDA_IDX']
    print(usda_idx)
    dist = row['DCA_ID']

    # Check if row['mini_idx'] is None or NaN
    if pd.isna(row['mini_idx']):
        print(f"No mini_idx of {usda_idx}")
        return None

    # Filter data_ids by USDA_IDX
    ids_data = data_ids[data_ids['USDA_IDX'] == usda_idx]
    refdm_data = row

    # Extract info from 'mini_idx'
    idx, year, dist_type = extract_info(row['mini_idx'])

    # Construct the path to the NetCDF file
    cube_file_path = f"/Net/Groups/BGI/scratch/fmueller/Data/s2_region8_nc_256px_vi/{dist_type}/{year}/{row['mini_idx']}.nc"

    # Load the corresponding NetCDF file
    try:
        mc = xr.open_dataset(cube_file_path, engine='netcdf4')
    except Exception as e:
        print(f"Error loading file {row['mini_idx']}")
        print(e)
        return None

    # Check if CRS is None and assign if needed
    if mc.rio.crs is None:
        mc.rio.write_crs("epsg:4326", inplace=True)

    # Set the CRS for the xarray dataset
    mc = mc.rio.write_crs("epsg:4326", inplace=True)

    shape_usda = gpd.GeoSeries(ids_data.geometry).buffer(0.00001)
    shape_s1 = gpd.GeoSeries(refdm_data.geometry).buffer(0.00001)

    # Check if the geometries are valid
    if not shape_usda.is_valid.all() or not shape_s1.is_valid.all():
        mc.close()
        return None

    # Check if the geometries are non-empty
    if shape_usda.is_empty.any() or shape_s1.is_empty.any():
        mc.close()
        return None

    # Clip the data to the polygon extent
    try:
        clipped_data_ids = mc.rio.clip(shape_usda.geometry.apply(mapping), drop=True)
        clipped_data_refdm = mc.rio.clip(shape_s1.geometry.apply(mapping), drop=True)
    except (ValueError, NoDataInBounds) as e:
        mc.close()
        return None

    # Compute the median for the clipped data
    median_ids = clipped_data_ids.median(dim=['x', 'y'])
    median_refdm = clipped_data_refdm.median(dim=['x', 'y'])
    p25_ids = clipped_data_ids.quantile(0.25, dim=['x', 'y'])
    p75_ids = clipped_data_ids.quantile(0.75, dim=['x', 'y'])
    p25_refdm = clipped_data_refdm.quantile(0.25, dim=['x', 'y'])
    p75_refdm = clipped_data_refdm.quantile(0.75, dim=['x', 'y'])

    # Analyze each variable
    result_dicts = []
    for var in median_ids.data_vars:
        if 's2' not in var:
            slope_ids, distance_ids = calculate_trend_and_distance(median_ids, var)
            slope_refdm, distance_refdm = calculate_trend_and_distance(median_refdm, var)

            # Prepare a dictionary to store results for this variable
            result_dict = {
                'USDA_IDX': usda_idx,
                'Variable': var,
                'Median_IDS': median_ids[var].values,
                'Median_REFDM': median_refdm[var].values,
                '25_percentiles_IDS': p25_ids[var].values,
                '75_percentiles_IDS': p75_ids[var].values,
                '25_percentiles_REFDM': p25_refdm[var].values,
                '75_percentiles_REFDM': p75_refdm[var].values,
                'Slope_IDS': slope_ids,
                'Distance_IDS': distance_ids,
                'Slope_REFDM': slope_refdm,
                'Distance_REFDM': distance_refdm
            }
            result_dicts.append(result_dict)

    # Close the dataset to free resources
    mc.close()

    return result_dicts

def extract_median_percentiles_slope_distance_per_VI(minicube, data_ids, data_refdm, result_path):
    # Create an empty list to store results
    results_list = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_row, row, data_ids): idx for idx, row in data_refdm.iterrows()}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            result = future.result()
            if result:
                results_list.extend(result)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results DataFrame to CSV
    results_df.to_csv(result_path, index=False)

    print(f"Saved extracted data to {result_path}")


def calculate_slope_distance_per_vegitation_index(df):
    # Calculate median, 25th and 75th percentiles for Slope and Distance for IDS and REFDM for each variable
    quantiles = df.groupby('Variable').agg({
        'Slope_IDS': ['median', lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 75)],
        'Slope_REFDM': ['median', lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 75)],
        'Distance_IDS': ['median', lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 75)],
        'Distance_REFDM': ['median', lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 75)]
    }).reset_index()

    # Rename columns for easier access
    quantiles.columns = ['Variable', 'Slope_IDS_Median', 'Slope_IDS_P25', 'Slope_IDS_P75', 
                        'Slope_REFDM_Median', 'Slope_REFDM_P25', 'Slope_REFDM_P75', 
                        'Distance_IDS_Median', 'Distance_IDS_P25', 'Distance_IDS_P75', 
                        'Distance_REFDM_Median', 'Distance_REFDM_P25', 'Distance_REFDM_P75']

    # Calculate the absolute differences for median, P25, and P75
    quantiles['Slope_Median_Diff'] = np.abs(quantiles['Slope_IDS_Median'] - quantiles['Slope_REFDM_Median'])
    quantiles['Slope_P25_Diff'] = np.abs(quantiles['Slope_IDS_P25'] - quantiles['Slope_REFDM_P25'])
    quantiles['Slope_P75_Diff'] = np.abs(quantiles['Slope_IDS_P75'] - quantiles['Slope_REFDM_P75'])

    quantiles['Distance_Median_Diff'] = np.abs(quantiles['Distance_IDS_Median'] - quantiles['Distance_REFDM_Median'])
    quantiles['Distance_P25_Diff'] = np.abs(quantiles['Distance_IDS_P25'] - quantiles['Distance_REFDM_P25'])
    quantiles['Distance_P75_Diff'] = np.abs(quantiles['Distance_IDS_P75'] - quantiles['Distance_REFDM_P75'])

    # Normalize the median values and differences
    scaler = MinMaxScaler()

    quantiles[['Norm_Slope_IDS_Median', 'Norm_Slope_REFDM_Median', 'Norm_Slope_Median_Diff',
            'Norm_Distance_IDS_Median', 'Norm_Distance_REFDM_Median', 'Norm_Distance_Median_Diff']] = scaler.fit_transform(
        quantiles[['Slope_IDS_Median', 'Slope_REFDM_Median', 'Slope_Median_Diff',
                'Distance_IDS_Median', 'Distance_REFDM_Median', 'Distance_Median_Diff']])

    # Define a simple scoring system (summing the normalized values)
    quantiles['Slope_Importance_Score'] = (quantiles['Norm_Slope_IDS_Median'] +
                                        quantiles['Norm_Slope_REFDM_Median'] +
                                        2*quantiles['Norm_Slope_Median_Diff'])

    quantiles['Distance_Importance_Score'] = (quantiles['Norm_Distance_IDS_Median'] +
                                            quantiles['Norm_Distance_REFDM_Median'] +
                                            2*quantiles['Norm_Distance_Median_Diff'])

    quantiles['Overall_Importance_Score'] = quantiles['Slope_Importance_Score'] + quantiles['Distance_Importance_Score']

    # Rank the variables based on the overall importance score
    quantiles['Rank'] = quantiles['Overall_Importance_Score'].rank(ascending=False)

    # Sort the dataframe by the rank
    quantiles = quantiles.sort_values(by='Rank')

    return quantiles

def load_ids(path, disturbance_type):
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf_ids = gdf.rename(columns={'index_usda': 'USDA_IDX'})
    disturbance_ids = gdf_ids[gdf_ids['DCA_ID'] == disturbance_type]
    return disturbance_ids


def load_refdm(path, disturbance_type):
    # Load the shapefile using geopandas
    refdm_dataset = gpd.read_file(path)
    # Filter the dataset for DCA_ID 'drought'
    drought_refdm = refdm_dataset[refdm_dataset['DCA_ID'] == disturbance_type]
    dissolved_disturbance_refdm = drought_refdm.dissolve(by='USDA_IDX')
    dissolved_disturbance_refdm = dissolved_disturbance_refdm.reset_index()
    return dissolved_disturbance_refdm


def load_minicubes(path, disturbance_type):
    # Load the shapefile using geopandas
    minicubes_grid = gpd.read_file(path)
    disturbance_minicube_grid = minicubes_grid[minicubes_grid['dist_type'] == disturbance_type]
    return disturbance_minicube_grid


def plot_distance_and_slope_per_disturbancetype_ids_refdm_diff(quantiles, figure_path):
    # Create subplots for slopes and distances
    fig, axs = plt.subplots(2, 3, figsize=(30, 15), constrained_layout=True)

    # Variables for x-axis positioning
    x = np.arange(len(quantiles))

    # Define colors for each percentile
    colors = ['orange', 'blue']

    # Plot Slope Medians for IDS
    axs[0, 0].bar(x, quantiles['Slope_IDS_Median'], color=colors[0])
    axs[0, 0].set_title('Slope Medians for IDS')
    axs[0, 0].set_ylabel('Slope Median')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max slope median difference for IDS
    max_slope_median_ids = quantiles['Slope_IDS_Median'].max()
    axs[0, 0].axhline(max_slope_median_ids, color='red', linestyle='--', label=f'Max Median IDS ({max_slope_median_ids})')
    axs[0, 0].legend()

    # Plot Slope Medians for REFDM
    axs[0, 1].bar(x, quantiles['Slope_REFDM_Median'], color=colors[1])
    axs[0, 1].set_title('Slope Medians for REFDM')
    axs[0, 1].set_ylabel('Slope Median')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max slope median difference for REFDM
    max_slope_median_refdm = quantiles['Slope_REFDM_Median'].max()
    axs[0, 1].axhline(max_slope_median_refdm, color='red', linestyle='--', label=f'Max Median REFDM ({max_slope_median_refdm})')
    axs[0, 1].legend()

    # Plot Slope Differences
    axs[0, 2].bar(x, quantiles['Slope_Median_Diff'], color=colors[0])
    axs[0, 2].set_title('Slope Differences between IDS and REFDM')
    axs[0, 2].set_ylabel('Absolute Difference')
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max slope median difference
    max_slope_diff = quantiles['Slope_Median_Diff'].max()
    max_slope_diff_var = quantiles.loc[quantiles['Slope_Median_Diff'].idxmax(), 'Variable']
    axs[0, 2].axhline(max_slope_diff, color='red', linestyle='--', label=f'Max Median Diff ({max_slope_diff_var})')
    axs[0, 2].legend()

    # Plot Distance Medians for IDS
    axs[1, 0].bar(x, quantiles['Distance_IDS_Median'], color=colors[0])
    axs[1, 0].set_title('Distance Medians for IDS')
    axs[1, 0].set_ylabel('Distance Median')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max distance median difference for IDS
    max_distance_median_ids = quantiles['Distance_IDS_Median'].max()
    axs[1, 0].axhline(max_distance_median_ids, color='red', linestyle='--', label=f'Max Median IDS ({max_distance_median_ids})')
    axs[1, 0].legend()

    # Plot Distance Medians for REFDM
    axs[1, 1].bar(x, quantiles['Distance_REFDM_Median'], color=colors[1])
    axs[1, 1].set_title('Distance Medians for REFDM')
    axs[1, 1].set_ylabel('Distance Median')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max distance median difference for REFDM
    max_distance_median_refdm = quantiles['Distance_REFDM_Median'].max()
    axs[1, 1].axhline(max_distance_median_refdm, color='red', linestyle='--', label=f'Max Median REFDM ({max_distance_median_refdm})')
    axs[1, 1].legend()

    # Plot Distance Differences
    axs[1, 2].bar(x, quantiles['Distance_Median_Diff'], color=colors[0])
    axs[1, 2].set_title('Distance Differences between IDS and REFDM')
    axs[1, 2].set_ylabel('Absolute Difference')
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels(quantiles['Variable'], rotation=90)

    # Highlight max distance median difference
    max_distance_diff = quantiles['Distance_Median_Diff'].max()
    max_distance_diff_var = quantiles.loc[quantiles['Distance_Median_Diff'].idxmax(), 'Variable']
    axs[1, 2].axhline(max_distance_diff, color='red', linestyle='--', label=f'Max Median Diff ({max_distance_diff_var})')
    axs[1, 2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(figure_path)
    plt.show()

def plot_vegitation_indecies_imporance_rank(quantiles, figure_path, color):
    
    # Plot the importance scores with VIs on y-axis
    fig, ax = plt.subplots(figsize=(6, 8))

    # Bar plot for overall importance scores
    y = np.arange(len(quantiles))
    ax.barh(y, quantiles['Overall_Importance_Score'], color=color)
    ax.set_title(f'Overall Importance Scores for Vegetation Indices')
    ax.set_xlabel('Overall Importance Score')
    ax.set_ylabel('Vegetation Index')
    ax.set_yticks(y)
    ax.set_yticklabels(quantiles['Variable'])

    # Highlight the top-ranked index
    top_ranked_var = quantiles.iloc[0]['Variable']
    top_ranked_score = quantiles.iloc[0]['Overall_Importance_Score']
    ax.axvline(top_ranked_score, color='red', linestyle='--', label=f'Top Ranked ({top_ranked_var})')
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(figure_path)
    plt.show()


# Function to get color for a given dist_type
def get_color(dist_type):
    return custom_colors.get(dist_type, '#000000')  # Default to black if dist_type not found

custom_colors = {
    'wind': '#1f77b4',      # tab:blue
    'fire': '#d62728',      # tab:red
    'defoliators': '#2ca02c',  # tab:green
    'drought': '#FFBA08', # tab:yellow
    'bark_beetle': '#714709'  # tab:brown
}

def main(ids_path, refdm_path, minicubes_shape_path, disturbance_type):
    logger = logging.getLogger(__name__)
    
    logger.info("Define Paths...")
    result_path = f'/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/{disturbance_type}_slope_distance_results_dissolved.csv'

    logger.info(f"Load {disturbance_type} minicubes ...")
    minicubes_grid = load_minicubes(path=minicubes_shape_path, disturbance_type=disturbance_type)
    logger.info(f"Load {disturbance_type} ids ...")
    ids = load_ids(path=ids_path, disturbance_type=disturbance_type)
    logger.info(f"Load {disturbance_type} refdm ...")
    refdm = load_refdm(path=refdm_path, disturbance_type=disturbance_type)

    logger.info(f"Extract median, percentiles, slope and distance for each vegetation index ...")
    extract_median_percentiles_slope_distance_per_VI(minicube=minicubes_grid,
                                                     data_ids=ids, 
                                                     data_refdm=refdm, 
                                                     result_path=result_path) 
    
    logger.info(f"Load result ...")
    results_df = pd.read_csv(result_path)

    logger.info(f"Calculate best score ...")
    data = calculate_slope_distance_per_vegitation_index(results_df)

    logger.info(f"Plot result ...")
    plot_distance_and_slope_per_disturbancetype_ids_refdm_diff(data, figure_path=f"/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/figures/{disturbance_type}_slope_distance_VIs_ids_refdm_difference.png")
    plot_vegitation_indecies_imporance_rank(data, figure_path=f"/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/figures/{disturbance_type}_vegetation_indices_importance_scores.png", color = get_color(disturbance_type))

    logger.info("Finished")

if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fire_vi_script.log"),
        logging.StreamHandler()
    ]
)
    
    if len(sys.argv) != 5:
        logging.error("Usage: python main.py <ids_path> <refdm_path> <minicubes_shape_path> <disturbance_type>")
        sys.exit(1)
    
    ids_path = sys.argv[1]
    refdm_path = sys.argv[2]
    minicubes_shape_path = sys.argv[3]
    disturbance_type = sys.argv[4]
    
    main(ids_path, refdm_path, minicubes_shape_path, disturbance_type)