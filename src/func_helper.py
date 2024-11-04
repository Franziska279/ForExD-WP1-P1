import json
from scipy.stats import zscore
import xarray as xr
import numpy as np

########################################################################################
#######                             Helper Functions                             #######
########################################################################################
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


# def save_as_zarr(da, path: str):
#     """
#     Triggers dask compute and saves chunks whenever they have been
#     processed. Empty chunks are not written. Chunks are compressed with
#     lz4. 

#     Parameters
#     ----------
#     da : xr.DataArray
#         DataArray that should be saved as zarr.
#     path : str
#         Specifies where save path of the zarr file.    
#     """
#     # Check if the folder exists, then delete it along with all its subfolders
#     if os.path.exists(path):
#         logger.info(f"Deleting folder and subfolders at: {path}")
#         shutil.rmtree(path)
#     else:
#         logger.info(f"Folder does not exist: {path}")

#     logger.info(f"Saving data at: {path}")
#     # NOTE the compression may not be optimal, need to benchmark
#     store = zarr.storage.DirectoryStore(path, dimension_separator=".")
#     da.to_zarr(store=store, mode="w-",
#                                 compute=True
#                                 )
#     logger.info(f"Succesfully saved at: {path}")
########################################################################################
#######                           Plotting Functions                             #######
########################################################################################