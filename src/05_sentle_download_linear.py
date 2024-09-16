import geopandas as gpd
from sentle import sentle
import os
import geopandas as gpd
from dotenv import load_dotenv
import os
from pathlib import Path
import torch


def load_sentle(grid_path, idx, res):
    """
    Load Sentinel data using the Sentle library for a given grid.
    """
    intersected_gdf_equi7 = gpd.read_file(grid_path)
    bounds = idx + 1
    bounds = intersected_gdf_equi7[idx:bounds].geometry.iloc[0].bounds
    bound_left = int(bounds[0])
    bound_bottom = int(bounds[1])
    bound_right = int(bounds[2])
    bound_top = int(bounds[3])
    equi7_crs = intersected_gdf_equi7.crs
    print(f"Resolution: {res}")

    da = sentle.process(
        target_crs=equi7_crs,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        datetime="2015-01-01/2024-07-31",
        target_resolution=res,
        S2_mask_snow=True,
        S2_cloud_classification=True,
        S2_cloud_classification_device="cuda",
        S1_assets=["vv", "vh"],
        S2_apply_snow_mask=True,
        S2_apply_cloud_mask=True,
        time_composite_freq="7d",
        num_workers=40,
    )
    return da


def main():

    try:
        # Load the Environment variables
        env_path = Path('/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/environment/.env')
        load_dotenv(dotenv_path=env_path)

        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print(f"> Available CUDA devices: {torch.cuda.device_count()}")
        
        res = 10
        grid_path=f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_{res}_512.shp"

        start_idx = 10
        end_idx = 10
        for idx in range(start_idx, end_idx + 1):
            print(f"> Load the Minicube {idx} ...")
            da = load_sentle(grid_path=grid_path, idx = idx, res=res)
            print("> Save the Minicube ...")
            output_zarr_path = f"{os.getenv('SENTINEL2_MINICUBES')}/{idx}_{res}_512_20152024_equi7_NA.zarr"
            sentle.save_as_zarr(da, path=output_zarr_path)
            print(f"> Sucessfully saved the Minicube {idx} at {output_zarr_path} ...")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()