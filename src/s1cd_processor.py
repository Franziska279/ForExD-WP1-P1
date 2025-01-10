import os
import logging
from pathlib import Path
import geopandas as gpd
from dotenv import load_dotenv
import concurrent.futures
import xarray as xr
import rioxarray
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from affine import Affine
import rasterio
from shapely.geometry import box, shape
import numpy as np
from func_data_preprocessing import extract_s1cd_filename_part, calculate_area_in_km2_s1cd
from func_file_io import load_data
import shutil
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box, shape
import pandas as pd


class S1CDProcessor:

    def __init__(self, env_path, buffer_years=2, max_jobs=4):
        self.max_jobs = max_jobs
        self._set_up_logging()
        self._load_env_variables(env_path)
        self.dataset = None
        self.buffer_years = buffer_years
        self.metadata_table = []  

    def _set_up_logging(self):
        """Set up logging to file with timestamp."""
        logging.basicConfig(filename='scheduler.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def save_metadata_table(self, output_path):
        """Save the metadata table to a CSV file."""
        try:
            metadata_df = pd.DataFrame(self.metadata_table)
            metadata_df.to_csv(output_path, index=False)
            logging.info(f"Metadata table saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving metadata table: {e}")

    def _load_env_variables(self, env_path):
        """Load environment variables from a .env file."""
        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)
        self.region = os.getenv('REGION')
        if self.region is None:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)
        
        self.tcc_dir = os.getenv('TCC_PATH')
        self.input_dir = os.getenv('SENTINEL1_TILES')
        self.output_dir = os.getenv('RESULTS')
        if not all([self.tcc_dir, self.input_dir, self.output_dir]):
            raise ValueError("TCC_PATH, SENTINEL1_TILES, or RESULTS environment variables are not set")
        
        # Ensure required directories exist
        self.shapefile_dir = Path(f"{self.output_dir}/03_s1cd_polygons")
        self.shapefile_dir.mkdir(parents=True, exist_ok=True)
        
        self.s1dm_dir = Path(f"{self.output_dir}/s1dm")
        self.s1dm_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_crs = "EPSG:4326"

    def process_files(self):
        """Processes input files and updates progress bar."""

        logging.info("Starting batch processing...")

        input_files = [f for f in os.listdir(self.input_dir) if not f.endswith('.py')]
        total_files = min(6, len(input_files))  # Limit to the first 20 files (or fewer if there are less than 20)

        with tqdm(total=total_files, desc="Processing Files") as pbar:
            success_count, error_count = 0, 0
            with ProcessPoolExecutor(max_workers=self.max_jobs) as executor:
                # Only submit the first 20 files (or fewer if less than 20)
                futures = [executor.submit(self.run_extraction_script, f) for f in input_files[:total_files]]
                
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                    pbar.update(1)
                    pbar.set_description(f"Success: {success_count}, Errors: {error_count}")

        # Step to merge after all files have been processed
        logging.info("All individual files processed. Starting merge of shapefiles...")
        merged_gdf = self.merge_shapefiles(self.shapefile_dir)
        
        # Calculate area and filter the merged GeoDataFrame
        filtered_gdf = self.calculate_and_filter_area(merged_gdf)
        
        # Save the final merged and filtered result
        self.save_result(filtered_gdf, self.output_dir, output_filename=f'radar_enhanced_forest_disturbance_mapping_region_{self.region_id}.shp')

    # Cleanup: Remove the directory with individual shapefiles (optional)
    # try:
    #     shutil.rmtree(self.shapefile_dir)
    #     logging.info(f"Successfully removed directory and all contents: {self.shapefile_dir}")
    # except OSError as e:
    #     logging.error(f"Error removing directory and all contents: {self.shapefile_dir} - {e}")

    # logging.info("Batch processing completed.")

    def run_extraction_script(self, input_file):
        """Process individual file and log result."""
        input_path = os.path.join(self.input_dir, input_file)
        ids_path = f"{self.output_dir}/region_{self.region_id}_dca_filtered_ids_usda_polygons.shp"
        tcc_path = os.path.join(self.tcc_dir, "wp1_nlcd_tcc_conus_2017_v2021_4_20m_EPSG_4326_cropped_normalized_region_08.tif")

        try:
            if self.extract_polygons_from_raster(input_path, ids_path, tcc_path, self.target_crs, self.shapefile_dir):
                logging.info(f"Extraction successful for {input_file}")
                return True
            else:
                logging.error(f"Extraction failed for {input_file}")
                return False
        except Exception as e:
            logging.error(f"Error during polygon extraction for {input_file}: {str(e)}")
            return False

    def extract_year_from_filename(self, input_path):
        """Extract year from filename with pattern '_year_'."""
        try:
            filename = os.path.basename(input_path)
            return int(filename.split('_year_')[-1].split('_')[0])
        except (IndexError, ValueError) as e:
            logging.error(f"Error extracting year from filename {input_path}: {e}")
            return None

    def drop_unnecessary_vars(self):
        """Drop unnecessary variables from the dataset."""
        self.dataset = self.dataset.drop_vars(["x_bnds", "y_bnds"], errors='ignore')
        return self.dataset

    def rename_variables(self):
        """Rename variables for consistency."""
        if 'unnamed' in self.dataset.variables:
            self.dataset = self.dataset.rename({'unnamed': 'layer'})
        if 'X' in self.dataset.variables and 'Y' in self.dataset.variables:
            self.dataset = self.dataset.rename({'X': 'x', 'Y': 'y'})
        return self.dataset

    def reproject_to_wgs84(self):
        """Reproject the dataset to WGS 84 CRS."""
        
        crs_azimuthal_equidistant = "+proj=aeqd +lat_0=52 +lon_0=-97.5 +x_0=8264722.17686 +y_0=4867518.35323 +datum=WGS84 +units=m +no_defs"
        crs_wgs84 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        self.dataset.rio.write_crs(crs_azimuthal_equidistant, inplace=True)
    
        return self.dataset.rio.reproject(crs_wgs84)

    def load_and_preprocess_dataset(self, input_file):
        """Load and preprocess the raster dataset."""
        self.dataset = xr.open_dataset(input_file)
        self.dataset = self.drop_unnecessary_vars()
        self.dataset = self.rename_variables()
        self.dataset = self.reproject_to_wgs84()
        return self.dataset 

    def apply_tcc_mask(self, tcc_path):
        """
        Applies a Tree Canopy Cover (TCC) mask to the dataset.
        
        Parameters:
        - tcc_path (str): Path to the TCC 2017 raster file.
        
        Returns:
        - xarray.DataArray: Masked dataset.
        """
        logging.info("Opening TCC file...")
        try:
            tcc = rioxarray.open_rasterio(tcc_path)
        except Exception as e:
            logging.error(f"Error opening TCC file: {e}")
            return None
        
        # Step 2: Define spatial extent based on the dataset's coordinates
        logging.info("Extracting spatial extent from the dataset...")
        min_lon, max_lon = self.dataset['x'].min(), self.dataset['x'].max()
        min_lat, max_lat = self.dataset['y'].min(), self.dataset['y'].max()

        # Check dataset bounds and TCC data bounds
        # logging.info(f"Dataset bounds - min_lon: {min_lon}, max_lon: {max_lon}, min_lat: {min_lat}, max_lat: {max_lat}")
        # logging.info(f"TCC data bounds - min_lon: {tcc.coords['x'].min()}, max_lon: {tcc.coords['x'].max()}, min_lat: {tcc.coords['y'].min()}, max_lat: {tcc.coords['y'].max()}")

        # Step 3: Subset the TCC data to match the spatial extent of the dataset
        logging.info("Selecting subset from TCC data...")
        subset = tcc.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))
        
        # Check if the subset is empty
        if subset.isnull().all():
            logging.error(f"Subset extraction for coordinates {min_lon}-{max_lon}, {max_lat}-{min_lat} is empty.")
            return None

        # Step 4: Interpolate normalized subset to match dataset's coordinates
        logging.info("Interpolating normalized subset to match dataset's coordinates...")
        try:
            normalized_subset_interp = subset.interp(x=self.dataset.coords['x'], y=self.dataset.coords['y'], method='nearest')
        except Exception as e:
            logging.error(f"Error during interpolation: {e}")
            return None

        # Step 5: Apply mask to dataset where normalized values are above threshold
        logging.info("Applying mask to dataset...")
        masked_dataset = self.dataset.where(normalized_subset_interp > 0.3, 0).fillna(0)
        
        return masked_dataset

    def extract_polygons_from_mask(self, filename, masked_data_array):
        """
        Extracts polygons from a masked data array and stores them in a GeoDataFrame 
        with year and tile metadata extracted from the filename.

        Parameters:
        - filename (str): The filename containing year and tile information.
        - masked_data_array (xarray.DataArray): The masked data array with 'x' and 'y' coordinates 
        and a 'layer' attribute representing the mask layer.

        Returns:
        - GeoDataFrame: A GeoDataFrame containing extracted polygons with year and tile metadata.
        """
        
        logging.info(f"Polygon extraction completed successfully for {filename}.")
        # Step 1: Parse year and tile information from filename
        year = int(filename.split('_year_')[-1].split('_')[0])
        tile_name = filename[13:23]  # Extract tile name based on naming convention
        
        # Step 2: Initialize an empty GeoDataFrame to store results
        results_gdf = gpd.GeoDataFrame(columns=['geometry', 'S1_YEAR', 'S1_TILE'], crs="EPSG:4326")
        
        # Step 3: Extract the bounding box of the masked data array
        min_x, max_x = masked_data_array['x'].min().item(), masked_data_array['x'].max().item()
        min_y, max_y = masked_data_array['y'].min().item(), masked_data_array['y'].max().item()
        
        # Step 4: Create a bounding box GeoDataFrame for the spatial extent
        bounds_gdf = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs="EPSG:4326")
        
        # Step 5: Drop the 'band' dimension if present, keeping only relevant layers
        cropped_mask = masked_data_array.squeeze("band")
        
        # Step 6: Define the affine transform for geospatial information
        # This transformation maps array coordinates to spatial coordinates
        transform = (
            Affine.translation(cropped_mask.x[0], cropped_mask.y[0]) * 
            Affine.scale(cropped_mask.x[1] - cropped_mask.x[0], cropped_mask.y[1] - cropped_mask.y[0])
        )
        
        # Step 7: Generate a binary mask from the data array layer
        # Convert values greater than zero to 1, else to 0
        binary_mask = (cropped_mask['layer'] > 0).astype(np.uint8)
        
        # Step 8: Extract polygon shapes from the binary mask using the affine transform
        # Each shape corresponds to a contiguous region of 1s in the mask
        extracted_shapes = list(rasterio.features.shapes(binary_mask.values, transform=transform))
        
        # Step 9: Filter the extracted shapes to keep only those with a value of 1
        polygons = [shape(geom) for geom, value in extracted_shapes if value == 1]
        
        # Step 10: Create a GeoDataFrame from the list of polygons
        polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=cropped_mask.spatial_ref)
        
        # Step 11: Add metadata columns for year and tile
        polygons_gdf['S1_YEAR'] = year
        polygons_gdf['S1_TILE'] = tile_name
        
        # Step 12: Append the new polygons to the results GeoDataFrame
        results_gdf = pd.concat([results_gdf, polygons_gdf], ignore_index=True)
        logging.info(f"Polygon extraction completed successfully for {filename}.")
        return results_gdf

    def process_and_filter_polygons(self, ids_usda_path, s1_year, year_buffer, file, target_crs, output_dir):
        logging.info("Step 1: Loading and buffering USDA polygons...")
        try:
            ids_usda_gdf = load_data(ids_usda_path)
            ids_usda_gdf['geometry'] = ids_usda_gdf['geometry'].buffer(0.005)  # 500m buffer
        except Exception as e:
            logging.error(f"Error loading USDA polygons: {e}")
            return
        
        logging.info(f"Step 2: Filtering USDA polygons within ±{year_buffer} years of {s1_year}...")
        try:
            ids_usda_filtered = ids_usda_gdf[
                (ids_usda_gdf['SURVEY_Y'] >= s1_year - year_buffer) & 
                (ids_usda_gdf['SURVEY_Y'] <= s1_year + year_buffer)
            ]
        except Exception as e:
            logging.error(f"Error filtering USDA polygons: {e}")
            return

        logging.info("Step 3: Calculating area before intersection...")
        polygons_gdf = self.dataset
        try:
            # Compute total area before intersection
            area_gdf = calculate_area_in_km2_s1cd(polygons_gdf)
            area_before_intersection = area_gdf['area'].sum()
        except Exception as e:
            logging.error(f"Error calculating area before intersection: {e}")
            return
        
        logging.info("Step 4: Performing spatial join to find intersecting polygons...")
        try:
            intersecting_gdf = gpd.sjoin(polygons_gdf, ids_usda_filtered, predicate='intersects')
            intersecting_gdf = intersecting_gdf.rename(columns={'index_right': 'S1CD_IDX'})
        except Exception as e:
            logging.error(f"Error during spatial join: {e}")
            return
        
        logging.info("Step 5: Calculating area after intersection...")
        try:
            # Compute total area after intersection
            #area_after_intersection = intersecting_gdf['geometry'].area.sum()
            area_after_gdf = calculate_area_in_km2_s1cd(intersecting_gdf)
            area_after_intersection = area_after_gdf['area'].sum()
        except Exception as e:
            logging.error(f"Error calculating area after intersection: {e}")
            return

        # Add metadata to the table
        tile_name = extract_s1cd_filename_part(os.path.basename(file))
        self.metadata_table.append({
            'Tile': tile_name,
            'Year': s1_year,
            'Area Before Intersection': area_before_intersection,
            'Area After Intersection': area_after_intersection
        })

        logging.info("Step 6: Aggregating geometries by USDA_IDX...")
        try:
            aggregated_gdf = intersecting_gdf.dissolve(by='IDX_D')
            aggregated_gdf.reset_index(inplace=True)
        except Exception as e:
            logging.error(f"Error during geometry aggregation: {e}")
            return

        logging.info("Step 7: Reprojecting and saving output shapefile...")
        try:
            aggregated_gdf.set_crs(epsg=4326, inplace=True)
            aggregated_gdf = aggregated_gdf.to_crs(target_crs)
            os.makedirs(output_dir, exist_ok=True)
            shapefile_path = os.path.join(output_dir, f"{tile_name}.shp")
            aggregated_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
        except Exception as e:
            logging.error(f"Error saving shapefile: {e}")


    def extract_polygons_from_raster(self, input_file, ids_usda_path, tcc_path, target_crs, output_dir):
        try:
            s1_year = self.extract_year_from_filename(input_file)
            filename = os.path.basename(input_file)
            logging.info(f"Processing file: {input_file} for year {s1_year}")

            self.dataset = self.load_and_preprocess_dataset(input_file)
            self.dataset = self.apply_tcc_mask(tcc_path)
            self.dataset = self.extract_polygons_from_mask(filename, self.dataset)
            self.process_and_filter_polygons(ids_usda_path,s1_year, self.buffer_years, input_file, target_crs, output_dir)
            
            logging.info(f"Polygon extraction successful for {input_file}, saved to {output_dir}")
            return True
        except Exception as e:
            logging.error(f"Error during polygon extraction for {input_file}: {e}")
            return False


    def merge_shapefiles(self, input_dir):
        """
        Merge all shapefiles in the specified directory into a single GeoDataFrame.
        """
        logging.info(f"Merging shapefiles from {input_dir}")
        files = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
        gdf_list = []
        
        for file in tqdm(files, desc="Merging shapefiles"):
            filepath = os.path.join(input_dir, file)
            gdf = gpd.read_file(filepath)
            
            # Ensure CRS is set
            if gdf.crs is None:
                raise ValueError(f"CRS not defined for file: {filepath}. Please define CRS for all shapefiles.")
            
            gdf_list.append(gdf)
        
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        logging.info("Shapefiles merged successfully.")
        return merged_gdf

    def calculate_and_filter_area(self, gdf):
        """
        Calculate area in square kilometers and filter out polygons larger than 15 km².
        """
        logging.info("Calculating area and filtering polygons...")
        gdf = gdf.to_crs(self.target_crs)
        
        # Reproject to a CRS with meters (e.g., EPSG:3857) for accurate area calculation
        projected_gdf = gdf.to_crs('EPSG:27705')
        
        # Calculate the area in square meters and convert to km²
        projected_gdf['area_km2'] = projected_gdf.geometry.area / 1e6
        
        # Add the calculated area to the original GeoDataFrame and filter
        gdf['area_km2'] = projected_gdf['area_km2']
        filtered_gdf = gdf[gdf['area_km2'] <= 15]
        
        logging.info("Area calculated and polygons filtered.")
        return filtered_gdf

    def save_result(self, gdf, output_dir, output_filename):

        """
        Save the resulting GeoDataFrame to a new shapefile.
        """
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Saving result to {output_path}...")
        gdf.to_file(output_path)
        logging.info("Result saved successfully.")
