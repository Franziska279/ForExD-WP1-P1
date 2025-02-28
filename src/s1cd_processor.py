import os
import logging
import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from dotenv import load_dotenv
import geopandas as gpd
from shapely.geometry import box
import logging
from func_file_io import load_data
from func_data_preprocessing import extract_s1cd_filename_part, calculate_area_in_km2_s1cd
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from func_s1cd_preprocessing import (
    drop_unnecessary_vars,
    rename_variables,
    merge_shapefiles,
    process_and_filter_usda_polygons,
    extract_polygons_from_mask,
    apply_tcc_mask,
    load_and_preprocess_dataset,
    extract_year_from_s1cd_filename,
    calculate_and_filter_area
)

class S1CDProcessor:

    def __init__(self, env_path, buffer_years, spatial_buffer, max_jobs):
        """Initialize the S1CDProcessor with environment variables, buffer settings, and logging."""
        self._set_up_logging()
        self._load_env_variables(env_path)

        self.buffer_years = buffer_years  # Buffer for year filtering
        self.spatial_buffer = spatial_buffer  
        self.max_jobs = max_jobs

    def _set_up_logging(self):
        """Set up logging to a file with timestamps for tracking the process."""
        logging.basicConfig(
            filename='log_s1cd_processor.log',
            level=logging.INFO, 
            format='%(asctime)s - %(message)s'
        )

    def _load_env_variables(self, env_path):
        """Load required environment variables from a .env file and ensure the paths exist."""
        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        # Load environment variables and validate
        self.region = os.getenv('REGION')
        if not self.region:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)

        # Set target CRS (Coordinate Reference System)
        self.target_crs = os.getenv('TARGET_CRS')
        self.equi7_crs = os.getenv('EQUI7_NA_EPSG')

        # Set directory paths from environment variables
        self.input_dir = os.getenv('SENTINEL1_TILES_DIR')
        self.ids_filtered = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        self.tcc_normalized = os.path.join(os.getenv('TCC_DIR'), os.getenv('TCC_NORMALIZED_RASTER_TEMPLATE').format(region_id=self.region_id))
        self.s1_tiles_boundary_path =  os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id))
    
        self.intermediate_dir = os.getenv('INTERMEDIATE_FILES_DIR')
        self.metadata_dir = os.getenv('METADATA_FILES_DIR')

        # Validate that the required directories exist
        self._validate_directory(self.input_dir, "Input directory")
        self._validate_directory(self.intermediate_dir, "Intermediate directory")
        self._validate_directory(self.metadata_dir, "Metadata directory")

    def _validate_directory(self, directory_path, dir_name):
        """Check if a directory exists, create it if not, and log the status."""
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"{dir_name} directory was missing. Created at {dir_path}")
        else:
            logging.info(f"{dir_name} is valid at {dir_path}")

    def process_files(self):
        """
        Process all input files concurrently.
        """
        logging.info("============================================")
        logging.info("Starting processing of S1 Boundary Outlines...")

        self._collect_and_save_s1_tile_boundary(self.input_dir, self.region_id, self.s1_tiles_boundary_path, self.equi7_crs)
        logging.info("---------------------------------------------")
        logging.info("Starting batch processing of input files...")

        # List all files in the input directory excluding Python files
        input_files = [f for f in os.listdir(self.input_dir) if not f.endswith('.py')]
        total_files = len(input_files)
        
        logging.info(f"Total files identified for processing: {total_files}")

        # Handle case where no files are found for processing
        if total_files == 0:
            logging.warning("No valid input files found in the directory. Exiting process.")
            return

        success_count, error_count = 0, 0

        # Start processing files concurrently using ProcessPoolExecutor
        logging.info(f"Processing {total_files} files concurrently with {self.max_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=self.max_jobs) as executor:
            # Using executor to process files concurrently
            results = list(executor.map(self._process_file_wrapper, input_files))

            # Summing up the results to count successes and errors
            success_count = sum(results)
            error_count = len(results) - success_count

        # Log summary of the processing results
        logging.info(f"Batch processing complete.")
        logging.info(f"Successfully processed {success_count} files.")
        logging.error(f"Processing failed for {error_count} files.")

        # If all files were processed successfully, proceed with merging and saving
        
        logging.info(f"{success_count} files processed successfully. Proceeding with merging and saving shapefiles.")
        self._merge_and_save_shapefiles()
    

    def _merge_and_save_shapefiles(self):
        """
        Merges shapefiles from each buffer directory, calculates areas, filters the resulting GeoDataFrame, 
        and saves the final merged shapefile.
        """
        logging.info("Starting the merge and save process for shapefiles...")

        # Loop through each buffer and process its shapefiles
        for buffer in self.spatial_buffer:
            logging.info(f"Processing buffer: {buffer} m")

            # Define the directory containing shapefiles for the current buffer
            buffer_dir = os.path.join(self.intermediate_dir, f'buffer_{buffer}')
            
            # Merge shapefiles for the current buffer
            logging.info(f"Merging shapefiles in directory: {buffer_dir}")
            merged_gdf = merge_shapefiles(buffer_dir, self.target_crs)

            if merged_gdf.empty:
                logging.warning(f"No shapefiles found or failed to merge in: {buffer_dir}")
                continue

            # Calculate and filter the merged GeoDataFrame based on area
            logging.info(f"Calculating area and filtering the merged GeoDataFrame for buffer: {buffer}")
            filtered_gdf = calculate_and_filter_area(merged_gdf, self.target_crs)

            if filtered_gdf.empty:
                logging.warning(f"No valid data left after filtering for buffer: {buffer}. Skipping save.")
                continue

            # Define the output path for the final shapefile
            shapefile_output_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('S1DM_SHAPE_FILE').format(region_id=self.region_id, buffer=buffer))
            logging.info(f"Output path: {shapefile_output_path}")
            # Ensure the directory for the shapefile exists
            os.makedirs(os.path.dirname(shapefile_output_path), exist_ok=True)

            # Save the filtered GeoDataFrame to a shapefile
            try:
                logging.info(f"Saving filtered shapefile for buffer: {buffer} to {shapefile_output_path}")
                filtered_gdf.to_file(shapefile_output_path, driver='ESRI Shapefile')
                logging.info(f"Shapefile saved successfully for buffer {buffer} at {shapefile_output_path}")
            except Exception as e:
                logging.error(f"Error saving shapefile for buffer {buffer} to {shapefile_output_path}: {e}")


            # Cleanup: Delete the buffer directory after saving
            try:
                shutil.rmtree(buffer_dir)
                logging.info(f"Successfully deleted buffer directory: {buffer_dir}")
            except OSError as e:
                logging.error(f"Error deleting buffer directory {buffer_dir}: {e}")

        logging.info("Merge and save process completed for all buffers.")

    def _process_file_wrapper(self, file_name):
        try:
            if self._run_extraction_script(file_name):
                logging.info(f"Successfully processed {file_name}")
                return True
            logging.error(f"Failed to process {file_name}")
            return False
        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")
            return False

    def _run_extraction_script(self, input_file):
        """
        Run the extraction process for a single input file.
        Includes applying a TCC mask, extracting polygons, and filtering.
        """
        input_path = os.path.join(self.input_dir, input_file)
    
        if not Path(input_path).exists():
            logging.error(f"Input file {input_file} does not exist at {input_path}.")
            return False

        logging.info(f"Extracting polygons from raster for {input_file}.")
        if self._extract_polygons_from_raster(
            input_path, 
            self.ids_filtered, 
            self.tcc_normalized
            ):
            logging.info(f"Extraction successful for {input_file}.")
            return True
        else:
            logging.error(f"Extraction failed for {input_file}")
            return False

    def _extract_polygons_from_raster(self, input_file, ids_usda_path, tcc_path):
        """
        Extract polygons from a raster file and process them by applying a Tree Canopy Cover (TCC) mask, 
        filtering USDA polygons, and calculating areas before and after intersections. 
        The results are saved as shapefiles and metadata tables.

        Parameters:
        - input_file (str): Path to the input raster file.
        - ids_usda_path (str): Path to the USDA polygons shapefile.
        - tcc_path (str): Path to the Tree Canopy Cover (TCC) raster file.

        Returns:
        - bool: True if processing was successful, False if an error occurred.
        """
        try:
            # Step 1: Extract year from the Sentinel-1 Cloud Data filename
            s1_year = extract_year_from_s1cd_filename(input_file)
            filename = os.path.basename(input_file)
            logging.info(f"Processing file: {input_file} for year {s1_year}")

            # Step 2: Load and preprocess the input raster dataset
            logging.info(f"Loading and preprocessing dataset from {input_file}...")
            dataset = load_and_preprocess_dataset(input_file)

            # Step 3: Apply Tree Canopy Cover (TCC) mask if path is provided
            if tcc_path:
                logging.info(f"Applying TCC mask from {tcc_path}...")
                dataset = apply_tcc_mask(dataset, tcc_path)

            # Step 4: Extract polygons from the masked dataset
            logging.info("Extracting polygons from the masked dataset...")
            dataset = extract_polygons_from_mask(filename, dataset)
            
            # Step 5: Process and filter USDA polygons for each spatial buffer size
            for buffer in self.spatial_buffer:
                logging.info(f"Processing with buffer size: {buffer} meters...")
                
                # Define the output shapefile path
                tile_name = extract_s1cd_filename_part(os.path.basename(input_file))
                output_shapefile = os.path.join(self.intermediate_dir, f"buffer_{buffer}", f"{tile_name}.shp")
                
                # Define the output metadata CSV path
                output_metadata = os.path.join(self.metadata_dir, f"buffer_{buffer}", f"metadata_table_{tile_name}.csv")

                # Ensure output directories exist
                os.makedirs(os.path.dirname(output_shapefile), exist_ok=True)
                os.makedirs(os.path.dirname(output_metadata), exist_ok=True)

                # Step 6: Process USDA polygons, filter by year, and apply spatial operations
                logging.info(f"Processing and filtering USDA polygons for year {s1_year}...")
                process_and_filter_usda_polygons(
                    dataset, ids_usda_path, s1_year, self.buffer_years, buffer, input_file, 
                    self.target_crs, output_shapefile, output_metadata, tile_name
                )

            # Step 7: Log success message
            logging.info(f"Polygon extraction successful for {input_file}. Results saved to {output_shapefile} and metadata to {output_metadata}")

            return True

        except Exception as e:
            # Log the error message if any exception occurs during the processing
            logging.error(f"Error during polygon extraction for {input_file}: {e}")
            return False

    def _merge_geometries_and_keep_columns(self, gdf):
        """
        Merge geometries by 'IDX_D' and 'S1_YEAR' into a single geometry (MultiPolygon) and
        keep the first value for all other columns.

        Parameters:
        - gdf (GeoDataFrame): The input GeoDataFrame with geometries and other columns.
        
        Returns:
        - GeoDataFrame: A new GeoDataFrame with merged geometries and first values of other columns.
        """
        grouped_gdf = gdf.groupby(['IDX_D', 'S1_YEAR']).apply(
            lambda group: group.unary_union  # Merge the geometries within each group
        ).reset_index(name='geometry')

        # Step 2: For other columns, keep the first value
        for column in gdf.columns:
            if column not in ['IDX_D', 'S1_YEAR', 'geometry']:  # Skip 'IDX_D', 'S1_YEAR', and 'geometry'
                # Ensure the aggregation keeps the first value for each group
                grouped_gdf[column] = gdf.groupby(['IDX_D', 'S1_YEAR'])[column].first().values

        # Step 3: Convert to GeoDataFrame and ensure geometries are MultiPolygons if they aren't already
        grouped_gdf = gpd.GeoDataFrame(grouped_gdf, geometry='geometry')

        # Ensure the geometries are MultiPolygons if they aren't already
        grouped_gdf['geometry'] = grouped_gdf['geometry'].apply(
            lambda geom: MultiPolygon([geom]) if not isinstance(geom, MultiPolygon) else geom
        )

        # Step 4: Set CRS (coordinate reference system) if needed
        grouped_gdf.set_crs(gdf.crs, allow_override=True, inplace=True)

        return grouped_gdf



    def _collect_and_save_s1_tile_boundary(self, input_dir, region_id, s1_tiles_boundary_path, crs):
        """
        Collects spatial boundaries of Sentinel-1 tiles and saves them as a shapefile.

        Parameters:
        - input_dir (str): Directory containing Sentinel-1 tile NetCDF files.
        - region_id (str): Region identifier (not currently used but kept for future use).
        - s1_tiles_boundary_path (str): Output shapefile path.

        Returns:
        - None
        """

        input_files = [f for f in os.listdir(input_dir) if not f.endswith('.py')]
        total_files = len(input_files)
        outline_s1cd_files = None
        logging.info(f"Total files identified for processing: {total_files}")

            
        for file in input_files:
            file_path = os.path.join(input_dir, file)

            
            try:
                # Open dataset
                dataset = xr.open_dataset(file_path)
                dataset = drop_unnecessary_vars(dataset)
                dataset = rename_variables(dataset)

                # Extract spatial bounds
                dataset_lon_min, dataset_lon_max = float(dataset['x'].min()), float(dataset['x'].max())
                dataset_lat_min, dataset_lat_max = float(dataset['y'].min()), float(dataset['y'].max())

                logging.info(f"Processing {file}: Longitude [{dataset_lon_min}, {dataset_lon_max}], "
                            f"Latitude [{dataset_lat_min}, {dataset_lat_max}]")

                # Create a bounding box polygon
                bounding_box = box(dataset_lon_min, dataset_lat_min, dataset_lon_max, dataset_lat_max)

                # Create a GeoDataFrame
                new_gdf = gpd.GeoDataFrame(geometry=[bounding_box], crs=crs)

                # Append to existing GeoDataFrame
                if outline_s1cd_files is None:
                    outline_s1cd_files = new_gdf
                else:
                    outline_s1cd_files = pd.concat([outline_s1cd_files, new_gdf], ignore_index=True)

            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
        
        # Ensure we have a valid dataset before saving
        if outline_s1cd_files is not None and not outline_s1cd_files.empty:
            try:
                outline_s1cd = outline_s1cd_files.drop_duplicates().reset_index(drop=True)
                
                # Ensure CRS is explicitly set
                if outline_s1cd.crs is None:
                    logging.warning("GeoDataFrame has no CRS; setting it to Traget Crs.")
                    outline_s1cd.set_crs(crs, inplace=True)
                
                # Save to shapefile
                outline_s1cd.to_file(s1_tiles_boundary_path, driver="ESRI Shapefile")

                # Log saved CRS
                saved_gdf = gpd.read_file(s1_tiles_boundary_path)
                logging.info(f"Boundary shapefile saved successfully at {s1_tiles_boundary_path}")
                logging.info(f"Shapefile saved with CRS: {saved_gdf.crs}")

            except Exception as e:
                logging.error(f"Failed to save boundary shapefile: {e}")
        else:
            logging.warning("No valid dataset boundaries were processed. No shapefile created.")
