import os
import logging
import shutil
from pathlib import Path
import geopandas as gpd
from shapely.geometry import MultiPolygon
from dotenv import load_dotenv
import geopandas as gpd
from shapely.geometry import box
import logging
from func_file_io import load_data
from func_data_preprocessing import extract_s1cd_filename_part, calculate_area_in_km2_s1cd
from concurrent.futures import ProcessPoolExecutor
from func_s1cd_preprocessing import (
    merge_shapefiles,
    process_and_filter_usda_polygons,
    extract_polygons_from_mask,
    apply_tcc_mask,
    load_and_preprocess_dataset,
    extract_year_from_s1cd_filename,
    calculate_and_filter_area
)

class S1DM_S2_Processor:

    def __init__(self, env_path, buffer_years, spatial_buffer, max_jobs):
        """Initialize the S1CDProcessor with environment variables, buffer settings, and logging."""
        self._set_up_logging()
        self._load_env_variables(env_path)

        self.buffer_years = buffer_years  # Buffer for year filtering
        self.spatial_buffer = spatial_buffer  
        self.max_jobs = max_jobs
        self.outline_s1cd_files = None


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

    def ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def calulate_eventwise_timeseries(aggregation_modes, reference_years, dca_keys, year_keys, method_keys, variables):

        # List of aggregation modes and reference years
        aggregation_modes = ["earliest"] #"earliest", 
        reference_years = ['S1_YEAR', 'SURVEY_Y']
        dca_keys = ["defoliators", "wind", "bark_beetle", "fire", "drought"]
        year_keys = list(range(2016, 2022))
        method_keys = ['percentile']
        variables = ['nbr', 'ndvi']

        buffer = 100


        print(f"Processing for buffer: {buffer}")

        # Reference file for the buffer
        s1dm_path = f"{os.getenv('RESULTS')}/radar_enhanced_forest_disturbance_mapping_region_{region_id}_buffer_{buffer}.shp"
        s1dm_gdf = load_data(s1dm_path)
        ids_gdf = load_data(ids_path)


        # Create individual log file for each buffer
        log_file = f'./log_s1dm_s2_buffer_{buffer}_test.log'
        logger = setup_logger(log_file)

        # Log the length of refdm_gdf
        s1dm_length = len(s1dm_gdf)
        logger.info(f"Buffer: {buffer} - Length of refdm_gdf: {s1dm_length}")
        data_path = os.path.join(netcdf_data_path, f"buffer_{buffer}")
        figure_path = os.path.join(s1dm_s1_figure_path, f"buffer_{buffer}")

        _calculate_save_netcdf_events_anomalys(ids, s1dm, grid, data_path, figure, s2_folder, equi7_crs, tcc_file_path, logger)

    def _calculate_save_netcdf_events_anomalys(ids_gdf, s1dm_gdf, grid, data_path, figure_path, s2_minicube_folder, 
                                                equi7_crs, tcc_file, logger, aggregation_modes, reference_years, dca_keys, year_keys, method_keys, variables):
        """
        Führt die Aggregation und Gruppierung mit mehreren Varianten durch und speichert die Ergebnisse.
        1. Aggregation 'earliest' mit Referenzjahr 'S1_YEAR'.
        2. Aggregation 'nearest' mit Referenzjahr 'S1_YEAR'.
        3. Aggregation 'nearest' mit Referenzjahr 'SURVEY_Y'.

        Args:
            ids_gdf (GeoDataFrame): Eine Liste von IDs oder Schlüsseln für die Verarbeitung.
            s1dm_gdf (GeoDataFrame): Das Eingabe-GeoDataFrame.
            data_path (str): Pfad zum Speichern der Daten.
            figure_path (str): Pfad zum Speichern der Abbildungen.
        """


        tcc = rioxarray.open_rasterio(tcc_file)
        # Transform to the desired CRS (equi7_crs)
        if s1dm_gdf.crs != equi7_crs:
            s1dm_gdf = s1dm_gdf.to_crs(equi7_crs)
            logger.info(f"🌍 Transformed S1DM to CRS {equi7_crs}.")

        if ids_gdf.crs != equi7_crs:
            ids_gdf = ids_gdf.to_crs(equi7_crs)
            logger.info(f"🌍 Transformed IDS to CRS {equi7_crs}.")

        # Iterate over all combinations of aggregation modes and reference years
        for agg_mode in aggregation_modes:
            for ref_year in reference_years:
                # log_file = f'./logs/s1dm_s2_minicube_timeseries_agg_{agg_mode}_ref_{ref_year}.log'
                # logger = setup_logger(log_file)
                # Perform aggregation
                logger.info(f"Processing aggregation_mode={agg_mode} with reference_year={ref_year}...")
                
                disturbance_counts_df, s1dm_gdf_aggregated = process_geodataframe(
                    s1dm_gdf, 
                    aggregation_mode=agg_mode, 
                    reference_year_column=ref_year
                    )
                
                for dca_key, year_key in itertools.product(dca_keys, year_keys):
                
                    logger.info("\n" + "=" * 70)
                    logger.info(f"   🚀 Starting Processing for DCA_ID: {dca_key}")
                    logger.info(f"   📅 Year: {year_key}")
                    logger.info("=" * 70 + "\n")

                    # Filter the GeoDataFrame based on DCA_ID and reference year
                    gdf = s1dm_gdf_aggregated[
                        (s1dm_gdf_aggregated['DCA_ID'] == dca_key) &
                        (s1dm_gdf_aggregated[ref_year] == year_key)
                    ]

                    # Extract unique IDX_D values
                    unique_idx_d_values = gdf['IDX_D'].unique()
                    unique_idx_d_df = pd.DataFrame(unique_idx_d_values, columns=['IDX_D'])

                    # Iterate through unique IDX_D values
                    logger.info(f"🔍 Found {len(unique_idx_d_df)} unique IDX_D values for DCA_ID: {dca_key}")
                    for idx, row in unique_idx_d_df.iterrows():
                        logger.info(f"   👉 Processing IDX_D {idx + 1}/{len(unique_idx_d_df)}: {row['IDX_D']}")
                        
                        try:
                            # Get minicubes for the group
                            cubes, idx_d = get_unique_minicube_FID(
                                i=idx,
                                df=unique_idx_d_df,
                                s1dm=s1dm_gdf,
                                ids=ids_gdf,
                                grid=grid
                                )

                            # Subset the data
                            s1dm = subset(idx_d, s1dm_gdf)
                            ids = subset(idx_d, ids_gdf)
                            square_bbox = generate_square_bbox_with_buffer(s1dm, ids, buffer=0.02)

                            # Validate geometries
                            if ids.geometry.is_empty.any() or not ids.geometry.is_valid.all():
                                logger.warning(f"⚠️ Invalid or missing geometries in IDS for {idx_d}. Skipping...")
                                continue  # Skip to next IDX_D
                            if s1dm.geometry.is_empty.any() or not s1dm.geometry.is_valid.all():
                                logger.warning(f"⚠️ Invalid or missing geometries in S1DM for {idx_d}. Skipping...")
                                continue  # Skip to next IDX_D

                            ids_mc, s1dm_mc, bbox_mc = [], [], []

                            # Process minicubes
                            for index, row in cubes.iterrows():
                                try:
                                    i = row['FID']
                                    logger.info(f"      🔄 Processing minicube FID {i}...")

                                    path = f"{s2_minicube_folder}/{i}_10_512_20152024_equi7_NA.nc"
                                    mc = load_netcdf(path)

                                    if not mc.rio.crs:
                                        mc = mc.rio.write_crs(equi7_crs)

                                    
                                    min_lon, max_lon = mc['x'].min(), mc['x'].max()
                                    min_lat, max_lat = mc['y'].min(), mc['y'].max()
                                    tcc_subset = tcc.sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))
                                    normalized_subset_interp = tcc_subset.interp(x=mc.coords['x'], y=mc.coords['y'], method='nearest')

                                    # Apply mask with NaN instead of 0
                                    masked_mc = mc.where(normalized_subset_interp > 0.2)
                                    logger.info(f"      🌳 TCC subset successfully processed and applied for FID {i}.")
                                    # Check for required data variables
                                    if all(var in masked_mc.data_vars for var in ['nbr', 'ndvi', 'kndvi']):
                                        masked_mc = masked_mc[['nbr', 'ndvi','kndvi']]
                                    else:
                                        logger.warning(f"⚠️ Missing required variables ['nbr', 'ndvi', 'kndvi'] in FID {i}. Skipping...")
                                        continue

                                    # Clip IDS and S1DM geometries
                                    clipped_ids, clipped_s1dm, clipped_bbox = None, None, None
                                    ids_shape = gpd.GeoSeries(ids.geometry)
                                    s1dm_shape = gpd.GeoSeries(s1dm.geometry)
                                    square_bbox_shape = gpd.GeoSeries(square_bbox.geometry)

                                    try:
                                        clipped_ids = masked_mc.rio.clip(ids_shape.geometry.apply(mapping), drop=True)
                                        if clipped_ids:
                                            ids_mc.append(clipped_ids)
                                        logger.info(f"         ✅ Successfully clipped IDS for FID {i}.")
                                    except NoDataInBounds:
                                        logger.warning(f"         ⚠️ No data found in bounds for IDS in FID {i}. Skipping IDS.")

                                    try:
                                        clipped_s1dm = masked_mc.rio.clip(s1dm_shape.geometry.apply(mapping), drop=True)
                                        if clipped_s1dm:
                                            s1dm_mc.append(clipped_s1dm)
                                        logger.info(f"         ✅ Successfully clipped S1DM for FID {i}.")
                                    except NoDataInBounds:
                                        logger.warning(f"         ⚠️ No data found in bounds for S1DM in FID {i}. Skipping S1DM.")

                                    try:
                                        clipped_bbox = masked_mc.rio.clip(square_bbox_shape.geometry.apply(mapping), drop=True)
                                        if clipped_bbox:
                                            bbox_mc.append(clipped_bbox) 
                                        logger.info(f"         ✅ Successfully clipped Box for FID {i}.")
                                    except NoDataInBounds:
                                        logger.warning(f"         ⚠️ No data found in bounds for S1DM in FID {i}. Skipping Box.")

                                    # Ensure 'time' dimension is present
                                    if 'time' not in masked_mc.dims:
                                        logger.warning(f"⚠️ No 'time' dimension in NetCDF for FID {i}. Skipping...")
                                        continue

                                except Exception as e:
                                    logger.warning(f"❌ Error processing FID {i}: {e}")
                                    continue

                            # Log the count of clipped datasets
                            logger.info(f"   📊 Processed minicubes summary for IDX_D {idx_d}:")
                            logger.info(f"         📌 Nₖ(IDS) minicubes: {len(ids_mc)}")
                            logger.info(f"         📌 Nₖ(S1DM) minicubes count: {len(s1dm_mc)}")
                            logger.info(f"         📌 Nₖ(Box) minicubes count: {len(bbox_mc)}")

                            # Skip if no valid minicubes
                            if not ids_mc or not s1dm_mc:
                                logger.warning(f"⚠️ No valid data for IDX_D {idx_d}. Skipping...")
                                continue

                            # Merge datasets
                            try:
                                merged_ids = xr.merge(ids_mc)
                                merged_s1dm = xr.merge(s1dm_mc)
                                merged_bbox = xr.merge(bbox_mc)
                                logger.info(f"   ✅ Successfully merged datasets for IDX_D {idx_d}.")
                            except ValueError as e:
                                logger.error(f"❌ Failed to merge datasets for IDX_D {idx_d}: {e}")
                                continue
                            
                            # Loop through the season calculation methods
                            for mode in method_keys:
                                logger.info(f"   🌀 Calculating Perfect Season using method: {mode}")

                                # Define paths for saving data and figures
                                save_data_dir = os.path.join(data_path, f"Event_Timeseries_Anomalys/Ref_Col_{ref_year}/Aggregation_{agg_mode}/Anomaly_{mode}/{dca_key}/{year_key}/")
                                os.makedirs(save_data_dir, exist_ok=True)

                                save_fig_dir = os.path.join(figure_path, f"Event_Timeseries_Anomalys/Ref_Col_{ref_year}/Aggregation_{agg_mode}/Anomaly_{mode}/{dca_key}/{year_key}/")
                                os.makedirs(save_fig_dir, exist_ok=True)
                                try:
                                    # Calculate Perfect Season for IDS and S1DM
                                    ids_diff, ids_perfect, ids_mc_reprocessed = calculatePerfectSaison(merged_ids, 2016, mode)
                                    s1dm_diff, s1dm_perfect, s1dm_mc_reprocessed = calculatePerfectSaison(merged_s1dm, 2016, mode)
                                    bbox_diff, bbox_perfect, bbox_reprocessed = calculatePerfectSaison(merged_bbox, 2016, mode)

                                    # Compute the mean differences over spatial dimensions and interpolate missing values
                                    ids_mean_diff = ids_diff.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')
                                    s1dm_mean_diff = s1dm_diff.mean(dim=['x', 'y']).interpolate_na(dim='time', method='linear')

                                    # Create output file paths based on method
                                    output_path_ids = os.path.join(save_data_dir, f"ids_event_{idx_d}_{dca_key}_{year_key}_{mode}_diff.nc")
                                    output_path_s1dm = os.path.join(save_data_dir, f"s1dm_event_{idx_d}_{dca_key}_{year_key}_{mode}_diff.nc")

                                    # Save NetCDF files
                                    logger.info(f"      📂 Saving IDS differences to: {output_path_ids}")
                                    ids_mean_diff.to_netcdf(output_path_ids)

                                    logger.info(f"      📂 Saving S1DM differences to: {output_path_s1dm}")
                                    s1dm_mean_diff.to_netcdf(output_path_s1dm)

                                    logger.info(f"   ✅ Successfully calculated and saved outputs for method: {mode}")

                                    # Plotting loop for each variable
                                    for variable in variables:
                                        logger.info(f"   🔄 Processing variable: {variable}")
                                        result = get_max_diff_timestamp_index(ids_mean_diff, s1dm_mean_diff, bbox_diff, year_key, ref_years_filter=1, variable=variable)
                                        time_index = result['time_idx']
                                        logger.info(f"    📅 Index in {ref_year} for the largest {variable.upper()} difference: {time_index}")

                                        # Create output file paths based on method
                                        output_path_fig = os.path.join(save_fig_dir, f"timeseries_spatial_event_{idx_d}_{dca_key}_{ref_year}_{year_key}_{mode}_diff_{variable}.png")
                                        
                                        plot_spatial_and_timeseries(
                                            bbox_difference=bbox_diff,
                                            ids=ids,
                                            s1dm=s1dm,
                                            ids_mean_difference=ids_mean_diff,
                                            s1dm_mean_difference=s1dm_mean_diff,
                                            ref_year=year_key,
                                            variable=variable,
                                            dca=dca_key,
                                            idx_d=idx_d,
                                            time_index=time_index,
                                            save_path=output_path_fig
                                        )
                                        logger.info(f"    📂 📊 Saved plot for variable: {variable} at: {output_path_fig}")

                                except Exception as e:
                                    logger.error(f"   ❌ Error calculating season for method {mode}: {e}")
                                    continue

                        except Exception as e:
                            logger.error(f"❌ Critical error during processing of IDX_D {idx_d}: {e}")
                            continue


        print("Processing complete for all combinations.")


    def _