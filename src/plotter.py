# plotter.py
from func_plots import (
    run_manual_validation,
    plot_overlap_omission,
    plot_study_area,
    plot_radar_reduction_potential,
    plot_size_position_comparison,
    plot_detection_year_lag,
    plot_spatial_overlap_histograms,
    plot_annual_event_counts,
    plot_manual_disturbance_examples,
)
from func_helper import (
    parse_color_map,
    format_label,
    add_convex_hull_area,
    aggregate_detections_by_event,
    drop_disturbance_types,
    dissolve_to_event_level,
)
from func_file_io import load_data
from func_preprocessing import calculate_size_shift_difference
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np


class Plotter:
    def __init__(self, env_path, spatial_buffer):
        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        self.region = os.getenv('REGION')
        if not self.region:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)
        self.target_crs = os.getenv('TARGET_CRS')
        self.custom_colors = parse_color_map(os.getenv('COLORS'))

        self.usa_filepath = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        self.ids_path = os.path.join(os.getenv('RESULTS_DIR'), os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        crs = os.getenv('TCC_CRS')
        crs_number = crs.split(":")[-1] if crs else None
        self.tcc_downsampled = os.path.join(
            os.getenv('TCC_DIR'),
            os.getenv('TCC_DOWNSAMPLED_RASTER_TEMPLATE').format(
                region_id=self.region_id, crs=crs_number, tcc_year=2017
            )
        )
        self.s1_tiles_boundary_path = os.path.join(
            os.getenv('RESULTS_DIR'), os.getenv('S1CD_TILES_BOUNDS_FILE').format(region_id=self.region_id)
        )

        self._figures_dir = os.getenv('FIGURES_DIR')
        self._results_dir = os.getenv('RESULTS_DIR')
        self._manual_dir = os.getenv('MANUAL_DIR')
        self._figure_significance_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_SIGNIFICANCE'))
        self._figure_manual_path = os.path.join(os.getenv('FIGURES_DIR'), os.getenv('FIGURE_MANUAL'))
        self._figure_radar_reduction_tmpl = os.getenv('FIGURE_RADAR_REDUCTION')
        self._figure_year_lag_tmpl = os.getenv('FIGURE_YEAR_LAG')
        self._figure_size_change_tmpl = os.getenv('FIGURE_SIZE_CHANGE')
        self._figure_overlap_tmpl = os.getenv('FIGURE_OVERLAP')
        self._figure_potential_dca_year_tmpl = os.getenv('FIGURE_POTENTIAL_DCA_YEAR')
        self._s1dm_shape_tmpl = os.getenv('S1DM_SHAPE_FILE')
        self._figure_overlap_omission_path = os.path.join(self._figures_dir, f"p1_f11_overlap_omission_{self.region_id}.png")
        self._summary_overlap_omission_path = os.path.join(self._results_dir, f"region_{self.region_id}_overlap_omission_summary.csv")
        self._stats_size_shift_tmpl = os.path.join(self._results_dir, f"region_{self.region_id}_size_shift_stats_buffer_{{buffer}}.csv")

        self.spatial_buffer = spatial_buffer

        os.makedirs(self._figures_dir, exist_ok=True)
        os.makedirs(self._results_dir, exist_ok=True)

    def plot(self):
        """
        Plots the processed data.

        :param data: The processed data to be plotted
        """

        # Get figure file paths from environment variables with dynamic buffer and region ID replacement
        figure_study_area_path = os.path.join(self._figures_dir, os.getenv('FIGURE_STUDY_AREA').format(region_id=self.region_id))
        ids_gdf = load_data(self.ids_path)
        ids_clean = drop_disturbance_types(ids_gdf, ["drought","fire"])
        plot_study_area(
                self.usa_filepath,
                self.region_id,
                self.tcc_downsampled,
                self.s1_tiles_boundary_path,
                ids_clean,
                self.custom_colors,
                figure_study_area_path,
                logging)

        for buffer in self.spatial_buffer:
            logging.info(f"Processing for buffer: {buffer}")

            figure_radar_reduction_potential_path = os.path.join(self._figures_dir, self._figure_radar_reduction_tmpl.format(buffer=buffer))
            figure_year_lag_path = os.path.join(self._figures_dir, self._figure_year_lag_tmpl.format(buffer=buffer))
            figure_size_position_change_path = os.path.join(self._figures_dir, self._figure_size_change_tmpl.format(buffer=buffer))
            figure_overlap_percentage_path = os.path.join(self._figures_dir, self._figure_overlap_tmpl.format(buffer=buffer))
            figure_radar_reduction_potential_year_dca_path = os.path.join(self._figures_dir, self._figure_potential_dca_year_tmpl.format(buffer=buffer))
            figure_manual_significance_path = self._figure_significance_path
            figure_manual_examples_path = self._figure_manual_path

            s1dm_path = os.path.join(self._results_dir, self._s1dm_shape_tmpl.format(region_id=self.region_id, buffer=buffer))
            manual_base_folder = self._manual_dir

            ids_gdf = load_data(self.ids_path)
            s1dm_gdf = load_data(s1dm_path)

            logging.info("Plot overlap/omission summary")
            figure_overlap_omission_path = self._figure_overlap_omission_path.replace(".png", f"_buffer_{buffer}.png")
            summary_overlap_omission_path = self._summary_overlap_omission_path.replace(".csv", f"_buffer_{buffer}.csv")
            plot_overlap_omission(ids_gdf, s1dm_gdf, figure_overlap_omission_path, summary_overlap_omission_path)

            s1dm_gdf_yearly_aggregated = dissolve_to_event_level(s1dm_gdf)
            s1dm_frequency = aggregate_detections_by_event(s1dm_gdf_yearly_aggregated, self.target_crs)
            #s1dm_cleaned = drop_drought(s1dm_frequency)
            s1dm_cleaned = drop_disturbance_types(s1dm_frequency, ["drought","fire"])
            s1dm_cleaned = s1dm_cleaned[s1dm_cleaned["DCA_ID"] != "fire"]
            logging.info('Claculate size shift difference')
            gdf = calculate_size_shift_difference(ids_gdf, s1dm_cleaned)
            # Minimum outerline areas
            s1dm_convex = add_convex_hull_area(s1dm_cleaned)[['geometry', 'area_km2', 'DCA_ID']]
            ids_convex = add_convex_hull_area(ids_gdf)[['geometry', 'area_km2', 'DCA_ID']]

            # Remove drought disturbances
            s1dm_no_drought = drop_disturbance_types(s1dm_cleaned, ["drought","fire"]) #drop_drought(s1dm_cleaned)
            s1dm_no_drought_gdf = drop_disturbance_types(s1dm_gdf, ["drought","fire"]) #drop_drought(s1dm_gdf)


            # Plot radar reduction potential
            logging.info('Plot radar reduction potential')
            plot_radar_reduction_potential(
                s1dm_frequency,
                ids_gdf,
                save_path=figure_radar_reduction_potential_path,
                plot_reduction=True
            )

            logging.debug("gdf geometry valid: %s", gdf.geometry.is_valid.value_counts().to_dict())
            logging.debug("gdf geometry empty: %s", gdf.geometry.is_empty.value_counts().to_dict())
            logging.debug("gdf area_km2 NaN: %d  centroid_shift_m NaN: %d",
                          gdf['area_km2'].isna().sum(), gdf['centroid_shift_m'].isna().sum())
            plot_size_position_comparison(
                gdf,
                ids_gdf,
                s1dm_convex,
                ids_convex,
                self.custom_colors,
                save_path=figure_size_position_change_path,
                stats_path=self._stats_size_shift_tmpl.format(buffer=buffer),
            )

            # Plot signal counts
            logging.info('Plot signal counts')
            plot_detection_year_lag(
                #s1dm_no_drought_gdf,
                s1dm_no_drought,
                self.custom_colors,
                save_path=figure_year_lag_path
            )

            #Calculate and plot overlap percentages
            logging.info('Calculate and plot overlap percentages')
            plot_spatial_overlap_histograms(
                ids_gdf,
                s1dm_no_drought_gdf,
                self.custom_colors,
                figure_overlap_percentage_path
            )

            logging.info('Calculate and plot DCA, Year Potential')
            plot_annual_event_counts(s1dm_path, self.ids_path,
                                    exclude_types=['fire', 'drought'],
                                    ordered_types=['wind', 'bark_beetle', 'defoliators'],
                                    custom_colors=self.custom_colors,
                                    output_file=figure_radar_reduction_potential_year_dca_path)

            # Calculate the significance , manual Plot
            logging.info('Calculate the significance')
            run_manual_validation(self.ids_path,
                                                 s1dm_path,
                                                 manual_base_folder,
                                                 self._results_dir,
                                                 figure_manual_significance_path)
        
            # disturbance_files = {
            #                     "wind": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/Wind_0_larger_RGB_psscene_visual/composite_file_format.tif",
            #                         "idx": "wind_2018_1764",
            #                         "date": "2018-04-16"
            #                     },
            #                     "defoliators": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/Defoliators_17_psscene_visual/composite_file_format.tif",
            #                         "idx": "defoliators_2019_2009",
            #                         "date": "2021-05-14"
            #                     },
            #                     "bark_beetle": {
            #                         "file": "/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/data/Planet/BarkBeetle_10_psscene_visual/composite_file_format.tif",
            #                         "idx": "bark_beetle_2018_1136",
            #                         "date": "2018-04-28"
            #                     }
            #                 }

            # logging.info('Calculate Manual Examples')

            # plot_manual_disturbance_examples(
            #     disturbance_files,
            #     manual_base_folder,
            #     ids_gdf,
            #     s1dm_gdf,
            #     figure_path=figure_manual_examples_path)
