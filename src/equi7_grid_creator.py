import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from pathlib import Path
import xarray as xr
from equi7grid_lite import Equi7Grid
from tqdm import tqdm  # For progress bars
from dotenv import load_dotenv
from func_helper import load_and_extract_region




class Equi7GridCreator:

    def __init__(self, resolution, pixel_size, env_path):

        self._set_up_logging()
        self.resolution = resolution
        self.pixel_size = pixel_size
        self._load_env_variables(env_path)
        

    def _set_up_logging(self):
        """Set up logging to file with timestamp."""
        logging.basicConfig(
            filename='log_equi7_gridCreation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Logging initialized.")

    def _load_env_variables(self, env_path):
        """Load environment variables from a .env file."""
        logging.info("Loading environment variables.")
        if not env_path.exists():
            logging.error(f"The .env file does not exist at {env_path}")
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        self.region = os.getenv('REGION')
        if self.region is None:
            logging.error("The 'REGION' environment variable is not set.")
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)
        self.equi7_crs = os.getenv('EQUI7_NA_EPSG')
        self.usa_filepath = f"{os.getenv('REGION_SHAPE')}/S_USA.AdministrativeRegion.shp"
        self.output_paths = {
            "grid_output_path": f"{os.getenv('EQUI7_GRIDS')}/grid_equi7_region_{self.region_id}.shp",
            "grid_figure_output_path": f"{os.getenv('FIGURES')}/grid_equi7_region_{self.region_id}.png",
            "convex_hulls_output_path": f"{os.getenv('EQUI7_GRIDS')}/convex_hulls_region_{self.region_id}.shp",
            "intersection_output_path": f"{os.getenv('EQUI7_GRIDS')}/intersected_grid_{self.resolution}_{self.pixel_size}_region_{self.region_id}.shp",
            "intersection_figure_output_path": f"{os.getenv('FIGURES')}/intersection_region_{self.region_id}.png",
            "s1dm_path": f"{os.getenv('RESULTS')}/radar_enhanced_forest_disturbance_mapping_region_{self.region_id}.shp",
            "ids_path": f"{os.getenv('RESULTS')}region_{self.region_id}_dca_filtered_ids_usda_polygons.shp",
            "output_path_s1dm": f"{os.getenv('RESULTS')}radar_enhanced_forest_disturbance_mapping_region_{self.region_id}.shp",
            "output_path_ids": f"{os.getenv('RESULTS')}region_{self.region_id}_dca_filtered_ids_usda_polygons.shp",
            "batches_figure_output_path": f"{os.getenv('FIGURES')}/intersection_batches_region_{self.region_id}.png"
        }
        logging.info("Environment variables loaded successfully.")

    def create_grid(self):
        """Main public method to create the Equi7 grid, perform intersection and save results."""
        logging.info("Starting grid creation process.")
        try:
            region_shape = self._get_region_shape(self.usa_filepath, self.region_id)
            grid = self._generate_equi7_grid()
            convex_hulls = self._create_convex_hulls()
            reprojected_convex_hulls = convex_hulls.to_crs(self.equi7_crs)
            intersected = self._intersect_grid(reprojected_convex_hulls, grid, region_shape)
            s1dm_gdf = gpd.read_file(self.output_paths["s1dm_path"])
            ids_gdf = gpd.read_file(self.output_paths["ids_path"])
            self._add_minicube_index_s1dm(intersected, s1dm_gdf)
            self._add_minicube_index_ids(intersected, ids_gdf=ids_gdf)
            self._plot_intersection_batches(intersected, region_shape)
        except Exception as e:
            logging.error(f"An error occurred during grid creation: {e}")
            raise
        logging.info("Grid creation process completed successfully.")

    def _get_region_shape(self, path, region_id):
        logging.info(f"Extracting region shape for REGION={region_id}.")
        try:
            usa = gpd.read_file(path)
            country = usa[usa.REGION == region_id]
            region = country.explode()[0:1]
            logging.info("Region shape extracted successfully.")
            return region
        except Exception as e:
            logging.error(f"Error while extracting region shape: {e}")
            raise

    def _generate_equi7_grid(self):
        logging.info("Generating Equi7 grid.")
        try:
            size = self.resolution * self.pixel_size
            grid_system = Equi7Grid(min_grid_size=size)
            region = load_and_extract_region(self.usa_filepath, self.region_id)
            grid = grid_system.create_grid(level=0, zone="NA", mask=region)
            grid.to_file(self.output_paths["grid_output_path"])
            grid.boundary.plot()
            plt.savefig(self.output_paths["grid_figure_output_path"])
            logging.info("Equi7 grid generated and saved successfully.")
            return grid
        except Exception as e:
            logging.error(f"Error while generating Equi7 grid: {e}")
            raise

    def _create_convex_hulls(self):
        logging.info("Creating convex hulls.")
        try:
            s1dm_gdf = gpd.read_file(self.output_paths["s1dm_path"])
            ids_gdf = gpd.read_file(self.output_paths["ids_path"])
            dissolved_s1dm = s1dm_gdf[['IDX_D', 'geometry']].dissolve(by='IDX_D').reset_index()
            merged_gdf = gpd.sjoin(dissolved_s1dm, ids_gdf, how='left', predicate='intersects')
            merged_geometries = merged_gdf.groupby('IDX_D_left')['geometry'].apply(lambda x: x.unary_union)
            convex_hulls = merged_geometries.apply(lambda geom: MultiPolygon([geom.convex_hull]))
            convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=s1dm_gdf.crs).reset_index()
            convex_hulls_gdf.to_file(self.output_paths["convex_hulls_output_path"])
            logging.info("Convex hulls created and saved successfully.")
            return convex_hulls_gdf
        except Exception as e:
            logging.error(f"Error while creating convex hulls: {e}")
            raise

    def _intersect_grid(self, convex_hulls_gdf, grid_gdf, region_shape):
        logging.info("Intersecting grid with convex hulls.")
        try:
            intersected_gdf = gpd.sjoin(grid_gdf, convex_hulls_gdf, how='inner', predicate='intersects')
            intersected_gdf = intersected_gdf.drop(columns=['index_right', 'IDX_D_left', 'level', 'land', 'zone']).drop_duplicates().reset_index(drop=True)
            intersected_gdf.to_file(self.output_paths["intersection_output_path"])
            fig, ax = plt.subplots(figsize=(12, 12))
            intersected_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6, label='Intersected Grids')
            region_shape = region_shape.to_crs(self.equi7_crs)
            region_shape.boundary.plot(ax=ax, color='red', linewidth=0.5, linestyle='--', label=f'Region {self.region_id} Boundary')
            ax.legend()
            plt.savefig(self.output_paths["intersection_figure_output_path"], dpi=300)
            plt.show()
            logging.info("Grid intersection completed successfully.")
            return intersected_gdf
        except Exception as e:
            logging.error(f"Error during grid intersection: {e}")
            raise

    def _add_minicube_index_s1dm(self, intersected_grid, s1dm_gdf):
        logging.info("Adding minicube index to s1dm data.")
        try:
            reprojected_s1dm = s1dm_gdf.to_crs(self.equi7_crs)
            intersected_grid = intersected_grid.to_crs(reprojected_s1dm.crs)
            reprojected_s1dm['minicube_index'] = reprojected_s1dm['geometry'].apply(lambda geom: intersected_grid[intersected_grid.intersects(geom)].index.tolist())
            reprojected_s1dm['cube_amount'] = reprojected_s1dm['minicube_index'].apply(len)
            reprojected_s1dm.to_file(self.output_paths["output_path_s1dm"])
            logging.info("Minicube index added and saved successfully for s1dm.")
        except Exception as e:
            logging.error(f"Error while adding minicube index to s1dm: {e}")
            raise

    def _add_minicube_index_ids(self, intersected_grid, ids_gdf):
        """Private method to add minicube index to s1dm data."""
        logging.info("Adding minicube index to ids data.")
        try:
            reprojected_ids = ids_gdf.to_crs(self.equi7_crs)
            intersected_grid = intersected_grid.to_crs(reprojected_ids.crs)

            reprojected_ids['minicube_index'] = reprojected_ids['geometry'].apply(lambda geom: intersected_grid[intersected_grid.intersects(geom)].index.tolist())
            reprojected_ids['cube_amount'] = reprojected_ids['minicube_index'].apply(len)
            reprojected_ids.to_file(self.output_paths["output_path_ids"])
        except Exception as e:
            logging.error(f"Error while adding minicube index to ids: {e}")
            raise

    def _plot_intersection_batches(self, intersection, region_shape):
        """Private method to plot intersected grid in batches of 100 cells."""
        logging.info("plot intersected grid in batches of 100 cells.")
        fig, ax = plt.subplots(figsize=(12, 12))
        region_shape.boundary.plot(ax=ax, edgecolor='#264653', linewidth=2, linestyle='--', label='Region Boundary')
        intersection.boundary.plot(ax=ax, edgecolor='#E76F51', linewidth=1, label='Intersection Data')

        def _add_convex_hull(ax, geometries, batch_idx):
            combined_geom = geometries.geometry.unary_union.convex_hull
            if combined_geom.is_empty:
                return
            x, y = combined_geom.exterior.xy
            ax.plot(x, y, color='#2A9D8F', lw=2)
            ax.text(combined_geom.centroid.x, combined_geom.centroid.y, f'{batch_idx}', fontsize=20, color='#264653', ha='center')

        batch_size = 100
        for i in range(0, len(intersection), batch_size):
            batch_geometries = intersection.iloc[i:i + batch_size]
            _add_convex_hull(ax, batch_geometries, batch_idx=i // batch_size)

        plt.savefig(self.output_paths["batches_figure_output_path"], dpi=300)
        plt.show()
