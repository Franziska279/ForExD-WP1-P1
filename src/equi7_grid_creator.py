import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import xarray as xr
from equi7grid_lite import Equi7Grid
from tqdm import tqdm  # For progress bars
from dotenv import load_dotenv
from func_helper import load_and_extract_region


class Equi7GridCreator:
    def __init__(self, resolution, pixel_size, buffer, env_path):
        """Initialize the Equi7GridCreator with parameters and environment variables."""
        self._setup_logging()
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.buffer = buffer
        self._load_env_variables(env_path)

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename='log_equi7_grid_creation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Logging initialized.")

    def _load_env_variables(self, env_path):
        """Load environment variables from the .env file."""
        logging.info("Loading environment variables.")
        if not env_path.exists():
            logging.error(f"The .env file does not exist at {env_path}")
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)

        self.region_id = str(os.getenv('REGION', '01')).zfill(2)
        self.target_crs = os.getenv('TARGET_CRS')
        self.equi7_crs = os.getenv('EQUI7_NA_EPSG')
        self.region_shapefile = os.path.join(os.getenv('REGION_SHAPE_DIR'), os.getenv('REGION_SHAPE_FILE'))
        
        base_dir = os.getenv('EQUI7_GRIDS_DIR')
        self.grid_filepath = os.path.join(base_dir, os.getenv('EQUI7_GRID').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))
        self.convex_hull_filepath = os.path.join(base_dir, os.getenv('CONVEX_HULL').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))
        self.intersected_grid_filepath = os.path.join(base_dir, os.getenv('INTERSECTED_GRIDS').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))
        
        results_dir = os.getenv('RESULTS_DIR')
        self.filtered_ids_filepath = os.path.join(results_dir, os.getenv('IDS_FILTERED_FILE').format(region_id=self.region_id))
        self.s1dm_filepath = os.path.join(results_dir, os.getenv('S1DM_SHAPE_FILE').format(region_id=self.region_id, buffer=self.buffer))
        
        figures_dir = os.getenv('FIGURES_DIR')
        self.grid_figure_filepath = os.path.join(figures_dir, os.getenv('FIGURE_EQUI7').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))
        self.intersected_grid_figure_filepath = os.path.join(figures_dir, os.getenv('FIGURE_EQUI7_INTERSECTED').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))
        self.batch_grid_figure_filepath = os.path.join(figures_dir, os.getenv('FIGURE_EQUI7_INTERSECTED_BATCH').format(resolution=self.resolution, pixel_size=self.pixel_size, region_id=self.region_id, buffer=self.buffer))

        # Ensure required directories exist
        for path in [
            base_dir, figures_dir
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        logging.info("Environment variables loaded successfully.")

    def create_grid(self):
        """Main method to create and process the Equi7 grid."""
        logging.info("Starting grid creation process.")
        try:
            region_shape = self._get_region_shape()
            grid = self._generate_equi7_grid()
            convex_hulls = self._create_convex_hulls()
            convex_hulls = convex_hulls.to_crs(self.equi7_crs)
            intersected_grid = self._intersect_grid(convex_hulls, grid, region_shape)
            self._plot_intersection_batches(intersected_grid, region_shape)
            return intersected_grid
        except Exception as e:
            logging.error(f"Grid creation process failed: {e}")
            raise
        logging.info("Grid creation process completed successfully.")


    def _get_region_shape(self):
        """Extract the region shape from the shapefile."""
        logging.info("Extracting region shape.")
        try:
            usa = gpd.read_file(self.region_shapefile)
            region = usa[usa.REGION == self.region_id].explode(index_parts=False).iloc[0:1]
            logging.info("Region shape extracted successfully.")
            return region
        except Exception as e:
            logging.error(f"Failed to extract region shape: {e}")
            raise

    def _generate_equi7_grid(self):
        """Generate the Equi7 grid."""
        logging.info("Generating Equi7 grid.")
        try:
            grid_system = Equi7Grid(min_grid_size=self.resolution * self.pixel_size)
            region = load_and_extract_region(self.region_shapefile, self.region_id)
            grid = grid_system.create_grid(level=0, zone="NA", mask=region)
            grid.to_file(self.grid_filepath)

            # Polt grid
            grid.to_file(self.grid_filepath)
            grid.boundary.plot()
            plt.savefig(self.grid_figure_filepath)

            logging.info("Equi7 grid generated successfully.")
            return grid
        except Exception as e:
            logging.error(f"Failed to generate Equi7 grid: {e}")
            raise


    def _create_convex_hulls(self):
        """Generate convex hulls from spatial data."""
        logging.info("Creating convex hulls.")
        try:
            s1dm_gdf = gpd.read_file(self.s1dm_filepath)
            ids_gdf = gpd.read_file(self.filtered_ids_filepath)
            dissolved_s1dm = s1dm_gdf[['IDX_D', 'geometry']].dissolve(by='IDX_D').reset_index()
            merged_gdf = gpd.sjoin(dissolved_s1dm, ids_gdf, how='left', predicate='intersects')
            merged_geometries = merged_gdf.groupby('IDX_D_left')['geometry'].apply(lambda x: x.unary_union)
            convex_hulls = merged_geometries.apply(lambda geom: MultiPolygon([geom.convex_hull]))
            convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=s1dm_gdf.crs).reset_index()
            convex_hulls_gdf.to_file(self.convex_hull_filepath)
            logging.info("Convex hulls created successfully.")
            return convex_hulls_gdf
        except Exception as e:
            logging.error(f"Failed to create convex hulls: {e}")
            raise

    def _intersect_grid(self, convex_hulls, grid, region_shape):
        """Intersect the grid with convex hulls."""
        logging.info("Performing grid intersection.")
        try:
            intersected = gpd.sjoin(grid, convex_hulls, how='inner', predicate='intersects').drop(columns=['index_right', 'IDX_D_left', 'level', 'land', 'zone']).drop_duplicates().reset_index(drop=True)
            intersected.to_file(self.intersected_grid_filepath)
            logging.info("Grid intersection completed successfully.")

            fig, ax = plt.subplots(figsize=(12, 12))
            intersected.boundary.plot(ax=ax, color='black', linewidth=0.6, label='Intersected Grids')
            region_shape = region_shape.to_crs(self.equi7_crs)
            region_shape.boundary.plot(ax=ax, color='red', linewidth=0.5, linestyle='--', label=f'Region {self.region_id} Boundary')
            ax.legend()
            plt.savefig(self.intersected_grid_figure_filepath, dpi=300)
            plt.show()
            logging.info("Grid intersection completed successfully.")

            return intersected
        except Exception as e:
            logging.error(f"Grid intersection failed: {e}")
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

        plt.savefig(self.batch_grid_figure_filepath, dpi=300)
        plt.show()


    # def _add_minicube_index_s1dm(self, intersected_grid, s1dm_gdf):
    #     logging.info("Adding minicube index to s1dm data.")
    #     try:
    #         reprojected_s1dm = s1dm_gdf.to_crs(self.equi7_crs)
    #         intersected_grid = intersected_grid.to_crs(reprojected_s1dm.crs)
    #         reprojected_s1dm['minicube_index'] = reprojected_s1dm['geometry'].apply(lambda geom: intersected_grid[intersected_grid.intersects(geom)].index.tolist())
    #         reprojected_s1dm['cube_amount'] = reprojected_s1dm['minicube_index'].apply(len)
    #         reprojected_s1dm.to_file(self.output_paths["output_path_s1dm"])
    #         logging.info("Minicube index added and saved successfully for s1dm.")
    #     except Exception as e:
    #         logging.error(f"Error while adding minicube index to s1dm: {e}")
    #         raise

    # def _add_minicube_index_ids(self, intersected_grid, ids_gdf):
    #     """Private method to add minicube index to s1dm data."""
    #     logging.info("Adding minicube index to ids data.")
    #     try:
    #         reprojected_ids = ids_gdf.to_crs(self.equi7_crs)
    #         intersected_grid = intersected_grid.to_crs(reprojected_ids.crs)

    #         reprojected_ids['minicube_index'] = reprojected_ids['geometry'].apply(lambda geom: intersected_grid[intersected_grid.intersects(geom)].index.tolist())
    #         reprojected_ids['cube_amount'] = reprojected_ids['minicube_index'].apply(len)
    #         reprojected_ids.to_file(self.output_paths["output_path_ids"])
    #     except Exception as e:
    #         logging.error(f"Error while adding minicube index to ids: {e}")
    #         raise