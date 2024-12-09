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

    def __init__(self, usa_filepath, resolution, pixel_size, env_path, output_paths):

        self._set_up_logging()
        self._load_env_variables(env_path)
        self.usa_filepath = usa_filepath
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.output_paths = output_paths

    def _set_up_logging(self):
        """Set up logging to file with timestamp."""
        logging.basicConfig(filename='gridCreation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def _load_env_variables(self, env_path):
        """Load environment variables from a .env file."""
        if not env_path.exists():
            raise FileNotFoundError(f"The .env file does not exist at {env_path}")
        load_dotenv(dotenv_path=env_path)
        
        self.region = os.getenv('REGION')
        if self.region is None:
            raise ValueError("The 'REGION' environment variable is not set.")
        self.region_id = str(self.region).zfill(2)
        self.equi7_crs = os.getenv('EQUI7_NA_EPSG')
        

    def create_grid(self):
        """Main public method to create the Equi7 grid, perform intersection and save results."""
        region_shape = self._get_region_shape()

        # Step 1: Generate the Equi7 grid
        grid = self._generate_equi7_grid()

        # Step 2: Create convex hulls
        convex_hulls = self._create_convex_hulls()

        # Step 3: Reproject convex hulls
        reprojected_convex_hulls = convex_hulls.to_crs(self.equi7_crs)

        # Step 4: Intersect grid and convex hulls
        intersected = self._intersect_grid(reprojected_convex_hulls, grid, region_shape)

        # Step 5: Add minicube index to REFDM
        refdm_gdf = gpd.read_file(self.output_paths["refdm_path"])
        self._add_minicube_index(intersected, refdm_gdf)

        # Step 6: Plot batches of intersections
        self._plot_intersection_batches(intersected, region_shape)


    def _generate_equi7_grid(self):
        """Private method to generate the Equi7 grid."""
        size = self.resolution * self.pixel_size
        grid_system = Equi7Grid(min_grid_size=size)

        region = load_and_extract_region(self.region_id, self.output_paths["grid_output_path"])

        grid = grid_system.create_grid(level=0, zone="NA", mask=region)
        grid.to_file(self.output_paths["grid_output_path"])
        grid.boundary.plot()
        plt.savefig(self.output_paths["grid_figure_output_path"])
        return grid

    def _create_convex_hulls(self):
        """Private method to create convex hulls from REFDM and USDA polygons."""
        refdm_gdf = gpd.read_file(self.output_paths["refdm_path"])
        ids_gdf = gpd.read_file(self.output_paths["ids_path"])

        dissolved_refdm = refdm_gdf[['IDX_D', 'geometry']].dissolve(by='IDX_D').reset_index()
        merged_gdf = gpd.sjoin(dissolved_refdm, ids_gdf, how='left', predicate='intersects')
        merged_geometries = merged_gdf.groupby('IDX_D')['geometry'].apply(lambda x: x.unary_union)
        convex_hulls = merged_geometries.apply(lambda geom: MultiPolygon([geom.convex_hull]))

        convex_hulls_gdf = gpd.GeoDataFrame(geometry=convex_hulls, crs=refdm_gdf.crs).reset_index()
        convex_hulls_gdf.to_file(self.output_paths["convex_hulls_output_path"])
        return convex_hulls_gdf

    def _intersect_grid(self, convex_hulls_gdf, grid_gdf, region_shape):
        """Private method to intersect convex hulls with the grid."""
        intersected_gdf = gpd.sjoin(grid_gdf, convex_hulls_gdf, how='inner', predicate='intersects')
        intersected_gdf = intersected_gdf.drop(columns=['index_right', 'IDX_D', 'level', 'land', 'zone']).drop_duplicates().reset_index(drop=True)
        intersected_gdf.to_file(self.output_paths["intersection_output_path"])

        fig, ax = plt.subplots(figsize=(12, 12))
        intersected_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6, label='Intersected Grids')
        region_shape = region_shape.to_crs(self.equi7_crs)
        region_shape.boundary.plot(ax=ax, color='red', linewidth=0.5, linestyle='--', label=f'Region {self.region_id} Boundary')
        ax.legend()
        plt.savefig(self.output_paths["intersection_figure_output_path"], dpi=300)
        plt.show()
        return intersected_gdf

    def _add_minicube_index(self, intersected_grid, refdm_gdf):
        """Private method to add minicube index to REFDM data."""
        reprojected_refdm = refdm_gdf.to_crs(self.equi7_crs)
        intersected_grid = intersected_grid.to_crs(reprojected_refdm.crs)

        reprojected_refdm['minicube_index'] = reprojected_refdm['geometry'].apply(lambda geom: intersected_grid[intersected_grid.intersects(geom)].index.tolist())
        reprojected_refdm['cube_amount'] = reprojected_refdm['minicube_index'].apply(len)
        reprojected_refdm.to_file(self.output_paths["output_path_refdm"])

    def _plot_intersection_batches(self, intersection, region_shape):
        """Private method to plot intersected grid in batches of 100 cells."""
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
