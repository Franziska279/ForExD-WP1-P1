import geopandas as gpd
from tqdm import tqdm



# Define a function to process each hull row
def process_hull(row, minicubes_grid):
    intersecting_usda_idx = None
    max_intersection_area = 0.0
    is_perfect_coverage = False
    intersect_count = 0
    
    # Iterate over minicubes_grid and check intersection
    for mc_idx, mc_row in minicubes_grid.iterrows():
        if row['geometry'].intersects(mc_row['geometry']):
            intersection = row['geometry'].intersection(mc_row['geometry'])
            intersection_area = intersection.area
            hull_area = row['geometry'].area
            intersect_count += 1
            
            # Check if intersection area equals hull area (perfect coverage)
            if abs(intersection_area - hull_area) < 1e-6:
                is_perfect_coverage = True
            
            # Update if larger intersection area found
            if intersection_area > max_intersection_area:
                max_intersection_area = intersection_area
                intersecting_usda_idx = mc_row['USDA_IDX']
    
    return intersecting_usda_idx, max_intersection_area, is_perfect_coverage, intersect_count


def main():
    hulls =gpd.read_file("/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/convex_hulls_refdm.shp")
    minicubes_shape_path="/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/s2_minicube_bounderies_all.shp"
    minicubes_grid = gpd.read_file(minicubes_shape_path)
    # Create new columns in hulls to store intersecting USDA_IDX, intersection area, and check for perfect coverage
    hulls['usda_idx'] = None
    hulls['max_intersection_area'] = 0.0
    hulls['is_perfect_coverage'] = False
    hulls['intersection_count'] = 0

    # Apply the process_hull function to each row in hulls with tqdm progress bar
    tqdm.pandas(desc="Processing hulls")
    hulls[['usda_idx', 'max_intersection_area', 'is_perfect_coverage', 'intersection_count']] = hulls.progress_apply(
        lambda row: process_hull(row, minicubes_grid),
        axis=1, result_type='expand'
    )

    # Count occurrences of perfect coverage
    perfect_coverage_count = hulls['is_perfect_coverage'].sum()

    # Count occurrences of each intersection count
    intersection_counts = hulls['intersection_count'].value_counts().sort_index()

    # Print the counts
    print("Intersection counts:")
    for count, num_rows in intersection_counts.items():
        print(f"{num_rows} hulls have {count} intersections.")

    # Print the count of perfect coverage
    print(f"Perfect coverage occurs {perfect_coverage_count}/{len(hulls)} times.")

    # Save hulls GeoDataFrame to file
    output_file = "/Net/Groups/BGI/scratch/fmueller/ForExD-WP1-P1/results/hulls_hull_minicube_indices_2.shp"  # Update with your desired file path and format
    
    hulls.to_file(output_file)


if __name__ == "__main__":
    main()