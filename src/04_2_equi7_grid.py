# Import the Equi7Grid class from the equi7grid_lite module
from equi7grid_lite import Equi7Grid

# Create an instance of Equi7Grid with the specified minimum grid size
grid_system = Equi7Grid(min_grid_size=2560)

# Print out details about the grid system
print("Equi7Grid instance created with the following settings:")
print(f"Minimum Grid Size: {grid_system.min_grid_size} meters")
print("Levels: 0, 1, ..., 7, 8")
print("Zones: AN, NA, OC, SA, AF, EU, AS")
print(f"Max Grid Size: {grid_system.max_grid_size} meters")

# Example usage (optional)
# You can uncomment the following lines to see how the grid system can be used
# Example coordinate (latitude, longitude)
# lat, lon = 48.8588443, 2.2943506
# tile = grid_system.lonlat2tile(lon, lat)
# print(f"Tile for coordinates ({lat}, {lon}): {tile}")
