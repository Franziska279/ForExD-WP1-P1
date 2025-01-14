# main.py
from pathlib import Path
from ids_processor import IDSProcessor
from tcc_processor import TCCProcessor
from s1cd_processor import S1CDProcessor
from equi7_grid_creator import Equi7GridCreator
import argparse

def main(env_path, metadata_output):
    """
    Main function to initialize the IDSProcessor, load data, process it, 
    filter and analyze it, and then save and plot the results.
    """
    # Specify the path to the environment file
    #env_path = Path('/work/sy58xupo-CleaningSpace/ForExD-WP1-P1/environment/.env')
    
    # # Initialize the IDSProcessor with the environment path
    # processor = IDSProcessor(env_path)
    
    # # Load data
    # processor.load_data()

    # # Process and clean data, including handling overlaps
    # processor.exclude_include_overlapping_entries()
    
    # # Filter data based on defined criteria
    # processor.filter_data()
    
    # # Print a status summary of the processed data
    # processor.print_status()
    
    # # Save and plot the final data results
    # processor.save_and_plot()


    # Create an instance of TCCProcessor and run the processing
    # tcc_processor = TCCProcessor(env_path)
    # tcc_processor.process()

    # s1cd_processor = S1CDProcessor(env_path, buffer_years=2, max_jobs=4)  # Example: region 8
    # s1cd_processor.process_files()

    grid_creator = Equi7GridCreator(resolution = 10, pixel_size = 512, env_path=env_path)
    grid_creator.create_grid()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sentinel-1 Data Processor")
    parser.add_argument("--env", required=True, help="Path to the .env file")
    parser.add_argument("--metadata-output", required=True, help="Path to save metadata table (CSV)")
    args = parser.parse_args()
    
    main(env_path=Path(args.env), metadata_output=args.metadata_output)
