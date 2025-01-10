# main.py
from pathlib import Path
from ids_processor import IDSProcessor
from tcc_processor import TCCProcessor
from s1cd_processor import S1CDProcessor

def main():
    """
    Main function to initialize the IDSProcessor, load data, process it, 
    filter and analyze it, and then save and plot the results.
    """
    # Specify the path to the environment file
    env_path = Path('/home/sc.uni-leipzig.de/sy58xupo/ForExD-WP1-P1/environment/.env')
    
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


    # # Create an instance of TCCProcessor and run the processing
    # tcc_processor = TCCProcessor(env_path)
    # tcc_processor.process()

    s1cd_processor = S1CDProcessor(env_path, buffer_years=2)  # Example: region 8
    s1cd_processor.process_files()
    # Save the metadata table after processing all files
    s1cd_processor.save_metadata_table("/net/projects/forexd/WP1/02_ImprovedLabels/Scripts/ForExD-WP1-P1/results_clean/metadata_table.csv")


if __name__ == "__main__":
    main()
