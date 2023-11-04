# csv_combiner.py

import os
import pandas as pd
import fire
from typing import List

class CSVCombiner:
    def combine_csv_files(self, input_directory: str, output_filename: str) -> None:
        """
        Combine CSV files from the input directory into a single CSV file.

        Args:
            input_directory (str): The directory containing CSV files to combine.
            output_filename (str): The name of the output combined CSV file.

        Returns:
            None
        """
        # Check if the input directory exists
        if not os.path.exists(input_directory):
            print(f"Input directory '{input_directory}' does not exist.")
            return

        # Initialize an empty DataFrame to store the combined data
        combined_data = pd.DataFrame()

        # Loop through CSV files in the input directory and concatenate them
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    # Specify column names when reading the CSV file
                    df = pd.read_csv(file_path, names=["slices", "formula", "crystal"])
                    combined_data = pd.concat([combined_data, df], ignore_index=True)

        # Save the combined data to the output CSV file
        combined_data.to_csv(output_filename, index=False)
        print(f"Combined CSV files saved to '{output_filename}'.")

if __name__ == "__main__":
    fire.Fire(CSVCombiner)
