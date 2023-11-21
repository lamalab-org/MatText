import os
import pandas as pd
import fire
from sklearn.model_selection import train_test_split

def save_slice_column( input_file: str, output_file: str) -> None:
        """
        Load an input CSV file, select and save only the "slice" column to a separate CSV file.

        Args:
            input_file (str): The path to the input CSV file.
            output_file (str): The name of the output CSV file where the "slice" column will be saved.

        Returns:
            None
        """
        # Load the input CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Select and save only the "slice" column to a new CSV file
        df.columns = df.columns.str.strip()
        slice_column = df['slices']
        slice_column.to_csv(output_file, header=["slices"], index=False)

def split_csv(input_file: str, train_output_file: str, val_output_file: str, split_ratio: float = 0.8, random_state: int = None) -> None:
        """
        Split a CSV file into training and validation sets.

        Args:
            input_file (str): Path to the input CSV file.
            train_output_file (str): Path to save the training CSV file.
            val_output_file (str): Path to save the validation CSV file.
            split_ratio (float): Ratio to split the data into training and validation sets (default: 0.8).
            random_state (int): Random seed for reproducibility (default: None).

        Returns:
            None
        """
        # Load the CSV data
        try:
            data = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"File '{input_file}' not found.")
            return

        # Split the data
        train_data, val_data = train_test_split(data, test_size=(1 - split_ratio), random_state=random_state)

        # Save the split datasets
        train_data.to_csv(train_output_file, index=False)
        val_data.to_csv(val_output_file, index=False)

        print(f"Split complete: Train data saved to '{train_output_file}', Validation data saved to '{val_output_file}'.")

def combine_csv_files(input_directory: str, output_filename: str) -> None:
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
    fire.Fire({
        'save_slice_column': save_slice_column,
        'split_csv': split_csv,
        'combine_csv_files': combine_csv_files
    })


