import json
import os
from matbench.bench import MatbenchBenchmark
import numpy as np

# Define the available benchmarks
available_benchmarks = [
    "matbench_mp_is_metal",
]

def convert_structure_to_serializable(pymatgen_structure):
    """
    Convert a pymatgen Structure object to a serializable format (CIF).
    """
    return pymatgen_structure.to(fmt="cif")

def convert_label_to_serializable(label):
    """
    Convert labels to 0 or 1, specifically converting numpy booleans to Python integers.
    """
    return int(label)

def download_benchmark_data(benchmark_name, save_path):
    """
    Download and save the Matbench benchmark data as JSON files.

    Args:
        benchmark_name (str): The name of the benchmark to download.
        save_path (str): The directory path where the JSON files will be saved.
    """
    if benchmark_name not in available_benchmarks:
        raise ValueError(
            f"Invalid benchmark name. Available benchmarks: {', '.join(available_benchmarks)}"
        )

    # Load the MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    
    # Create the save directory if it does not exist
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print(f"Directory '{save_path}' already exists.")

    # Load the benchmark data
    benchmark = getattr(mb, benchmark_name)
    benchmark.load()

    # Process each fold in the benchmark
    for fold in benchmark.folds:
        # Get train inputs and outputs
        train_inputs, train_outputs = benchmark.get_train_and_val_data(fold)
        test_inputs = benchmark.get_test_data(fold)

        # Create the train data
        train_data = [
            {
                "mbid": index,  # Add material ID (index)
                "structure": convert_structure_to_serializable(train_inputs[index]),
                "labels": convert_label_to_serializable(train_outputs[index]),  # Convert bool to 0 or 1
            }
            for index in train_inputs.index
        ]

        # Save the train data as a JSON file
        train_dataset_name = f"train_{benchmark_name}_{fold}.json"
        with open(os.path.join(save_path, train_dataset_name), "w") as train_file:
            json.dump(train_data, train_file)

        print(f"Train data saved to {save_path}/{train_dataset_name}")

        # Create the test data
        test_data = [
            {
                "mbid": index,  # Add material ID (index)
                "structure": convert_structure_to_serializable(test_inputs[index])
            }
            for index in test_inputs.index
        ]

        # Save the test data as a JSON file
        test_dataset_name = f"test_{benchmark_name}_{fold}.json"
        with open(os.path.join(save_path, test_dataset_name), "w") as test_file:
            json.dump(test_data, test_file)

        print(f"Test data saved to {save_path}/{test_dataset_name}")

if __name__ == "__main__":
    # Define the benchmark name and the directory to save the data
    benchmark_name = "matbench_mp_is_metal"
    save_path = "./benchmark_data_is_metal"

    # Download and save the benchmark data
    download_benchmark_data(benchmark_name, save_path)
