import hydra
from omegaconf import DictConfig

import pandas as pd
import os
from invcryrep.invcryrep import InvCryRep

from matbench.bench import MatbenchBenchmark


# Check if the specified benchmark exists
available_benchmarks = [
    'matbench_dielectric',
    'matbench_expt_gap',
    'matbench_expt_is_metal',
    'matbench_glass',
    'matbench_jdft2d',
    'matbench_log_gvrh',
    'matbench_log_kvrh',
    'matbench_mp_e_form',
    'matbench_mp_gap',
    'matbench_mp_is_metal',
    'matbench_perovskites',
    'matbench_phonons',
    'matbench_steels'
]


# Define the function to convert a structure to slices
def give_slice(structure):
    backend = InvCryRep(check_results=True)
    return backend.structure2SLICES(structure)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    benchmarks = cfg.matbench.benchmarks.dataset
    path = cfg.matbench.path.save_path
    print(path)
    os.mkdir(path)
    for benchmark_name in benchmarks:
        if benchmark_name not in available_benchmarks:
            raise ValueError(f"Invalid benchmark name. Available benchmarks: {', '.join(available_benchmarks)}")
        
    for benchmark_name in benchmarks:

        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, benchmark_name)
        benchmark.load()

        for fold in benchmark.folds:

            # Get train inputs and outputs
            train_inputs, train_outputs = benchmark.get_train_and_val_data(fold)
            test_inputs = benchmark.get_test_data(fold)

            # Create the train data
            train_data = [(give_slice(train_inputs[index]), train_outputs[index]) for index in range(len(train_inputs))]
            

            # Create a DataFrame from the train data
            train_df = pd.DataFrame(train_data)
            train_df.columns = ["slices", "labels"]

            # Save the DataFrame as a CSV file
            train_dataset_name = f"train_{benchmark_name}_{fold}"
            train_df.to_csv(f"{path}/{train_dataset_name}", index=False)

            print(f"Train data saved to {path}/{train_dataset_name}")


            test_data = [give_slice(test_inputs[index]) for index in range(len(test_inputs))]

            test_df = pd.DataFrame(test_data)
            test_df.columns = ["slices"]

            # Save the DataFrame as a CSV file
            test_dataset_name = f"test_{benchmark_name}_{fold}"
            test_df.to_csv(f"{path}/{test_dataset_name}", index=False)

            print(f"Test data saved to {path}/{test_dataset_name}")
          


if __name__ == "__main__":
    main()