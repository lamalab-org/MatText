import json
import os

import hydra
from matbench.bench import MatbenchBenchmark
from omegaconf import DictConfig

# Check if the specified benchmark exists
available_benchmarks = [
    "matbench_dielectric",
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_glass",
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_phonons",
    "matbench_steels",
]


def convert_structure_to_serializable(pymatgen_structure):
    # Assuming Structure has 'data' and 'metadata' attributes
    cif_content = pymatgen_structure.to(fmt="cif")
    return cif_content


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mb = MatbenchBenchmark(autoload=False)
    benchmarks = cfg.matbench.benchmarks.dataset
    path = cfg.matbench.path.save_path
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print(f"Directory '{path}' already exists.")
    for benchmark_name in benchmarks:
        if benchmark_name not in available_benchmarks:
            raise ValueError(
                f"Invalid benchmark name. Available benchmarks: {', '.join(available_benchmarks)}"
            )

    for benchmark_name in benchmarks:
        benchmark = getattr(mb, benchmark_name)
        benchmark.load()

        for fold in benchmark.folds:
            # Get train inputs and outputs
            train_inputs, train_outputs = benchmark.get_train_and_val_data(fold)
            test_inputs = benchmark.get_test_data(fold)

            # Create the train data
            train_data = [
                {
                    "structure": convert_structure_to_serializable(train_inputs[index]),
                    "labels": train_outputs[index],
                }
                for index in range(len(train_inputs))
            ]

            # Save the train data as a JSON file
            train_dataset_name = f"train_{benchmark_name}_{fold}.json"
            with open(f"{path}/{train_dataset_name}", "w") as train_file:
                json.dump(train_data, train_file)

            print(f"Train data saved to {path}/{train_dataset_name}")

            test_data = [
                convert_structure_to_serializable(test_inputs[index])
                for index in range(len(test_inputs))
            ]

            # Save the test data as a JSON file
            test_dataset_name = f"test_{benchmark_name}_{fold}.json"
            with open(f"{path}/{test_dataset_name}", "w") as test_file:
                json.dump(test_data, test_file)

    print(f"Test data saved to {path}/{test_dataset_name}")


if __name__ == "__main__":
    main()
