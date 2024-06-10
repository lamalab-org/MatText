import json
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List

import fire
from pymatgen.core import Structure
from xtal2txt.core import TextRep

from mattext.analysis.xtal2pot import Xtal2Pot

linearpotential = Xtal2Pot()


def read_json(json_file: str) -> List[Dict]:
    """Read JSON data from a file.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries containing the JSON data.
    """
    with open(json_file) as file:
        data = json.load(file)
    return data


class TimeoutException(Exception):
    """Custom exception class for timeouts."""


def timeout_handler(signum, frame):
    """Custom signal handler for timeouts."""
    raise TimeoutException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


def process_entry_train_matbench(entry: dict, timeout: int, alphas=None) -> dict:
    """Process as entry for Matbench train dataset with a timeout.

    Args:
        entry (dict): The entry to process.
        timeout (int): The timeout in seconds.

    Returns:
        dict: The processed entry, or None if an error occurred.
    """
    try:
        signal.alarm(timeout)  # Start the timer
        try:
            structure = Structure.from_str(str(entry["structure"]), "cif")
        except Exception as e:
            print(e)
        # text_reps = linearpotential.get_total_energy(structure, alphas=alphas)
        text_reps = {}
        # text_reps["crystal_llm_rep"] = TextRep.from_input(
        #     entry["structure"]
        # ).get_crystal_llm_rep()
        list_of_rep = [
            "zmatrix",
            "atoms_params",
            "local_env",
            "cif_p1",
            "composition",
            "atoms_params",
            "crystal_llm_rep",
        ]
        text_reps = TextRep.from_input(entry["structure"]).get_requested_text_reps(
            list_of_rep
        )
        text_reps["labels"] = entry["labels"]
        text_reps["mbid"] = entry["mbid"]
        composition_energy, geometry_energy = linearpotential.get_potential(structure)
        text_reps["composition_energy"] = composition_energy
        text_reps["geometry_energy"] = geometry_energy
        signal.alarm(0)  # Reset the timer
        return text_reps
    except TimeoutException:
        print("Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        print(entry["structure"])
        return None


def process_entry_test_matbench(entry: dict, timeout: int, alphas=None) -> dict:
    """Process an entry for Matbench test dataset with a timeout.

    Args:
        entry (dict): The entry to process.
        timeout (int): The timeout in seconds.

    Returns:
        dict: The processed entry, or None if an error occurred.
    """
    try:
        signal.alarm(timeout)  # Start the timer
        structure = Structure.from_str(str(entry["structure"]), "cif")
        list_of_rep = [
            "zmatrix",
            "atoms_params",
            "local_env",
            "cif_p1",
            "composition",
            "atoms_params",
            "crystal_llm_rep",
        ]
        text_reps = TextRep.from_input(entry["structure"]).get_requested_text_reps(
            list_of_rep
        )
        text_reps["mbid"] = entry["mbid"]
        composition_energy, geometry_energy = linearpotential.get_potential(structure)
        text_reps["composition_energy"] = composition_energy
        text_reps["geometry_energy"] = geometry_energy
        signal.alarm(0)  # Reset the timer
        return text_reps
    except TimeoutException:
        print("Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        print(entry["structure"])
        return None


def process_batch(num_workers, batch, timeout, process_entry_func, alphas):
    """Process a batch of entries in parallel.

    Args:
        num_workers (int): The number of worker processes.
        batch (list): The batch of entries to process.
        timeout (int): The timeout in seconds for each entry.
        process_entry_func (function): The function to process an entry.

    Returns:
        list: The processed entries.
    """
    process_entry_with_timeout = partial(
        process_entry_func, timeout=timeout, alphas=alphas
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry_with_timeout, batch))

    return [result for result in results if result is not None]


def process_json_to_json(
    json_file: str,
    output_json_file: str,
    log_file_path: str,
    process_entry: str = "test",
    num_workers: int = 48,
    timeout: int = 600,
    save_interval: int = 100,
    last_processed_entry: int = 0,
    alphas: List[float] = None,
):
    """Prepare Matbench dataset with different representation as implemented in Xtal2txt."""
    # Your main processing function here
    num_cpus = multiprocessing.cpu_count()

    process_entry_funcs = {
        "test": process_entry_test_matbench,
        "train": process_entry_train_matbench,
    }
    # Get the selected function
    process_entry_func = process_entry_funcs[process_entry]

    print(f"json file: {json_file}")
    print(f"number of cpus: {num_cpus}")
    print(f"number of workers: {num_workers}")
    print(f"last processed entry: {last_processed_entry}")
    print(f"save_interval: {save_interval}")

    data = read_json(json_file)
    batch_size = num_workers * 4

    if last_processed_entry > 0:
        data = data[last_processed_entry:]

    processed_entries = []

    batch_iterator = (data[i : i + batch_size] for i in range(0, len(data), batch_size))

    for i, batch_data in enumerate(batch_iterator, start=1):
        batch_results = process_batch(
            num_workers, batch_data, timeout, process_entry_func, alphas
        )

        processed_entries.extend(batch_results)

        last_processed_entry += len(batch_data)
        if i % save_interval == 0:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Last processed entry index: {last_processed_entry}\n")
                log_file.write(f"Last processed batch number: {i}\n")

    with open(output_json_file, "w") as f:
        json.dump(processed_entries, f)

    print(f"Finished !!! logging at {log_file_path}")


if __name__ == "__main__":
    fire.Fire(process_json_to_json)
