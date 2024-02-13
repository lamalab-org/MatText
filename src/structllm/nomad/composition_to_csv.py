import json
import pandas as pd
import fire

from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
from functools import partial
from pymatgen.core.structure import Structure

from typing import List, Dict
from functools import partial
from typing import List

def read_json(json_file: str) -> List[Dict]:
    """Read JSON data from a file.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries containing the JSON data.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def give_composition(cif:str)-> str:
    """Return composition in hill format from a CIF string.

    Args:
        cif (str): The CIF string.

    Returns:
        str: The composition in hill format.
    """
    pymatgen_struct = Structure.from_str(cif, "cif")
    composition_string = pymatgen_struct.composition.hill_formula
    composition= composition_string.replace(" ", "")
    return composition


def process_entry_train_matbench(entry: dict, timeout: int) -> dict:
    
    # Ensure the give_slice function and necessary data are picklable
    try:
        slices_result = give_composition(entry["structure"])
        return {
                'slice': slices_result,
                'labels': entry["labels"]
            }
    except TimeoutError:
        print(f"Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        return None


    
def process_entry_test_matbench(entry: List, timeout: int) -> dict:
    # Ensure the give_slice function and necessary data are picklable
    try:

        slice_result = give_composition(entry)
        return {
                'slice': slice_result,
            }
    except TimeoutError:
        print(f"Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        return None


def process_batch(num_workers, batch, timeout, process_entry_func):

    process_entry_with_timeout = partial(process_entry_func, timeout=timeout)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry_with_timeout, batch))

    return [result for result in results if result is not None]



def process_json_to_csv(json_file: str, csv_file: str, log_file_path: str,  process_entry: str = 'test', num_workers: int = 4, timeout: int = 600, save_interval: int = 100,  last_processed_entry: int = 0):
    

    num_cpus = multiprocessing.cpu_count()
    #num_workers = num_cpus

    process_entry_funcs = {
        'test': process_entry_test_matbench,
        'train': process_entry_train_matbench
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

    
    batch_iterator = (data[i:i+batch_size] for i in range(0, len(data), batch_size))

    for i, batch_data in enumerate(batch_iterator, start=1):
        batch_results = process_batch(num_workers,batch_data, timeout, process_entry_func)
        intermediate_df = pd.DataFrame(batch_results)
        intermediate_df.to_csv(csv_file, index=False, mode='a', header=False)

        last_processed_entry += len(batch_data)
        if i % save_interval == 0:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Last processed entry index: {last_processed_entry}\n")
                log_file.write(f"Last processed batch number: {i}\n")

    print(f"Finished !!! logging at {log_file_path}")



if __name__ == "__main__":
    fire.Fire(process_json_to_csv)

