import json
import pandas as pd
import fire

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure

from typing import List, Dict
from functools import partial

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

def give_slice(cif: str) -> str:
    """Calculate a slice from a CIF string.

    Args:
        cif (str): The CIF string.

    Returns:
        str: The calculated slice.
    """
    backend = InvCryRep()
    pymatgen_struct = Structure.from_str(cif, "cif")
    return backend.structure2SLICES(pymatgen_struct)



def process_entry(entry: Dict, timeout: int) -> Dict:
    """Process a dictionary entry with a timeout.

    Args:
        entry (Dict): A dictionary containing data.
        timeout (int): Maximum time limit for give_slice in seconds.

    Returns:
        Dict: A dictionary containing processed data, or None for rows with errors or timeouts.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(give_slice, entry['cif'])
            slice_result = future.result(timeout=timeout)
            return {
                'slice': slice_result,
                'chemical_formula': entry['chemical_formula'],
                'crystal_system': entry['structural_info']['crystal_system']
            }
    except TimeoutError:
        print(f"Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        return None


def process_json_to_csv(json_file: str, csv_file: str, num_workers: int = 4, timeout: int = 600, save_interval: int = 100):
    data = read_json(json_file)

    process_entry_with_timeout = partial(process_entry, timeout=timeout)
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, entry in enumerate(data):
            result = process_entry_with_timeout(entry)
            if result is not None:
                results.append(result)

            # Save the results to a CSV file after processing a specified number of entries
            if idx > 0 and idx % save_interval == 0:
                intermediate_df = pd.DataFrame(results)
                intermediate_df.to_csv(csv_file, index=False, mode='a', header=False)
                results = []  # Clear the results list

    # Save any remaining results to the final CSV file
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, mode='a', header=False)

if __name__ == "__main__":
    fire.Fire(process_json_to_csv)