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


def process_json_to_csv(json_file: str, csv_file: str, num_workers: int = 4, timeout: int = 600):
    data = read_json(json_file)

    process_entry_with_timeout = partial(process_entry, timeout=timeout)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry_with_timeout, data))

    # Remove None values from the results
    results = [result for result in results if result is not None]

    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
    else:
        print("No valid data to write to the CSV file.")

if __name__ == "__main__":
    fire.Fire(process_json_to_csv)
