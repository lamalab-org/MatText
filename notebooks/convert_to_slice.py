import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import fire
from typing import List, Dict


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

def process_entry(entry: Dict) -> Dict:
    """Process a dictionary entry.

    Args:
        entry (Dict): A dictionary containing data.

    Returns:
        Dict: A dictionary containing processed data.
    """
    slice_result = give_slice(entry['cif'])
    return {
        'slice': slice_result,
        'chemical_formula': entry['chemical_formula'],
        'crystal_system': entry['structural_info']['crystal_system']
    }

def process_json_to_csv(json_file: str, csv_file: str, num_workers: int = 4):
    """Process JSON data and save it as a CSV file.

    Args:
        json_file (str): The path to the JSON file.
        csv_file (str): The path to the CSV file.
        num_workers (int): Number of worker threads for parallel processing.
    """
    data = read_json(json_file)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry, data))

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    fire.Fire(process_json_to_csv)
