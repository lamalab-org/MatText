import json
import pandas as pd
import fire

from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
from functools import partial
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
    primitive_structure = pymatgen_struct.get_primitive_structure() # convert to primitive structure
    return backend.structure2SLICES(primitive_structure)

def get_agmented_slice(struct_str: str) -> str:
    """
    Get the canonical slice for a given CIF structure string.

    Args:
        struct_str (str): CIF structure string.

    Returns:
        str: Canonical slice string.
    """
    try:
        original_structure = Structure.from_str(struct_str, "cif")
        backend = InvCryRep(graph_method='econnn')
        slices_list = backend.structure2SLICESAug(structure=original_structure, num=2000)
        return slices_list
    
    except ValueError as e:
        if str(e) == "Invalid CIF file with no structures!":
            return None  # Handle the case of an invalid CIF file with no structures
        else:
            #raise e  # Raise any other ValueErrors encountered
            return None
    except IndexError:
        return None

def get_canonical_slice(struct_str: str) -> str:
    """
    Get the canonical slice for a given CIF structure string.

    Args:
        struct_str (str): CIF structure string.

    Returns:
        str: Canonical slice string.
    """
    try:
        original_structure = Structure.from_str(struct_str, "cif")
        backend = InvCryRep(graph_method='econnn')
        slices_list = backend.structure2SLICESAug(structure=original_structure, num=2000)
        slices_list_unique = list(set(slices_list))
        cannon_slices_list = []
        for i in slices_list_unique:
            try:
                
                slice = backend.get_canonical_SLICES(i)
                #print(slice)
                cannon_slices_list.append(slice)
            except Exception as e:
                print(e)
                continue

        return list(set(cannon_slices_list))[0]
    
    except ValueError as e:
        if str(e) == "Invalid CIF file with no structures!":
            return None  # Handle the case of an invalid CIF file with no structures
        else:
            #raise e  # Raise any other ValueErrors encountered
            return None
    except IndexError:
        return None

def process_entry(entry: dict, timeout: int) -> dict:
    
    # Ensure the give_slice function and necessary data are picklable
    try:
        print(f"strating{entry['chemical_formula']}")
        slice_result = get_canonical_slice(entry['cif'])
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

from typing import List

def process_entry_train_matbench(entry: dict, timeout: int) -> dict:
    
    # Ensure the give_slice function and necessary data are picklable
    try:
        slices_result = give_slice(entry["structure"])
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


# def process_entry_train_matbench(entry: dict, timeout: int) -> List[dict]:
#     try:
#         slices_result = give_slice(entry["structure"])
#         processed_entries = []
#         for slice_result in slices_result:
#             processed_entry = {
#                 'slice': slice_result,
#                 'labels': entry["labels"]
#             }
#             processed_entries.append(processed_entry)
#         return processed_entries
#     except TimeoutError:
#         print(f"Timeout error processing a row")
#         return []
#     except Exception as e:
#         print(f"Error processing a row: {e}")
#         return []
    
def process_entry_test_matbench(entry: List, timeout: int) -> dict:
    # Ensure the give_slice function and necessary data are picklable
    try:

        slice_result = get_canonical_slice(entry)
        return {
                'slice': slice_result,
            }
    except TimeoutError:
        print(f"Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        return None


def process_batch(num_workers,batch, timeout):

    process_entry_with_timeout = partial(process_entry_train_matbench, timeout=timeout)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry_with_timeout, batch))

    return [result for result in results if result is not None]


def process_json_to_csv(json_file: str, csv_file: str, log_file_path: str,  num_workers: int = 4, timeout: int = 600, save_interval: int = 100,  last_processed_entry: int = 0):
    

    num_cpus = multiprocessing.cpu_count()
    #num_workers = num_cpus

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
        batch_results = process_batch(num_workers,batch_data, timeout)
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

