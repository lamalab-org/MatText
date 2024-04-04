import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from functools import partial
from typing import Dict, List

import fire
from xtal2txt.core import TextRep


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



def process_entry(entry: dict) -> dict:
    try:
        print(f"strating{entry['material_id']}")
        text_reps = TextRep.from_input(entry['cif_p1']).get_requested_text_reps([ "atoms", "atoms_params"])
        #text_reps['mbid'] = entry["mbid"] # Use get_all_text_reps to get various text representations # Add chemical formula to the dictionary
        #text_reps['crystal_system'] = entry['structural_info']['crystal_system']  # Add crystal system to the dictionary
        text_reps['material_id'] = entry['material_id']  # Add material id to the dictionary
        return text_reps  # Return the entire dictionary
    except TimeoutError:
        print("Timeout error processing a row")
        return None
    except Exception as e:
        print(f"Error processing a row: {e}")
        return None



def process_batch(num_workers,batch, timeout):

    #process_entry_with_timeout = partial(process_entry, timeout=timeout)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_entry, batch))

    return [result for result in results if result is not None]



def process_json_to_json(json_file: str, output_json_file: str, log_file_path: str, num_workers: int = 48, timeout: int = 600, save_interval: int = 100, last_processed_entry: int = 0):

    num_cpus = multiprocessing.cpu_count()
    print(num_workers)

    print(f"json file: {json_file}")
    print(f"number of cpus: {num_cpus}")
    print(f"number of workers: {num_workers}")
    print(f"last processed entry: {last_processed_entry}")
    print(f"save_interval: {save_interval}")

    data = read_json(json_file)
    batch_size = num_workers * 4

    if last_processed_entry > 0:
        data = data[last_processed_entry:]

    batch_iterator = (data[i:i + batch_size] for i in range(0, len(data), batch_size))

    for i, batch_data in enumerate(batch_iterator, start=1):
        batch_results = process_batch(num_workers, batch_data, timeout)

        # Append batch_results to the output JSON file
        with open(output_json_file, 'a') as f:
            for result in batch_results:
                json.dump(result, f)
                f.write('\n')

        last_processed_entry += len(batch_data)
        if i % save_interval == 0:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Last processed entry index: {last_processed_entry}\n")
                log_file.write(f"Last processed batch number: {i}\n")

    print(f"Finished !!! logging at {log_file_path}")


if __name__ == "__main__":
    fire.Fire(process_json_to_json)


