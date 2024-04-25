import copy
import glob
import json
from typing import Dict, List

import fire


def read_json_file(filename: str) -> Dict:
    """
    Read JSON file and return its content as dictionary.

    Args:
    filename (str): Path to the JSON file.

    Returns:
    dict: Content of the JSON file.
    """
    with open(filename) as f:
        data = json.load(f)
    return data

def get_bid_mbid(matbench_data_path: str) -> List[str]:
    """
    Extract 'mbid' from matbench data.

    Args:
    matbench_data_path (str): Path to the matbench data.

    Returns:
    List[str]: List of 'mbid' values.
    """
    data = read_json_file(matbench_data_path)
    mbids = [item['mbid'] for item in data]
    return mbids

def process_matbench_data(matbench_data: Dict, reports_path: str) -> Dict:
    """
    Process matbench data by removing entries based on reports.

    Args:
    matbench_data (Dict): Matbench data as dictionary.
    reports_path (str): Path to the reports.

    Returns:
    Dict: Modified matbench data.
    """
    small_matbench = copy.deepcopy(matbench_data)

    for prop in matbench_data['splits'].keys():
        for sp in ['train', 'test']:
            for fold in range(5):
                report_files = glob.glob(f'{reports_path}/{sp}_{prop}_{fold}.json_report.json')
                if len(report_files) > 0:
                    mbids = get_bid_mbid(report_files[0])

                    print(f"No of big structures in {sp}_{prop}_{fold} are : ", len(mbids))
                    fold_key = f"fold_{fold}"
                    ids = matbench_data['splits'][prop][fold_key][sp]

                    print("Total number of structures in matbench :", len(ids))

                    # Remove entries where ID is in mbid
                    matbench_data['splits'][prop][fold_key][sp] = [id for id in ids if id not in mbids]

                    print("--------------------")

    return matbench_data

def save_modified_data(matbench_data: Dict, output_file: str) -> None:
    """
    Save modified matbench data as JSON.

    Args:
    matbench_data (Dict): Modified matbench data.
    output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(matbench_data, file)

def run(reports_path: str, input_file: str, output_file: str) -> None:
    """
    Run the processing pipeline.

    Args:
    reports_path (str): Path to the reports.
    input_file (str): Path to the input JSON file.
    output_file (str): Path to the output JSON file.
    """
    matbench_data = read_json_file(input_file)
    modified_data = process_matbench_data(matbench_data, reports_path)
    save_modified_data(modified_data, output_file)

if __name__ == '__main__':
    fire.Fire(run)
