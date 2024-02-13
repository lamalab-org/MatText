import json
import csv
from pymatgen.core.structure import Structure
from typing import List, Dict
import fire

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

def extract_info(data: List[Dict]) -> List[Dict]:
    """Extract required information from JSON data.

    Args:
        data (List[Dict]): List of dictionaries containing JSON data.

    Returns:
        List[Dict]: List of dictionaries containing extracted information.
    """
    extracted_data = []
    for item in data:
        try:
            entry = {}
            entry['material_id'] = item.get('material_id')
            entry['massdensity'] = item.get('mass_density', 'None')
            structure = Structure.from_str(item['cif'], "cif")
            composition_string = structure.composition.hill_formula
            composition= composition_string.replace(" ", "")   # remove spaces
            entry['composition'] = composition
            extracted_data.append(entry)
        except Exception as e:
            print(f"Skipping entry with invalid CIF: {item['material_id']}. Error: {e}")
    return extracted_data


def write_csv(data: List[Dict], output_file: str):
    """Write extracted data to a CSV file.

    Args:
        data (List[Dict]): List of dictionaries containing extracted information.
        output_file (str): Path to the output CSV file.
    """
    keys = data[0].keys()
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def generate_csv(json_file: str, output_file: str):
    """Generate CSV file from JSON data.

    Args:
        json_file (str): Path to the JSON file.
        output_file (str): Path to the output CSV file.
    """
    json_data = read_json(json_file)
    extracted_data = extract_info(json_data)
    write_csv(extracted_data, output_file)

if __name__ == "__main__":
    fire.Fire(generate_csv)
