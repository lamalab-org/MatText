import pandas as pd
import json
import fire


def create_csv_from_json(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Creating lists to store values for each column
    cif = []
    chemical_formula = []
    space_group_symbol = []
    space_group_number = []
    crystal_system = []
    point_group = []
    spin_polarized = []
    energy_fermi = []
    energy_highest_occupied = []
    energy_lowest_unoccupied = []
    material_id = []
    mass_density = []
    fermi = []
    total_energy = []

    for entry in data:
        # Collecting values for each column
        cif.append(entry.get('cif', None))
        chemical_formula.append(entry.get('chemical_formula', None))
        space_group_symbol.append(entry['structural_info'].get('space_group_symbol', None))
        space_group_number.append(entry['structural_info'].get('space_group_number', None))
        crystal_system.append(entry['structural_info'].get('crystal_system', None))
        point_group.append(entry['structural_info'].get('point_group', None))
        spin_polarized.append(entry['electronic'].get('spin_polarized', None))
        energy_fermi.append(entry['electronic'].get('energy_fermi', None))
        energy_highest_occupied.append(entry['electronic'].get('energy_highest_occupied', None))
        energy_lowest_unoccupied.append(entry['electronic'].get('energy_lowest_unoccupied', None))
        material_id.append(entry.get('material_id', None))
        mass_density.append(entry.get('mass_density', None))
        fermi.append(entry.get('fermi', None))
        total_energy.append(entry.get("total_energy",None))

    # Creating a DataFrame with collected values
    df = pd.DataFrame({
        "cif": cif,
        "chemical_formula": chemical_formula,
        "space_group_symbol": space_group_symbol,
        "space_group_number": space_group_number,
        "crystal_system": crystal_system,
        "point_group": point_group,
        "spin_polarized": spin_polarized,
        "energy_fermi": energy_fermi,
        "energy_highest_occupied": energy_highest_occupied,
        "energy_lowest_unoccupied": energy_lowest_unoccupied,
        "material_id": material_id,
        "mass_density": mass_density,
        "fermi": fermi
    })

    # Writing the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    fire.Fire(create_csv_from_json)
