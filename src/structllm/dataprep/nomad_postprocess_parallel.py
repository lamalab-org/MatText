import json
import pickle
from math import pi
from multiprocessing import Pool

import fire
import lmdb
from pymatgen.core import Element, Lattice, Structure
from tqdm import tqdm

scale_factor : int = 1e10 #length in nomad is in meters, scale to angstrom



class Dataset:
  """
  Custom class for reading NOMAD dataset from MatSciML Zenodo
  """

  def __init__(self, lmdb_path, max_readers=1, transform=None, pre_transform=None):
    """
    Constructor for dataset
    param: lmdb_path -> path to lmdb_file
    param: max_readers -> maximum number of concurrent read processes accessing lmdb file
    """
    self.env = lmdb.open(lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers)
    self.txn = self.env.begin()

  def len(self):

    return self.txn.stat()['entries']


  def get(self, index):
    """
    Return a  datapoint
    """
    # Select graph sample
    id = f"{index}".encode("ascii")
    datapoint = pickle.loads(self.txn.get(id))

    return datapoint


def create_lattice(lattice_params : dict):

    lattice_abc = (
                    lattice_params["a"] * scale_factor,
                    lattice_params["b"] * scale_factor,
                    lattice_params["c"] * scale_factor,
                )
    lattice_angles = (
                    lattice_params["alpha"],
                    lattice_params["beta"],
                    lattice_params["gamma"],
                )
    a, b, c  = lattice_abc
    alpha, beta, gamma = lattice_angles
    lattice = Lattice.from_parameters(
                a, b, c, alpha * 180/pi, beta * 180/pi, gamma * 180/pi
            )
    return lattice


def create_cif(mat_dict:dict):

    if 'structure_primitive' in mat_dict['properties']['structures']:
        crystal = mat_dict['properties']['structures']['structure_primitive']
    else:
        crystal = mat_dict["properties"]["structures"]["structure_original"]


    atom_species = crystal["species_at_sites"]
    cartesian_pos = crystal['cartesian_site_positions']


    scale_factor : int = 1e10
    atom_positions = [
            [v * scale_factor for v in row] for row in cartesian_pos
        ]


    lattice_params =crystal[
            "lattice_parameters"
        ]
    lattice = create_lattice(lattice_params)



    # Handle atom species which could be either symbols or atomic numbers
    atom_symbols = []
    for species in atom_species:
        if isinstance(species, int) and species != 0:
            element = Element.from_Z(species)
            atom_symbols.append(element.symbol)
        else:
            atom_symbols.append(species)


    # Create a pymatgen Structure
    pymatgen_structure = Structure(lattice, species=atom_symbols, coords=atom_positions )

    # Generate CIF content using pymatgen
    cif_content = pymatgen_structure.to(fmt="cif")
    # Remove the comment line if present
    cif_lines = cif_content.split('\n')

    if cif_lines[0].startswith("# generated using pymatgen"):
        cif_lines = cif_lines[1:]

    # Join the lines back together to form the CIF content without the comment
    cif_content_without_comment = '\n'.join(cif_lines)
    return cif_content_without_comment

def prepare_dict(mat_dict:dict):

    cif_content = create_cif(mat_dict)

    # crystal structure properties
    material_id = mat_dict['material']['material_id']
    chemical_formula_descriptive = mat_dict['material']['chemical_formula_descriptive']
    chemical_formula_reduced = mat_dict['material']['chemical_formula_reduced']
    chemical_formula_hill = mat_dict['material']['chemical_formula_hill']

    # Crystal structure properties
    try:
        structure = mat_dict['material']['symmetry']

        structural = {
            "space_group_symbol": structure.get('space_group_symbol', None),
            "space_group_number": structure.get('space_group_number', None),
            "crystal_system": structure.get('crystal_system', None),
            "point_group": structure.get('point_group', None)
        }
        symmetry = mat_dict['material']['symmetry']
    except KeyError:
        structural = {
            "space_group_symbol": None,
            "space_group_number": None,
            "crystal_system": None,
            "point_group": None
        }

    # Fetching mass_density
    mass_density = mat_dict['properties']['structures']['structure_original'].get('mass_density', None)

    # Electronic structure properties
    try:
        elec_structure = mat_dict['properties']['electronic']['dos_electronic']

        electronic = {
            "spin_polarized": elec_structure.get('spin_polarized', None),
            "energy_fermi": elec_structure.get('energy_fermi', None),
            "energy_highest_occupied": elec_structure['band_gap'][0].get('energy_highest_occupied', None),
            "energy_lowest_unoccupied": elec_structure['band_gap'][0].get('energy_lowest_unoccupied', None)
        }
    except KeyError:
        # If 'properties' or 'electronic' or 'dos_electronic' or 'band_gap' is missing
        electronic = {
            "spin_polarized": None,
            "energy_fermi": None,
            "energy_highest_occupied": None,
            "energy_lowest_unoccupied": None
        }


    # Extracting energy values
    total_energy = mat_dict['energies']['total'].get('value', None)
    fermi = mat_dict['energies'].get('fermi', None)

    # Extracting the method
    method = mat_dict.get('method')

    return {
        "material_id": material_id,
        "cif": cif_content,
        "chemical_formula_descriptive": chemical_formula_descriptive,
        "chemical_formula_reduced": chemical_formula_reduced,
        "chemical_formula_hill": chemical_formula_hill,
        "symmetry": symmetry,
        "method":method,
        "mass_density":mass_density,
        "total_energy":total_energy,
        "fermi":fermi ,
        "structural_info" :structural,
        "electronic" :electronic,
    }

def process_data(args):
    lmdb_path, index = args
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    with env.begin() as txn:
        id = f"{index}".encode("ascii")
        datapoint = pickle.loads(txn.get(id))

    if datapoint is not None:
        try:
            data_dict = prepare_dict(datapoint)
            return data_dict
        except Exception as e:
            print(f"An error occurred for index {index}: {e}. Skipping this entry.")
            return None
    else:
        print(f"Data point for index {index} is None. Skipping this entry.")
        return None

def prep_data(lmdb_path: str, output_file: str, num_processes: int = 4) -> None:
    materials_list = []
    dataset = Dataset(lmdb_path, 1)
    total_entries = dataset.len()

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_data, [(lmdb_path, i) for i in range(total_entries)]), total=total_entries))

    skipped_count = 0
    for result in results:
        if result is not None:
            materials_list.append(result)
        else:
            skipped_count += 1

    with open(output_file, 'w') as json_file:
        json.dump(materials_list, json_file)
    print(f"Total number of skipped datapoints: {skipped_count}")  # Print the total number of skipped datapoints

if __name__ == "__main__":
    fire.Fire(prep_data)
    print("Finished")
