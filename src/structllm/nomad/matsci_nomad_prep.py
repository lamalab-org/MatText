import lmdb
import pickle
import json
import fire
import numpy as np

from tqdm import tqdm 
from math import pi
from pymatgen.core import Structure, Lattice



scale_factor : int = 1e10



class Dataset():
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

    atom_species = mat_dict["properties"]["structures"]["structure_original"]["species_at_sites"]
    cartesian_pos = mat_dict['properties']["structures"]["structure_original"]['cartesian_site_positions']


    scale_factor : int = 1e10
    atom_positions = [
            [v * scale_factor for v in row] for row in cartesian_pos
        ]
    

    lattice_params = mat_dict["properties"]["structures"]["structure_original"][
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
    return cif_content


            
def prepare_dict(mat_dict:dict):

    cif_content = create_cif(mat_dict)

    # crystal structure properties
    material_name = mat_dict['material']['material_name']
    chemical_formula = mat_dict['material']['chemical_formula_descriptive']
    structural = {
        "space_group_symbol": mat_dict['material']['symmetry']['space_group_symbol'],
        "crystal_system": mat_dict['material']['symmetry']['crystal_system'],
        "mass_density": mat_dict['properties']['structures']['structure_original']['mass_density']
    } 

    # Electronic structure properties
    elec_structure = mat_dict['properties']['electronic']['dos_electronic']
    electronic = {
        "spin_polarized" : elec_structure['spin_polarized'] ,
        "energy_fermi" : elec_structure['energy_fermi'],
        # "energy_highest_occupied" : elec_structure['band_gap'][0]['energy_highest_occupied'],
        # "energy_lowest_unoccupied" :elec_structure['band_gap'][0]['energy_lowest_unoccupied']
    }

    # Energy
    total_energy = mat_dict['energies']['total']['value']
    fermi = mat_dict['energies']['fermi']

    #method 
    method = mat_dict['method']

    return {"material_name": material_name, "method":method, "total_energy":total_energy,"fermi":fermi , "chemical_formula":chemical_formula, "structural_info" :structural,"electronic": electronic, "cif": cif_content}
    
        
def prep_data(lmdb_path:str,output_file:str)->None:
    materials_list = []
    dataset = Dataset(lmdb_path, 1)

    #loop through data points in lmdb
    for index in tqdm(range(10)):
        datapoint = dataset.get(index)
        data_dict = prepare_dict(datapoint)
        materials_list.append(data_dict)


    with open(output_file, 'w') as json_file:
        json.dump(materials_list, json_file)


if __name__ == "__main__":

   fire.Fire(prep_data)

