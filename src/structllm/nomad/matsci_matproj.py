import lmdb
import pickle
import json
import fire
import numpy as np

from tqdm import tqdm 
from math import pi
from pymatgen.core import Structure, Lattice





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


from pymatgen.core import Structure, Element
def create_cif(structure:dict):

    atom_species = structure.atomic_numbers
    cartesian_pos = structure.cart_coords

    lattice = structure.lattice
    
    
    
    # Handle atom species which could be either symbols or atomic numbers
    atom_symbols = []
    for species in atom_species:
        if isinstance(species, int) and species != 0:
            element = Element.from_Z(species)
            atom_symbols.append(element.symbol)
        else:
            atom_symbols.append(species)


    # Create a pymatgen Structure
    pymatgen_structure = Structure(lattice, species=atom_symbols, coords=cartesian_pos )

    # Generate CIF content using pymatgen
    cif_content = pymatgen_structure.to(fmt="cif")
    return cif_content



            
def prepare_dict(mat_dict:dict):

    cif_content = create_cif(mat_dict['structure'])
    energy_per_atom = mat_dict['energy_per_atom']
    formation_energy_per_atom = mat_dict['formation_energy_per_atom']
    is_stable = mat_dict['is_stable']
    band_gap = mat_dict['band_gap']
    efermi =mat_dict['efermi']
    crystal_type = mat_dict['symmetry'].crystal_system.value
    space_group = mat_dict['structure'].get_space_group_info()
    chemical_formula = mat_dict['structure'].formula

   
    return {"formation_energy_per_atom":formation_energy_per_atom , 
            "is_stable":is_stable, 
            "chemical_formula":chemical_formula,
            "space_group":space_group,
            "crystal_type":crystal_type,
            "band_gap":band_gap,
            "efermi":efermi,
            "energy_per_atom":energy_per_atom,
            "cif": cif_content}
    


    
def prep_data(lmdb_path:str,output_file:str)->None:
    materials_list = []
    dataset = Dataset(lmdb_path, 1)

    #loop through data points in lmdb
    for index in tqdm(range(dataset.len())):
        datapoint = dataset.get(index)
        data_dict = prepare_dict(datapoint)
        materials_list.append(data_dict)


    with open(output_file, 'w') as json_file:
        json.dump(materials_list, json_file)


if __name__ == "__main__":

   fire.Fire(prep_data)
   print("Finished")

