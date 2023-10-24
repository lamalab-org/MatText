from pymatgen.core import Structure, Lattice
import numpy as np
import requests
import json


max_entries : int = 50000
entry_count : int = 0
scale_factor : int = 1e10
cif_json_save_path : str = f"cifs_{max_entries}.json"

url = "https://nomad-lab.eu/prod/v1/api/v1/entries/archive/query"
query =  {
    "results.material.structural_type:any": [
      "bulk"
    ],
    "results.material.n_elements": {
      "gte": 2,
      "lte": 3
    },
    "results.method.simulation.program_name:any": [
      "VASP"
    ],
    "results.method.simulation.dft.basis_set_type:any": [
      "plane waves"
    ],
    "results.method.simulation.dft.core_electron_treatment:any": [
      "pseudopotential"
    ],
    "results.method.simulation.dft.xc_functional_type:any": [
      "GGA"
    ],
    "results.method.simulation.dft.xc_functional_names:any": [
      "GGA_C_PBE",
      "GGA_X_PBE"
    ],
    "results.properties.available_properties:all": [
      "geometry_optimization"
]
}


required={ 
    'run': {
      'system': {
       'chemical_composition_reduced': '*',
       'chemical_composition': '*',
       'atoms': {
          'labels': '*',
          'species': '*',
          'lattice_vectors': '*',
          'positions': '*'
        }
      },

  },
}



from pymatgen.core.periodic_table import Element

def convert_to_cif(response_json):
    # Extract relevant information from the response JSON
    entry_id = response_json['entry_id']
    lattice_vectors = [
        [v * scale_factor for v in row] for row in response_json['archive']['run'][0]['system'][0]['atoms']['lattice_vectors']
    ]
    atom_labels = response_json['archive']['run'][0]['system'][0]['atoms']['labels']
    atom_species = response_json['archive']['run'][0]['system'][0]['atoms']['species']
    atom_positions = [
        [v * scale_factor for v in row] for row in response_json['archive']['run'][0]['system'][0]['atoms']['positions']
    ] 

    # Handle atom species which could be either symbols or atomic numbers
    atom_symbols = []
    for species in atom_species:
        if isinstance(species, int) and species != 0:
            element = Element.from_Z(species)
            atom_symbols.append(element.symbol)
        else:
            atom_symbols.append(species)

    # Construct the lattice using lattice vectors
    lattice_matrix = np.array(lattice_vectors )
    lattice = Lattice(lattice_matrix)

    # Create a pymatgen Structure
    pymatgen_structure = Structure(lattice, species=atom_symbols, coords=atom_positions )

    # Generate CIF content using pymatgen
    cif_content = pymatgen_structure.to(fmt="cif")

    return {"material_id": entry_id, "cif": cif_content}



def remove_none_values(data):
    if isinstance(data, dict):
        return {key: remove_none_values(value) for key, value in data.items() if value is not None}
    elif isinstance(data, list):
        return [remove_none_values(item) for item in data if item is not None]
    else:
        return data

result_list = []
page_after_value = None

while entry_count < max_entries:
    response = requests.post(
        url=url,
        json=dict(query=query,
                  required=required,
                  pagination=dict(
                      page_size = 100,
                      page_after_value=page_after_value
                  )))

    data = response.json()
    if len(data['data']) == 0:
        break

    page_after_value = data['pagination']['next_page_after_value']

    filtered_response = remove_none_values(data['data'])

    for entry in filtered_response:
        try:
            print(
                entry_count,
                entry['entry_id'],
                entry['archive']['run'][0]['system'][0]['chemical_composition_reduced'],
            )
            result_list.append(convert_to_cif(entry))
        except KeyError as e:
            print(f"KeyError: {e}")
            print("Entry content:")
            print(entry)
        entry_count += 1
        if entry_count >= max_entries:
            break


# Save the result list to a JSON file
with open(cif_json_save_path, "w") as json_file:
    json.dump(result_list, json_file, indent=4)