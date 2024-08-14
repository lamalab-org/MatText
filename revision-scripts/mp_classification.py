import lmdb
import pickle
import json
import os
from pymatgen.core import Structure
import fire

class Dataset:
    def __init__(self, lmdb_path, max_readers=1):
        self.env = lmdb.open(lmdb_path,
                             subdir=False,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False,
                             max_readers=max_readers)
        self.txn = self.env.begin()

    def __len__(self):
        return self.txn.stat()['entries']

    def get(self, index):
        id = f"{index}".encode("ascii")
        return pickle.loads(self.txn.get(id))

def create_json_from_lmdb(lmdb_path, output_dir):
    dataset = Dataset(lmdb_path)
    output_data = []

    for i in range(len(dataset)):
        d = dataset.get(i)
        
        # Convert structure to CIF
        structure = d['structure']
        cif = structure.to(fmt="cif")

        entry = {
            "structure": cif,
            "is_stable": d['is_stable'],
            "is_metal": d['is_metal'],
            "is_magnetic": d['is_magnetic']
        }
        
        output_data.append(entry)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file
    output_file = os.path.join(output_dir, "mp_test.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"JSON file created: {output_file}")

if __name__ == "__main__":
    fire.Fire(create_json_from_lmdb)