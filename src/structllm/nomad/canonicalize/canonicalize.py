import concurrent.futures
import pandas as pd
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import fire


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
                cannon_slices_list.append(backend.get_canonical_SLICES(i))
            except Exception as e:
                if str(e) == "Error: wrong edge label":
                    # Handle the specific "Error: wrong edge label" exception
                    continue 

        if cannon_slices_list:
            return list(set(cannon_slices_list))[0]
        else:
            return None  # Return None if cannon_slices_list is empty

    except ValueError as e:
        if str(e) == "Invalid CIF file with no structures!":
            return None  # Handle the case of an invalid CIF file with no structures
        else:
            return None  # Return None for other ValueErrors encountered
    except IndexError:
        return None  # Return None for IndexError




def process_slice(struct_str: str) -> str:
    """
    Process the CIF structure string to get its canonical slice.

    Args:
        struct_str (str): CIF structure string.

    Returns:
        str: Canonical slice string.
    """
    return get_canonical_slice(struct_str)


def apply_parallel(func, iterable, max_workers=None):
    """
    Apply a function in parallel using concurrent.futures.

    Args:
        func: Function to be applied.
        iterable: Iterable data to process.
        max_workers (int, optional): Number of workers for parallel processing.

    Returns:
        list: Results of applying the function.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, iterable))
    return results


def main(path: str,  output_path: str , max_workers: int = 12,):
    """
    Main function to read data, process it in parallel, and update the dataframe.

    Args:
        path (str): Path to the CSV file.
        max_workers (int, optional): Number of workers for parallel processing.
    """
    data = pd.read_csv(path)
    df = data.drop_duplicates()

    # Use concurrent.futures to apply the function in parallel
    results = apply_parallel(process_slice, df['cif'], max_workers=max_workers)
    df['slices'] = results
    
    # Save the updated DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}.")


if __name__ == "__main__":
    fire.Fire(main)
