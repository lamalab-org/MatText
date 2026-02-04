#!/usr/bin/env python3
"""
Script to fill in missing robocrys_rep entries in the MatText dataset.
Uses robocrystallographer to generate descriptions from CIF structures.
Parallelized for better performance.
"""

import os
import warnings
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from robocrys import StructureCondenser, StructureDescriber
from tqdm import tqdm

# Configuration
NUM_WORKERS = 64 #  max(1, cpu_count() - 12)  # Leave one core free

# Setup logging
log_filename = f"robocrys_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="No oxidation states specified on sites!",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="CrystalNN: cannot locate an appropriate radius",
    category=UserWarning,
)


def process_single_entry(args: tuple) -> tuple[int, str | None, str | None]:
    """
    Worker function to process a single entry.
    Must be a top-level function for multiprocessing.

    Args:
        args: Tuple of (idx, cif_symmetrized, cif_p1)

    Returns:
        Tuple of (idx, description, error_message)
    """
    idx, cif_symmetrized, cif_p1 = args

    # Suppress specific warnings in worker processes
    warnings.filterwarnings(
        "ignore",
        message="No oxidation states specified on sites!",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="CrystalNN: cannot locate an appropriate radius",
        category=UserWarning,
    )

    cif_sources = [
        ("cif_symmetrized", cif_symmetrized),
        ("cif_p1", cif_p1),
    ]

    errors = []
    for cif_name, cif_string in cif_sources:
        if cif_string:
            try:
                structure = Structure.from_str(cif_string, fmt="cif")
                condenser = StructureCondenser()
                describer = StructureDescriber()
                condensed_structure = condenser.condense_structure(structure)
                description = describer.describe(condensed_structure)
                return idx, description, None
            except Exception as e:
                errors.append(f"{cif_name}: {type(e).__name__}: {e}")
        else:
            errors.append(f"{cif_name}: empty/null")

    error_msg = f"Index {idx}: Failed - {'; '.join(errors)}"
    return idx, None, error_msg


def main():
    # Configuration
    dataset_name = "n0w0f/MatText"
    subset_name = "pretrain300k"
    output_dir = "./mattext_pretrain300k_updated"

    logger.info(f"Loading dataset: {dataset_name}, subset: {subset_name}")
    logger.info(f"Using {NUM_WORKERS} worker processes")

    # Load the dataset
    dataset = load_dataset(dataset_name, subset_name)

    # Get the split (usually 'train')
    split_name = list(dataset.keys())[0]
    logger.info(f"Working with split: {split_name}")
    data = dataset[split_name]

    # Count missing entries and prepare work items
    total_entries = len(data)
    work_items = []  # (idx, cif_symmetrized, cif_p1) for entries needing processing
    existing_robocrys = {}  # idx -> existing robocrys_rep

    for idx, entry in enumerate(data):
        current_robocrys = entry.get("robocrys_rep")
        if current_robocrys is not None and current_robocrys != "":
            existing_robocrys[idx] = current_robocrys
        else:
            work_items.append((
                idx,
                entry.get("cif_symmetrized"),
                entry.get("cif_p1"),
            ))

    missing_count = len(work_items)
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Already have robocrys_rep: {len(existing_robocrys)}")
    logger.info(f"Missing robocrys_rep entries to process: {missing_count}")

    if missing_count == 0:
        logger.info("No missing entries to fill. Exiting.")
        return

    # Process missing entries in parallel
    results = {}  # idx -> description
    failed_indices = []

    with Pool(processes=NUM_WORKERS) as pool:
        for idx, description, error in tqdm(
            pool.imap_unordered(process_single_entry, work_items),
            total=len(work_items),
            desc="Processing entries",
        ):
            if description:
                results[idx] = description
            else:
                failed_indices.append(idx)
                logger.warning(error)

    success_count = len(results)
    logger.info(f"Successfully generated {success_count} new descriptions")
    logger.info(f"Failed to generate {len(failed_indices)} descriptions")

    # Build final robocrys_rep list in order
    final_robocrys = []
    for idx in range(total_entries):
        if idx in existing_robocrys:
            final_robocrys.append(existing_robocrys[idx])
        elif idx in results:
            final_robocrys.append(results[idx])
        else:
            final_robocrys.append(None)

    # Update the dataset with new robocrys_rep values
    def update_robocrys(example, idx):
        example["robocrys_rep"] = final_robocrys[idx]
        return example

    updated_data = data.map(update_robocrys, with_indices=True)

    # Save the updated dataset locally
    logger.info(f"Saving updated dataset to: {output_dir}")
    updated_data.save_to_disk(output_dir)

    # Log summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Previously missing: {missing_count}")
    logger.info(f"Successfully filled: {success_count}")
    logger.info(f"Still missing (failed): {len(failed_indices)}")
    logger.info(f"Failed indices logged to: {log_filename}")
    logger.info(f"Updated dataset saved to: {output_dir}")

    # Write failed indices to a separate file for easy reference
    if failed_indices:
        failed_file = f"failed_indices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(failed_file, "w") as f:
            for idx in sorted(failed_indices):
                f.write(f"{idx}\n")
        logger.info(f"Failed indices also saved to: {failed_file}")


if __name__ == "__main__":
    import robocrys.condense.molecule

    # Patch MoleculeNamer's __init__ to default to no PubChem
    original_init = robocrys.condense.molecule.MoleculeNamer.__init__
    def patched_init(self, *args, **kwargs):
        kwargs['use_online_pubchem'] = False  # Force False, overriding any default or passed value
        original_init(self, *args, **kwargs)
    robocrys.condense.molecule.MoleculeNamer.__init__ = patched_init
    main()
