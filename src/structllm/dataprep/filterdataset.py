"""
filterdataset.py

This module provides functions for processing and filtering datasets based on the length of their tokenized representations. 
The main function, `process_datasets`, takes as input a directory of JSON datasets and a dictionary mapping representations to their context lengths. It processes each dataset, filtering out entries whose tokenized representation exceeds the specified context length. The filtered datasets are saved to a specified output directory, along with a report of the entries that were removed.
The module uses a map of tokenizers (`_TOKENIZER_MAP`), each corresponding to a different type of representation. The tokenization process is handled by the `_tokenize_without_truncation` function, which tokenizes texts without truncation based on the specified representation type.
The `filter_dataset_with_context` function is used to filter a dataset based on the context lengths for each representation. It applies a filter function to each entry in the dataset, removing entries whose tokenized representation exceeds the context length.
This module is designed to be run as a script using the Fire library. When run as a script, it calls the `main` function, which processes datasets in an input directory and saves the filtered datasets and reports to an output directory.
"""


import json
import os
from typing import Any, Dict, List, Tuple, Type

import fire
from datasets import DatasetDict, load_dataset
from xtal2txt.tokenizer import (
    CifTokenizer,
    CompositionTokenizer,
    CrysllmTokenizer,
    RobocrysTokenizer,
    SliceTokenizer,
)

# Mapping of representation types to their corresponding tokenizer classes
_TOKENIZER_MAP: Dict[str, Type] = {
    "slice": SliceTokenizer,
    "composition": CompositionTokenizer,
    "cif_symmetrized": CifTokenizer,
    "cif_p1": CifTokenizer,
    "cif_bonding": CifTokenizer,
    "crystal_llm_rep": CrysllmTokenizer,
    "robocrys_rep": RobocrysTokenizer,
    "wycoff_rep": None,
}

# Default special tokens for tokenization
_DEFAULT_SPECIAL_TOKENS: Dict[str, str] = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
}

def _tokenize_without_truncation(texts: Dict[str, Any], representation:str) -> Dict[str, Any]:
    """
    Tokenizes the given texts without truncation.

    Args:
        texts: The texts to tokenize.
        representation: The type of representation to use for tokenization.

    Returns:
        The tokenized texts.
    """
    tokenizer_class = _TOKENIZER_MAP[representation]
    tokenizer = tokenizer_class(model_max_length=512, truncation=False, padding=False)
    tokenizer.add_special_tokens(special_tokens=_DEFAULT_SPECIAL_TOKENS)
    return tokenizer(texts['representation'], truncation=False, padding=False)

def filter_dataset_with_context(dataset, representation_context: Dict[str, int]) -> Tuple[DatasetDict, List[Dict[str, Any]]]:
    """
    Filters the given dataset based on the context lengths for each representation.

    Args:
        dataset: The dataset to filter.
        representation_context: A dictionary mapping representations to their context lengths.

    Returns:
        A tuple containing the filtered dataset and a list of removed entries.
    """
    removed_entries = []
    for representation, context_length in representation_context.items():
        def filter_func(example):
            tokens = _tokenize_without_truncation({'representation': example[representation]}, representation)
            if len(tokens['input_ids']) > context_length:
                removed_entries.append({'mbid': example['mbid'], 'representation': representation, 'length': len(tokens['input_ids'])})
                return False
            return True
        dataset = dataset.filter(filter_func)
    return dataset, removed_entries

def process_datasets(input_dir: str, output_dir: str, representation_context: Dict[str, int]) -> None:
    """
    Processes the datasets in the given input directory and saves the filtered datasets and reports to the output directory.

    Args:
        input_dir: The directory containing the input datasets.
        output_dir: The directory to save the filtered datasets and reports to.
        representation_context: A dictionary mapping representations to their context lengths.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            print(f"Processing {filename}...")
            dataset = load_dataset("json", data_files=os.path.join(input_dir, filename))
            filtered_dataset, report = filter_dataset_with_context(dataset, representation_context)
            filtered_dataset['train'].to_json(os.path.join(output_dir, filename))
            with open(os.path.join(output_dir, filename + '_report.json'), 'w') as f:
                json.dump(report, f)
            print(f"Saved filtered dataset and report for {filename}.")



def main(input_dir: str, output_dir: str) -> None:
    """
    Main function to process datasets.

    Args:
        input_dir: The directory containing the input datasets.
        output_dir: The directory to save the filtered datasets and reports to.
    """
    representation_context = {
        "cif_p1" : 1024,
        "cif_symmetrized" : 1024,
        "slice" : 512,
        "crystal_llm_rep" : 512,
        "composition" : 32
    }
    process_datasets(input_dir, output_dir, representation_context)

if __name__ == '__main__':
    fire.Fire(main)
