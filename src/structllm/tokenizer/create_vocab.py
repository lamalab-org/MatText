from itertools import product
from typing import List
import fire


def generate_vocab_file(vocab_file_path: str = 'periodic_table_vocab.txt') -> None:
    """
    Generate a vocabulary file with direction symbols as in SLICES, elements, and numbers.

    Args:
        vocab_file_path (str): Path to the vocabulary file. Default is 'periodic_table_vocab.txt'.
    """
    elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ar',
        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
        'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
        'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
        'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
        'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]

    symbols = ['o', '+', '-']
    # Generate all combinations of length 3
    combinations = [' '.join(combination) for combination in product(symbols, repeat=3)]
    # Numbers
    numbers = [str(i) for i in range(10)]

    all_tokens = combinations + elements + numbers

    # Write all tokens to the vocabulary file
    with open(vocab_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(all_tokens))

    print(f"Vocabulary file '{vocab_file_path}' created successfully.")


if __name__ == "__main__":
    fire.Fire(generate_vocab_file)
