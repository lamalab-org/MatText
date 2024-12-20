# MatText Representations

## Creating Text Representations for Crystal Structures

Converting structures into text representations can be done using our [`TextRep`](api.md#mattext.representations.TextRep) class

```python
from mattext.representations import TextRep
from pymatgen.core import Structure


# Load structure from a CIF file
from_file = "InCuS2_p1.cif"
structure = Structure.from_file(from_file, "cif")

# Initialize TextRep Class
text_rep = TextRep.from_input(structure)
```

### `get_requested_text_reps` Method

The [`get_requested_text_reps`](api.md#mattext.representations.TextRep.get_requested_text_reps) method retrieves the requested text representations of the crystal structure and returns them in a dictionary.

For example, to obtain the `cif_p1`, `slice`, `atoms_params`, `crystal_llm_rep`, and `zmatrix` representations, use the following code:

```python
text_rep = TextRep(structure)

# Define a list of requested representations
requested_reps = [
    "cif_p1",
    "slice",
    "atoms_params",
    "crystal_llm_rep",
    "zmatrix"
]

# Get the requested text representations
requested_text_reps = text_rep.get_requested_text_reps(requested_reps)
print(requested_text_reps)
```

??? success "output"

    ```bash
    {'cif_p1': "data_InCuS2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.52\n_cell_length_b   5.52\n_cell_length_c   6.8\n_cell_angle_alpha   113.96\n_cell_angle_beta   113.96\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   InCuS2\n_chemical_formula_sum   'In2 Cu2 S4'\n_cell_volume   169.53\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  In3+  3.0\n  Cu+  1.0\n  S2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Cu+  Cu4  1  0.25  0.75  0.5  1.0\n  Cu+  Cu5  1  0.0  0.0  0.0  1.0\n  In3+  In0  1  0.5  0.5  0.0  1.0\n  In3+  In1  1  0.75  0.25  0.5  1.0\n  S2-  S8  1  0.9  0.88  0.25  1.0\n  S2-  S9  1  0.62  0.1  0.75  1.0\n  S2-  S10  1  0.35  0.38  0.25  1.0\n  S2-  S11  1  0.12  0.65  0.75  1.0\n", 'slice': 'Cu Cu In In S S S S 0 7 o o o 0 4 - o o 0 6 o o o 0 5 o + o 1 4 - - o 1 5 - o - 1 7 o - - 1 6 o o o 2 6 o o o 2 7 o o - 2 5 o o - 2 4 o o o 3 5 o o o 3 6 o o o 3 4 o - o 3 7 + o o ', 'atoms_params': 'Cu Cu In In S S S S 5.52 5.52 6.8 113 113 90', 'crystal_llm_rep': '5.5 5.5 6.8\n113 113 90\nCu+\n0.25 0.75 0.50\nCu+\n0.00 0.00 0.00\nIn3+\n0.50 0.50 0.00\nIn3+\n0.75 0.25 0.50\nS2-\n0.90 0.87 0.25\nS2-\n0.62 0.10 0.75\nS2-\n0.35 0.37 0.25\nS2-\n0.12 0.65 0.75', 'zmatrix': 'Cu\nCu 1 3.9\nIn 2 3.9 1 60\nIn 1 3.9 2 60 3 -71\nS 3 2.5 4 90 1 93\nS 4 2.5 2 90 1 93\nS 1 2.3 2 32 3 -35\nS 1 2.3 7 111 6 -32'}

    ```

### Supported Text Representations

The [`TextRep`](api.md#mattext.representations.TextRep) class currently supports the following text representations:

- **SlICES** (`slices`): SLICE representation of the crystal structure.
- **Composition** (`composition`): Chemical composition in hill format.
- **CIF Symmetrized** (`cif_symmetrized`): Multi-line CIF representation with symmetrized structure and rounded float numbers.
- **CIF $P_1$** (`cif_p1`): Multi-line CIF representation with the conventional unit cell and rounded float numbers.
- **Crystal-text-LLM Representation** (`crystal_text_llm`): Representation following the format specified in the cited work.
- **Robocrystallographer Representation** (`robocrys_rep`): Representation generated by Robocrystallographer.
- **Atom Sequences** (`atom_sequences`): List of atoms inside the unit cell.
- **Atoms Squences++** (`atom_sequences_plusplus`): List of atoms with lattice parameters.
- **Z-Matrix** (`zmatrix`): Z-Matrix representation of the crystal structure.
- **Local-Env** (`local_env`):  List of Wyckoff label and SMILES separated by line breaks for each local environment.

For more details on each representation and how to obtain them, refer to the respective method documentation in the `TextRep` class.


## Transformations 


The `TextRep` class supports various transformations that can be applied to the input structure.



### Permute Structure 

The `permute_structure` transformation randomly permutes the order of atoms in a structure. 


```python
from mattext.representations import TextRep
from pymatgen.core.structure import Structure

structure_1 = Structure.from_file("N2_p1.cif", "cif")

transformations = [("permute_structure", {"seed": 42})]

text_rep = TextRep.from_input(structure_1, transformations)
text_representations_requested = ["atoms", "crystal_llm_rep"]
print("Permuted Pymatgen Structure:")
print(text_rep.structure)
print("Permuted Text Representations:")
print(text_rep.get_requested_text_reps(text_representations_requested))
```

??? success "output"

    ```bash
    Permuted Pymatgen Structure:
    Full Formula (N4)
    Reduced Formula: N2
    abc   :   5.605409   5.605409   5.605409
    angles:  90.000000  90.000000  90.000000
    pbc   :       True       True       True
    Sites (4)
    #  SP          a        b        c
    ---  ----  -------  -------  -------
    0  N0+   0.02321  0.02321  0.02321
    1  N0+   0.97679  0.52321  0.47679
    2  N0+   0.52321  0.47679  0.97679
    3  N0+   0.47679  0.97679  0.52321
    Permuted Text Representations:
    {'atoms': 'N N N N', 'crystal_llm_rep': '5.6 5.6 5.6\n90 90 90\nN0+\n0.02 0.02 0.02\nN0+\n0.98 0.52 0.48\nN0+\n0.52 0.48 0.98\nN0+\n0.48 0.98 0.52'}

    ```

### Translate Structure 

The `translate_structure` transformation randomly translates all atoms in a structure by a specified vector. This can simulate small displacements in the structure.

```python
transformations = [("translate_structure", {"seed": 42, "vector": [0.1, 0.1, 0.1]})]

text_rep = TextRep.from_input(structure_1, transformations)
text_representations_requested = ["crystal_llm_rep"]
print("Translated Crystal-text-LLM Representations:")
print(text_rep.get_requested_text_reps(text_representations_requested))
```
??? success "output"

    ```bash
    Translated Crystal-text-LLM Representations:
    {'crystal_llm_rep': '5.6 5.6 5.6\n90 90 90\nN0+\n0.58 0.08 0.62\nN0+\n0.08 0.62 0.58\nN0+\n0.12 0.12 0.12\nN0+\n0.62 0.58 0.08'}
    ```

### Augmenting Data

In principle, we can generate valid text representations with random transformations with physically meaningful parameters. Dummy example is shown below

```python
from mattext.representations import TextRep

# Define transformations
translation_vectors = np.random.uniform(low=0.1, high=0.5, size=(3, 3))
for vector in translation_vectors:
    transformations = [
        ("permute_structure", {"seed": 42}),
        ("perturb_structure", {"seed": 42, "max_distance": 0.1}),
        ("translate_structure", {"seed": 42, "vector": vector.tolist()})
    ]
    text_rep = TextRep.from_input(structure_2, transformations)
    text_representations_requested = ["crystal_llm_rep"]
    print("Translated Text Representations:")
    print(text_rep.get_requested_text_reps(text_representations_requested))

```

??? success "output"

    ```bash
    Translated Text Representations:{'crystal_llm_rep': '3.9 3.9 3.9\n90 90 90\nO2-\n0.76 0.98 0.41\nTi4+\n0.77 0.98 0.89\nO2-\n0.76 0.49 0.89\nO2-\n0.26 0.97 0.88\nSr2+\n0.25 0.47 0.38'}

    Translated Text Representations:{'crystal_llm_rep': '3.9 3.9 3.9\n90 90 90\nO2-\n0.85 0.66 0.18\nTi4+\n0.86 0.66 0.66\nO2-\n0.85 0.17 0.66\nO2-\n0.35 0.65 0.65\nSr2+\n0.34 0.15 0.15'}

    Translated Text Representations:{'crystal_llm_rep': '3.9 3.9 3.9\n90 90 90\nO2-\n0.63 0.94 0.35\nTi4+\n0.64 0.94 0.84\nO2-\n0.64 0.45 0.84\nO2-\n0.13 0.94 0.83\nSr2+\n0.12 0.43 0.33'}

    ```

???+ example 

    More examples are available as notebook in the repository.



The following transformations are available for transforming structures:

#### Randomly permute structure

[`permute_structure`](api.md#mattext.representations.transformations.TransformationCallback.permute_structure) randomly permutes the order of atoms in a structure.

#### Randomly Translate Single Atom
[`translate_single_atom`](api.md#mattext.representations.transformations.TransformationCallback.translate_single_atom) randomly translates one or more atoms in a structure.


#### Randomly Perturb Structure

[`perturb_structure`](api.md#mattext.representations.transformations.TransformationCallback.perturb_structure) randomly perturbs atoms in a structure.

#### Randomly Translate Structure

[`translate_structure`](api.md#mattext.representations.transformations.TransformationCallback.translate_structure) randomly translates the atoms in a structure.

???+ tip

    This transformation supports additional keyword arguments for fine-tuning the translation.



MatText leverages methods from pymatgen and support all the keyword arguments in `Structure.translate_sites` method.


All transformations utilize a common seed value for reproducibility and accept additional parameters for customization.

For more details on each transformation and its parameters, refer to the respective function documentation.



