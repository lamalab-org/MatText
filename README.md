
<p align="center">
  <img src="https://github.com/lamalab-org/mattext/raw/main/docs/static/logo.png" height="150">
</p>


<h1 align="center">
  MatText: A framework for text-based materials modeling
</h1>

<p align="center">
    <a href="https://github.com/lamalab-org/mattext/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/lamalab-org/mattext/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/mattext">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/mattext" />
    </a>
    <a href="https://pypi.org/project/mattext">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/mattext" />
    </a>
    <a href="https://github.com/lamalab-org/mattext/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/mattext" />
    </a>
</p>



## Local Installation

We recommend that you create a virtual conda environment on your computer in which you install the dependencies for this package. To do so head over to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions there.


<!-- ### Install latest release

```bash
pip install mattext
``` -->

### Install development version

Clone this repository (you need `git` for this, if you get a `missing command` error for `git` you can install it with `sudo apt-get install git`)

```bash
git clone https://github.com/lamalab-org/mattext.git
cd mattext
```

```bash
pip install -e .
```


## Getting started

### Converting crystals into text

```python
from mattext.representations import TextRep
from pymatgen.core import Structure


# Load structure from a CIF file
from_file = "InCuS2_p1.cif"
structure = Structure.from_file(from_file, "cif")

# Initialize TextRep Class
text_rep = TextRep.from_input(structure)

requested_reps = [
    "cif_p1",
    "slice",
    "atoms_params",
    "crystal_llm_rep",
    "zmatrix"
]

# Get the requested text representations
requested_text_reps = text_rep.get_requested_text_reps(requested_reps)
```

### Running a benchmark 

```bash
python main.py \
    model=finetune_filtered \
    +model.representation=composition \ # can be any MatText Representation
    +model.dataset=gvrh \ # can be any MatText Benchmark property 
    +model.finetune.path.pretrained_checkpoint=composition_30k_ft/checkpoint-1000 #path to pretrain checkpoint
``