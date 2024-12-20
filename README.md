
<p align="center">
  <img src="https://github.com/lamalab-org/mattext/raw/main/docs/static/logo.png" height="150">
</p>


<h1 align="center">
  MatText: A framework for text-based materials modeling
</h1>

<p align="center">
    <a href="https://github.com/lamalab-org/mattext/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/lamalab-org/MatText/actions/workflows/tests.yml/badge.svg" />
    </a>
    <a href="https://lamalab-org.github.io/MatText/">
        <img alt="Docs"src="https://img.shields.io/badge/docs-GitHub_Pages-blue" / >
    </a>
</p>


MatText is a framework for text-based materials modeling. It supports 

- conversion of crystal structures in to text representations 
- transformations of crystal structures for sensitivity analyses
- decoding of text representations to crystal structures
- tokenization of text-representation of crystal structures
- pre-training, finetuning and testing of language models on text-representations of crystal structures 
- analysis of language models trained on text-representations of crystal structures



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

If you want to use the Local Env representation, you will also need to install OpenBabel, e.g. using 

```bash 
conda install openbabel -c conda-forge
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
    "slices",
    "atom_sequences",
    "atom_sequences_plusplus",
    "crystal_text_llm",
    "zmatrix"
]

# Get the requested text representations
requested_text_reps = text_rep.get_requested_text_reps(requested_reps)
```


### Pretrain

```bash
python main.py -cn=pretrain model=pretrain_example +model.representation=composition +model.dataset_type=pretrain30k +model.context_length=32
```

### Running a benchmark 

```bash
python main.py -cn=benchmark model=benchmark_example +model.dataset_type=filtered +model.representation=composition +model.dataset=perovskites +model.checkpoint=path/to/checkpoint  
```

The `+` symbol before a configuration key indicates that you are adding a new key-value pair to the configuration. This is useful when you want to specify parameters that are not part of the default configuration.

To override the existing default configuration, use `++`, for e.g., `++model.pretrain.training_arguments.per_device_train_batch_size=32`. Refer to the [docs](https://lamalab-org.github.io/MatText/) for more examples and advanced ways to use the configs with config groups.


### Using data 

The MatText datasets can be easily obtained from [HuggingFace](https://huggingface.co/datasets/n0w0f/MatText), for example

```
from datasets import load_dataset

dataset = load_dataset("n0w0f/MatText", "pretrain300k")
```


## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/lamalab-org/xtal2txt/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### Citation 

If you use MatText in your work, please cite 

```
@misc{alampara2024mattextlanguagemodelsneed,
      title={MatText: Do Language Models Need More than Text & Scale for Materials Modeling?}, 
      author={Nawaf Alampara and Santiago Miret and Kevin Maik Jablonka},
      year={2024},
      eprint={2406.17295},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
      url={https://arxiv.org/abs/2406.17295}, 
}
```


### ⚖️ License

The code in this package is licensed under the MIT License.

### 💰 Funding

This project has been supported by the [Carl Zeiss Foundation](https://www.carl-zeiss-stiftung.de/en/) as well as Intel and Merck.