# Installation

**Requirements:**
- Python 3.10 or 3.11 (tested and supported)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management.

The most recent code and data can be installed directly from GitHub with:

```shell
$ uv pip install git+https://github.com/lamalab-org/mattext.git
```

To install in development mode, use the following:

```shell
$ git clone https://github.com/lamalab-org/mattext.git
$ cd mattext
$ uv venv --python 3.10
$ source .venv/bin/activate  # On Windows: .venv\Scripts\activate
$ uv pip install -e ".[dev]"
```

Install pre-commit hooks (optional, for development):

```shell
$ pre-commit install
```


If you want to use the Local Env representation, you will also need to install OpenBabel. You can install it via conda/mamba:

```bash
conda install openbabel -c conda-forge
```

or on Ubuntu/Debian:

```bash
sudo apt-get install openbabel
```
