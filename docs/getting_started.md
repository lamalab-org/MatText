# Installation


We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management.

The most recent code and data can be installed directly from GitHub with:


```shell
$ uv pip install git+https://github.com/lamalab-org/mattext.git
```

To install in development mode, use the following:

```shell
$ git clone git+https://github.com/lamalab-org/mattext.git
$ cd mattext
$ uv pip install -e .
```


If you want to use the Local Env representation, you will also need to install OpenBabel. You can install it via conda/mamba:

```bash
conda install openbabel -c conda-forge
```

or on Ubuntu/Debian:

```bash
sudo apt-get install openbabel
```
