
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


### Clone our repository

Clone this repository (you need `git` for this, if you get a `missing command` error for `git` you can install it with `sudo apt-get install git`)

```bash
git clone https://github.com/lamalab-org/mattext.git
cd mattext
```

### Install our package


```bash
pip install -e .
```


### Configuring experiments and model

All config file can be found here, inside respective directories
```bash
cd /src/mattext/conf/
```
The main configuration for the run is in config.yaml and other configs are grouped in respective folders. An example directory structure of configs is below.
```bash
├── conf
│   ├── config.yaml
│   ├── pretrain30k
│   │   ├── cifp1.yaml
│   │   ├── cifsymmetrized.yaml
│   │   ├── composition.yaml
│   │   ├── crystal_llm.yaml
│   │   └── slice.yaml
│   ├── finetune30k
│   │   ├── cifp1.yaml
│   │   ├── cifsymmetrized.yaml
│   │   ├── composition.yaml
│   │   ├── crystal_llm.yaml
│   │   └── slice.yaml
│   └── model
│       ├── finetune_template_dielectric.yaml
│       ├── finetune_template_gvrh.yaml
│       ├── finetune_template.yaml
│       └── pretrain_template.yaml
├── main.py
├── models
```

>Configs in the group __models__ are the _base config templates_ and other groups (__pretrain30K / finetune30K__ ) are experiment groups. Experiment group extends from the base template and add/override additional configs. read more about extending configs [here](https://hydra.cc/docs/patterns/extending_configs/).


configs inside __pretrain30k__ extends the `base pretrain template` and override the   `representation name`, `batch size`, `etc` in the base pretrain_template.
```yaml
# @package _global_
model:
  logging:
    wandb_project: pt_30k_test
  
  representation: cif_p1
  pretrain:
    name: pt_30k_test
    context_length:  1024
    training_arguments:
      per_device_train_batch_size: 32
    path:
      data_root_path: </path/to/dataset>
      
```

### Pretraining and Finetuning experiments

You can start a multirun job parallely with the below cli script. 

```bash
python /path/to/main.py --multirun model=pretrain_template ++hydra.launcher.gres=gpu:1 +pretrain30k=cifp1,cifsym,composition,crystal_llm,slice

```
We use HF Trainer and hence by default it support DP but for DDP support 
```bash
python -m torch.distributed.run --nproc_per_node=4  /path/to/main.py --multirun model=pretrain_template ++hydra.launcher.gres=gpu:1 +pretrain30k=cifp1,cifsym,composition,crystal_llm,slice

```

Here `model=pretrain_template` select pretrain_template as the base config and override/extend it with `+pretrain30k=cifp1`. This would essentially start pretraining with cifp1 representation for the dataset-30K

For finetuning select `model=finetune_template_<property>` as the base template

Note `+pretrain30k=cifp1,cifsym,composition,crystal_llm,slice` will launch 5 jobs parallely each of them with pretrain_template as the base config and corresponding experiment template extending them.

>By default we use [hydra submitit slurm launcher](https://hydra.cc/docs/plugins/submitit_launcher/). you can override it from cli / or change it in the main config file. For kubernetes based infrastructures [hydra submitit local launcher](https://hydra.cc/docs/plugins/submitit_launcher/) is ideal for parallel jobs. Or you can use the default hydra multirun launcher, which will run jobs sequentially.
You can configure the launcher configurations in main config file.

### Adding new experiments
New experiments can be easily added with the following step. 

1. Create an experiment config group inside `conf/` . Make a new directory and add experiment template inside it. 
2. Add / Edit the configs that you want for the new experiments. eg: override the pretrain checkpoints to new pretrained checkpoint
3. Launch runs similarly but now with new experiment group 

```bash
python main.py --multirun model=pretrain_template ++hydra.launcher.gres=gpu:1 +<new_exp_group>=<new_exp_template_1>,<new_exp_template_2>, ..

```
<!-- 
### Run with Docker 

Build Docker image

```bash 
cd docker
docker build --build-arg GITHUB_PAT=<your_token> -t mattext .
```

```bash
docker exec -it --gpus all -v /path/to/host/mattext:/app/mattext/ mattext python main.py hydra/launcher=submitit_local --multirun +pretrain30k=cifp1,cifsym,composition,crystal_llm,slice

``` -->