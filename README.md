Struct_llm
==============================


LLMs applied on materials using slice representation


## Local Installation

We recommend that you create a virtual conda environment on your computer in which you install the dependencies for this exercise. To do so head over to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions there.


### Clone our repository

Clone this repository (you need `git` for this, if you get a `missing command` error for `git` you can install it with `sudo apt-get install git`)

```bash
git clone https://github.com/lamalab-org/structllm.git
cd structllm
```

### Install our package


```bash
pip install -e .
```


### configuring experiments and model

All config file can be found here, inside respective directories
```bash
cd structllm/src/structllm/conf/
```


### Steps for Pretraining / Finetuning

1. Modify the config for Pretraining/Finetuning experiments 

```bash
cd structllm/src/structllm/conf/model/bert.yaml

#modify the relevant params

exp_name: 
traindata: ["...csv"] # currently the model accept training/eval data in csv format
evaldata: 
root_path: 
```
additional training_arguments can also be defined in huggingface library format.


```bash
cd structllm/src/structllm/conf/config.yaml #main config file

defaults:
  - model: bert
  - tokenizer: bpe
  - matbench: matbench
  - logging: wandb

runs:

  - name: pretrain_run
    tasks: [pretrain]

#   - name: finetune_run
#     tasks: [finetune]

#   - name: matbench_predict
#     tasks: [matbench_predict]

```


Running the experiment

```bash
python main.py 
```
