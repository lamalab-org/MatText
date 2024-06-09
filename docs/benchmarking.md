# Modeling and Benchmarking 

MatText have  pipelines for seamless pretraining([`pretrain`](api.md#mattext.models.pretrain)) and benchmarking ([`benchmark`](api.md#mattext.models.benchmark)) with finetuning ([`finetune`](api.md#mattext.models.finetune)) on multiple MatText representations. We use the Hydra framework to dynamically create hierarchical configurations based on the pipeline and representations that we want to use.


### Pretraining on Single MatText Representation

```bash
python main.py -cn=pretrain model=pretrain_example +model.representation=composition +model.dataset_type=pretrain30k +model.context_length=32
```

Here, `model=pretrain_example` would select `pretrain_example` as the base config for pretrain run `-cn=pretrain`.

All config file can be found here, inside respective directories.
```bash
cd /conf/
```
Base configs can be found at `/conf/model`

`+model.representation` can be any MatText representation. The pipeline would use data represented using that particular representation for training.
`+model.dataset_type` can be one of the MatText pretraining datasets.
`+model.context_length` would define the context length.

>Note: Use meaningful context length according to the representation to avoid truncation.

The `+` symbol before a configuration key indicates that you are adding a new key-value pair to the configuration. This is useful when you want to specify parameters that are not part of the default configuration.


In order to override the existing default configuration from CLI, use `++`, for e.g, `++model.pretrain.training_arguments.per_device_train_batch_size=32`. 


For advanced usage (changing architecture, training arguments, or modeling parameters), it would be easier to make the changes in the base config file which is `/conf/model/pretrain_example`, than having to override parameters with lengthy CLI commands!



### Running Benchmark on a Single MatText Representation

```bash
python main.py -cn=benchmark model=benchmark_example +model.dataset_type=filtered +model.representation=composition +model.dataset=perovskites +model.checkpoint=path/to/checkpoint  
```


Here for the benchmarking pipeline(`-cn=benchmark`) base config is `benchmark_example.yaml`. 
You can define the parameters for the experiment hence at `\conf\model\benchmark_example.yaml`.

> Here +model.dataset_type=filtered would select the type of benchmark. It can be `filtered` (avoid having truncated structure in train and test set, Only relatively small structures are present here, but this would also mean having less number of sampels to train on ) or `matbench` (complete dataset, there are few big structures , which would be trunated if the context length for modelling is less than `2048`).


> `+model.dataset_type=filtered` would produce the report compatible with matbench leaderboard.


Benchmark report is saved to the path defined in the base config. By default to `"${hydra:runtime.cwd}/../../results/${now:%Y-%m-%d}/${now:%H-%M-%S}/$`

### Pretraining or Benchmarking Multiple MatText Representation

The easiest way to model multiple representation in one run would be by using `config-groups` and multirun

```bash
python main.py --multirun -cn=benchmark model=benchmark_example +model.dataset_type=matbench +group-test=slices,composition
```

Here, we create a config group (directory with config files for different representations) at `/conf/<config group name>`

In the above example, we have two config files (`slices.yaml, composition.yaml`) inside the config group `group-test.`
with `--multirun` enabled we can launch the pipeline parallely or sequentially (by default) for the representations, Here two but representations, but once can add more.

The `child config` (config inside the `config group` ) would override or add the key value pair on top of the  `base config` (here `benchmark_example`).

>configs inside __group-test__ extends the `benchmark_example` and override the   `representation name`, `batch size`, `etc` in the `base config`.

example `child config`
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

Read more about extending configs [here](https://hydra.cc/docs/patterns/extending_configs/).

### Configuring experiments and model


The main configuration for the run is in `config.yaml` and other configs are grouped in respective folders. An example directory structure of configs is below.
```bash
├── conf
│   ├── pretrain.yaml
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
│       ├── finetune_template.yaml
│       └── pretrain_template.yaml
├── main.py
├── models
```




We use the HF Trainer, and hence, by default, it supports DP. For DDP support you can run
```bash
python -m torch.distributed.run --nproc_per_node=4  /path/to/main.py --multirun model=pretrain_template +pretrain30k=cifp1,cifsym,composition,crystal_llm,slice

```

Here `model=pretrain_template` selects `pretrain_template` as the base config and override/extend it with `+pretrain30k=cifp1`. This would essentially start pretraining with cifp1 representation for the dataset-30K


Note `+pretrain30k=cifp1,cifsym,composition,crystal_llm,slice` will launch 5 jobs parallelly, each of them with `pretrain_template` as the base config and corresponding experiment template extending them.

>For launching runs parallely checkout [hydra submitit slurm launcher](https://hydra.cc/docs/plugins/submitit_launcher/). you can override it from cli / or change it in the main config file. For kubernetes based infrastructures [hydra submitit local launcher](https://hydra.cc/docs/plugins/submitit_launcher/) is ideal for parallel jobs. Or you can use the default hydra multirun launcher, which will run jobs sequentially.
You can configure the launcher configurations in the main config file.

### Adding new experiments
New experiments can be easily added with the following step. 

1. Create an experiment config group inside `conf/` . Make a new directory and add experiment template inside it. 
2. Add / Edit the configs that you want for the new experiments. eg: override the pretrain checkpoints to new pretrained checkpoint
3. Launch runs similarly but now with new experiment group 

```bash
python main.py --multirun model=pretrain_template ++hydra.launcher.gres=gpu:1 +<new_exp_group>=<new_exp_template_1>,<new_exp_template_2>, ..

```

## Running a benchmark 

```bash
python main.py -cn=benchmark model=benchmark_example +model.dataset_type=filtered +model.representation=composition +model.dataset=perovskites +model.checkpoint=path/to/checkpoint  
```

## Finetuning LLM 

```bash
python main.py -cn=llm_sft model=llama_example +model.representation=composition +model.dataset_type=filtered +model.dataset=perovskites  
```

The `+` symbol before a configuration key indicates that you are adding a new key-value pair to the configuration. This is useful when you want to specify parameters that are not part of the default configuration.

To override the existing default configuration, use `++`, for e.g., `++model.pretrain.training_arguments.per_device_train_batch_size=32`. Refer to the [docs](https://lamalab-org.github.io/MatText/) for more examples and advanced ways to use the configs with config groups.

>Define the number of folds for n-fold cross validation in the config or through cli. For Matbench benchmarks however number of folds  should be 5. Default value for all experiments are set to 5.

## Using data 

The MatText datasets can be easily obtained from [HuggingFace](https://huggingface.co/datasets/n0w0f/MatText), for example

```python
from datasets import load_dataset

dataset = load_dataset("n0w0f/MatText", "pretrain300k")
```

## Using Pretrained MatText Models 

The pretrained MatText models can be easily loaded from [HuggingFace](https://huggingface.co/collections/n0w0f/mattext-665fe18e5eec38c2148ccf7a), for example

```python
from transformers import AutoModel

model = AutoModel.from_pretrained ("n0w0f/MatText−cifp1−2m")
```
>This would need the code to pull the model from HF HUB and require internet.


## Training Other Language Models Using Mattext Pipeline

You can pretrain models that are compatible with AutoModelForMaskedLM from Hugging Face using our framework. The path to the model or the model name (model name in Hugging Face) can be defined using the +model.pretrain.model_name_or_path argument. Here’s an example:

```
python main.py -cn=pretrain model=pretrain_other_models +model.representation=slices +model.dataset_type=pretrain30k +model.context_length=32 +model.pretrain.model_name_or_path="FacebookAI/roberta-base"
```

For better manageability, you can define the model name and configuration in the base config file. This way, you don't need to specify the model and model configs in the CLI.

```yaml
pretrain:
  name: test-pretrain
  exp_name: "${model.representation}_${model.pretrain.name}"
  model_name_or_path: "FacebookAI/roberta-base" 
  dataset_name: "${model.dataset_type}"
  context_length: "${model.context_length}"

  model_config:
    hidden_size: 768
    num_hidden_layers: 4
    num_attention_heads: 8
    is_decoder: False
    add_cross_attention: False
    max_position_embeddings: 768
```
