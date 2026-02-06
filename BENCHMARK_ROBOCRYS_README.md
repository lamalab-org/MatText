# Benchmarking n0w0f/MatText-robocrys_30k

This guide explains how to benchmark the `n0w0f/MatText-robocrys_30k` checkpoint on three datasets from `jablonkagroup/MatText`.

## Created Config Files

Three benchmark configuration files have been created in `conf/model/`:

1. **benchmark_robocrys_gvrh.yaml** - GVRH dataset benchmark
2. **benchmark_robocrys_kvrh.yaml** - KVRH dataset benchmark
3. **benchmark_robocrys_perovskites.yaml** - Perovskites dataset benchmark

## Configuration Details

All configs use:
- **Checkpoint**: `n0w0f/MatText-robocrys_30k` (from HuggingFace Hub)
- **Representation**: `robocrys_rep`
- **Data Repository**: `jablonkagroup/MatText`
- **Folds**: 5 (fold_0 through fold_4)
- **Dataset Type**: filtered

### Training Parameters
- Context length: 512
- Batch size: 64
- Learning rate: 2e-4
- Max epochs: 50
- Early stopping: Enabled (patience=10)

### Datasets
Each config will automatically load the correct train/test splits from HuggingFace:

| Config | Train Dataset | Test Dataset |
|--------|---------------|--------------|
| gvrh | mattext_gvrh-train-filtered | mattext_gvrh-test-filtered |
| kvrh | mattext_kvrh-train-filtered | mattext_kvrh-test-filtered |
| perovskites | mattext_perovskites-train-filtered | mattext_perovskites-test-filtered |

## How to Launch

### Option 1: Run All Benchmarks (Sequential)

Use the provided shell script to run all three benchmarks one after another:

```bash
# From the MatText root directory
./scripts/run_robocrys_benchmarks.sh
```

### Option 2: Run Individual Benchmarks

Run each benchmark separately from the `src/mattext` directory:

```bash
cd src/mattext

# GVRH
python main.py -cn=benchmark model=benchmark_robocrys_gvrh

# KVRH
python main.py -cn=benchmark model=benchmark_robocrys_kvrh

# Perovskites
python main.py -cn=benchmark model=benchmark_robocrys_perovskites
```

### Option 3: Run Multiple Benchmarks in Parallel

If you have multiple GPUs and want to run benchmarks in parallel:

```bash
cd src/mattext

# Terminal 1
CUDA_VISIBLE_DEVICES=0 python main.py -cn=benchmark model=benchmark_robocrys_gvrh

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python main.py -cn=benchmark model=benchmark_robocrys_kvrh

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python main.py -cn=benchmark model=benchmark_robocrys_perovskites
```

### Option 4: Use Hydra Multirun

Run all three benchmarks using Hydra's multirun feature (will run sequentially by default):

```bash
cd src/mattext
python main.py --multirun -cn=benchmark \
    model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh,benchmark_robocrys_perovskites
```

## Configuration Overrides

You can override any config parameter from the command line:

```bash
# Change batch size
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.finetune.training_arguments.per_device_train_batch_size=32

# Change learning rate
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    ++model.finetune.training_arguments.learning_rate=1e-4

# Change context length
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.finetune.context_length=256

# Change number of epochs
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.finetune.training_arguments.num_train_epochs=100

# Use a different checkpoint
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.checkpoint=path/to/your/checkpoint
```

## What Happens During Benchmarking

The benchmark process includes:

1. **Fine-tuning** (5 folds):
   - Loads the pretrained checkpoint from HuggingFace
   - Fine-tunes on each of the 5 training folds
   - Saves checkpoints to `results/<timestamp>/<model_name>/checkpoints/`
   - Logs metrics to Weights & Biases

2. **Inference** (5 folds):
   - Loads each fine-tuned checkpoint
   - Runs inference on corresponding test folds
   - Saves predictions to `predictions/`

3. **Results**:
   - Final benchmark scores are saved to the root results path
   - All runs are tracked in W&B projects:
     - `benchmark-robocrys-gvrh`
     - `benchmark-robocrys-kvrh`
     - `benchmark-robocrys-perovskites`

## Output Structure

```
results/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        └── robocrys-{dataset}/
            ├── checkpoints/
            │   ├── finetuned_train_robocrys_rep_mattext_{dataset}-train-filtered_0/
            │   ├── finetuned_train_robocrys_rep_mattext_{dataset}-train-filtered_1/
            │   ├── ... (one per fold)
            │   └── finetuned_train_robocrys_rep_mattext_{dataset}-train-filtered_4/
            └── logs/
                └── ... (tensorboard logs)

predictions/
└── checkpoints/
    ├── inferencetest_robocrys_rep_mattext_{dataset}-train-filtered_0/
    ├── ... (one per fold)
    └── inferencetest_robocrys_rep_mattext_{dataset}-train-filtered_4/
```

## Requirements

Before running, ensure you have:

1. **Environment**: MatText conda/pip environment activated
2. **Weights & Biases**:
   ```bash
   export WANDB_API_KEY="your_api_key"
   # Or run: wandb login
   ```
3. **GPU**: CUDA-enabled GPU (recommended)
4. **HuggingFace Access**: Ensure you can access both repositories:
   - `n0w0f/MatText-robocrys_30k` (checkpoint)
   - `jablonkagroup/MatText` (datasets)

## Troubleshooting

### Out of Memory
If you encounter OOM errors, reduce the batch size:
```bash
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.finetune.training_arguments.per_device_train_batch_size=32
```

### Context Length Too Long
Robocrys representations can be long. If needed, reduce context length:
```bash
python main.py -cn=benchmark model=benchmark_robocrys_gvrh \
    +model.finetune.context_length=256
```

### Dataset Access Issues
Make sure you can access the dataset:
```python
from datasets import load_dataset
ds = load_dataset("jablonkagroup/MatText", "mattext_gvrh-train-filtered")
```

### W&B Login
If W&B isn't configured:
```bash
wandb login
# Or disable W&B:
export WANDB_MODE=disabled
```

## Monitoring

Track your runs in real-time on Weights & Biases:
- https://wandb.ai/your-username/benchmark-robocrys-gvrh
- https://wandb.ai/your-username/benchmark-robocrys-kvrh
- https://wandb.ai/your-username/benchmark-robocrys-perovskites

## Notes

- Each benchmark runs 5-fold cross-validation
- The checkpoint will be automatically downloaded from HuggingFace on first run
- Datasets will be cached locally after first download
- Training progress is saved every epoch
- Best model (based on validation loss) is automatically loaded at the end
