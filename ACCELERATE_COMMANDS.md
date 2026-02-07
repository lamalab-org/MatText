# Accelerate Command Reference

## Your Original Command Converted

### torchrun (old)
```bash
python -m torch.distributed.run --nproc_per_node=4 main.py \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh,benchmark_robocrys_perovskites
```

### accelerate (new)
```bash
accelerate launch --config_file accelerate_config.yaml -m mattext.main \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh,benchmark_robocrys_perovskites
```

## Common Commands

### 1. Run All Benchmarks (Multirun)
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh,benchmark_robocrys_perovskites
```

Or use the convenience script:
```bash
./run_benchmark_multirun.sh
```

### 2. Run Single Benchmark
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh
```

### 3. Run Specific Benchmarks (Subset)
```bash
# Just GVRH and KVRH
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh
```

Or with script:
```bash
./run_benchmark_multirun.sh "benchmark_robocrys_gvrh,benchmark_robocrys_kvrh"
```

### 4. Run Pretraining
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=pretrain_robocrys
```

### 5. Run Fine-tuning
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=finetune \
  model=your_model_config
```

### 6. Run Classification
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=classification_example
```

### 7. Run with Different Number of GPUs

Edit `accelerate_config.yaml` and change:
```yaml
num_processes: 4  # Change this to 2, 8, etc.
```

Or use command line override:
```bash
accelerate launch \
  --num_processes=2 \
  --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh
```

### 8. Run on CPU (for testing)
```bash
accelerate launch \
  --cpu \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh
```

### 9. Run Single GPU (no DDP)
```bash
# Just use python directly
python -m mattext.main -cn=benchmark model=benchmark_robocrys_gvrh
```

### 10. Run with Custom Hydra Overrides
```bash
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh \
  model.finetune.training_arguments.num_train_epochs=100 \
  model.finetune.training_arguments.per_device_train_batch_size=64
```

## Hydra Multirun Examples

### Run Multiple Configs in Sequence
```bash
# Run 3 different benchmarks, one after another
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh,benchmark_robocrys_perovskites
```

### Sweep Over Hyperparameters
```bash
# Try different learning rates
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh \
  --multirun model.finetune.training_arguments.learning_rate=1e-5,5e-5,1e-4
```

### Combine Multiple Parameters
```bash
# Grid search over batch size and learning rate
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh \
  --multirun \
  model.finetune.training_arguments.per_device_train_batch_size=32,64 \
  model.finetune.training_arguments.learning_rate=1e-5,5e-5
```

## Debugging Commands

### 1. Dry Run (Check Config Without Training)
```bash
python -m mattext.main -cn=benchmark model=benchmark_robocrys_gvrh --cfg job
```

### 2. Print Resolved Config
```bash
python -m mattext.main -cn=benchmark model=benchmark_robocrys_gvrh --cfg all
```

### 3. Run with Verbose Logging
```bash
ACCELERATE_LOG_LEVEL=debug \
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  model=benchmark_robocrys_gvrh
```

### 4. Check Accelerate Environment
```bash
accelerate env
```

### 5. Test Accelerate Setup
```bash
accelerate test --config_file accelerate_config.yaml
```

## Environment Variables

### Useful Environment Variables
```bash
# Set before running

# NCCL debugging (if you get timeout errors)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Increase NCCL timeout (default is 30 minutes)
export NCCL_TIMEOUT=3600000  # 1 hour in ms

# Disable NCCL P2P (if you have issues)
export NCCL_P2P_DISABLE=1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Wandb settings
export WANDB_PROJECT=your_project
export WANDB_ENTITY=your_entity
```

### Example with Environment Variables
```bash
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file accelerate_config.yaml \
  -m mattext.main \
  -cn=benchmark \
  --multirun model=benchmark_robocrys_gvrh,benchmark_robocrys_kvrh
```

## Quick Comparison Table

| Task | torchrun | accelerate |
|------|----------|------------|
| **Single run** | `python -m torch.distributed.run --nproc_per_node=4 main.py` | `accelerate launch --config_file accelerate_config.yaml -m mattext.main` |
| **Multirun** | `... main.py --multirun ...` | `... -m mattext.main --multirun ...` |
| **Specify GPUs** | `--nproc_per_node=N` | `num_processes: N` in config |
| **CPU mode** | Not easy | `accelerate launch --cpu` |
| **Mixed precision** | Manual setup | `--mixed_precision=fp16` |
| **Config file** | N/A | `--config_file path/to/config.yaml` |

## Tips

1. **Always use `-m mattext.main`** instead of `main.py` to maintain Python module structure

2. **Config file is recommended** - Easier to manage than command-line args

3. **Hydra args come AFTER accelerate args**:
   ```bash
   accelerate launch [ACCELERATE_ARGS] -m mattext.main [HYDRA_ARGS]
   ```

4. **Use convenience scripts** for common tasks (like `run_benchmark_multirun.sh`)

5. **Test single GPU first**:
   ```bash
   python -m mattext.main -cn=benchmark model=benchmark_robocrys_gvrh
   ```
   Then scale up to multi-GPU:
   ```bash
   accelerate launch --config_file accelerate_config.yaml -m mattext.main -cn=benchmark model=benchmark_robocrys_gvrh
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'mattext'"
Make sure you're in the right directory or PYTHONPATH is set:
```bash
cd /Users/nalampara/n0w0f/dev/MatText
export PYTHONPATH=/Users/nalampara/n0w0f/dev/MatText/src:$PYTHONPATH
```

### "Could not locate best model"
Already fixed! The code now disables `load_best_model_at_end` in DDP mode.

### Hanging at end of training
Already fixed! Removed manual barriers, let Trainer handle everything.

### Multiple checkpoint folders
This is expected for multirun - each run gets its own folder with timestamp.
