# Simple DDP Training Guide

## What I Changed (Summary)

### Removed:
- ❌ All manual `torch.distributed.barrier()` calls
- ❌ All debug print statements
- ❌ Manual model unwrapping (`model.module`)
- ❌ Complex synchronization logic
- ❌ All manual barriers

### Kept:
- ✅ `is_main` check for wandb operations
- ✅ Device placement (`model.to(local_rank)`)
- ✅ Disabling `load_best_model_at_end` in DDP mode
- ✅ Using `trainer.save_model()` instead of `model.save_pretrained()`

### Key Principle:
**Trainer handles ALL DDP synchronization automatically.** We just need to:
1. Not manually wrap models
2. Only do wandb/logging on rank 0
3. Let Trainer manage checkpoints

## How to Run

### Option 1: Using Accelerate (Recommended)

```bash
# Run with accelerate
accelerate launch --config_file accelerate_config.yaml -m mattext.main
```

### Option 2: Using torchrun

```bash
# Run with torchrun
torchrun --nproc_per_node=4 -m mattext.main
```

### Option 3: Single GPU (for testing)

```bash
# Regular single GPU run
python -m mattext.main
```

## What Trainer Does Automatically

When you call `trainer.train()`, it automatically:

1. **Detects DDP environment** - Checks if `torch.distributed.is_initialized()`
2. **Wraps model** - Applies `DistributedDataParallel` internally
3. **Synchronizes gradients** - Handles all-reduce operations
4. **Adds barriers** - Before evaluation, before checkpoint loading
5. **Saves checkpoints** - Only on rank 0
6. **Distributes data** - Uses `DistributedSampler` automatically

## Expected Behavior

### During Training:
```
[Rank 0] WARNING: Disabling load_best_model_at_end in DDP mode
[Rank 1] WARNING: Disabling load_best_model_at_end in DDP mode
[Rank 2] WARNING: Disabling load_best_model_at_end in DDP mode
[Rank 3] WARNING: Disabling load_best_model_at_end in DDP mode

Training: 100%|████████| 700/700 [03:00<00:00]
{'train_runtime': '180.5', 'train_loss': '0.15', ...}

[Only rank 0 saves checkpoints]
Writing model shards: 100%|████████| 1/1
```

### What You Should See:
- ✅ Only ONE set of checkpoint files (saved by rank 0)
- ✅ Only ONE wandb run (logged by rank 0)
- ✅ No hanging at end of training
- ✅ Clean completion

### What You Should NOT See:
- ❌ "Could not locate the best model" errors
- ❌ Multiple checkpoint directories with same name
- ❌ Hanging/freezing at end of training
- ❌ NCCL timeout errors

## Troubleshooting

### Still Hanging?

Check your config YAML:
```yaml
training_arguments:
  load_best_model_at_end: true  # ← Set to false for DDP
  save_on_each_node: false  # ← Keep as false
```

### Multiple Checkpoint Folders?

This is normal IF you're running multiple experiments. Each experiment gets:
- Its own timestamp folder
- Its own checkpoint subdirectories

But you should NOT see duplicate files within the same experiment.

### NCCL Timeouts?

Usually caused by:
1. One rank crashing while others wait
2. File I/O on all ranks (should be rank 0 only)
3. Mismatched collective operations

The simplified code fixes all these issues.

## Files Changed

1. **potential.py** - Removed manual barriers, simplified saving
2. **finetune.py** - Removed manual barriers, use trainer.save_model()
3. **llama.py** - Removed manual barriers
4. **llama_sft.py** - Removed manual barriers, cleaned up inference
5. **benchmark.py** - Removed debug prints, kept only essential logic

## Key Code Pattern

```python
# Simple, clean DDP pattern
def finetune(self):
    # Setup model (no manual DDP wrapping!)
    model = AutoModelForSequenceClassification.from_pretrained(...)

    if self.local_rank is not None:
        model = model.to(self.local_rank)

    # Disable problematic features in DDP
    config_dict = dict(config_train_args)
    if self.local_rank is not None:
        config_dict['load_best_model_at_end'] = False

    trainer = Trainer(model=model, args=TrainingArguments(**config_dict), ...)

    # Only rank 0 does logging
    is_main = self.local_rank is None or self.local_rank == 0
    if is_main:
        wandb.init(...)

    # Trainer handles ALL DDP
    trainer.train()

    # Only rank 0 saves and logs
    if is_main:
        trainer.save_model(...)
        wandb.log(...)
        wandb.finish()

    return checkpoint_path
```

## That's It!

The code is now:
- ✅ **Simple** - No complex manual DDP logic
- ✅ **Clean** - Trainer handles synchronization
- ✅ **Robust** - Fewer places for bugs
- ✅ **Maintainable** - Easy to understand

Just run with accelerate or torchrun and it should work!
