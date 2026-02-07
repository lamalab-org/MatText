# Clean DDP Solution - Start Fresh

## The Problem
We've been manually managing DDP synchronization, which is error-prone and complex.

## The Solution
**Use Accelerate CLI** - it handles everything automatically.

## Step-by-Step Clean Implementation

### 1. Launch with Accelerate (instead of torchrun)

```bash
# Old way (manual, complex)
torchrun --nproc_per_node=4 -m mattext.main

# New way (simple, automatic)
accelerate launch --config_file accelerate_config.yaml -m mattext.main
```

### 2. Simplify Code - Remove Manual DDP Handling

The HuggingFace `Trainer` already handles DDP correctly IF you:
- ✅ Don't manually wrap models in DDP
- ✅ Use `local_rank` correctly
- ✅ Let Trainer manage checkpoints
- ✅ Only do wandb/file I/O on rank 0

### 3. Core Principle: Trainer Handles It

When using `accelerate launch`:
- ✅ Environment variables are set correctly
- ✅ Process groups are initialized automatically
- ✅ Model wrapping happens automatically
- ✅ Checkpoint saving/loading works correctly
- ✅ Barriers are added automatically where needed

### 4. Minimal Code Changes Needed

**What to keep:**
```python
# Check if main process (for wandb, logging, saving)
is_main = self.local_rank is None or self.local_rank == 0

if is_main:
    wandb.init(...)
    wandb.log(...)
    wandb.finish()
```

**What to remove:**
```python
# ❌ Remove manual DDP wrapping
model = nn.parallel.DistributedDataParallel(model, ...)

# ❌ Remove manual barriers (Trainer adds them)
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# ❌ Remove complex debug prints
print(f"[Rank {rank}] ...")
```

**What Trainer already handles:**
- Model wrapping in DDP
- Gradient synchronization
- Checkpoint saving (only rank 0)
- Loading best model at end
- Barrier synchronization

### 5. Fix load_best_model_at_end Properly

The issue: Non-rank-0 processes can't load checkpoints saved only by rank 0.

**Solution 1: Use save_on_each_node (simplest)**
```yaml
# In your config YAML
training_arguments:
  save_on_each_node: false  # Default, keep it this way
  load_best_model_at_end: true
```

**Solution 2: Disable in DDP mode (code)**
```python
# In training setup
if torch.distributed.is_initialized():
    training_args.load_best_model_at_end = False
```

### 6. Clean Training Function Template

```python
def finetune(self):
    # Setup
    model = AutoModelForSequenceClassification.from_pretrained(...)

    # Move to device (no manual DDP wrapping!)
    if self.local_rank is not None:
        model = model.to(self.local_rank)
    else:
        model = model.to("cuda")

    # Training args - let Trainer handle DDP
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=32,
        load_best_model_at_end=False,  # Disable in DDP to avoid issues
        save_on_each_node=False,  # Only rank 0 saves
        # ... other args
    )

    # Create trainer (it detects DDP automatically)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Only rank 0 logs to wandb
    is_main = self.local_rank is None or self.local_rank == 0
    if is_main:
        wandb.init(...)

    # Train (Trainer handles all DDP internally)
    trainer.train()

    # Only rank 0 saves and logs
    if is_main:
        trainer.save_model("./final_model")
        eval_results = trainer.evaluate()
        wandb.log(eval_results)
        wandb.finish()

    return "./final_model"
```

### 7. What Trainer Does Automatically

When you call `trainer.train()`:

```
✅ Wraps model in DDP if distributed detected
✅ Synchronizes gradients across GPUs
✅ Adds barriers before evaluation
✅ Only rank 0 saves checkpoints
✅ Only rank 0 logs to tensorboard/wandb (if using TrainerCallback)
✅ Handles data distribution (DistributedSampler)
✅ Manages device placement
```

### 8. Custom Callbacks (Better than Manual Checks)

Instead of manual rank checks everywhere, use callbacks:

```python
from transformers import TrainerCallback

class RankZeroCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.local_rank in [-1, 0]:  # Main process only
            wandb.log(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if args.local_rank in [-1, 0]:
            wandb.log(metrics)

# Use it
trainer = Trainer(
    model=model,
    callbacks=[RankZeroCallback()],
    ...
)
```

## Complete Clean Example

### Before (Complex, Manual DDP):
```python
# 50+ lines of manual DDP handling
if local_rank is not None:
    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

trainer.train()

if torch.distributed.is_initialized():
    torch.distributed.barrier()

if is_main:
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(...)

if torch.distributed.is_initialized():
    torch.distributed.barrier()
```

### After (Clean, Accelerate):
```python
# 5 lines, Trainer handles everything
if local_rank is not None:
    model = model.to(local_rank)

trainer.train()

if is_main:
    trainer.save_model(...)
```

## How to Run

```bash
# Using accelerate (recommended)
accelerate launch --config_file accelerate_config.yaml -m mattext.main

# Or using torchrun (if you prefer)
torchrun --nproc_per_node=4 -m mattext.main
```

## Key Takeaways

1. **Don't fight the framework** - Trainer already does DDP correctly
2. **Trust Trainer** - It wraps models, adds barriers, manages checkpoints
3. **Only handle wandb manually** - That's the one thing Trainer doesn't manage
4. **Use callbacks** - Better than manual checks scattered everywhere
5. **Disable load_best_model_at_end** - Simplest fix for the hanging issue

## Files to Simplify

Remove complexity from:
- `potential.py` - Remove manual DDP wrapping, remove manual barriers
- `finetune.py` - Keep only rank-0 wandb checks
- `llama.py` - Simplify to basic pattern
- `llama_sft.py` - Remove manual barriers
- `benchmark.py` - Keep as-is (already mostly clean)
- `main.py` - Simplify rank detection

## Testing

```bash
# 1. Single GPU (sanity check)
python -m mattext.main

# 2. Multi-GPU with accelerate
accelerate launch --config_file accelerate_config.yaml -m mattext.main

# Should see:
# - No hanging
# - Clean logs
# - Only one checkpoint directory
# - Training completes successfully
```
