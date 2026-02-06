#!/usr/bin/env python3
"""
Wrapper script to run benchmarks in single-GPU mode.
This ensures local_rank is None instead of 0 when not using distributed training.
"""

import os
import sys

# Ensure LOCAL_RANK is not set for single-GPU mode
if 'LOCAL_RANK' in os.environ:
    del os.environ['LOCAL_RANK']

# Import after unsetting LOCAL_RANK
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/mattext'))

# Patch main.py's local_rank assignment
import main
original_main = main.main.__wrapped__

def patched_main(cfg):
    """Patched main function that sets local_rank to None for single-GPU mode"""
    import os
    from hydra import utils

    print(f"Working directory : {os.getcwd()}")
    import hydra
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    # Set local_rank to None for single-GPU mode
    local_rank = None
    task_runner = main.TaskRunner()
    task_runner.initialize_wandb()

    if cfg.runs:
        print(cfg)
        runs = utils.instantiate(cfg.runs)
        print(runs)
        for run in runs:
            print(run)
            task_runner.run_task(run["tasks"], task_cfg=cfg, local_rank=local_rank)

# Replace the main function
main.main.__wrapped__ = patched_main

if __name__ == "__main__":
    main.main()
