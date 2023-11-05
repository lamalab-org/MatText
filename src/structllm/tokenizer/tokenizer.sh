#!/bin/bash
#SBATCH --job-name=tokenizer
#SBATCH --output=token1_out.log
#SBATCH --error=token1_err.log
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48


module load nvidia/cuda/11.7
source /home/so87pot/miniconda3/bin/activate
conda activate structllm

python tokenizer.py
