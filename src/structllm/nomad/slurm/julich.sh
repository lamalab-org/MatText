#!/bin/bash
#SBATCH --account=hai_structllm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --output=out.%j
#SBATCH --error=err.%j

source source /p/home/jusers/alampara1/juwels/miniconda3/bin/activate
conda activate chemcaption

python -u --lmdb_path /p/project/hai_structllm/n0w0f/material_db/nomad/all/data.lmdb --output_file /p/project/hai_structllm/n0w0f/structllm/data/output_test.json