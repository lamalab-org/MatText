#!/bin/bash
#SBATCH --job-name nomad_1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --output nomad_prep_test.out.%j
#SBATCH --error nomad_prep_test.err.%j

source /home/so87pot/miniconda3/bin/activate
conda activate structllm

python convert_to_slice.py \
    --json_file /home/so87pot/n0w0f/structllm/data/output_1.json \
    --csv_file /home/so87pot/n0w0f/structllm/data/output_1.csv \
    --num_workers 32 \
    --timeout 600 
