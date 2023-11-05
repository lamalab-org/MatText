#!/bin/bash
#SBATCH --job-name nomad_prep_test
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --output nomad_prep_test.out.%j
#SBATCH --error nomad_prep_test.err.%j

source /home/so87pot/miniconda3/bin/activate
conda activate structllm

python /home/so87pot/n0w0f/structllm/src/structllm/nomad/matsci_nomad_prep.py  --lmdb_path /home/so87pot/n0w0f/material_db/nomad/all/data.lmdb --output_file /home/so87pot/n0w0f/structllm/data/output_test.json
