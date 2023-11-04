#!/bin/bash
name="slice5"
#SBATCH --job-name=${name}
#SBATCH --output=${name}_output.log
#SBATCH --error=${name}_error.log
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

# Assign input and output file paths
input_json="/home/so87pot/n0w0f/structllm/data/output_5.json"
output_csv="/home/so87pot/n0w0f/structllm/data/output_5.csv"

source /home/so87pot/miniconda3/bin/activate
conda activate structllm

# Run your Python script with the assigned input JSON and output CSV files
python slice.py \
    --json_file "$input_json" \
    --csv_file "$output_csv" \
    --save_interval 100 \
    --num_workers 36 \  # Line continuation with a backslash
    --timeout 600
