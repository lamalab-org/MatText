#!/bin/bash
#SBATCH --job-name=slice
#SBATCH --output=slice_output_%j.log  # %j is replaced by the job ID
#SBATCH --error=slice_error_%j.log
#SBATCH --partition=standard  # Use the "standard" partition
#SBATCH --nodes=1  # Request 1 node per job
#SBATCH --cpus-per-task=48  # Use all available cores on the node

# Calculate the task ID based on the Slurm job ID (if running as an array job)
#task_id=$((SLURM_ARRAY_TASK_ID))


# Assign the corresponding input JSON and output CSV files based on the task ID
case $SLURM_JOB_ID in
    1)
        input_json="/home/so87pot/n0w0f/structllm/data/output_1.json"
        output_csv="/home/so87pot/n0w0f/structllm/data/output_1.csv"
        ;;
    2)
        input_json="/home/so87pot/n0w0f/structllm/data/output_2.json"
        output_csv="/home/so87pot/n0w0f/structllm/data/output_2.csv"
        ;;
    3)
       input_json="/home/so87pot/n0w0f/structllm/data/output_3.json"
       output_csv="/home/so87pot/n0w0f/structllm/data/output_3.csv"
        ;;
    4)
       input_json="/home/so87pot/n0w0f/structllm/data/output_4.json"
       output_csv="/home/so87pot/n0w0f/structllm/data/output_4.csv"
        ;;
    5)
       input_json="/home/so87pot/n0w0f/structllm/data/output_5.json"
       output_csv="/home/so87pot/n0w0f/structllm/data/output_5.csv"
        ;;
    # Add more cases for additional jobs as needed
    *)
        echo "Invalid SLURM_JOB_ID: $SLURM_JOB_ID"
        exit 1
        ;;
esac


# Run your Python script with the assigned input JSON and output CSV files
python slice.py \
    --json_file "$input_json" \
    --csv_file "$output_csv" \
    --save_interval 100 \
    --num_workers 36  # Use all available cores
    --timeout 600 \

