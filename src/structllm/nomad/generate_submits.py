import os


if __name__ == "__main__":

    # Specify the path to the original YAML file
    json_file = "/home/so87pot/n0w0f/structllm/data/all/"
    output_dir = "/home/so87pot/n0w0f/structllm/data/all/slice"

    # Specify the number of desired chunks
    num_chunks = 10


    # Generate Slurm job scripts for each chunk
    for i in range(num_chunks):
        job_script = os.path.join(output_dir, f"slice_job_{i}.sh")
        with open(job_script, "w") as file:
            file.write(f"""#!/bin/bash
                            #SBATCH --job-name=slice_{i}
                            #SBATCH --nodes=1
                            #SBATCH --mem=64GB
                            #SBATCH --partition=long
                            #SBATCH --time=48:00:00
                            #SBATCH --cpus-per-task=48
                            #SBATCH --output slice_{i}.out.%j
                            #SBATCH --error slice_{i}.err.%j
                            
                            source /home/so87pot/miniconda3/bin/activate
                            conda activate structllm
                            
                            # Specify the YAML file for this job's chunk
                            JSON_FILE="{os.path.join(json_file, f'lmdb_{i}.json')}"
                            
                            # Specify the LMDB directory for this job
                            OUTPUT_CSV="{os.path.join(output_dir, f'slice_{i}.csv')}"
                            
                            # Run your Python script to query and download data
                            python slice_2.py  --json_file "$JSON_FILE" --csv_file "$OUTPUT_CSV" --num_workers 48 --timeout 600
                            """)

