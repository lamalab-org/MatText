import json
import os
import fire

def split_json(input_file, output_dir, num_files):
    try:
        with open(input_file, 'r') as infile:
            data = json.load(infile)

        total_items = len(data)
        items_per_file = total_items // num_files
        remainder = total_items % num_files

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_idx = 0
        for i in range(num_files):
            end_idx = start_idx + items_per_file + (1 if i < remainder else 0)
            output_file = os.path.join(output_dir, f'output_{i}.json')
            with open(output_file, 'w') as outfile:
                json.dump(data[start_idx:end_idx], outfile)
            start_idx = end_idx

        print(f'Successfully split {input_file} into {num_files} JSON files in {output_dir}.')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    fire.Fire(split_json)
