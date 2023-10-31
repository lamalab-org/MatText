import json
import csv

def read_json_and_save_as_csv(json_file, csv_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extract the keys from the first data entry to use as column headers in CSV
    if data:
        fieldnames = data[0].keys()
    
    with open(csv_file, 'w', newline='') as csv_output:
        writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in data:
            writer.writerow(entry)




# Usage:
json_file = '/home/nawaf/n0w0f/structllm/data/output_test.json' 
csv_file = '/home/nawaf/n0w0f/structllm/data/output_test.csv'  
read_json_and_save_as_csv(json_file, csv_file)
