import os
import shutil
import re

def organize_json_files(base_path, output_path):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Walk through the base directory
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Check if it's a JSON file
            if file.endswith('.json'):
                # Extract components from the path
                # Example path: base/llama2-13b/llama2-robocrys-gvrh-13b/checkpoints/file.json
                parts = root.split(os.sep)
                
                # Determine Model Size (e.g., llama-2-7b, llama-2-13b)
                model_size = "unknown_model"
                for part in parts:
                    if 'llama2-' in part:
                        # Standardize name to llama-2-Xb
                        size_match = re.search(r'(\d+b)', part)
                        if size_match:
                            model_size = f"llama-2-{size_match.group(1)}"
                        elif part == 'llama2-7b' or 'llama2-robocrys' in part: 
                            # Fallback for the 7b folders which might just be labeled 'llama2-7b'
                            if '7b' in part or 'llama2-robocrys' in root:
                                model_size = "llama-2-7b"

                # Determine Property (gvrh, kvrh, perovskites)
                property_type = "other"
                if 'gvrh' in root.lower():
                    property_type = "gvrh"
                elif 'kvrh' in root.lower():
                    property_type = "kvrh"
                elif 'perovskites' in root.lower():
                    property_type = "perovskites"

                # Construct destination path: output/model_size/property/file.json
                dest_dir = os.path.join(output_path, model_size, property_type)
                
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                source_file = os.path.join(root, file)
                destination_file = os.path.join(dest_dir, file)

                # Copy the file
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {file} -> {model_size}/{property_type}/")

if __name__ == "__main__":
    # --- CHANGE THESE PATHS ---
    SOURCE = "/data/alamparan/Mattext_robocrys_warmup"
    DESTINATION = "/data/alamparan/Organized_JSONs_warmups"
    
    organize_json_files(SOURCE, DESTINATION)
    print("\nOrganization complete!")