import json
import os
import random
from sklearn.model_selection import KFold
import fire

def split_dataset(input_json, output_dir, n_splits=5, random_state=42):
    # Load the data
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Perform the split
    for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        # Save train data
        train_file = os.path.join(output_dir, f'train_mp_classification_fold_{fold}.json')
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)

        # Save test data
        test_file = os.path.join(output_dir, f'test_mp_classification_fold_{fold}.json')
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"Fold {fold} created: {train_file} and {test_file}")

    print("Dataset splitting completed.")

if __name__ == "__main__":
    fire.Fire(split_dataset)