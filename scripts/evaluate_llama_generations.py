"""
Evaluate LLAMA model generations against matbench test data.

This script:
1. Loads LLAMA generation files from a directory
2. Matches predictions with ground truth from matbench datasets
3. Computes MAE and RMSE metrics
4. Saves results to JSON files, one per model

Usage:
    python evaluate_llama_generations.py \
        --generations_dir /path/to/LLAMA_Generations \
        --output_dir /path/to/results \
        --data_repository jablonkagroup/MatText
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fire
from datasets import load_dataset
from matbench.data_ops import load as load_matbench
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# Mapping from short names to matbench dataset names
MATTEXT_MATBENCH = {
    "kvrh": "matbench_log_kvrh",
    "gvrh": "matbench_log_gvrh",
    "perovskites": "matbench_perovskites",
    "bandgap": "matbench_mp_gap",
    "form_energy": "matbench_mp_e_form",
    "is-metal": "matbench_mp_is_metal",
    "mattext_kvrh": "matbench_log_kvrh",
    "mattext_gvrh": "matbench_log_gvrh",
    "mattext_perovskites": "matbench_perovskites",
}

# Mapping from short names to column names
MATMINER_COLUMNS = {
    "kvrh": "log10(K_VRH)",
    "gvrh": "log10(G_VRH)",
    "perovskites": "e_form",
    "is-metal": "is_metal",
    "bandgap": "gap pbe",
    "form_energy": "e_form",
    "mattext_kvrh": "log10(K_VRH)",
    "mattext_gvrh": "log10(G_VRH)",
    "mattext_perovskites": "e_form",
}


def parse_generation_filename(filepath: Path) -> Dict[str, str]:
    """
    Parse generation filename to extract metadata.

    Expected format:
    finetuned_train_robocrys_rep_mattext_{property}-train-filtered_fold_{fold}_fold_{fold}_predictions.json

    Args:
        filepath: Path to generation file

    Returns:
        Dictionary with keys: model_size, property, fold
    """
    # Extract model size from parent directories
    parts = filepath.parts
    model_size = None
    for part in parts:
        if part.startswith("llama-2-"):
            model_size = part.replace("llama-2-", "")
            break

    # Extract property from parent directory
    property_name = filepath.parent.name

    # Extract fold number from filename
    filename = filepath.name
    fold_match = re.search(r"fold_(\d+)_fold_\1_predictions", filename)
    fold = fold_match.group(1) if fold_match else None

    return {
        "model_size": model_size,
        "property": property_name,
        "fold": fold,
    }


def load_predictions(filepath: Path) -> List[float]:
    """
    Load predictions from LLAMA generation file.

    The file format is a list of lists, where each inner list contains
    a dictionary with "generated_text" key.

    Args:
        filepath: Path to generation file

    Returns:
        List of predicted values
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    predictions = []
    for item in data:
        if not item or len(item) == 0:
            predictions.append(float("nan"))
            continue

        generated_text = item[0].get("generated_text", "")

        # Try to parse the generated text as a float
        try:
            # Clean up the text (remove any non-numeric characters except . and -)
            cleaned = re.sub(r"[^0-9.\-]", "", generated_text.strip())
            if cleaned:
                value = float(cleaned)
                predictions.append(value)
            else:
                predictions.append(float("nan"))
        except (ValueError, AttributeError):
            predictions.append(float("nan"))

    return predictions


def load_true_scores(
    dataset: str,
    fold: str,
    data_repository: str = "jablonkagroup/MatText"
) -> Tuple[List[str], List[float]]:
    """
    Load ground truth scores for a given dataset and fold.

    Args:
        dataset: Short dataset name (e.g., "kvrh", "gvrh")
        fold: Fold number as string
        data_repository: HuggingFace dataset repository

    Returns:
        Tuple of (mbids, true_scores)
    """
    # Map short name to matbench dataset name
    matbench_dataset = MATTEXT_MATBENCH[dataset]

    # Load matbench dataset (returns DataFrame directly)
    df = load_matbench(matbench_dataset)
    column = MATMINER_COLUMNS[dataset]

    # Load test dataset from HuggingFace to get the correct ordering
    hf_dataset_name = f"mattext_{dataset}-test-filtered"
    try:
        # Load the specific fold split (folds are already split in the dataset)
        fold_split_name = f"fold_{fold}"
        hf_dataset = load_dataset(data_repository, hf_dataset_name, split=fold_split_name)

        # Get mbids
        mbids = hf_dataset["mbid"]

        # Get true scores in the same order as mbids (mbid is the index in matbench df)
        true_scores = [df.loc[mbid, column] for mbid in mbids]

        return mbids, true_scores
    except Exception as e:
        print(f"Warning: Could not load from HuggingFace ({e}), falling back to local files")
        # Fallback: try loading from local JSON files
        return load_true_scores_from_local(dataset, fold)


def load_true_scores_from_local(
    dataset: str,
    fold: str
) -> Tuple[List[str], List[float]]:
    """
    Load ground truth scores from local JSON files (fallback method).

    Args:
        dataset: Short dataset name (e.g., "kvrh", "gvrh")
        fold: Fold number as string

    Returns:
        Tuple of (mbids, true_scores)
    """
    # Try to find local test file
    local_test_path = Path(f"dataset/normalized_data/test_{dataset}_{fold}.json")

    if not local_test_path.exists():
        raise FileNotFoundError(
            f"Could not find test data at {local_test_path}. "
            "Please ensure HuggingFace datasets library is installed or local files exist."
        )

    # Load matbench dataset for true scores (returns DataFrame directly)
    matbench_dataset = MATTEXT_MATBENCH[dataset]
    df = load_matbench(matbench_dataset)
    column = MATMINER_COLUMNS[dataset]

    # Load local test data
    mbids = []
    with open(local_test_path, "r") as f:
        for line in f:
            data = json.loads(line)
            mbids.append(data["mbid"])

    # Get true scores (mbid is the index in matbench df)
    true_scores = [df.loc[mbid, column] for mbid in mbids]

    return mbids, true_scores


def compute_metrics(predictions: List[float], true_scores: List[float]) -> Dict[str, float]:
    """
    Compute MAE and RMSE metrics.

    Args:
        predictions: List of predicted values
        true_scores: List of true values

    Returns:
        Dictionary with mae and rmse
    """
    # Filter out NaN predictions
    valid_pairs = [
        (pred, true)
        for pred, true in zip(predictions, true_scores)
        if not math.isnan(pred)
    ]

    if not valid_pairs:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "num_valid": 0,
            "num_total": len(predictions),
        }

    valid_predictions, valid_true = zip(*valid_pairs)

    mae = mean_absolute_error(valid_true, valid_predictions)
    rmse = math.sqrt(mean_squared_error(valid_true, valid_predictions))

    return {
        "mae": mae,
        "rmse": rmse,
        "num_valid": len(valid_pairs),
        "num_total": len(predictions),
    }


def evaluate_generation_file(
    filepath: Path,
    data_repository: str = "jablonkagroup/MatText"
) -> Dict:
    """
    Evaluate a single generation file.

    Args:
        filepath: Path to generation file
        data_repository: HuggingFace dataset repository

    Returns:
        Dictionary with metadata and metrics
    """
    # Parse filename
    metadata = parse_generation_filename(filepath)

    print(f"\nEvaluating {filepath.name}")
    print(f"  Model: llama-2-{metadata['model_size']}")
    print(f"  Property: {metadata['property']}")
    print(f"  Fold: {metadata['fold']}")

    # Load predictions
    predictions = load_predictions(filepath)
    print(f"  Loaded {len(predictions)} predictions")

    # Load ground truth
    try:
        mbids, true_scores = load_true_scores(
            metadata["property"],
            metadata["fold"],
            data_repository
        )
        print(f"  Loaded {len(true_scores)} ground truth scores")
    except Exception as e:
        print(f"  Error loading ground truth: {e}")
        return None

    # Check length match
    if len(predictions) != len(true_scores):
        print(f"  Warning: Length mismatch! Predictions: {len(predictions)}, True: {len(true_scores)}")
        # Truncate to shorter length
        min_len = min(len(predictions), len(true_scores))
        predictions = predictions[:min_len]
        true_scores = true_scores[:min_len]
        mbids = mbids[:min_len]

    # Compute metrics
    metrics = compute_metrics(predictions, true_scores)

    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Valid predictions: {metrics['num_valid']}/{metrics['num_total']}")

    return {
        "file": str(filepath),
        "model_size": metadata["model_size"],
        "property": metadata["property"],
        "fold": metadata["fold"],
        "metrics": metrics,
        "num_predictions": len(predictions),
        "num_ground_truth": len(true_scores),
    }


def aggregate_results_by_model(results: List[Dict]) -> Dict:
    """
    Aggregate results by model (across folds).

    Args:
        results: List of evaluation results

    Returns:
        Dictionary mapping model keys to aggregated results
    """
    from collections import defaultdict

    # Group by model and property
    grouped = defaultdict(list)
    for result in results:
        if result is None:
            continue
        key = f"llama-2-{result['model_size']}_{result['property']}"
        grouped[key].append(result)

    # Aggregate metrics
    aggregated = {}
    for key, fold_results in grouped.items():
        maes = [r["metrics"]["mae"] for r in fold_results if not math.isnan(r["metrics"]["mae"])]
        rmses = [r["metrics"]["rmse"] for r in fold_results if not math.isnan(r["metrics"]["rmse"])]

        if maes and rmses:
            aggregated[key] = {
                "model_size": fold_results[0]["model_size"],
                "property": fold_results[0]["property"],
                "num_folds": len(fold_results),
                "mean_mae": sum(maes) / len(maes),
                "std_mae": math.sqrt(sum((x - sum(maes)/len(maes))**2 for x in maes) / len(maes)) if len(maes) > 1 else 0.0,
                "mean_rmse": sum(rmses) / len(rmses),
                "std_rmse": math.sqrt(sum((x - sum(rmses)/len(rmses))**2 for x in rmses) / len(rmses)) if len(rmses) > 1 else 0.0,
                "fold_results": fold_results,
            }

    return aggregated


def main(
    generations_dir: str,
    output_dir: str,
    data_repository: str = "jablonkagroup/MatText",
    pattern: str = "**/*fold_*_fold_*_predictions.json",
):
    """
    Evaluate LLAMA generations.

    Args:
        generations_dir: Directory containing LLAMA generation files
        output_dir: Directory to save results
        data_repository: HuggingFace dataset repository (default: jablonkagroup/MatText)
        pattern: Glob pattern to match generation files (default: **/*fold_*_fold_*_predictions.json)
                 This excludes _merged files
    """
    generations_path = Path(generations_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Searching for generation files in: {generations_path}")
    print(f"Pattern: {pattern}")

    # Find all generation files
    # Use rglob if pattern starts with **/
    if pattern.startswith("**/"):
        generation_files = list(generations_path.rglob(pattern[3:]))
    else:
        generation_files = list(generations_path.glob(pattern))
    print(f"Found {len(generation_files)} generation files")

    if not generation_files:
        print("No generation files found!")
        return

    # Evaluate each file
    results = []
    for filepath in tqdm(generation_files, desc="Evaluating generations"):
        result = evaluate_generation_file(filepath, data_repository)
        if result:
            results.append(result)

    # Aggregate results by model
    aggregated = aggregate_results_by_model(results)

    # Save results
    print(f"\n{'='*80}")
    print("Summary of Results:")
    print(f"{'='*80}")

    for model_key, model_results in aggregated.items():
        print(f"\nModel: {model_key}")
        print(f"  Property: {model_results['property']}")
        print(f"  Folds evaluated: {model_results['num_folds']}")
        print(f"  Mean MAE: {model_results['mean_mae']:.4f} ± {model_results['std_mae']:.4f}")
        print(f"  Mean RMSE: {model_results['mean_rmse']:.4f} ± {model_results['std_rmse']:.4f}")

        # Save individual model results
        output_file = output_path / f"{model_key}_results.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Save all results
    all_results_file = output_path / "all_results.json"
    with open(all_results_file, "w") as f:
        json.dump(
            {
                "aggregated": aggregated,
                "individual_files": results,
            },
            f,
            indent=2,
        )
    print(f"\nAll results saved to: {all_results_file}")


if __name__ == "__main__":
    fire.Fire(main)
