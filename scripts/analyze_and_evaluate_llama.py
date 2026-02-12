"""
Analyze LLAMA generations for anomalies and evaluate excluding outliers.

This script:
1. Identifies anomalous predictions (>5x or <0.2x ground truth)
2. Generates anomaly report with mbid, prediction, ground truth
3. Evaluates models excluding anomalies
4. Saves both anomaly report and clean evaluation results
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from datasets import load_dataset
from matbench.data_ops import load as load_matbench
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# ============================================================================
# CONFIGURATION - HARDCODED PATHS
# ============================================================================

GENERATIONS_DIR = "/Users/nalampara/n0w0f/dev/MatText/LLAMA_Generations"
OUTPUT_DIR = "/Users/nalampara/n0w0f/dev/MatText/LLAMA_Results_2"
DATA_REPOSITORY = "jablonkagroup/MatText"

# ============================================================================
# ANOMALY DETECTION THRESHOLDS
# ============================================================================

# Minimum absolute difference to consider for anomaly detection
# Predictions with |pred - truth| < MIN_ABSOLUTE_DIFFERENCE are never flagged as anomalies
# This prevents zero or near-zero predictions from being flagged
MIN_ABSOLUTE_DIFFERENCE = 1.0  # Only check ratios if absolute difference exceeds this

# Ratio-based thresholds (only applied if absolute difference > MIN_ABSOLUTE_DIFFERENCE)
# If prediction/truth > UPPER_THRESHOLD or prediction/truth < LOWER_THRESHOLD, flag as anomaly
UPPER_THRESHOLD = 5.0  # 5x larger than truth
LOWER_THRESHOLD = 0.2  # 5x smaller than truth (1/5)

# Relative difference threshold
# If |prediction - truth| / |truth| > RATIO_THRESHOLD, flag as anomaly
RATIO_THRESHOLD = 4.0  # 400% difference


# ============================================================================
# DATASET MAPPINGS (from original script)
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS (from original script)
# ============================================================================

def parse_generation_filename(filepath: Path) -> Dict[str, str]:
    """Parse generation filename to extract metadata."""
    parts = filepath.parts
    model_size = None
    for part in parts:
        if part.startswith("llama-2-"):
            model_size = part.replace("llama-2-", "")
            break

    property_name = filepath.parent.name
    filename = filepath.name

    # Handle both fold_X_fold_X and fold_X__fold_X patterns
    fold_match = re.search(r"fold_(\d+)_+fold_\1_predictions", filename)
    fold = fold_match.group(1) if fold_match else None

    return {
        "model_size": model_size,
        "property": property_name,
        "fold": fold,
    }


def load_predictions(filepath: Path) -> List[float]:
    """Load predictions from LLAMA generation file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    predictions = []
    for item in data:
        if not item or len(item) == 0:
            predictions.append(float("nan"))
            continue

        generated_text = item[0].get("generated_text", "")

        try:
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
    """Load ground truth scores for a given dataset and fold."""
    matbench_dataset = MATTEXT_MATBENCH[dataset]
    df = load_matbench(matbench_dataset)
    column = MATMINER_COLUMNS[dataset]

    hf_dataset_name = f"mattext_{dataset}-test-filtered"
    try:
        fold_split_name = f"fold_{fold}"
        hf_dataset = load_dataset(data_repository, hf_dataset_name, split=fold_split_name)
        mbids = hf_dataset["mbid"]
        true_scores = [df.loc[mbid, column] for mbid in mbids]
        return mbids, true_scores
    except Exception as e:
        print(f"Warning: Could not load from HuggingFace ({e})")
        raise


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def detect_anomalies(
    predictions: List[float],
    true_scores: List[float],
    mbids: List[str],
    upper_threshold: float = UPPER_THRESHOLD,
    lower_threshold: float = LOWER_THRESHOLD,
) -> Tuple[List[int], List[Dict]]:
    """
    Detect anomalous predictions.

    Returns:
        Tuple of (anomaly_indices, anomaly_details)
    """
    anomaly_indices = []
    anomaly_details = []

    for i, (pred, truth, mbid) in enumerate(zip(predictions, true_scores, mbids)):
        # Skip NaN predictions
        if math.isnan(pred):
            continue

        # Calculate absolute difference first
        abs_diff = abs(pred - truth)

        # Skip if absolute difference is too small - these are not anomalies
        # This allows zero predictions when ground truth is small
        if abs_diff < MIN_ABSOLUTE_DIFFERENCE:
            continue

        # Handle near-zero ground truth to avoid division by zero
        if abs(truth) < 1e-10:
            # Only flag if absolute difference is large (already checked above)
            anomaly_indices.append(i)
            anomaly_details.append({
                "index": i,
                "mbid": mbid,
                "prediction": pred,
                "ground_truth": truth,
                "ratio": float("inf") if pred != 0 else 1.0,
                "absolute_diff": abs_diff,
                "reason": f"ground_truth_near_zero_but_large_prediction_abs_diff_{abs_diff:.2f}"
            })
            continue

        # Calculate ratio and relative difference
        ratio = abs(pred / truth)
        relative_diff = abs_diff / abs(truth)

        # Check if anomalous based on ratio thresholds
        is_anomaly = False
        reason = []

        if ratio > upper_threshold:
            is_anomaly = True
            reason.append(f"ratio_{ratio:.2f}x_exceeds_upper_{upper_threshold}x")
        elif ratio < lower_threshold:
            is_anomaly = True
            reason.append(f"ratio_{ratio:.2f}x_below_lower_{lower_threshold}x")

        if relative_diff > RATIO_THRESHOLD:
            is_anomaly = True
            reason.append(f"relative_diff_{relative_diff:.2f}_exceeds_{RATIO_THRESHOLD}")

        if is_anomaly:
            anomaly_indices.append(i)
            anomaly_details.append({
                "index": i,
                "mbid": mbid,
                "prediction": pred,
                "ground_truth": truth,
                "ratio": ratio,
                "absolute_diff": abs_diff,
                "relative_diff": relative_diff,
                "reason": "; ".join(reason)
            })

    return anomaly_indices, anomaly_details


# ============================================================================
# EVALUATION WITH ANOMALY FILTERING
# ============================================================================

def compute_metrics_filtered(
    predictions: List[float],
    true_scores: List[float],
    exclude_indices: List[int] = None
) -> Dict[str, float]:
    """Compute MAE and RMSE metrics, optionally excluding certain indices."""
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)

    valid_pairs = [
        (pred, true)
        for i, (pred, true) in enumerate(zip(predictions, true_scores))
        if not math.isnan(pred) and i not in exclude_indices
    ]

    if not valid_pairs:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "num_valid": 0,
            "num_total": len(predictions),
            "num_excluded": len(exclude_indices),
        }

    valid_predictions, valid_true = zip(*valid_pairs)

    mae = mean_absolute_error(valid_true, valid_predictions)
    rmse = math.sqrt(mean_squared_error(valid_true, valid_predictions))

    return {
        "mae": mae,
        "rmse": rmse,
        "num_valid": len(valid_pairs),
        "num_total": len(predictions),
        "num_excluded": len(exclude_indices),
    }


def analyze_generation_file(
    filepath: Path,
    data_repository: str = "jablonkagroup/MatText"
) -> Dict:
    """Analyze a single generation file for anomalies and compute metrics."""
    metadata = parse_generation_filename(filepath)

    print(f"\nAnalyzing {filepath.name}")
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
        min_len = min(len(predictions), len(true_scores))
        predictions = predictions[:min_len]
        true_scores = true_scores[:min_len]
        mbids = mbids[:min_len]

    # Detect anomalies
    anomaly_indices, anomaly_details = detect_anomalies(predictions, true_scores, mbids)
    print(f"  Found {len(anomaly_indices)} anomalies")

    # Compute metrics with and without anomalies
    metrics_all = compute_metrics_filtered(predictions, true_scores, exclude_indices=None)
    metrics_clean = compute_metrics_filtered(predictions, true_scores, exclude_indices=anomaly_indices)

    print(f"  Metrics (all data):")
    print(f"    MAE: {metrics_all['mae']:.4f}, RMSE: {metrics_all['rmse']:.4f}")
    print(f"  Metrics (excluding anomalies):")
    print(f"    MAE: {metrics_clean['mae']:.4f}, RMSE: {metrics_clean['rmse']:.4f}")

    return {
        "file": str(filepath),
        "model_size": metadata["model_size"],
        "property": metadata["property"],
        "fold": metadata["fold"],
        "metrics_all": metrics_all,
        "metrics_clean": metrics_clean,
        "anomalies": anomaly_details,
        "num_anomalies": len(anomaly_indices),
    }


def aggregate_results_by_model(results: List[Dict]) -> Dict:
    """Aggregate results by model (across folds)."""
    grouped = defaultdict(list)
    for result in results:
        if result is None:
            continue
        key = f"llama-2-{result['model_size']}_{result['property']}"
        grouped[key].append(result)

    aggregated = {}
    for key, fold_results in grouped.items():
        # Aggregate metrics for all data
        maes_all = [r["metrics_all"]["mae"] for r in fold_results if not math.isnan(r["metrics_all"]["mae"])]
        rmses_all = [r["metrics_all"]["rmse"] for r in fold_results if not math.isnan(r["metrics_all"]["rmse"])]

        # Aggregate metrics for clean data
        maes_clean = [r["metrics_clean"]["mae"] for r in fold_results if not math.isnan(r["metrics_clean"]["mae"])]
        rmses_clean = [r["metrics_clean"]["rmse"] for r in fold_results if not math.isnan(r["metrics_clean"]["rmse"])]

        # Count total anomalies
        total_anomalies = sum(r["num_anomalies"] for r in fold_results)

        if maes_all and rmses_all:
            aggregated[key] = {
                "model_size": fold_results[0]["model_size"],
                "property": fold_results[0]["property"],
                "num_folds": len(fold_results),
                "total_anomalies": total_anomalies,
                "metrics_all": {
                    "mean_mae": sum(maes_all) / len(maes_all),
                    "std_mae": math.sqrt(sum((x - sum(maes_all)/len(maes_all))**2 for x in maes_all) / len(maes_all)) if len(maes_all) > 1 else 0.0,
                    "mean_rmse": sum(rmses_all) / len(rmses_all),
                    "std_rmse": math.sqrt(sum((x - sum(rmses_all)/len(rmses_all))**2 for x in rmses_all) / len(rmses_all)) if len(rmses_all) > 1 else 0.0,
                },
                "metrics_clean": {
                    "mean_mae": sum(maes_clean) / len(maes_clean),
                    "std_mae": math.sqrt(sum((x - sum(maes_clean)/len(maes_clean))**2 for x in maes_clean) / len(maes_clean)) if len(maes_clean) > 1 else 0.0,
                    "mean_rmse": sum(rmses_clean) / len(rmses_clean),
                    "std_rmse": math.sqrt(sum((x - sum(rmses_clean)/len(rmses_clean))**2 for x in rmses_clean) / len(rmses_clean)) if len(rmses_clean) > 1 else 0.0,
                },
                "fold_results": fold_results,
            }

    return aggregated


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis and evaluation pipeline."""
    generations_path = Path(GENERATIONS_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LLAMA Generations Analysis and Evaluation")
    print("="*80)
    print(f"Generations directory: {generations_path}")
    print(f"Output directory: {output_path}")
    print(f"\nAnomaly Detection Configuration:")
    print(f"  Min absolute difference: {MIN_ABSOLUTE_DIFFERENCE}")
    print(f"  Ratio thresholds: {LOWER_THRESHOLD}x < ratio < {UPPER_THRESHOLD}x")
    print(f"  Relative difference threshold: {RATIO_THRESHOLD}")
    print("="*80)

    # Find all generation files (excluding _merged files)
    all_json_files = list(generations_path.rglob("*predictions.json"))
    generation_files = [
        f for f in all_json_files
        if "_merged" not in f.name and re.search(r"fold_\d+_+fold_\d+_predictions\.json$", f.name)
    ]
    print(f"\nFound {len(generation_files)} generation files (excluding merged files)")

    if not generation_files:
        print("No generation files found!")
        return

    # Analyze each file
    results = []
    for filepath in tqdm(generation_files, desc="Analyzing generations"):
        result = analyze_generation_file(filepath, DATA_REPOSITORY)
        if result:
            results.append(result)

    # Aggregate results by model
    aggregated = aggregate_results_by_model(results)

    # ========================================================================
    # Save anomaly report
    # ========================================================================
    anomaly_report = []
    for result in results:
        for anomaly in result["anomalies"]:
            anomaly_report.append({
                "model_size": result["model_size"],
                "property": result["property"],
                "fold": result["fold"],
                "mbid": anomaly["mbid"],
                "prediction": anomaly["prediction"],
                "ground_truth": anomaly["ground_truth"],
                "ratio": anomaly["ratio"],
                "absolute_diff": anomaly["absolute_diff"],
                "relative_diff": anomaly.get("relative_diff", None),
                "reason": anomaly["reason"],
            })

    anomaly_report_file = output_path / "anomaly_report.json"
    with open(anomaly_report_file, "w") as f:
        json.dump(anomaly_report, f, indent=2)
    print(f"\nAnomaly report saved to: {anomaly_report_file}")
    print(f"Total anomalies found: {len(anomaly_report)}")

    # ========================================================================
    # Print summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("Summary of Results:")
    print(f"{'='*80}")

    for model_key, model_results in aggregated.items():
        print(f"\nModel: {model_key}")
        print(f"  Property: {model_results['property']}")
        print(f"  Folds evaluated: {model_results['num_folds']}")
        print(f"  Total anomalies: {model_results['total_anomalies']}")
        print(f"  Metrics (all data):")
        print(f"    Mean MAE: {model_results['metrics_all']['mean_mae']:.4f} ± {model_results['metrics_all']['std_mae']:.4f}")
        print(f"    Mean RMSE: {model_results['metrics_all']['mean_rmse']:.4f} ± {model_results['metrics_all']['std_rmse']:.4f}")
        print(f"  Metrics (clean data, anomalies excluded):")
        print(f"    Mean MAE: {model_results['metrics_clean']['mean_mae']:.4f} ± {model_results['metrics_clean']['std_mae']:.4f}")
        print(f"    Mean RMSE: {model_results['metrics_clean']['mean_rmse']:.4f} ± {model_results['metrics_clean']['std_rmse']:.4f}")

    # ========================================================================
    # Save results
    # ========================================================================

    # Save individual model results
    for model_key, model_results in aggregated.items():
        output_file = output_path / f"{model_key}_results_with_anomaly_analysis.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved {model_key} results to: {output_file}")

    # Save all results
    all_results_file = output_path / "all_results_with_anomaly_analysis.json"
    with open(all_results_file, "w") as f:
        json.dump(
            {
                "configuration": {
                    "min_absolute_difference": MIN_ABSOLUTE_DIFFERENCE,
                    "upper_threshold": UPPER_THRESHOLD,
                    "lower_threshold": LOWER_THRESHOLD,
                    "ratio_threshold": RATIO_THRESHOLD,
                },
                "aggregated": aggregated,
                "individual_files": results,
                "total_anomalies": len(anomaly_report),
            },
            f,
            indent=2,
        )
    print(f"\nAll results saved to: {all_results_file}")

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
