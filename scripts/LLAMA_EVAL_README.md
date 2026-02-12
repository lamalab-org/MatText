# LLAMA Generations Analysis and Evaluation

This document describes the anomaly detection and evaluation script for LLAMA model generations.

## Overview

The `analyze_and_evaluate_llama.py` script performs:

1. **Anomaly Detection**: Identifies predictions that are anomalously far from ground truth
2. **Detailed Reporting**: Creates a report with mbid, prediction, and ground truth for each anomaly
3. **Clean Evaluation**: Computes metrics both with and without anomalies
4. **Results Aggregation**: Aggregates results across folds for each model-property combination

## Anomaly Detection Criteria

A prediction is flagged as anomalous if **any** of the following conditions are met:

### 1. Ratio-based Detection
- **Too large**: `prediction / ground_truth > 5.0` (5x larger)
- **Too small**: `prediction / ground_truth < 0.2` (5x smaller, i.e., 1/5)

### 2. Relative Difference Detection
- **Large relative error**: `|prediction - ground_truth| / |ground_truth| > 4.0` (400% error)

### 3. Special Cases
- If ground truth is near zero (<1e-10) and prediction is >1.0, flagged as anomaly

## Configuration

Edit these constants at the top of the script to adjust:

```python
# Paths
GENERATIONS_DIR = "/Users/nalampara/n0w0f/dev/MatText/LLAMA_Generations"
OUTPUT_DIR = "/Users/nalampara/n0w0f/dev/MatText/LLAMA_Results"
DATA_REPOSITORY = "jablonkagroup/MatText"

# Thresholds
UPPER_THRESHOLD = 5.0  # Flag if prediction > 5x ground truth
LOWER_THRESHOLD = 0.2  # Flag if prediction < 0.2x ground truth
RATIO_THRESHOLD = 4.0  # Flag if relative error > 400%
```

## Usage

```bash
cd /Users/nalampara/n0w0f/dev/MatText/scripts
python analyze_and_evaluate_llama.py
```

No command-line arguments needed - paths are hardcoded at the top of the script.

## Output Files

The script creates the following files in `LLAMA_Results/`:

### 1. Anomaly Report
**File**: `anomaly_report.json`

Contains all detected anomalies with details:
```json
[
  {
    "model_size": "7b",
    "property": "kvrh",
    "fold": "0",
    "mbid": "mp-12345",
    "prediction": 15.6,
    "ground_truth": 2.3,
    "ratio": 6.78,
    "absolute_diff": 13.3,
    "relative_diff": 5.78,
    "reason": "ratio_6.78x_exceeds_upper_5.0x"
  }
]
```

### 2. Individual Model Results
**Files**: `llama-2-{size}_{property}_results_with_anomaly_analysis.json`

Per-model results including:
- Metrics for all data (including anomalies)
- Metrics for clean data (excluding anomalies)
- Fold-by-fold breakdown
- List of all anomalies for that model-property combination

### 3. Complete Results
**File**: `all_results_with_anomaly_analysis.json`

Complete analysis including:
- Configuration used (thresholds)
- Aggregated results for all models
- Individual file results
- Total anomaly count

## Metrics Comparison

For each model-property combination, you'll see:

**Metrics (all data)**:
- Mean MAE/RMSE across folds including all predictions
- Useful for understanding raw model performance

**Metrics (clean data, anomalies excluded)**:
- Mean MAE/RMSE across folds excluding anomalies
- Better representation of typical model performance
- More fair comparison across models

## Example Output

```
Model: llama-2-7b_kvrh
  Property: kvrh
  Folds evaluated: 5
  Total anomalies: 12
  Metrics (all data):
    Mean MAE: 0.3456 ± 0.0234
    Mean RMSE: 0.4789 ± 0.0345
  Metrics (clean data, anomalies excluded):
    Mean MAE: 0.2891 ± 0.0189
    Mean RMSE: 0.3967 ± 0.0267
```

## Adjusting Thresholds

If you find too many/few anomalies:

- **Too many false positives**: Increase `UPPER_THRESHOLD` (e.g., to 10.0) or decrease `RATIO_THRESHOLD` (e.g., to 5.0)
- **Missing real anomalies**: Decrease `UPPER_THRESHOLD` (e.g., to 3.0) or increase `RATIO_THRESHOLD` (e.g., to 3.0)

## Dependencies

Same as the original evaluation script:
- datasets
- matbench
- sklearn
- fire
- tqdm
