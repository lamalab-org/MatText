"""Compile all inference results into a single dataset and push to HuggingFace."""

import json
import os
from glob import glob

from huggingface_hub import hf_hub_download, HfApi

RESULTS_ROOT = "/home/alamparan/results/2026-02-07/n0w0f"

# Map: (size, task) -> local path
LOCAL_FILES = {}
for size in ["30k", "100k", "300k"]:
    for task, dirname in [("gvrh", "robocrys-gvrh"), ("kvrh", "robocrys-kvrh"), ("perovskites", "robocrys-perovskites")]:
        pattern = os.path.join(RESULTS_ROOT, f"MatText-robocrys_{size}", dirname, "inference_results_*.json")
        matches = glob(pattern)
        if matches:
            LOCAL_FILES[(size, task)] = matches[0]

# Download 2m results from HuggingFace
HF_2M_REPOS = {
    ("2m", "gvrh"): "n0w0f/MatText_Robocrys_gvrh_2m",
    ("2m", "kvrh"): "n0w0f/MatText_Robocrys_kvrh_2m",
    ("2m", "perovskites"): "n0w0f/MatText_Robocrys_perovskites_2m",
}

SIZE_ORDER = {"30k": 30000, "100k": 100000, "300k": 300000, "2m": 2000000}

compiled = []

# Process local files
for (size, task), filepath in sorted(LOCAL_FILES.items()):
    print(f"Reading local: {size} / {task}")
    with open(filepath) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    compiled.append({
        "task_name": data.get("task_name", f"mattext_{task}"),
        "task": task,
        "checkpoint": data.get("checkpoint", f"MatText-robocrys_{size}"),
        "size": size,
        "size_num": SIZE_ORDER[size],
        "is_classification": data.get("is_classification", False),
        "num_folds": data.get("num_folds", summary.get("num_folds", 5)),
        "mean_mae": summary.get("mean_mae"),
        "std_mae": summary.get("std_mae"),
        "mean_rmse": summary.get("mean_rmse"),
        "std_rmse": summary.get("std_rmse"),
        "per_fold": summary.get("per_fold", {}),
    })

# Process 2m from HuggingFace
for (size, task), repo_id in HF_2M_REPOS.items():
    print(f"Downloading from HF: {repo_id}")
    try:
        filepath = hf_hub_download(repo_id=repo_id, filename="data/inference_results.jsonl", repo_type="dataset")
        with open(filepath) as f:
            data = json.loads(f.readline())

        summary = data.get("summary", {})
        compiled.append({
            "task_name": data.get("task_name", f"mattext_{task}"),
            "task": task,
            "checkpoint": data.get("checkpoint", f"MatText-robocrys_{size}"),
            "size": size,
            "size_num": SIZE_ORDER[size],
            "is_classification": data.get("is_classification", False),
            "num_folds": data.get("num_folds", summary.get("num_folds", 5)),
            "mean_mae": summary.get("mean_mae"),
            "std_mae": summary.get("std_mae"),
            "mean_rmse": summary.get("mean_rmse"),
            "std_rmse": summary.get("std_rmse"),
            "per_fold": summary.get("per_fold", {}),
        })
    except Exception as e:
        print(f"  Could not fetch {repo_id}: {e}")

# Sort by size then task
compiled.sort(key=lambda x: (x["size_num"], x["task"]))

# Print summary table
print(f"\n{'='*80}")
print(f"{'Size':<8} {'Task':<15} {'MAE':>12} {'RMSE':>12} {'Folds':>6}")
print(f"{'='*80}")
for r in compiled:
    mae_str = f"{r['mean_mae']:.4f}±{r['std_mae']:.4f}" if r['mean_mae'] is not None else "N/A"
    rmse_str = f"{r['mean_rmse']:.4f}±{r['std_rmse']:.4f}" if r['mean_rmse'] is not None else "N/A"
    print(f"{r['size']:<8} {r['task']:<15} {mae_str:>12} {rmse_str:>12} {r['num_folds']:>6}")

# Save compiled results
out_path = os.path.join(RESULTS_ROOT, "compiled_results.json")
with open(out_path, "w") as f:
    json.dump(compiled, f, indent=2)
print(f"\nSaved to {out_path}")

# Push to HuggingFace
repo_id = "n0w0f/MatText_Robocrys_compiled_results"
print(f"\nPushing to {repo_id}...")
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=out_path,
    path_in_repo="compiled_results.json",
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"Done! https://huggingface.co/datasets/{repo_id}")
