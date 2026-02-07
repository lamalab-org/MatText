"""Standalone inference script.

Discovers finetuned checkpoints under a results root directory and runs
inference + scoring for each (checkpoint, task, fold) combination.

Usage:
    python run_inference.py /home/alamparan/results/2026-02-07/n0w0f
"""

import json
import math
import os
import sys
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModelForSequenceClassification, Trainer

from mattext.models.utils import TokenizerMixin

# Task definitions
TASKS = {
    "gvrh": {
        "dataset": "mattext_gvrh",
        "train_data": "mattext_gvrh-train-filtered",
        "test_data": "mattext_gvrh-test-filtered",
        "dir_name": "robocrys-gvrh",
    },
    "kvrh": {
        "dataset": "mattext_kvrh",
        "train_data": "mattext_kvrh-train-filtered",
        "test_data": "mattext_kvrh-test-filtered",
        "dir_name": "robocrys-kvrh",
    },
    "perovskites": {
        "dataset": "mattext_perovskites",
        "train_data": "mattext_perovskites-train-filtered",
        "test_data": "mattext_perovskites-test-filtered",
        "dir_name": "robocrys-perovskites",
    },
}

DATA_REPOSITORY = "jablonkagroup/MatText"
REPRESENTATION = "robocrys_rep"
CONTEXT_LENGTH = 1024
NUM_FOLDS = 5

# For scoring
MATTEXT_MATBENCH = {
    "mattext_gvrh": "matbench_log_gvrh",
    "mattext_kvrh": "matbench_log_kvrh",
    "mattext_perovskites": "matbench_perovskites",
}
MATMINER_COLUMNS = {
    "mattext_gvrh": "log10(G_VRH)",
    "mattext_kvrh": "log10(K_VRH)",
    "mattext_perovskites": "e_form",
}


class InferenceTokenizer(TokenizerMixin):
    """Minimal wrapper to get a tokenizer for inference."""

    def __init__(self):
        super().__init__(
            cfg=REPRESENTATION,
            special_tokens={
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "eos_token": "[EOS]",
                "bos_token": "[BOS]",
            },
            special_num_token=False,
        )


def load_true_scores(dataset_name, mbids):
    from matbench.data_ops import load

    data_frame = load(MATTEXT_MATBENCH[dataset_name])
    return [data_frame.loc[mbid][MATMINER_COLUMNS[dataset_name]] for mbid in mbids]


def run_inference_for_fold(checkpoint_path, test_data, fold_name, tokenizer_helper):
    """Run inference for a single fold and return predictions + ids."""
    dataset = load_dataset(DATA_REPOSITORY, test_data)
    filtered = dataset[fold_name].filter(
        lambda ex: ex[REPRESENTATION] is not None
    )
    tokenized = filtered.map(
        partial(
            tokenizer_helper._tokenize_pad_and_truncate,
            context_length=CONTEXT_LENGTH,
        ),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=1, ignore_mismatched_sizes=False
    )
    trainer = Trainer(model=model.to("cuda"), data_collator=None)
    preds = trainer.predict(tokenized)
    torch.cuda.empty_cache()

    predictions = pd.Series(preds.predictions.flatten())
    prediction_ids = tokenized["mbid"]

    return predictions, prediction_ids


def run_all(results_root):
    tokenizer_helper = InferenceTokenizer()

    # Discover checkpoints: results_root/{ckpt_name}/{task_dir}/checkpoints/finetuned_*
    ckpt_dirs = sorted(glob(os.path.join(results_root, "MatText-robocrys_*")))
    if not ckpt_dirs:
        print(f"No MatText-robocrys_* directories found in {results_root}")
        sys.exit(1)

    for ckpt_dir in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_dir)
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        for task_key, task_info in TASKS.items():
            task_dir = os.path.join(ckpt_dir, task_info["dir_name"])
            checkpoints_dir = os.path.join(task_dir, "checkpoints")

            if not os.path.exists(checkpoints_dir):
                print(f"  [{task_key}] No checkpoints directory, skipping")
                continue

            # Find finetuned checkpoints for each fold
            finetuned = sorted(glob(os.path.join(
                checkpoints_dir,
                f"finetuned_train_{REPRESENTATION}_{task_info['train_data']}_*",
            )))

            if not finetuned:
                print(f"  [{task_key}] No finetuned checkpoints found, skipping")
                continue

            print(f"  [{task_key}] Found {len(finetuned)} finetuned folds")

            folds_results = {}
            recorded_folds = []

            for fold_idx, ckpt_path in enumerate(finetuned):
                fold_name = f"fold_{fold_idx}"
                print(f"    Fold {fold_idx}: {os.path.basename(ckpt_path)}")

                predictions, prediction_ids = run_inference_for_fold(
                    ckpt_path, task_info["test_data"], fold_name, tokenizer_helper
                )

                # Convert to lists for JSON serialization
                pred_list = predictions.tolist()
                id_list = list(prediction_ids) if not isinstance(prediction_ids, list) else prediction_ids
                # Handle Arrow Column type
                if hasattr(id_list, "to_pylist"):
                    id_list = id_list.to_pylist()

                true_scores = load_true_scores(task_info["dataset"], id_list)

                mae = mean_absolute_error(true_scores, pred_list)
                rmse = math.sqrt(mean_squared_error(true_scores, pred_list))

                print(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                folds_results[fold_idx] = {
                    "prediction_ids": id_list,
                    "predictions": pred_list,
                    "true_scores": true_scores,
                    "mae": mae,
                    "rmse": rmse,
                }
                recorded_folds.append(fold_idx)

            # Aggregate metrics
            maes = [folds_results[f]["mae"] for f in recorded_folds]
            rmses = [folds_results[f]["rmse"] for f in recorded_folds]

            summary = {
                "checkpoint": ckpt_name,
                "task": task_key,
                "num_folds": len(recorded_folds),
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_rmse": float(np.mean(rmses)),
                "std_rmse": float(np.std(rmses)),
                "per_fold": {
                    f"fold_{f}": {"mae": folds_results[f]["mae"], "rmse": folds_results[f]["rmse"]}
                    for f in recorded_folds
                },
            }

            # Full results with predictions
            full_results = {
                "task_name": task_info["dataset"],
                "checkpoint": ckpt_name,
                "num_folds": len(recorded_folds),
                "is_classification": False,
                "folds_results": {str(k): v for k, v in folds_results.items()},
                "recorded_folds": recorded_folds,
                "summary": summary,
            }

            # Save results
            out_file = os.path.join(
                task_dir,
                f"inference_results_{REPRESENTATION}_{task_info['test_data']}.json",
            )
            os.makedirs(task_dir, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(full_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

            print(f"    Summary: MAE={summary['mean_mae']:.4f}+-{summary['std_mae']:.4f}, "
                  f"RMSE={summary['mean_rmse']:.4f}+-{summary['std_rmse']:.4f}")
            print(f"    Saved: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <results_root>")
        print("Example: python run_inference.py /home/alamparan/results/2026-02-07/n0w0f")
        sys.exit(1)

    run_all(sys.argv[1])
