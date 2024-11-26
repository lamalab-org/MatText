import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matbench.data_ops import load
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

MATTEXT_MATBENCH = {
    "kvrh": "matbench_log_kvrh",
    "gvrh": "matbench_log_gvrh",
    "perovskites": "matbench_perovskites",
    "bandgap": "matbench_mp_gap",
    "form_energy": "matbench_mp_e_form",
    "is-metal": "matbench_mp_is_metal",
}

MATMINER_COLUMNS = {
    "kvrh": "log10(K_VRH)",
    "gvrh": "log10(G_VRH)",
    "perovskites": "e_form",
    "is-metal": "is_metal",
    "bandgap": "gap pbe",
    "form_energy": "e_form",
}


def load_true_scores(dataset, mbids):
    data_frame = load(MATTEXT_MATBENCH[dataset])
    scores = []
    for mbid in mbids:
        score = data_frame.loc[mbid][MATMINER_COLUMNS[dataset]]
        scores.append(score)
    return scores


@dataclass
class MatTextTask:
    task_name: str
    num_folds: int = 5
    is_classification: bool = False
    num_classes: int = 2
    folds_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recorded_folds: List[int] = field(default_factory=list)

    def record_fold(
        self, fold: int, prediction_ids: List[str], predictions: List[float]
    ):
        if fold in self.recorded_folds:
            raise ValueError(f"Fold {fold} has already been recorded.")
        true_scores = load_true_scores(self.task_name, prediction_ids)

        if self.is_classification:
            self._calculate_classification_metrics(
                fold, prediction_ids, predictions, true_scores
            )
        else:
            self._calculate_regression_metrics(
                fold, prediction_ids, predictions, true_scores
            )

        self.recorded_folds.append(fold)

    def _calculate_regression_metrics(
        self, fold, prediction_ids, predictions, true_scores
    ):
        mae = mean_absolute_error(true_scores, predictions)
        rmse = math.sqrt(mean_squared_error(true_scores, predictions))
        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_scores": true_scores,
            "mae": mae,
            "rmse": rmse,
        }

    def _calculate_classification_metrics(
        self, fold, prediction_ids, predictions, true_labels
    ):
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted"
        )
        roc_auc = (
            roc_auc_score(true_labels, predictions[:, 1])
            if self.num_classes == 2
            else None
        )
        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_labels": true_labels,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    def get_final_results(self):
        if len(self.recorded_folds) < self.num_folds:
            raise ValueError(
                f"All {self.num_folds} folds must be recorded before getting final results."
            )
        return self._aggregate_results()

    def _aggregate_results(self):
        if self.is_classification:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            metrics = ["mae", "rmse"]

        final_scores = {metric: [] for metric in metrics}
        for fold in range(self.num_folds):
            for metric in metrics:
                if metric in self.folds_results[fold]:
                    final_scores[metric].append(self.folds_results[fold][metric])

        return {
            f"mean_{metric}": np.mean(scores)
            for metric, scores in final_scores.items()
            if scores
        } | {
            f"std_{metric}": np.std(scores)
            for metric, scores in final_scores.items()
            if scores
        }

    def to_file(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self, f, default=self._json_serializable)

    @staticmethod
    def from_file(file_path: str):
        with open(file_path) as f:
            data = json.load(f)
        task = MatTextTask(
            task_name=data["task_name"],
            num_folds=data["num_folds"],
            is_classification=data["is_classification"],
            num_classes=data["num_classes"],
        )
        task.folds_results = data["folds_results"]
        task.recorded_folds = data["recorded_folds"]
        return task

    @staticmethod
    def _json_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, MatTextTask):
            return {
                "task_name": obj.task_name,
                "num_folds": obj.num_folds,
                "is_classification": obj.is_classification,
                "num_classes": obj.num_classes,
                "folds_results": obj.folds_results,
                "recorded_folds": obj.recorded_folds,
            }
        raise TypeError(f"Type {type(obj)} not serializable")
