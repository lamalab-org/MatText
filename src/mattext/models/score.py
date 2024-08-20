import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import jsonpickle
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
class BaseMatTextTask:
    task_name: str
    num_folds: int = 5
    folds_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recorded_folds: List[int] = field(default_factory=list)

    def record_fold(
        self, fold: int, prediction_ids: List[str], predictions: List[float]
    ):
        if fold in self.recorded_folds:
            raise ValueError(f"Fold {fold} has already been recorded.")
        true_scores = load_true_scores(self.task_name, prediction_ids)
        self._calculate_metrics(fold, prediction_ids, predictions, true_scores)
        self.recorded_folds.append(fold)

    def _calculate_metrics(self, fold, prediction_ids, predictions, true_scores):
        raise NotImplementedError("Subclasses must implement this method")

    def get_final_results(self):
        if len(self.recorded_folds) < self.num_folds:
            raise ValueError(
                f"All {self.num_folds} folds must be recorded before getting final results."
            )
        return self._aggregate_results()

    def _aggregate_results(self):
        raise NotImplementedError("Subclasses must implement this method")

    def to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(jsonpickle.encode(self))

    @staticmethod
    def from_file(file_path: str):
        with open(file_path) as f:
            return jsonpickle.decode(f.read())


@dataclass
class MatTextTask(BaseMatTextTask):
    def _calculate_metrics(self, fold, prediction_ids, predictions, true_scores):
        mae = mean_absolute_error(true_scores, predictions)
        rmse = math.sqrt(mean_squared_error(true_scores, predictions))
        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_scores": true_scores,
            "mae": mae,
            "rmse": rmse,
        }

    def _aggregate_results(self):
        final_scores_mae = [
            self.folds_results[fold]["mae"] for fold in range(self.num_folds)
        ]
        final_scores_rmse = [
            self.folds_results[fold]["rmse"] for fold in range(self.num_folds)
        ]
        return {
            "mean_mae_score": np.mean(final_scores_mae),
            "std_mae_score": np.std(final_scores_mae),
            "mean_rmse_score": np.mean(final_scores_rmse),
            "std_rmse_score": np.std(final_scores_rmse),
        }


@dataclass
class MatTextClassificationTask(BaseMatTextTask):
    num_classes: int = 2

    def _calculate_metrics(self, fold, prediction_ids, predictions, true_labels):
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted"
        )
        roc_auc = roc_auc_score(true_labels, predictions[:, 1])
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

    def _aggregate_results(self):
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        final_scores = {metric: [] for metric in metrics}
        for fold in range(self.num_folds):
            for metric in metrics:
                final_scores[metric].append(self.folds_results[fold][metric])
        return {
            f"mean_{metric}": np.mean(scores) for metric, scores in final_scores.items()
        } | {f"std_{metric}": np.std(scores) for metric, scores in final_scores.items()}
