import json
import math
from dataclasses import asdict, dataclass, field
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
from sklearn.preprocessing import label_binarize

MATTEXT_MATBENCH = {
    "kvrh": "matbench_log_kvrh",
    "gvrh": "matbench_log_gvrh",
    "perovskites": "matbench_perovskites",
    "bandgap" : "matbench_mp_gap",
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

METRIC_MAP = {
    "mae": mean_absolute_error,
    "rmse": lambda true, pred: math.sqrt(mean_squared_error(true, pred)),
}


def fold_key_namer(fold_key):
    return f"fold_{fold_key}"


def load_true_scores(dataset, mbids):
    data_frame = load(MATTEXT_MATBENCH[dataset])
    print(MATMINER_COLUMNS)
    scores = []
    for mbid in mbids:
        # Get the score for the mbid
        score = data_frame.loc[mbid][MATMINER_COLUMNS[dataset]]
        scores.append(score)
    return scores


def mattext_score(prediction_ids, predictions, task_name):
    true = load_true_scores(task_name, prediction_ids)
    return mean_squared_error(true, predictions)


@dataclass
class MatTextTask:
    task_name: str
    num_folds: int = 5
    # metric: str
    folds_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recorded_folds: List[int] = field(default_factory=list)

    def record_fold(
        self, fold: int, prediction_ids: List[str], predictions: List[float]
    ):
        if fold in self.recorded_folds:
            raise ValueError(f"Fold {fold} has already been recorded.")
        true_scores = load_true_scores(self.task_name, prediction_ids)
        mae = mean_absolute_error(true_scores, predictions)
        rmse = math.sqrt(mean_squared_error(true_scores, predictions))
        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_scores": true_scores,
            "mae": mae,
            "rmse": rmse,
        }
        self.recorded_folds.append(fold)

    def get_final_results(self):
        if len(self.recorded_folds) < self.num_folds:
            raise ValueError(
                f"All {self.num_folds} folds must be recorded before getting final results."
            )
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
            "std_score": np.std(final_scores_mae),
        }

    def to_file(self, file_path: str):
        final_results = (
            self.get_final_results()
            if len(self.recorded_folds) == self.num_folds
            else {}
        )
        data_to_save = asdict(self)
        data_to_save["final_results"] = final_results
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, default=self._json_serializable)

    @staticmethod
    def from_file(file_path: str):
        with open(file_path) as f:
            data = json.load(f)
        task = MatTextTask(task_name=data["task_name"], metric=data["metric"])
        task.folds_results = data["folds_results"]
        task.recorded_folds = data["recorded_folds"]
        return task

    @staticmethod
    def _prepare_for_serialization(obj):
        if isinstance(obj, dict):
            return {
                k: MatTextTask._prepare_for_serialization(v) for k, v in obj.items()
            }
        elif (
            isinstance(obj, (list, pd.Series, np.ndarray))
        ):
            return MatTextTask._prepare_for_serialization(obj.tolist())
        else:
            return obj

    @staticmethod
    def _json_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")




@dataclass
class MatTextClassificationTask:
    task_name: str
    num_folds: int = 5
    num_classes: int = 2
    folds_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recorded_folds: List[int] = field(default_factory=list)

    def record_fold(
        self, fold: int, prediction_ids: List[str], predictions: List[float]
    ):
        if fold in self.recorded_folds:
            raise ValueError(f"Fold {fold} has already been recorded.")

        true_labels = self.load_true_labels(self.task_name, prediction_ids)
        pred_labels = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        roc_auc = roc_auc_score(true_labels, predictions[:, 1])

        # Compute ROC AUC
        # if self.num_classes == 2:
        #     roc_auc = roc_auc_score(true_labels, predictions[:, 1])
        # else:
        #     true_labels_binarized = label_binarize(true_labels, classes=range(self.num_classes))
        #     roc_auc = roc_auc_score(true_labels_binarized, predictions, average='weighted', multi_class='ovr')

        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_labels": true_labels,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
        self.recorded_folds.append(fold)

    def get_final_results(self):
        if len(self.recorded_folds) < self.num_folds:
            raise ValueError(
                f"All {self.num_folds} folds must be recorded before getting final results."
            )
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        final_scores = {metric: [] for metric in metrics}

        for fold in range(self.num_folds):
            for metric in metrics:
                final_scores[metric].append(self.folds_results[fold][metric])

        return {
            f"mean_{metric}": np.mean(scores) for metric, scores in final_scores.items()
        } | {
            f"std_{metric}": np.std(scores) for metric, scores in final_scores.items()
        }

    def to_file(self, file_path: str):
        final_results = (
            self.get_final_results()
            if len(self.recorded_folds) == self.num_folds
            else {}
        )
        data_to_save = asdict(self)
        data_to_save["final_results"] = final_results
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, default=self._json_serializable)

    @staticmethod
    def from_file(file_path: str):
        with open(file_path) as f:
            data = json.load(f)
        task = MatTextClassificationTask(task_name=data["task_name"], num_classes=data["num_classes"])
        task.folds_results = data["folds_results"]
        task.recorded_folds = data["recorded_folds"]
        return task

    @staticmethod
    def _prepare_for_serialization(obj):
        if isinstance(obj, dict):
            return {
                k: MatTextClassificationTask._prepare_for_serialization(v) for k, v in obj.items()
            }
        elif isinstance(obj, (list, pd.Series, np.ndarray)):
            return MatTextClassificationTask._prepare_for_serialization(obj.tolist())
        else:
            return obj

    @staticmethod
    def _json_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    # @staticmethod
    # def load_true_labels(dataset, mbids):
    #     raise NotImplementedError("load_true_labels method needs to be implemented")