import math

import numpy as np
from matbench.data_ops import load
import json
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from dataclasses import dataclass, field,asdict
from typing import List, Dict, Any


MATTEXT_MATBENCH = {
    "kvrh": "matbench_log_kvrh",
    "gvrh": "matbench_log_gvrh",
    "perovskites": "matbench_perovskites",
}

METRIC_MAP = {
    "mae": mean_absolute_error,
    "rmse": lambda true, pred: math.sqrt(mean_squared_error(true, pred)),
}


def fold_key_namer(fold_key):
    return f"fold_{fold_key}"


def load_true_scores(dataset, mbids):
    data_frame = load(MATTEXT_MATBENCH[dataset])
    scores = []
    for mbid in mbids:
        # Get the score for the mbid
        score = data_frame.loc[mbid]["log10(K_VRH)"]
        scores.append(score)
    return scores


def mattext_score(prediction_ids, predictions, task_name):
    true = load_true_scores(task_name, prediction_ids)
    return mean_squared_error(true, predictions)


# make a data class for property which has 5 folds and when called associated fold value is recorded


@dataclass
class Task:
    task_name: str
    metric: str
    folds_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recorded_folds: List[int] = field(default_factory=list)

    def record_fold(self, fold: int, prediction_ids: List[str], predictions: List[float]):
        if fold in self.recorded_folds:
            raise ValueError(f"Fold {fold} has already been recorded.")
        true_scores = load_true_scores(self.task_name, prediction_ids)
        metric_function = METRIC_MAP[self.metric]
        score = metric_function(true_scores, predictions)
        self.folds_results[fold] = {
            "prediction_ids": prediction_ids,
            "predictions": predictions,
            "true_scores": true_scores,
            "score": score
        }
        self.recorded_folds.append(fold)
    
    def get_final_results(self):
        if len(self.recorded_folds) < 5:
            raise ValueError("All 5 folds must be recorded before getting final results.")
        final_scores = [self.folds_results[fold]["score"] for fold in range(5)]
        final_result = {
            "mean_score": np.mean(final_scores),
            "std_score": np.std(final_scores)
        }
        return final_result

    def to_file(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, default=self._json_serializable)
    
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        task = Task(task_name=data["task_name"], metric=data["metric"])
        task.folds_results = data["folds_results"]
        task.recorded_folds = data["recorded_folds"]
        return task

    @staticmethod
    def _json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")