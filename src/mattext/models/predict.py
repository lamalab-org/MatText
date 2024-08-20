from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, Trainer, TrainerCallback

from mattext.models.utils import CustomWandbCallback_Inference, TokenizerMixin


class BaseInference(TokenizerMixin, ABC):
    def __init__(self, cfg: DictConfig, fold="fold_0"):
        super().__init__(
            cfg=cfg.model.representation,
            special_tokens=cfg.model.special_tokens,
            special_num_token=cfg.model.special_num_token,
        )
        self.fold = fold
        self.representation = cfg.model.representation
        self.data_repository = cfg.model.data_repository
        self.dataset_name = cfg.model.finetune.dataset_name
        self.cfg = cfg.model.inference
        self.context_length: int = self.cfg.context_length
        self.tokenized_test_datasets = self._prepare_datasets(self.cfg.path.test_data)
        self.prediction_ids = None

    def _prepare_datasets(self, path: str) -> DatasetDict:
        dataset = load_dataset(self.data_repository, path)
        filtered_dataset = dataset[self.fold].filter(
            lambda example: example[self.representation] is not None
        )
        return filtered_dataset.map(
            partial(
                self._tokenize_pad_and_truncate, context_length=self.context_length
            ),
            batched=True,
        )

    def _callbacks(self) -> List[TrainerCallback]:
        return [CustomWandbCallback_Inference()]

    @abstractmethod
    def get_model(self, pretrained_ckpt: str):
        pass

    @abstractmethod
    def process_predictions(self, predictions) -> Union[pd.Series, pd.DataFrame]:
        pass

    def predict(self) -> Tuple[Union[pd.Series, pd.DataFrame], List[str]]:
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._callbacks()

        model = self.get_model(pretrained_ckpt)

        trainer = Trainer(
            model=model.to("cuda"), data_collator=None, callbacks=callbacks
        )

        predictions = trainer.predict(self.tokenized_test_datasets)
        for callback in callbacks:
            callback.on_predict_end(None, None, None, model, predictions)
        torch.cuda.empty_cache()

        prediction_ids = self.tokenized_test_datasets["mbid"]
        self.prediction_ids = prediction_ids

        processed_predictions = self.process_predictions(predictions)

        return processed_predictions, prediction_ids


class Inference(BaseInference):
    def get_model(self, pretrained_ckpt: str):
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, num_labels=1, ignore_mismatched_sizes=False
        )

    def process_predictions(self, predictions) -> pd.Series:
        return pd.Series(predictions.predictions.flatten())


class InferenceClassification(BaseInference):
    def __init__(self, cfg: DictConfig, fold="fold_0"):
        super().__init__(cfg, fold)
        self.num_labels = 2  # You might want to make this configurable

    def get_model(self, pretrained_ckpt: str):
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, num_labels=self.num_labels, ignore_mismatched_sizes=False
        )

    def process_predictions(self, predictions) -> pd.DataFrame:
        probabilities = torch.nn.functional.softmax(
            torch.from_numpy(predictions.predictions), dim=-1
        ).numpy()
        return pd.DataFrame(
            probabilities, columns=[f"class_{i}" for i in range(self.num_labels)]
        )

    def evaluate(self, true_labels: List[int]) -> dict:
        predictions, _ = self.predict()
        pred_labels = np.argmax(predictions.values, axis=1)

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted"
        )

        if self.num_labels == 2:
            roc_auc = roc_auc_score(true_labels, predictions.iloc[:, 1])
        else:
            true_labels_binarized = label_binarize(
                true_labels, classes=range(self.num_labels)
            )
            roc_auc = roc_auc_score(
                true_labels_binarized,
                predictions,
                average="weighted",
                multi_class="ovr",
            )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }
