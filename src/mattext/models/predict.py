from functools import partial
from typing import List, Tuple

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


class Inference(TokenizerMixin):
    """Class to perform inference on a language model with a sequence classification head."""

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
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """
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
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_Inference()]

    def predict(self):
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._callbacks()

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, num_labels=1, ignore_mismatched_sizes=False
        )

        trainer = Trainer(
            model=model.to("cuda"), data_collator=None, callbacks=callbacks
        )

        predictions = trainer.predict(self.tokenized_test_datasets)
        for callback in callbacks:
            callback.on_predict_end(
                None, None, None, model, predictions
            )  # Manually trigger callback
        torch.cuda.empty_cache()

        # TODO: Save predictions to disk optional
        # os.makedirs(self.cfg.path.predictions, exist_ok=True)
        # predictions_path = os.path.join(self.cfg.path.predictions, 'predictions.npy')
        # np.save(predictions_path, predictions.predictions)
        prediction_ids = self.tokenized_test_datasets["mbid"]
        self.prediction_ids = prediction_ids

        return pd.Series(predictions.predictions.flatten()), prediction_ids


class InferenceClassification(TokenizerMixin):
    """Class to perform inference on a language model with a sequence classification head for classification tasks."""

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
        self.num_labels = cfg.model.num_labels
        self.tokenized_test_datasets = self._prepare_datasets(self.cfg.path.test_data)
        self.prediction_ids = None

    def _prepare_datasets(self, path: str) -> DatasetDict:
        """
        Prepare test datasets.

        Args:
            path (str): Path to the test data.

        Returns:
            DatasetDict: Dictionary containing the test dataset.
        """
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
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_Inference()]

    def predict(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform prediction on the test dataset.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing the predictions as a DataFrame
            and the prediction IDs as a list.
        """
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._callbacks()

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, num_labels=self.num_labels, ignore_mismatched_sizes=False
        )

        trainer = Trainer(
            model=model.to("cuda"), data_collator=None, callbacks=callbacks
        )

        predictions = trainer.predict(self.tokenized_test_datasets)
        for callback in callbacks:
            callback.on_predict_end(
                None, None, None, model, predictions
            )  # Manually trigger callback
        torch.cuda.empty_cache()

        prediction_ids = self.tokenized_test_datasets["mbid"]
        self.prediction_ids = prediction_ids

        # Convert predictions to probabilities
        probabilities = torch.nn.functional.softmax(
            torch.from_numpy(predictions.predictions), dim=-1
        ).numpy()

        # Create a DataFrame with prediction probabilities
        prediction_df = pd.DataFrame(
            probabilities, columns=[f"class_{i}" for i in range(self.num_labels)]
        )

        return prediction_df, prediction_ids

    def evaluate(self, true_labels: List[int]) -> dict:
        """
        Evaluate the model's predictions against true labels.

        Args:
            true_labels (List[int]): The true labels for the test set.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """

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
