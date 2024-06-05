from functools import partial
from typing import List

import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
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
