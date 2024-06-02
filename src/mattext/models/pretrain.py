from functools import partial
from typing import List

import wandb
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from mattext.models.utils import CustomWandbCallback_Pretrain, TokenizerMixin


class PretrainModel(TokenizerMixin):
    """Class to perform pretraining of a language model."""

    def __init__(self, cfg: DictConfig, local_rank=None):

        super().__init__(
            cfg=cfg.model.representation,
            special_tokens=cfg.model.special_tokens,
            special_num_token=cfg.model.special_num_token,
        )
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.data_repository = cfg.model.data_repository
        self.cfg = cfg.model.pretrain
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.tokenized_train_datasets,self.tokenized_eval_datasets = self._prepare_datasets(
            subset=self.cfg.dataset_name
        )


    def _prepare_datasets(self, subset: str) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """
        dataset = load_dataset(self.data_repository, subset)
        filtered_dataset = dataset.filter(
            lambda example: example[self.representation] is not None
        )

        return filtered_dataset['train'].map(
            partial(
                self._tokenize_pad_and_truncate, context_length=self.context_length
            ),
            batched=True,
        ), filtered_dataset['test'].map(
            partial(
                self._tokenize_pad_and_truncate, context_length=self.context_length
            ),
            batched=True,
        )

    def _callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for early stopping, and custom logging."""
        callbacks = []

        if self.callbacks.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.callbacks.early_stopping_patience,
                    early_stopping_threshold=self.callbacks.early_stopping_threshold,
                )
            )

        if self.callbacks.custom_logger:
            callbacks.append(CustomWandbCallback_Pretrain())

        return callbacks

    def pretrain_mlm(self) -> None:
        """Performs MLM pretraining of the language model."""
        config_mlm = self.cfg.mlm
        config_train_args = self.cfg.training_arguments
        config_model_args = self.cfg.model_config
        config_model_args["max_position_embeddings"] = self.context_length

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._wrapped_tokenizer,
            mlm=config_mlm.is_mlm,
            mlm_probability=config_mlm.mlm_probability,
        )

        callbacks = self._callbacks()

        config = AutoConfig.from_pretrained(
            self.model_name_or_path, **config_model_args
        )

        model = AutoModelForMaskedLM.from_config(config)

        if self.local_rank is not None:
            model = model.to(self.local_rank)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )
        else:
            model = model.to("cuda")

        training_args = TrainingArguments(**config_train_args)

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=self.tokenized_train_datasets,
            eval_dataset=self.tokenized_eval_datasets,
            args=training_args,
            callbacks=callbacks,
        )

        wandb.log({"config_details": str(config)})
        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(model)})

        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.path.finetuned_modelname)
