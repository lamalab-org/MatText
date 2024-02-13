from typing import Any, List, Dict, Union
import wandb
from omegaconf import DictConfig

from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import TrainerCallback


from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import EarlyStoppingCallback
from structllm.models.utils import CustomWandbCallback_Pretrain, TokenizerMixin
from functools import partial


class PretrainModel(TokenizerMixin):
    """Class to perform pretraining of a language model."""
    def __init__(self, cfg: DictConfig, local_rank=None):

        super().__init__(cfg.tokenizer)
        self.cfg = cfg.model.pretrain
        self.local_rank = local_rank
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.model_name_or_path: str = self.cfg.model_name_or_path

        self.tokenized_dataset = self._prepare_datasets

    @property
    def _prepare_datasets(self) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """

        # train_df = pd.read_csv(self.cfg.path.traindata,low_memory=False)
        # val_df = pd.read_csv(self.cfg.path.evaldata,low_memory=False)

        # datasets = DatasetDict({
        #     'train': Dataset.from_pandas(train_df),
        #     'test': Dataset.from_pandas(val_df)
        # })

        # # Tokenize, pad, and truncate the datasets
        # tokenized_datasets = {
        #     k: v.map(partial(self._tokenize_pad_and_truncate, context_length=self.context_length), batched=True)
        #     for k, v in datasets.items()
        # }

        # Load train and test datasets
        train_dataset = load_dataset("csv", data_files=self.cfg.path.traindata)
        eval_dataset = load_dataset("csv", data_files=self.cfg.path.evaldata)

        self.tokenized_train_datasets = train_dataset.map(partial(self._tokenize_pad_and_truncate, context_length=self.context_length), batched=True)
        self.tokenized_eval_datasets = eval_dataset.map(partial(self._tokenize_pad_and_truncate, context_length=self.context_length), batched=True)


    
    def _callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for early stopping, and custom logging."""
        callbacks = []

        if self.callbacks.early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.callbacks.early_stopping_patience,
                early_stopping_threshold=self.callbacks.early_stopping_threshold
            ))

        if self.callbacks.custom_logger:
            callbacks.append(CustomWandbCallback_Pretrain())

        return callbacks


    def pretrain_mlm(self) -> None:
        """Performs MLM pretraining of the language model."""
        config_mlm = self.cfg.mlm
        config_train_args = self.cfg.training_arguments
        config_model_args = self.cfg.model_config
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._wrapped_tokenizer,
            mlm=config_mlm.is_mlm,
            mlm_probability=config_mlm.mlm_probability
        )

        callbacks = self._callbacks()

        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            **config_model_args
        )
        
        model = AutoModelForMaskedLM.from_config(config)

        if self.local_rank is not None:
            model = model.to(self.local_rank)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
        else:
            model = model.to("cuda")
        
        training_args = TrainingArguments(
            **config_train_args
        )
    
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=self.tokenized_train_datasets['train'],
            eval_dataset=self.tokenized_eval_datasets['train'],
            args=training_args,
            callbacks= callbacks
        )

        wandb.log({"config_details": str(config)}) 
        wandb.log({"Training Arguments": str(config_train_args)}) 
        wandb.log({"model_summary": str(model)}) 
        
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.path.finetuned_modelname)

