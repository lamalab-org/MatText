import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TrainerCallback
from omegaconf import DictConfig
from typing import Any, Dict, List
import wandb

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback


from structllm.models.utils import CustomWandbCallback_FineTune, TokenizerMixin



class FinetuneModelwithEval(TokenizerMixin):
    """Class to perform finetuning of a language model."""
    def __init__(self, cfg: DictConfig,local_rank=None) -> None:

        super().__init__(cfg.tokenizer)
        self.cfg = cfg.model.finetune
        self.local_rank = local_rank
        self.context_length: int = self.cfg.context_length

        
    
        train_df = pd.read_csv(self.cfg.path.finetune_traindata)
        train_df, val_df = train_test_split(train_df, test_size=0.1)

        # Convert DataFrames to Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        
        self.tokenized_train_datasets = DatasetDict({
            'train': train_dataset,
            'test': val_dataset
        })

        # Tokenize, pad, and truncate the datasets
        self.tokenized_train_datasets = {
            k: v.map(self._tokenize_pad_and_truncate, batched=True)
            for k, v in self.tokenized_train_datasets.items()
        }


    def _wandb_callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""

        return [CustomWandbCallback_FineTune()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)
    

    def _compute_metrics(self, p: Any, eval=True) -> Dict[str, float]:
        preds = torch.tensor(p.predictions.squeeze())  # Convert predictions to PyTorch tensor
        label_ids = torch.tensor(p.label_ids)  # Convert label_ids to PyTorch tensor

        if eval:
            return {"eval_rmse": torch.sqrt(((preds - label_ids) ** 2).mean()).item()}
        else:
            return {"train_rmse": torch.sqrt(((preds - label_ids) ** 2).mean()).item()}




    def finetune(self) -> None:
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint

        config_train_args = self.cfg.training_arguments
        callbacks = self._wandb_callbacks()

        train_dataset = self.tokenized_train_datasets['train']
        eval_dataset = self.tokenized_train_datasets['test']
        early_stopping = EarlyStoppingCallback(early_stopping_patience=6)

    

        training_args = TrainingArguments(
            **config_train_args,
            metric_for_best_model="eval_rmse",  # Metric to use for determining the best model
            greater_is_better=False,  # Lower eval_rmse is better
        )

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_ckpt,
                                                                   num_labels=1,
                                                                   ignore_mismatched_sizes=False)
        
        # for param in model.base_model.parameters():
        #     param.requires_grad = False

        if self.local_rank is not None:
            model = model.to(self.local_rank)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
        else:
            model = model.to("cuda")


        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=None,
            compute_metrics=self._compute_metrics,
            tokenizer=self._wrapped_tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[early_stopping] #+ callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(model)})

        trainer.train()

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        wandb.log(eval_result)

        model.save_pretrained(self.cfg.path.finetuned_modelname)

    

