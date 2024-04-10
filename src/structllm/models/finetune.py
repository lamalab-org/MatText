from functools import partial
from typing import Any, Dict, List

import torch
import wandb
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from structllm.models.utils import CustomWandbCallback_FineTune, TokenizerMixin, EvaluateFirstStepCallback


class FinetuneModel(TokenizerMixin):
    """Class to perform finetuning of a language model.
        Initialize the FinetuneModel.

    Args:
        cfg (DictConfig): Configuration for the fine-tuning.
        local_rank (int, optional): Local rank for distributed training. Defaults to None.
    """
    def __init__(self, cfg: DictConfig,local_rank=None) -> None:

        super().__init__(cfg=cfg.model.representation,special_tokens=cfg.model.special_tokens)
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.num_data_points_to_train_on = cfg.model.num_data_points_to_train_on if hasattr(cfg.model, 'num_data_points_to_train_on') else None
        self.cfg = cfg.model.finetune
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.tokenized_dataset = self._prepare_datasets(self.cfg.path.finetune_traindata)


    def _prepare_datasets(self, path: str) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """

        ds = load_dataset("json", data_files=path,split="train")
        dataset = ds.train_test_split(shuffle=True, test_size=0.2, seed=42)
        #dataset= dataset.filter(lambda example: example[self.representation] is not None)

        if self.num_data_points_to_train_on:
            seed = 42
            dataset['train'] = dataset['train'].shuffle(seed=seed).select(range(self.num_data_points_to_train_on))

        print(dataset)
        return dataset.map(
            partial(self._tokenize_pad_and_truncate, context_length=self.context_length),
            batched=True)

    def _callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for early stopping, and custom logging."""
        callbacks = []

        if self.callbacks.early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.callbacks.early_stopping_patience,
                early_stopping_threshold=self.callbacks.early_stopping_threshold
            ))

        if self.callbacks.custom_logger:
            callbacks.append(CustomWandbCallback_FineTune())

        callbacks.append(EvaluateFirstStepCallback)

        return callbacks

    def _compute_metrics(self, p: Any, eval=True) -> Dict[str, float]:
        preds = torch.tensor(p.predictions.squeeze())  # Convert predictions to PyTorch tensor
        label_ids = torch.tensor(p.label_ids)  # Convert label_ids to PyTorch tensor

        if eval:
            # Calculate RMSE as evaluation metric
            eval_rmse = torch.sqrt(((preds - label_ids) ** 2).mean()).item()
            return {"eval_rmse": round(eval_rmse, 3)}
        else:
            # Calculate RMSE as training metric
            loss = torch.sqrt(((preds - label_ids) ** 2).mean()).item()
            return {"train_rmse": round(loss, 3), "loss": round(loss, 3)}


    def finetune(self) -> None:
        """
        Perform fine-tuning of the language model.
        """

        pretrained_ckpt = self.cfg.path.pretrained_checkpoint

        config_train_args = self.cfg.training_arguments
        callbacks = self._callbacks()


        training_args = TrainingArguments(
            **config_train_args,
            metric_for_best_model="eval_rmse",  # Metric to use for determining the best model
            greater_is_better=False,  # Lower eval_rmse is better
        )

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_ckpt,
                                                                   num_labels=1,
                                                                   ignore_mismatched_sizes=False)

        if self.cfg.freeze_base_model:
            for param in model.base_model.parameters():
                param.requires_grad = False

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
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(model)})

        trainer.train()

        eval_result = trainer.evaluate(eval_dataset=self.tokenized_dataset['test'])
        wandb.log(eval_result)

        model.save_pretrained(self.cfg.path.finetuned_modelname)
        wandb.finish()
        return self.cfg.path.finetuned_modelname

    def evaluate(self):
        """
        Evaluate the fine-tuned model on the test dataset.
        """
        ckpt = self.finetune()






