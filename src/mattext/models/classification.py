from functools import partial
from typing import Any, Dict, List

import numpy as np
import wandb
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from mattext.models.utils import (
    CustomWandbCallback_FineTune,
    EvaluateFirstStepCallback,
    TokenizerMixin,
)


class FinetuneClassificationModel(TokenizerMixin):
    """Class to perform finetuning of a language model.
        Initialize the FinetuneModel.

    Args:
        cfg (DictConfig): Configuration for the fine-tuning.
        local_rank (int, optional): Local rank for distributed training. Defaults to None.
    """

    def __init__(self, cfg: DictConfig, local_rank=None, fold="fold_0") -> None:
        super().__init__(
            cfg=cfg.model.representation,
            special_tokens=cfg.model.special_tokens,
            special_num_token=cfg.model.special_num_token,
        )
        self.fold = fold
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.data_repository = cfg.model.data_repository
        self.cfg = cfg.model.finetune
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.tokenized_dataset = self._prepare_datasets(
            self.cfg.path.finetune_traindata
        )

    def _prepare_datasets(self, subset: str) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """

        def replace_none(example, replacement="[PAD]"):
            for key, value in example.items():
                if value is None:
                    example[key] = replacement
            return example

        ds = load_dataset(self.data_repository, subset)
        dataset = ds[self.fold].train_test_split(shuffle=True, test_size=0.2, seed=42)
        dataset = dataset.filter(
            lambda example: example[self.representation] is not None
        )
        return dataset.map(
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
            callbacks.append(CustomWandbCallback_FineTune())

        callbacks.append(EvaluateFirstStepCallback)

        return callbacks

    def _compute_metrics(self, p: Any) -> Dict[str, float]:
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_argmax = np.argmax(preds, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_argmax, average='weighted')
        acc = accuracy_score(labels, preds_argmax)

        # Compute ROC AUC
        n_classes = preds.shape[1]
        if n_classes == 2:
            # Binary classification
            roc_auc = roc_auc_score(labels, preds[:, 1])
        else:
            # Multi-class classification
            labels_binarized = label_binarize(labels, classes=range(n_classes))
            roc_auc = roc_auc_score(labels_binarized, preds, average='weighted', multi_class='ovr')

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

    def finetune(self) -> None:
        """
        Perform fine-tuning of the language model.
        """

        pretrained_ckpt = self.cfg.path.pretrained_checkpoint

        config_train_args = self.cfg.training_arguments
        callbacks = self._callbacks()

        training_args = TrainingArguments(
            **config_train_args,
            metric_for_best_model="f1",  # or "accuracy", depending on your preference
            greater_is_better=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, num_labels=2, ignore_mismatched_sizes=False
        )

        if self.cfg.freeze_base_model:
            for param in model.base_model.parameters():
                param.requires_grad = False

        if self.local_rank is not None:
            model = model.to(self.local_rank)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )
        else:
            model = model.to("cuda")

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=None,
            compute_metrics=self._compute_metrics,
            tokenizer=self._wrapped_tokenizer,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(model)})

        trainer.train()

        eval_result = trainer.evaluate(eval_dataset=self.tokenized_dataset["test"])
        wandb.log(eval_result)

        model.save_pretrained(self.cfg.path.finetuned_modelname)
        wandb.finish()
        return self.cfg.path.finetuned_modelname

    def evaluate(self):
        """
        Evaluate the fine-tuned model on the test dataset.
        """
        ckpt = self.finetune()
