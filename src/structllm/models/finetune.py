import torch
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TrainerCallback, TrainerControl
import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Union
import wandb


class CustomWandbCallback_FineTune(TrainerCallback):
    """Custom W&B callback for logging during training."""
    def on_log(self, args: Any, state: Any, control: Any, model: Any, logs: Dict[str, Union[float, Any]], **kwargs: Any) -> None:
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss


class FinetuneModel:
    """Class to perform finetuning of a language model."""
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg.model.finetune
        self.tokenizer_cfg = cfg.tokenizer
        self.context_length: int = self.cfg.context_length

        # Load the custom tokenizer using tokenizers library
        self._tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_cfg.path.tokenizer_path)
        self._wrapped_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        train_dataset = load_dataset("csv", data_files=self.cfg.path.finetune_traindata)
        self.tokenized_train_datasets = train_dataset.map(self._tokenize_pad_and_truncate, batched=True)

    def _wandb_callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_FineTune()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)
    


    def _compute_metrics(self, p: Any) -> Dict[str, float]:
        preds = p.predictions.squeeze()
        return {"rmse": torch.sqrt(((preds - p.label_ids) ** 2).mean()).item()}

    def finetune(self) -> None:
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint

        config_train_args = self.cfg.training_arguments
        callbacks = self._wandb_callbacks()

        training_args = TrainingArguments(
            **config_train_args
        )

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_ckpt,
                                                                   num_labels=1,
                                                                   ignore_mismatched_sizes=True)
        
        # for param in model.base_model.parameters():
        #     param.requires_grad = False


        trainer = Trainer(
            model=model.to("cuda"),
            args=training_args,
            data_collator=None,
            compute_metrics=self._compute_metrics,
            tokenizer=self._wrapped_tokenizer,
            train_dataset=self.tokenized_train_datasets['train'],
            callbacks=callbacks,
            # shuffle=True
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(model)})

        trainer.train()

        model.save_pretrained(self.cfg.path.finetuned_modelname)

    

