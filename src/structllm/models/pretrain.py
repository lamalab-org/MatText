from typing import Any, List, Dict, Union

import os
import wandb
import hydra
from omegaconf import DictConfig

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import TrainerCallback, TrainerControl


class CustomWandbCallback(TrainerCallback):
    """Custom W&B callback for logging during training."""
    def on_log(self, args: Any, state: Any, control: Any, model: Any, logs: Dict[str, Union[float, Any]], **kwargs: Any) -> None:
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss


class PretrainModel:
    """Class to perform pretraining of a language model."""
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg.model.pretrain
        self.tokenizer_cfg = cfg.tokenizer
        self.context_length: int = self.cfg.context_length
        self.model_name_or_path: str = self.cfg.model_name_or_path
        
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

        train_dataset = load_dataset("csv", data_files=self.cfg.path.traindata)
        eval_dataset = load_dataset("csv", data_files=self.cfg.path.evaldata)

        self.tokenized_train_datasets = train_dataset.map(self._tokenize_pad_and_truncate, batched=True)
        self.tokenized_eval_datasets = eval_dataset.map(self._tokenize_pad_and_truncate, batched=True)

    def _wandb_callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

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

        callbacks = self._wandb_callbacks()

        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            **config_model_args
        )
         
        model = AutoModelForMaskedLM.from_config(config).to("cuda")
        
        training_args = TrainingArguments(
            **config_train_args
        )
    
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=self.tokenized_train_datasets['train'],
            eval_dataset=self.tokenized_eval_datasets['train'],
            args=training_args,
            callbacks=callbacks
        )

        wandb.log({"config_details": str(config)}) 
        wandb.log({"Training Arguments": str(config_train_args)}) 
        wandb.log({"model_summary": str(model)}) 
        
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.path.finetuned_modelname)

