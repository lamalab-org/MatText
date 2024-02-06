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
from structllm.tokenizer.slice_tokenizer import AtomVocabTokenizer
from datasets import load_dataset

from torch import nn
from torch.nn.parallel import DistributedDataParallel



class CustomWandbCallback(TrainerCallback):
    """Custom W&B callback for logging during training."""
    def on_log(self, args: Any, state: Any, control: Any, model: Any, logs: Dict[str, Union[float, Any]], **kwargs: Any) -> None:
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss


class PretrainModel:
    """Class to perform pretraining of a language model."""
    def __init__(self, cfg: DictConfig, local_rank=None):

        self.cfg = cfg.model.pretrain
        self.tokenizer_cfg = cfg.tokenizer
        self.local_rank = local_rank
        self.context_length: int = self.cfg.context_length
        self.model_name_or_path: str = self.cfg.model_name_or_path

        print("tokenizer")
        print(self.tokenizer_cfg.name)
        
        if self.tokenizer_cfg.name == "atom":
            tokenizer = AtomVocabTokenizer(self.tokenizer_cfg.path.tokenizer_path, model_max_length=512, truncation=False, padding=False)
        else:
            self._tokenizer: Tokenizer = Tokenizer.from_file(self.tokenizer_cfg.path.tokenizer_path)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self._tokenizer,
            )

        special_tokens = {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        tokenizer.add_special_tokens(special_tokens)
        self._wrapped_tokenizer = tokenizer
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
            callbacks=callbacks
        )

        wandb.log({"config_details": str(config)}) 
        wandb.log({"Training Arguments": str(config_train_args)}) 
        wandb.log({"model_summary": str(model)}) 
        
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.path.finetuned_modelname)

