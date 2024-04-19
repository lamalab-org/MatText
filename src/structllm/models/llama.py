import torch
import os

import numpy as np
import pandas as pd

from transformers import ( 
    LlamaTokenizer, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
)

from torch.utils.data import Dataset

from peft import (
    LoraConfig, 
    get_peft_model, 
)

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

from structllm.models.utils import CustomWandbCallback_FineTune, EvaluateFirstStepCallback


IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"




def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer),pad_to_multiple_of=8)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
     #   output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
     #   output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
     #   output_embeddings[-num_new_tokens:] = output_embeddings_avg


class FinetuneLLama():
    """Class to perform finetuning of a language model.
        Initialize the FinetuneModel.

    Args:
        cfg (DictConfig): Configuration for the fine-tuning.
        local_rank (int, optional): Local rank for distributed training. Defaults to None.
    """
    def __init__(self, cfg: DictConfig,local_rank=None) -> None:
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.cfg = cfg.model.finetune
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.model, self.tokenizer = self._setup_model_tokenizer()
        self.tokenized_dataset = self._prepare_datasets(self.cfg.path.finetune_traindata)

    def _setup_model_tokenizer(self) -> None:

        pretrained_ckpt = "meta-llama/Llama-2-7b-hf"

        llama_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_ckpt,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_ckpt,
                                                                   num_labels=1,
                                                                   ignore_mismatched_sizes=False,
                                                                   load_in_8bit=True,
                                                                    device_map={"": self.local_rank},)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        

        if self.cfg.freeze_base_model:
            for param in model.base_model.parameters():
                param.requires_grad = False

        
        special_tokens_dict = dict()
        if llama_tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if llama_tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if llama_tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if llama_tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        # llama_tokenizer.add_special_tokens(special_tokens_dict)

        

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            llama_tokenizer=llama_tokenizer,
            model=model,
        )
        # llama_tokenizer.pad_token = llama_tokenizer.eos_token
        # print(llama_tokenizer.pad_token)
        # print(llama_tokenizer.eos_token)
        print(len(llama_tokenizer))

        return model, llama_tokenizer
    
    def _tokenize(self, examples):
        # Tokenize the 'crystal_llm' column using the LAMA tokenizer
        tokenized_examples = self.tokenizer(examples["crystal_llm_rep"])
        return tokenized_examples
    


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
        return dataset.map(self._tokenize, batched=True)

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

        

        config_train_args = self.cfg.training_arguments
        callbacks = self._callbacks()

        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        training_args = TrainingArguments(
            fsdp=False,
            fp16=True,
            bf16=False,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
            **config_train_args,
            metric_for_best_model="eval_rmse",  # Metric to use for determining the best model
            greater_is_better=False,  # Lower eval_rmse is better
            dataloader_num_workers=8,
        )

        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=None,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(self.model)})

        trainer.train()

        eval_result = trainer.evaluate(eval_dataset=self.tokenized_dataset['test'])
        wandb.log(eval_result)

        self.model.save_pretrained(self.cfg.path.finetuned_modelname)
        wandb.finish()
        return self.cfg.path.finetuned_modelname

    def evaluate(self):
        """
        Evaluate the fine-tuned model on the test dataset.
        """
        ckpt = self.finetune()






