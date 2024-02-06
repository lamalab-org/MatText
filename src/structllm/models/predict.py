import json
import os
import torch
from torch.nn import DataParallel
import wandb
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import pipeline
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainerCallback, Trainer
from datasets import load_dataset
from omegaconf import DictConfig
from typing import Any, Dict, List, Union
from structllm.tokenizer.slice_tokenizer import AtomVocabTokenizer


class CustomWandbCallback_Inference(TrainerCallback):
    """Custom W&B callback for logging during inference."""

    def __init__(self):
        self.predictions = []

    def on_predict_end(self, args: Any, state: Any, control: Any, model: Any, predictions: Any, **kwargs: Any) -> None:
        wandb.log({"predictions": predictions.predictions, })

class TokenizerMixin:
    """Mixin class to handle tokenizer functionality."""

    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg
        self._wrapped_tokenizer = None

        if self.tokenizer_cfg.name == "atom":
            self._wrapped_tokenizer = AtomVocabTokenizer(
                self.tokenizer_cfg.path.tokenizer_path, model_max_length=512, truncation=False, padding=False
            )
        else:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_cfg.path.tokenizer_path)
            self._wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self._tokenizer)

        special_tokens = {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        self._wrapped_tokenizer.add_special_tokens(special_tokens)

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any], context_length: int) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=context_length)



class Inference:
    def __init__(self, cfg: DictConfig):

        super().__init__(cfg.tokenizer)
        self.cfg = cfg.model.inference
        self.tokenizer_cfg = cfg.tokenizer
        self.context_length: int = self.cfg.context_length

        
        test_data = load_dataset("csv", data_files=self.cfg.path.test_data)
        self.tokenized_test_datasets = test_data.map(self._tokenize_pad_and_truncate, batched=True)

    def _wandb_callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_Inference()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def predict(self):
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._wandb_callbacks()


        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, 
            num_labels=1, 
            ignore_mismatched_sizes=False
        )

        trainer = Trainer(
            model=model.to("cuda"),
            data_collator=None,
            callbacks = callbacks
        )

        predictions = trainer.predict(self.tokenized_test_datasets['train'])
        for callback in callbacks:
                callback.on_predict_end(None, None, None, model, predictions)  # Manually trigger callback
        torch.cuda.empty_cache()


        os.makedirs(self.cfg.path.predictions, exist_ok=True)
        predictions_path = os.path.join(self.cfg.path.predictions, 'predictions.npy')
        np.save(predictions_path, predictions.predictions)


        return pd.Series(predictions.predictions.flatten())

        

            

