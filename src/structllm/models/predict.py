import os
import torch
from torch.nn import DataParallel
import wandb
import pandas as pd
import numpy as np
from transformers import  AutoModelForSequenceClassification, TrainerCallback, Trainer
from datasets import load_dataset
from omegaconf import DictConfig
from typing import Any, Dict, List, Union

from structllm.models.utils import CustomWandbCallback_Inference, TokenizerMixin



class Inference(TokenizerMixin):
    def __init__(self, cfg: DictConfig):

        super().__init__(cfg.tokenizer)
        self.cfg = cfg.model.inference
        self.tokenizer_cfg = cfg.tokenizer
        self.context_length: int = self.cfg.context_length

        
        test_data = load_dataset("csv", data_files=self.cfg.path.test_data)
        self.tokenized_test_datasets = test_data.map(self._tokenize_pad_and_truncate, batched=True)

    def _callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_Inference()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def predict(self):
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._callbacks()


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

        

            

