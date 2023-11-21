import json
import torch
from torch.nn import DataParallel
import wandb
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainerCallback
from datasets import load_dataset
from omegaconf import DictConfig
from typing import Any, Dict, List, Union


class CustomWandbCallback_Inference(TrainerCallback):
    """Custom W&B callback for logging during inference."""

    def __init__(self):
        self.predictions = []

    def on_predict_end(self, args: Any, state: Any, control: Any, model: Any, predictions: Any, **kwargs: Any) -> None:
        self.predictions.append(predictions)
        wandb.log({"predictions": wandb.Table(dataframe=pd.Series(predictions))})


class Inference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.model.inference
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

        test_data = load_dataset("csv", data_files=self.cfg.path.test_data)
        self.tokenized_test_datasets = test_data.map(self._tokenize_pad_and_truncate, batched=True)

    def _wandb_callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for logging."""
        return [CustomWandbCallback_Inference()]

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def predict(self) -> pd.Series:
        pretrained_ckpt = self.cfg.path.pretrained_checkpoint
        callbacks = self._wandb_callbacks()

        input_ids = torch.tensor(self.tokenized_test_datasets['train']['input_ids'])
        attention_mask = torch.tensor(self.tokenized_test_datasets['train']['attention_mask'])

        # Check for available devices
        device_ids = [0]  # Set the device IDs for the GPUs you want to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_ckpt, 
            num_labels=1, 
            ignore_mismatched_sizes=True)
        
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
       
       # wandb.log({"model_summary": dict(model)})  # Log model summary

        predictions_path = f'{self.cfg.path.predictions}_predictions.json'
        

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)          
            predictions = outputs.logits.squeeze().cpu().numpy()
            for callback in callbacks:
                callback.on_predict_end(None, None, None, model, predictions)  # Manually trigger callback
        
        torch.cuda.empty_cache()

        print(pd.Series(predictions)) 
        predictions_list = predictions.tolist()
        with open(predictions_path, 'w') as json_file:
           json.dump(predictions_list, json_file)

                    
        predictions.to_json(predictions_path)
