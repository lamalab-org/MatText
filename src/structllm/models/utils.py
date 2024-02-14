from transformers import PreTrainedTokenizerFast, TrainerCallback
from tokenizers import Tokenizer
from typing import Any, List, Dict, Union
import wandb

from structllm.tokenizer.slice_tokenizer import AtomVocabTokenizer

#tokenizermap dictionary here

class TokenizerMixin:
    """Mixin class to handle tokenizer functionality."""

    def __init__(self, cfg):

        self.cfg = cfg
        self.tokenizer_cfg = cfg.model.tokenizer
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

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any],label:str, context_length: int) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts[label], truncation=True, padding="max_length", max_length=context_length)



class CustomWandbCallback_Inference(TrainerCallback):
    """Custom W&B callback for logging during inference."""

    def __init__(self):
        self.predictions = []

    def on_predict_end(self, args: Any, state: Any, control: Any, model: Any, predictions: Any, **kwargs: Any) -> None:
        wandb.log({"predictions": predictions.predictions, })
        

class CustomWandbCallback_Pretrain(TrainerCallback):
    """Custom W&B callback for logging during training."""
    def on_log(self, args: Any, state: Any, control: Any, model: Any, logs: Dict[str, Union[float, Any]], **kwargs: Any) -> None:
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss


class CustomWandbCallback_FineTune(TrainerCallback):
    """Custom W&B callback for logging during training."""
    def on_log(self, args: Any, state: Any, control: Any, model: Any, logs: Dict[str, Union[float, Any]], **kwargs: Any) -> None:
        if state.is_world_process_zero:
            step = state.global_step  # Retrieve the current step
            epoch = state.epoch  # Retrieve the current epoch
            print(f"Step: {step}, Epoch: {epoch}")

            if "loss" in logs and "eval_loss" in logs:  # Both training and evaluation losses are present
                wandb.log({"train_loss": logs.get("loss"), "eval_loss": logs.get("eval_loss")}, step=step)
            
            # if "eval_loss" not in logs:
            #     # Log eval_loss as NaN if it's missing to avoid issues with logging
            #     wandb.log({"eval_loss": float('nan')}, step=step)
