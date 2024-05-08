from typing import Any, Dict, Union

import wandb
from transformers import TrainerCallback
from xtal2txt.tokenizer import (
    CifTokenizer,
    CompositionTokenizer,
    CrysllmTokenizer,
    RobocrysTokenizer,
    SliceTokenizer,
)

_TOKENIZER_MAP = {
    "slice": SliceTokenizer,
    "composition": CompositionTokenizer,
    "cif_symmetrized": CifTokenizer,
    "cif_p1": CifTokenizer,
    "cif_bonding": CifTokenizer,
    "crystal_llm_rep": CrysllmTokenizer,
    "robocrys_rep": RobocrysTokenizer,
    "wycoff_rep": None,
    "atoms" : CompositionTokenizer,
    "atoms_params": CompositionTokenizer,
    "zmatrix": CrysllmTokenizer,
}

_DEFAULT_SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
}

class TokenizerMixin:
    """Mixin class to handle tokenizer functionality."""

    def __init__(self, cfg,special_tokens: Dict[str, str] = _DEFAULT_SPECIAL_TOKENS, special_num_token =False) -> None:

        self.representation = cfg
        self._wrapped_tokenizer = None

        self._tokenizer = _TOKENIZER_MAP.get(self.representation)
        if self._tokenizer is None:
            raise ValueError(f"Tokenizer not defined for {self.representation}")

        self._wrapped_tokenizer = self._tokenizer(
                    special_num_token=special_num_token,
                    special_tokens=special_tokens,
                    model_max_length=512, 
                    truncation=False, 
                    padding=False
                )
        print(f"special_tokens: {special_tokens}")
        print(self._wrapped_tokenizer.tokenize("Se2Se3"))

        #self._wrapped_tokenizer.add_special_tokens(special_tokens=special_tokens)

    def _tokenize_pad_and_truncate(self, texts: Dict[str, Any], context_length: int) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(texts[str(self.representation)], truncation=True, padding="max_length", max_length=context_length)



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
            print(f"Step: {step}, Epoch: {round(epoch,5)}")

            if "loss" in logs and "eval_loss" in logs:  # Both training and evaluation losses are present
                wandb.log({"train_loss": logs.get("loss"), "eval_loss": logs.get("eval_loss")}, step=step)


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True