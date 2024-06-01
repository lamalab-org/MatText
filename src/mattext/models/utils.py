from typing import Any, Dict, Union

import torch
import wandb
from tqdm import tqdm
from transformers import GenerationConfig, TrainerCallback
from transformers.integrations import WandbCallback
from mattext.tokenizer import (
    CifTokenizer,
    CompositionTokenizer,
    CrysllmTokenizer,
    RobocrysTokenizer,
    SliceTokenizer,
    SmilesTokenizer,
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
    "atoms": CompositionTokenizer,
    "atoms_params": CompositionTokenizer,
    "zmatrix": CrysllmTokenizer,
    "local_env": SmilesTokenizer,
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

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def assign_special_tokens(tokenizer):
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    Adapted from crystal-text-llm (https://github.com/facebookresearch/crystal-text-llm)
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer), pad_to_multiple_of=8)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg

    model.config.pad_token_id = llama_tokenizer.pad_token_id
    output_embeddings[-num_new_tokens:] = output_embeddings_avg


class TokenizerMixin:
    """Mixin class to handle tokenizer functionality."""

    def __init__(
        self,
        cfg,
        special_tokens: Dict[str, str] = _DEFAULT_SPECIAL_TOKENS,
        special_num_token=False,
    ) -> None:
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
            padding=False,
        )
        print(f"special_tokens: {special_tokens}")
        print(self._wrapped_tokenizer.tokenize("Se2Se3"))

        # self._wrapped_tokenizer.add_special_tokens(special_tokens=special_tokens)

    def _tokenize_pad_and_truncate(
        self, texts: Dict[str, Any], context_length: int
    ) -> Dict[str, Any]:
        """Tokenizes, pads, and truncates input texts."""
        return self._wrapped_tokenizer(
            texts[str(self.representation)],
            truncation=True,
            padding="max_length",
            max_length=context_length,
        )


class CustomWandbCallback_Inference(TrainerCallback):
    """Custom W&B callback for logging during inference."""

    def __init__(self):
        self.predictions = []

    def on_predict_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any,
        predictions: Any,
        **kwargs: Any,
    ) -> None:
        wandb.log(
            {
                "predictions": predictions.predictions,
            }
        )


class CustomWandbCallback_Pretrain(TrainerCallback):
    """Custom W&B callback for logging during training."""

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any,
        logs: Dict[str, Union[float, Any]],
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss


class CustomWandbCallback_FineTune(TrainerCallback):
    """Custom W&B callback for logging during training."""

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any,
        logs: Dict[str, Union[float, Any]],
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            step = state.global_step  # Retrieve the current step
            epoch = state.epoch  # Retrieve the current epoch
            print(f"Step: {step}, Epoch: {round(epoch,5)}")

            if (
                "loss" in logs and "eval_loss" in logs
            ):  # Both training and evaluation losses are present
                wandb.log(
                    {
                        "train_loss": logs.get("loss"),
                        "eval_loss": logs.get("eval_loss"),
                    },
                    step=step,
                )


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


class LLMSampleCB(WandbCallback):
    """A CallBack to log samples a wandb.Table during training"""

    def __init__(
        self,
        trainer,
        test_dataset,
        num_samples=5,
        max_new_tokens=10,
        log_model="checkpoint",
    ):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path, max_new_tokens=max_new_tokens
        )

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")[
            "input_ids"
        ].cuda()
        with torch.cuda.amp.autocast():
            output = self.model.generate(
                tokenized_prompt, generation_config=self.gen_config
            ).to("cuda")
        return self.tokenizer.decode(
            output[0][len(tokenized_prompt[0]) :], skip_special_tokens=True
        )

    def samples_table(self, examples):
        """Create a wandb.Table to store the generations"""
        records_table = wandb.Table(
            columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys())
        )
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(
                prompt, generation, *list(self.gen_config.to_dict().values())
            )
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        """Log the wandb.Table after calling trainer.evaluate"""
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})
        # "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})
