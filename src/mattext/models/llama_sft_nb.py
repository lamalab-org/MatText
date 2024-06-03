import gc
import json
from statistics import mean
from typing import List

import torch
import wandb

# from evaluate import load
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from mattext.models.utils import (
    EvaluateFirstStepCallback,
    LLMSampleCB,
    assign_special_tokens,
    smart_tokenizer_and_embedding_resize,
)

# Model
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
# Name of the new model
new_model = "TunedLlama-3-8B"
exp_name = "test_clm_kvrh_full"

# Dataset
root = "/work/so87pot/material_db/all_1/"
# dataset_path = f"{root}train_matbench_log_kvrh_1.json"
# test_response = f"{root}test_matbench_log_kvrh_1.json"


class FinetuneLLamaSFT:
    """Class to perform finetuning of a language model.
        Initialize the FinetuneModel.

    Args:
        cfg (DictConfig): Configuration for the fine-tuning.
        local_rank (int, optional): Local rank for distributed training. Defaults to None.
    """

    def __init__(self, cfg: DictConfig, local_rank=None) -> None:
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.add_special_tokens = cfg.model.add_special_tokens
        self.property_map = cfg.model.PROPERTY_MAP
        self.material_map = cfg.model.MATERIAL_MAP
        self.cfg = cfg.model.finetune
        self.context_length: int = self.cfg.context_length
        self.callbacks = self.cfg.callbacks
        self.ckpt = self.cfg.path.pretrained_checkpoint
        self.bnb_config = self.cfg.bnb_config
        self.model, self.tokenizer, self.peft_config = self._setup_model_tokenizer()
        self.property_ = self.property_map[self.cfg.dataset_name]
        self.material_ = self.material_map[self.cfg.dataset_name]
        self.dataset = self.prepare_data(
            self.cfg.path.finetune_traindata
        )

    def prepare_data(self,dataset_path):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.shuffle(seed=42)#.select(range(100))
        return dataset.train_test_split(test_size=0.1, seed=42)


    def _setup_model_tokenizer(self) -> None:

        # device_string = PartialState().process_index
        # compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

            # LoRA config
        peft_config = LoraConfig(
            r=32, # The rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
            lora_alpha=64, # LoRA scaling factor. It changes how the adaptation layer's weights affect the base model's
            lora_dropout=0.1, # Dropout is a regularization technique where a proportion of neurons (or parameters) are randomly “dropped out” or turned off during training to prevent overfitting.
            bias="none", # Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
            task_type="CAUSAL_LM", # Task to perform, Causal LM: Causal language modeling.
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model,use_fast=False,)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )

        return model, tokenizer, peft_config

    def formatting_prompts_func(self,example):
        output_texts = []
        for i in range(len(example[self.representation])):
            text = f"### What is the property of {example[self.representation][i]}\n ### Answer: {example['labels'][i]:.3f}@@@"
            output_texts.append(text)
        return output_texts

    def formatting_tests_func(self,example):
        output_texts = []
        for i in range(len(example[self.representation])):
            text = f"### What is the property of {example[self.representation][i]}\n "
            output_texts.append(text)
        return output_texts

    def _callbacks(self) -> List[TrainerCallback]:
        """Returns a list of callbacks for early stopping, and custom logging."""
        callbacks = []

        if self.callbacks.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.callbacks.early_stopping_patience,
                    early_stopping_threshold=self.callbacks.early_stopping_threshold,
                )
            )
        callbacks.append(EvaluateFirstStepCallback)
        return callbacks

    def finetune(self) -> None:
        """
        Perform fine-tuning of the language model.
        """

        config_train_args = self.cfg.training_arguments
        training_args = TrainingArguments(
            **config_train_args,
        )
        callbacks = self._callbacks()

        response_template = " ### Answer:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        packing = False
        max_seq_length = None
        if self.representation == "cif_p1":
            max_seq_length = 2048


        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            formatting_func=self.formatting_prompts_func,
            data_collator=collator,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=packing,
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(self.model)})

        # wandb_callback = LLMSampleCB(
        #     trainer, self.dataset['test'], num_samples=10, max_new_tokens=10
        # )
        # trainer.add_callback(wandb_callback)

        self.output_dir_ = (
            f"{self.cfg.path.finetuned_modelname}/llamav3-8b-lora-fine-tune"
        )
        trainer.train()

        testset = load_dataset("json", data_files=self.cfg.path.finetune_testdata, split="train")

        pipe = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=self.tokenizer,
            return_full_text = False,
            do_sample=False,
            max_new_tokens=4,
        )
        with torch.cuda.amp.autocast():
            pred = pipe(self.formatting_tests_func(testset))
        print(pred)

        with open(f"{self.cfg.path.finetuned_modelname}_predictions.json", 'w') as json_file:
            json.dump(pred, json_file)


        trainer.save_state()
        trainer.save_model(self.output_dir_)

       # Merge LoRA and base model
        merged_model = trainer.model.merge_and_unload()
        # Save the merged model
        merged_model.save_pretrained(
            f"{self.cfg.path.finetuned_modelname}/llamav3-8b-lora-save-pretrained",
            save_config=True,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(
            f"{self.cfg.path.finetuned_modelname}/llamav3-8b-lora-save-pretrained"
        )

        with torch.cuda.amp.autocast():
            merge_pred = pipe(self.formatting_tests_func(testset))
        print(merge_pred)

        with open(f"{self.cfg.path.finetuned_modelname}_predictions_merged.json", 'w') as json_file:
            json.dump(merge_pred, json_file)

        wandb.finish()
        return self.cfg.path.finetuned_modelname

