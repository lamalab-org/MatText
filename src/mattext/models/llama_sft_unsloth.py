import json
import os
from typing import List

import torch
import wandb
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel

from mattext.models.utils import EvaluateFirstStepCallback


class FinetuneLLamaSFT:
    def __init__(
        self, cfg: DictConfig, local_rank=None, fold="fold_0", test_sample_size=None
    ) -> None:
        self.fold = fold
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.data_repository = cfg.model.data_repository
        self.dataset_name = cfg.model.dataset
        self.add_special_tokens = cfg.model.add_special_tokens
        self.property_map = cfg.model.PROPERTY_MAP
        self.material_map = cfg.model.MATERIAL_MAP
        self.cfg = cfg.model.finetune
        self.train_data = self.cfg.dataset_name
        self.test_data = self.cfg.benchmark_dataset
        self.context_length: int = self.cfg.context_length
        self.dataprep_seed: int = self.cfg.dataprep_seed
        self.callbacks = self.cfg.callbacks
        self.ckpt = self.cfg.path.pretrained_checkpoint
        self.bnb_config = self.cfg.bnb_config
        self.dataset = self.prepare_data(self.train_data)
        self.testdata = self.prepare_test_data(self.test_data)
        self.model, self.tokenizer, self.peft_config = self._setup_model_tokenizer()
        self.property_ = self.property_map[self.dataset_name]
        self.material_ = self.material_map[self.dataset_name]
        self.test_sample_size = test_sample_size

    def prepare_test_data(self, subset):
        dataset = load_dataset(self.data_repository, subset)[self.fold]
        # if self.test_sample_size:
        #     dataset = dataset.select(range(self.test_sample_size))
        return dataset

    def prepare_data(self, subset):
        dataset = load_dataset(self.data_repository, subset)
        dataset = dataset.shuffle(seed=self.dataprep_seed)[self.fold]
        return dataset.train_test_split(test_size=0.1, seed=self.dataprep_seed)

    def _setup_model_tokenizer(self):
        max_seq_length = 2048 if self.representation == "cif_p1" else None
        dtype = getattr(torch, self.bnb_config.bnb_4bit_compute_dtype) 
        load_in_4bit = self.bnb_config.use_4bit  # Use 4bit quantization

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.ckpt,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.lora_config.r,
            #target_modules=self.cfg.lora_config.target_modules,
            lora_alpha=self.cfg.lora_config.lora_alpha,
            lora_dropout=self.cfg.lora_config.lora_dropout,
            bias=self.cfg.lora_config.bias,
            use_gradient_checkpointing="unsloth",  # "unsloth" for faster training
            random_state=self.cfg.training_arguments.seed,
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer, None

    def formatting_prompts_func(self, example):
        return [
            f"### What is the {self.property_} of {rep}\n ### Answer: {label:.3f}@@@"
            for rep, label in zip(example[self.representation], example["labels"])
        ]

    def formatting_tests_func(self, example):
        return [
            f"### What is the {self.property_} of {rep}\n "
            for rep in example[self.representation]
        ]

    def _callbacks(self) -> List[TrainerCallback]:
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

    def finetune(self) -> str:
        training_args = TrainingArguments(**self.cfg.training_arguments)
        callbacks = self._callbacks()

        collator = DataCollatorForCompletionOnlyLM(
            " ### Answer:", tokenizer=self.tokenizer
        )

        max_seq_length = 2048 if self.representation == "cif_p1" else None

        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            formatting_func=self.formatting_prompts_func,
            data_collator=collator,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=False,
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(self.cfg.training_arguments)})
        wandb.log({"model_summary": str(self.model)})

        output_dir = os.path.join(
            self.cfg.path.output_dir, f"finetuned_{self.dataset_name}_{self.fold}"
        )
        trainer.train()

        pipe = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=False,
            max_new_tokens=4,
        )

        self._save_predictions(pipe, "predictions.json")

        trainer.save_state()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(
            os.path.join(output_dir, "llamav3-8b-lora-save-pretrained")
        )

        # merged_model = trainer.model.merge_and_unload()
        # merged_model.save_pretrained(
        #     os.path.join(output_dir, "llamav3-8b-lora-save-pretrained"),
        #     save_config=True,
        #     safe_serialization=True,
        # )
        # self._save_predictions(pipe, "predictions_merged.json")

        wandb.finish()
        return output_dir

    def _save_predictions(self, pipe, filename):
        with torch.cuda.amp.autocast():
            pred = pipe(self.formatting_tests_func(self.testdata))
        logger.debug("Prediction: %s", pred)

        output_file = os.path.join(
            self.cfg.path.output_dir,
            f"finetuned_{self.dataset_name}_{self.fold}_{filename}",
        )
        with open(output_file, "w") as json_file:
            json.dump(pred, json_file)
