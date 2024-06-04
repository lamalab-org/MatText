import json
from typing import List

import torch
import wandb
from datasets import load_dataset
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


class FinetuneLLamaSFT:
    """Class to perform finetuning of a language model.
        Initialize the FinetuneModel.

    Args:
        cfg (DictConfig): Configuration for the fine-tuning.
        local_rank (int, optional): Local rank for distributed training. Defaults to None.
    """

    def __init__(self, cfg: DictConfig, local_rank=None, fold="fold_0") -> None:
        self.fold = fold
        self.local_rank = local_rank
        self.representation = cfg.model.representation
        self.data_repository = cfg.model.data_repository
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
        self.dataset = self.prepare_data(self.cfg.path.finetune_traindata)

    def prepare_data(self, subset):
        dataset = load_dataset(self.data_repository, subset)
        dataset = dataset.shuffle(seed=42)  # .select(range(100))
        return dataset.train_test_split(test_size=0.1, seed=42)

    def _setup_model_tokenizer(self) -> None:
        # device_string = PartialState().process_index
        # compute_dtype = getattr(torch, "float16")

        if self.bnb_config.use_4bit and self.bnb_config.use_8bit:
            raise ValueError(
                "You can't load the model in 8 bits and 4 bits at the same time"
            )

        elif self.bnb_config.use_4bit or self.bnb_config.use_8bit:
            compute_dtype = getattr(torch, self.bnb_config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.bnb_config.use_4bit,
                load_in_8bit=self.bnb_config.use_8bit,
                bnb_4bit_quant_type=self.bnb_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_config.use_nested_quant,
            )
        else:
            bnb_config = None

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # LoRA config
        peft_config = LoraConfig(**self.cfg.lora_config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt,
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            quantization_config=bnb_config,
            device_map="auto",
        )

        return model, tokenizer, peft_config

    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example[self.representation])):
            text = f"### What is the {self.property_} of {example[self.representation][i]}\n ### Answer: {example['labels'][i]:.3f}@@@"
            output_texts.append(text)
        return output_texts

    def formatting_tests_func(self, example):
        output_texts = []
        for i in range(len(example[self.representation])):
            text = f"### What is the {self.property_} of {example[self.representation][i]}\n "
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
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=self.tokenizer
        )

        packing = False
        max_seq_length = None
        if self.representation == "cif_p1":
            max_seq_length = 2048

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

        testset = load_dataset(
            "json", data_files=self.cfg.path.finetune_testdata, split="train"
        )

        pipe = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=False,
            max_new_tokens=4,
        )
        with torch.cuda.amp.autocast():
            pred = pipe(self.formatting_tests_func(testset))
        print(pred)

        with open(
            f"{self.cfg.path.finetuned_modelname}_{self.fold}_predictions.json", "w"
        ) as json_file:
            json.dump(pred, json_file)

        trainer.save_state()
        trainer.save_model(self.output_dir_)

        # Merge LoRA and base model
        merged_model = trainer.model.merge_and_unload()
        # Save the merged model
        merged_model.save_pretrained(
            f"{self.cfg.path.finetuned_modelname}_{self.fold}/llamav3-8b-lora-save-pretrained",
            save_config=True,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(
            f"{self.cfg.path.finetuned_modelname}_{self.fold}/llamav3-8b-lora-save-pretrained"
        )

        with torch.cuda.amp.autocast():
            merge_pred = pipe(self.formatting_tests_func(testset))
        print(merge_pred)

        with open(
            f"{self.cfg.path.finetuned_modelname}__{self.fold}_predictions_merged.json",
            "w",
        ) as json_file:
            json.dump(merge_pred, json_file)

        wandb.finish()
        return self.cfg.path.finetuned_modelname
