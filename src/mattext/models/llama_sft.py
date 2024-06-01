import json
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from peft import (
    LoraConfig,
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

IGNORE_INDEX = -100
MAX_LENGTH = 2048
use_flash_attention = True


def unplace_flash_attn_with_attn():
    import importlib

    import transformers

    print("Reloading llama model, unpatching flash attention")
    importlib.reload(transformers.models.llama.modeling_llama)


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
        self.trainset, self.evalset = self._prepare_datasets(
            self.cfg.path.finetune_traindata
        )
        self.testset = self._prepare_testset(self.cfg.path.finetune_testdata)

    def _setup_model_tokenizer(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.eos_token  #seperate pad token required for data - collator

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

        # device_map = "auto"  # {"": 0}
        # device_map={'':torch.cuda.current_device()}
        model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            use_cache=False,
            use_flash_attention_2=use_flash_attention,
            quantization_config=bnb_config,
            device_map="auto",
        )

        peft_config = LoraConfig(**self.cfg.lora_config)
        # model = prepare_model_for_kbit_training(model)  #confirm this - base model with peft config in SFT trainer equivalent to
        # model = get_peft_model(model, peft_config)        # peft model passed to SFT

        if self.add_special_tokens:
            special_tokens_dict = assign_special_tokens(tokenizer)
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                llama_tokenizer=tokenizer,
                model=model,
            )
        return model, tokenizer, peft_config

    def formatting_prompts_func(self, sample):
        return f"### What is the {self.property_} of {sample[self.representation]}\n ### Answer: {round(float(sample['labels']),3)}@@@"

    def formatting_test_func(self, sample):
        sample["text"] = (
            f"### What is the {self.property_} of {sample[self.representation]}\n "
        )
        return sample

    def template_dataset(self, sample):
        sample["text"] = self.formatting_prompts_func(sample)
        return sample

    def _prepare_datasets(self, path: str) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """

        dataset = load_dataset("json", data_files=path, split="train")
        dataset = dataset.train_test_split(shuffle=True, test_size=0.2, seed=42)
        trainset = dataset["train"].map(
            self.template_dataset, remove_columns=list(dataset["train"].features)
        )
        testset = dataset["test"].map(
            self.template_dataset, remove_columns=list(dataset["test"].features)
        )
        print(trainset[0]["text"])
        print(testset[0]["text"])
        return trainset, testset

    def _prepare_testset(self, path: str) -> DatasetDict:
        """
        Prepare testsets.
        """
        dataset = load_dataset("json", data_files=path, split="train")
        testset = dataset.map(
            self.formatting_test_func, remove_columns=list(dataset.features)
        )
        print(testset[0]["text"])
        return testset

    def generate_and_save(self, trainer):
        if use_flash_attention:
            # unpatch flash attention
            unplace_flash_attn_with_attn()
        pipe = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=False,
            temperature=None,
            max_new_tokens=10,
            batch_size=16,
            max_length=2048,
        )

        responses_dict = {}
        with torch.cuda.amp.autocast():
            resp = pipe(self.testset["text"])
            for j, (prompt, responses) in enumerate(zip(self.testset["text"], resp)):
                # print(prompt)
                generated_text = responses[0]["generated_text"]
                # Extract the response part by removing the prompt part
                complete_response = generated_text.replace(prompt, "").strip()
                parsed_answer = complete_response.replace("Answer:", "").strip()
                responses_dict[j] = {
                    "prompt": prompt,
                    "response": responses[0]["generated_text"],
                    "parsed_answer": parsed_answer,
                }
                print(parsed_answer)

            with open(
                f"{self.cfg.path.finetuned_modelname}/llama_evals_{self.representation}.json",
                "w",
            ) as f:
                json.dump(responses_dict, f, indent=4)

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

        max_seq_length = MAX_LENGTH
        packing = False
        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=self.trainset,
            eval_dataset=self.evalset,
            data_collator=collator,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=packing,
            callbacks=callbacks,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(self.model)})

        wandb_callback = LLMSampleCB(
            trainer, self.testset, num_samples=10, max_new_tokens=10
        )
        trainer.add_callback(wandb_callback)

        self.output_dir_ = (
            f"{self.cfg.path.finetuned_modelname}/llamav3-8b-lora-fine-tune"
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model(self.output_dir_)

        self.generate_once_and_save(trainer)
        wandb.finish()
        return self.cfg.path.finetuned_modelname
