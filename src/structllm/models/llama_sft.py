
import torch
import wandb
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
use_flash_attention = True


def unplace_flash_attn_with_attn():
    import importlib

    import transformers

    print("Reloading llama model, unpatching flash attention")
    importlib.reload(transformers.models.llama.modeling_llama)


# adapted from crystal-text-llm :TODO give credits
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
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
        self.trainset = self._prepare_datasets(self.cfg.path.finetune_traindata)

    def _setup_model_tokenizer(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )
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

        device_map = "auto"  # {"": 0}
        model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            use_cache=False,
            use_flash_attention_2=use_flash_attention,
            quantization_config=bnb_config,
            device_map=device_map,
        )

        peft_config = LoraConfig(**self.cfg.lora_config)
        # model = prepare_model_for_kbit_training(model)  #confirm this - base model with peft config in SFT trainer equivalent to
        # model = get_peft_model(model, peft_config)        # peft model passed to SFT

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            llama_tokenizer=tokenizer,
            model=model,
        )
        return model, tokenizer, peft_config

    def format_qstns(self, sample):
        question = f"""Question: What is the {self.property_} of the material {sample[self.representation]}?\n"""
        response = f"""Answer:{round(float(sample['labels']),3)}###"""
        return "".join([i for i in [question, response] if i is not None])

    def template_dataset(self, sample):
        sample["text"] = f"{self.format_qstns(sample)}"
        return sample

    def _prepare_datasets(self, path: str) -> DatasetDict:
        """
        Prepare training and validation datasets.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data.

        Returns:
            DatasetDict: Dictionary containing training and validation datasets.
        """

        ds = load_dataset("json", data_files=path, split="train")
        return ds.map(self.template_dataset, remove_columns=list(ds.features))

    def generate_once_and_save(self):
        if use_flash_attention:
            # unpatch flash attention
            unplace_flash_attn_with_attn()

        model = AutoPeftModelForCausalLM.from_pretrained(
            self.output_dir_,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir_)
        sample_prompt = self.trainset["text"][0][:-6]

        input_ids = tokenizer(
            sample_prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=100, do_sample=True, temperature=0.01
        )

        print(f"Prompt:\n{sample_prompt}\n")
        print(
            f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(sample_prompt):]}"
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            self.output_dir_,
            low_cpu_mem_usage=True,
        )

        # Merge LoRA and base model
        merged_model = model.merge_and_unload()
        # Save the merged model
        merged_model.save_pretrained(
            f"{self.cfg.path.finetuned_modelname}/llamav2-7b-lora-save-pretrain",
            save_config=True,
            safe_serialization=True,
        )
        tokenizer.save_pretrained("merged_model")

    def finetune(self) -> None:
        """
        Perform fine-tuning of the language model.
        """

        config_train_args = self.cfg.training_arguments
        training_args = TrainingArguments(
            **config_train_args,
        )

        response_template_with_context = "Answer:"
        response_template_ids = self.tokenizer.encode(
            response_template_with_context, add_special_tokens=False
        )
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=self.tokenizer
        )

        max_seq_length = MAX_LENGTH
        packing = False
        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=self.trainset,
            data_collator=data_collator,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=packing,
        )

        wandb.log({"Training Arguments": str(config_train_args)})
        wandb.log({"model_summary": str(self.model)})

        trainer.save_model(
            f"{self.cfg.path.finetuned_modelname}/llamav2-7b-no-fine-tune"
        )
        self.output_dir_ = (
            f"{self.cfg.path.finetuned_modelname}/llamav2-7b-lora-fine-tune"
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model(self.output_dir_)
        self.generate_once_and_save()
        wandb.finish()
        return self.cfg.path.finetuned_modelname
