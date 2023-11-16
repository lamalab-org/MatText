import os
import wandb
import hydra
from omegaconf import DictConfig

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import  AutoModelForMaskedLM, AutoConfig
from transformers import Trainer, TrainingArguments

from datasets import load_dataset



from transformers import TrainerCallback, TrainerControl

class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, model, logs, **kwargs):
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss



class PretrainModel:
    def __init__(self, pretrain_config: DictConfig , tokenizer_config: DictConfig ):
        self.cfg = pretrain_config
        self.tokenizer_cfg = tokenizer_config
        self.context_length = self.cfg.context_length
        self.model_name_or_path = self.cfg.model_name_or_path
        
        # Load the custom tokenizer using tokenizers library
        self._tokenizer = Tokenizer.from_file(self.tokenizer_cfg.path.tokenizer_path)
        self._wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        train_dataset = load_dataset("csv", data_files=self.cfg.path.traindata)
        eval_dataset = load_dataset("csv", data_files=self.cfg.path.evaldata)

        self.tokenized_train_datasets = train_dataset.map(self._tokenize_pad_and_truncate, batched=True)
        self.tokenized_eval_datasets = eval_dataset.map(self._tokenize_pad_and_truncate, batched=True)

    def _wandb_callbacks(self):
        # Use the custom callback for logging
        return [CustomWandbCallback()]

    
    def _tokenize_pad_and_truncate(self, texts):
        return self._wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def pretrain_mlm(self):
        config_mlm = self.cfg.mlm
        config_train_args = self.cfg.training_arguments
        config_model_args = self.cfg.model_config
        
        data_collator = DataCollatorForLanguageModeling(
                                    tokenizer = self._wrapped_tokenizer,
                                    mlm=config_mlm.is_mlm, 
                                    mlm_probability=config_mlm.mlm_probability
                                    )

        callbacks = self._wandb_callbacks()

       
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            **config_model_args)
         
        model = AutoModelForMaskedLM.from_config(
            config
            ).to("cuda")
        
        training_args = TrainingArguments(
            **config_train_args
           )
    
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=self.tokenized_train_datasets['train'],
            eval_dataset=self.tokenized_eval_datasets['train'],
            args=training_args,
            callbacks=callbacks
        )

        print("-------------------")
        print(config)
        print("-------------------")
        print(model)
        print("-------------------")
        
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.path.finetuned_modelname)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def pretrainer(cfg: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
    os.environ["WANDB_LOG_MODEL"] = cfg.logging.wandb_log_model

    # Initialize W&B session
    wandb.init(config= dict(cfg.model.pretrain),
            project= cfg.logging.wandb_project, 
            name= cfg.model.pretrain.exp_name,)

    pretrain_config = cfg.model.pretrain
    tokenizer_config = cfg.tokenizer
    pretrain = PretrainModel( pretrain_config, tokenizer_config)
    pretrain.pretrain_mlm()

if __name__ == "__main__": 
    # Retrieve the API key from the environment variable
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")
    pretrainer()