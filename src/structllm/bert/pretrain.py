from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
import hydra
from omegaconf import DictConfig
import os
import wandb

from transformers import TrainerCallback, TrainerControl

class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, model, logs, **kwargs):
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
            wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss



class PretrainBertModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.context_length = 128
        
        # Load the custom tokenizer using tokenizers library
        self.tokenizer = Tokenizer.from_file(self.cfg.tokenizer.path.tokenizer_path)
        self.wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        train_dataset = load_dataset("csv", data_files=self.cfg.model.pretrain.path.traindata)
        eval_dataset = load_dataset("csv", data_files=self.cfg.model.pretrain.path.evaldata)

        self.tokenized_train_datasets = train_dataset.map(self.tokenize_pad_and_truncate, batched=True)
        self.tokenized_eval_datasets = eval_dataset.map(self.tokenize_pad_and_truncate, batched=True)
        

    def tokenize_pad_and_truncate(self, texts):
        return self.wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def pretrain_mlm(self):
        config_path = self.cfg.model.pretrain.path
        config_mlm = self.cfg.model.pretrain.mlm
        config_params = self.cfg.model.pretrain.training_arguments
        bert_params = self.cfg.model.pretrain.bert_config
        
        data_collator = DataCollatorForLanguageModeling(
                                    tokenizer = self.wrapped_tokenizer,
                                    mlm=config_mlm.is_mlm, 
                                    mlm_probability=config_mlm.mlm_probability
        )

        config = BertConfig(self.wrapped_tokenizer.vocab_size, 
                            hidden_size= bert_params.hidden_size,
                            num_hidden_layers= bert_params.num_hidden_layers, 
                            num_attention_heads= bert_params.num_attention_heads, 
                            is_decoder= bert_params.is_decoder,
                            add_cross_attention= bert_params.add_cross_attention)
        
        model = BertForMaskedLM(config=config).to("cuda")
        
        training_args = TrainingArguments(
            output_dir= config_path.output_dir,
            overwrite_output_dir= True,
            logging_dir= config_path.logging_dir,
            label_names= ["labels"],
            num_train_epochs= config_params.epochs,
            per_device_train_batch_size= config_params.per_device_train_batch_size,
            save_steps= config_params.save_steps,
            save_total_limit= config_params.save_total_limit,
            report_to= config_params.report_to,
            evaluation_strategy = config_params.evaluation_strategy,          # check evaluation metrics at each epoch
            logging_steps =config_params.logging_steps,                    # we will log every 100 steps
            eval_steps = config_params.eval_steps,                      # we will perform evaluation every 500 steps
            load_best_model_at_end = config_params.load_best_model_at_end,
        )
        

        # Use the custom callback for logging
        callbacks = [CustomWandbCallback()]

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=self.tokenized_train_datasets['train'],
            eval_dataset=self.tokenized_eval_datasets['train'],
            args=training_args,
            callbacks=callbacks
        )

        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.model.pretrain.path.finetuned_modelname)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
    os.environ["WANDB_LOG_MODEL"] = cfg.logging.wandb_log_model

    # Initialize W&B session
    wandb.init(config=cfg.model.pretrain,
            project=cfg.logging.wandb_project, 
            name=cfg.model.pretrain.exp_name,)
    pretrain_bert = PretrainBertModel(cfg)
    pretrain_bert.pretrain_mlm()

if __name__ == "__main__":
    # Retrieve the API key from the environment variable
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        # Log in to W&B using the retrieved API key
        wandb.login(key=api_key)
    else:
        print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")
    main()