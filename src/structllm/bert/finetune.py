import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import hydra
from omegaconf import DictConfig


class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, model, logs, **kwargs):
        if state.is_world_process_zero:
            wandb.log({"train_loss": logs.get("loss")})  # Log training loss
           # wandb.log({"eval_loss": logs.get("eval_loss")})  # Log evaluation loss



class FinetuneBertModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.context_length = self.cfg.model.finetune.context_length
        
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

        slice_data = load_dataset("csv", data_files=self.cfg.model.finetune.path.finetune_traindata)
        self.tokenized_datasets = slice_data.map(self.tokenize_pad_and_truncate, batched=True)

    def tokenize_pad_and_truncate(self, texts):
        return self.wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length",  max_length=self.context_length)

    def compute_metrics(self, p):
        preds = p.predictions.squeeze()
        return {"rmse": torch.sqrt(((preds - p.label_ids) ** 2).mean()).item()}

    def fine_tune(self):
        config_path = self.cfg.model.finetune.path
        config_params = self.cfg.model.finetune.training_arguments

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=config_path.output_dir,  # Directory where model checkpoints and logs will be saved
            logging_dir=config_path.logging_dir,
            label_names=["labels"],
            save_total_limit=config_params.save_total_limit,  # Maximum number of checkpoints to save
            per_device_train_batch_size=config_params.per_device_train_batch_size,
            num_train_epochs=config_params.epochs,  # Number of training epochs
            learning_rate=config_params.learning_rate,
            save_steps= config_params.save_steps,
            save_total_limit= config_params.save_total_limit,
            report_to= config_params.report_to,
            logging_steps =config_params.logging_steps,                    # we will log every 100 steps
            #eval_steps = config_params.eval_steps,                      # we will perform evaluation every 500 steps
            load_best_model_at_end = config_params.load_best_model_at_end,
        )

        model = AutoModelForSequenceClassification.from_pretrained(config_path.pretrained_checkpoint,
                                                                 num_labels=1,
                                                                 ignore_mismatched_sizes=True)


        # Use the custom callback for logging
        callbacks = [CustomWandbCallback()]

        # Create a Trainer instance
        trainer = Trainer(
            model=model.to("cuda"),
            args=training_args,
            data_collator=None,  # We're using default data collation
            compute_metrics=self.compute_metrics,
            train_dataset=self.tokenized_datasets['train'],  # Use the training split
            callbacks=callbacks
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.model.finetune.path.finetuned_modelname)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
    os.environ["WANDB_LOG_MODEL"] = cfg.logging.wandb_log_model

    # Initialize W&B session
    wandb.init( config=cfg.model.finetune,
           project=cfg.logging.wandb_project, 
           name=cfg.model.finetune.exp_name,)

    finetune_bert = FinetuneBertModel(cfg)
    finetune_bert.fine_tune()

if __name__ == "__main__":
    # Retrieve the API key from the environment variable
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        # Log in to W&B using the retrieved API key
        wandb.login(key=api_key)
    else:
        print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")
    main()