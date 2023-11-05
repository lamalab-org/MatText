from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
import hydra
from omegaconf import DictConfig



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
        # train_dataset = slice_data.remove_columns('----9KNOtIZc9bDFEWxgjeSRsJrC')
        # train_dataset = train_dataset.rename_column(
        #     original_column_name="error", new_column_name="train"
        # )
        self.tokenized_datasets = train_dataset.map(self.tokenize_pad_and_truncate, batched=True)

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
        
        model = BertForMaskedLM(config=config)
        
        training_args = TrainingArguments(
            output_dir= config_path.output_dir,
            overwrite_output_dir= True,
            logging_dir= config_path.logging_dir,
            label_names= ["labels"],
            num_train_epochs= config_params.epochs,
            per_device_train_batch_size= config_params.per_device_train_batch_size,
            save_steps= config_params.save_steps,
            save_total_limit= config_params.save_total_limit,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_datasets['train'],
        )

        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(self.cfg.model.pretrain.path.finetuned_modelname)

        
        
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pretrain_bert = PretrainBertModel(cfg)
    pretrain_bert.pretrain_mlm()


if __name__ == "__main__":
    main()




