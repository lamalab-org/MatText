import torch 
import pandas as pd
from matbench.bench import MatbenchBenchmark
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

import hydra
from omegaconf import DictConfig

class MatbenchPredict:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.context_length = 128
        self.checkpoints = self.cfg.matbench.record_test.checkpoints
        self.benchmark = self.cfg.matbench.record_test.benchmark_dataset
        self.benchmark_save_path = self.cfg.matbench.record_test.benchmark_save_file
         
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

        self.model = [AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=1, ignore_mismatched_sizes=True) for ckpt in self.checkpoints]

        self.csv_datasets_list = cfg.matbench.record_test.list_test_data
        self.csv_datasets = [ load_dataset("csv", data_files=files)  for files in self.csv_datasets_list]   
        self.tokenized_datasets = [csv.map(self.tokenize_pad_and_truncate, batched=True) for csv in self.csv_datasets ] 
             
    def tokenize_pad_and_truncate(self, texts):
        return self.wrapped_tokenizer(texts["slices"], truncation=True, padding="max_length", max_length=self.context_length)

    def predict(self):

        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, self.benchmark)
        benchmark.load()
        

        for i, fold in enumerate(self.tokenized_datasets): 

            # Prepare the input data
            input_ids = torch.tensor(fold['train']['input_ids'])
            attention_mask = torch.tensor(fold['train']['attention_mask'])

            # Perform inference to get predictions
            model = self.model[i]
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.squeeze().numpy()

                predictions = pd.Series(predictions)

                benchmark.record(i,predictions)
            print(mb.is_recorded)
        
        benchmark.to_file(self.benchmark_save_path)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    record = MatbenchPredict(cfg)
    record.predict()


if __name__ == "__main__":
    main()