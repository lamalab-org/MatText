from typing import Dict, Any
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra import main as hydra_main

class TokenCount:
    """Class to count the number of tokens in the training data."""
    def __init__(self, cfg: DictConfig) -> None:
        self._tokenizer = Tokenizer.from_file(cfg.tokenizer.path.tokenizer_path)
        self._wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        self.context_length = cfg.model.finetune.context_length
        self.train_dataset = load_dataset("csv", data_files=cfg.model.finetune.path.finetune_traindata)

    def count_exceeding_entries(self) -> int:
        """Counts the number of entries that exceed the context length."""
        exceeding_entries = 0

        # Tokenize the training data
        tokenized_train_datasets = self.train_dataset.map(self._tokenize, batched=True)

        # Count the number of entries that exceed the context length
        for example in tokenized_train_datasets["train"]:
            if len(example["input_ids"]) > self.context_length:
                exceeding_entries += 1

        return exceeding_entries

    def _tokenize(self, texts: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes input texts without padding or truncation."""
        encoded_texts = self._tokenizer.encode_batch(texts["slices"])
        return {"input_ids": [text.ids for text in encoded_texts]}

@hydra_main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    token_count = TokenCount(cfg)
    exceeding_entries = token_count.count_exceeding_entries()
    print(f"Number of entries exceeding the context length: {exceeding_entries}")

if __name__ == "__main__":
    main()