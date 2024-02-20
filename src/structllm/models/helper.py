from pathlib import Path

import fire
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

from structllm.models.utils import TokenizerMixin


def count_tokens_and_plot(dataset_path: str, representation: str, context_length: int, report_path: str = "token_count_report.txt", plot_path: str = "token_count_plot.png"):
    tokenizer = TokenizerMixin(representation)
    ds = load_dataset("json", data_files=dataset_path,split='train')
    print(ds)
    print(representation)
    dataset = ds[representation]

    token_counts = []
    num_entries_exceeding_context = 0

    with open(report_path, "w") as report_file:
        for entry in tqdm(dataset):
            tokens = tokenizer._wrapped_tokenizer.tokenize(entry)
            tokenized_entry = tokenizer._wrapped_tokenizer(entry, truncation=False)
            num_tokens = len(tokenized_entry["input_ids"])
            token_counts.append(num_tokens)

            if num_tokens > context_length:
                num_entries_exceeding_context += 1

            report_file.write(f"ENTRY: {entry}\n")
            report_file.write(f"TOKENIZED: {tokens}\n")
            report_file.write(f"Number of tokens: {num_tokens}\n")
            report_file.write("-------------------\n")

    # Plot token count distribution
    plt.hist(token_counts, bins='auto')
    plt.title('Token Count Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.savefig(plot_path)
    plt.show()

    # Write summary to report file
    with open(report_path, "a") as report_file:
        report_file.write("\nSummary:\n")
        report_file.write(f"Total number of entries: {len(dataset)}\n")
        report_file.write(f"Total number of tokens: {sum(token_counts)}\n")
        report_file.write(f"Number of entries exceeding context length: {num_entries_exceeding_context}\n")

if __name__ == "__main__":
    fire.Fire(count_tokens_and_plot)
