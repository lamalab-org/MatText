import torch
from sklearn.cluster import KMeans
import numpy as np
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast
from transformers import AutoModel
from tokenizers import Tokenizer
import pandas as pd
from tqdm import tqdm

class EmbeddingClustering:
    def __init__(self, model_name: str, tokenizer: PreTrainedTokenizerFast, data: pd.DataFrame, num_clusters: int):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.data = data
        self.num_clusters = num_clusters

    def generate_embeddings(self):
        embeddings = []
        batch_size = 512  # Adjust this value to a smaller batch size

        for i in tqdm(range(0, len(self.data), batch_size)):
            batch = self.data['slices'].iloc[i:i+batch_size].tolist()
            tokenized_data = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            with torch.no_grad():
                outputs = self.model(**tokenized_data)
    
            # Extract embeddings from BERT
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            embeddings.extend(batch_embeddings)
    
        return np.array(embeddings)

    def cluster_embeddings(self):
        embeddings = self.generate_embeddings()
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(embeddings)
        return kmeans.labels_

    def save_clustered_results(self, labels: np.ndarray, output_file: str):
        self.data['cluster'] = labels
        self.data.to_csv(output_file, sep='\t', index=False)

    def save_clustered_results_with_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, output_file: str):
        df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
        df['cluster'] = labels
        df.to_csv(output_file, sep='\t', index=False)

# Example usage:
if __name__ == "__main__":
    
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    
    tokenizer = Tokenizer.from_file("/home/so87pot/n0w0f/structllm/src/structllm/tokenizer/tokenizer-slice_27k_train.json")
    PreTrainedTokenizerFast = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
    
    file_path = "/home/so87pot/n0w0f/structllm/data/130k/combined.csv"

    # Load data from the CSV file
    data = pd.read_csv(file_path)

    # Get unique crystal types and assign clusters accordingly
    unique_crystals = data['crystal'].unique()
    num_clusters = len(unique_crystals)

    model_path = "/home/so87pot/n0w0f/structllm/src/structllm/models/pretrain/checkpoints/27k_new/checkpoint-2000"
    
    clustering = EmbeddingClustering(model_path, PreTrainedTokenizerFast, data, num_clusters)
    
    cluster_labels = clustering.cluster_embeddings()
    embeddings = clustering.generate_embeddings()
    
    output_file = "clustered_embeddings_27_combined.csv"  # Change this to your desired file path
    clustering.save_clustered_results_with_embeddings(embeddings, cluster_labels, output_file)
