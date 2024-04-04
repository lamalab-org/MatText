import json
from functools import partial
from typing import Any, Dict, Type

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from matplotlib.colors import Normalize
from tqdm import tqdm
from transformers import AutoModel
from xtal2txt.tokenizer import (
    CifTokenizer,
    CompositionTokenizer,
    CrysllmTokenizer,
    RobocrysTokenizer,
    SliceTokenizer,
)

# Mapping of representation types to their corresponding tokenizer classes
_TOKENIZER_MAP: Dict[str, Type] = {
    "slice": SliceTokenizer,
    "composition": CompositionTokenizer,
    "cif_symmetrized": CifTokenizer,
    "cif_p1": CifTokenizer,
    "cif_bonding": CifTokenizer,
    "crystal_llm_rep": CrysllmTokenizer,
    "robocrys_rep": RobocrysTokenizer,
    "wycoff_rep": None,
}

_REPRESENTATION_CONTEXT = {
        "cif_p1" : 1024,
        "cif_symmetrized" : 1024,
        "slice" : 512,
        "crystal_llm_rep" : 512,
        "composition" : 32
    }

_DEFAULT_SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
    }


def save_json(average_weights, name):
    """Save the average weights to a JSON file."""
    average_weights_str_keys = {str(key): value for key, value in average_weights.items()}
    with open(name, 'w') as f:
        json.dump(average_weights_str_keys, f)

def num_layers(attention):
    return len(attention)


def num_heads(attention):
    return attention[0][0].size(0)

def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_tokenizer_from_rep(representation):
    tokenizer_class = _TOKENIZER_MAP[representation]
    tokenizer = tokenizer_class(model_max_length=512, truncation=False, padding=False)
    tokenizer.add_special_tokens(special_tokens=_DEFAULT_SPECIAL_TOKENS)
    return tokenizer


def get_dataset(path: str, representation: str):

    _wrapped_tokenizer = get_tokenizer_from_rep(representation)
    context_length = _REPRESENTATION_CONTEXT[representation]

    def _tokenize_pad_and_truncate(texts: Dict[str, Any], context_length: int) -> Dict[str, Any]:
        tokenized_texts = _wrapped_tokenizer(texts[representation], truncation=True, padding="max_length", max_length=context_length)
        analysis_results = [_wrapped_tokenizer.token_analysis(_wrapped_tokenizer.convert_ids_to_tokens(input_ids)) for input_ids in tokenized_texts['input_ids']]
        tokenized_texts['analysis'] = analysis_results
        return tokenized_texts



    dataset = load_dataset("json", data_files=path)
    return dataset.map(
                partial(_tokenize_pad_and_truncate, context_length=context_length),
                batched=True)


def mask_dict(tokens):
    unique_tokens = set(tokens)
    masks = {}

    # Create a mask for each unique token
    for token in unique_tokens:
        mask = np.array([np.array(tokens) == token] * len(tokens))
        mask = mask.astype(int)  # Convert boolean values to integers
        masks[token] = mask

    return masks

# def mask_dict(tokens):
#     unique_tokens = set(tokens)
#     masks = {}

#     # Create a mask for each unique token
#     for token in unique_tokens:
#         mask = np.array([np.array(tokens) == token] * len(tokens))
#         mask = (mask & mask.T).astype(int)
#         masks[token] = mask

#     return masks

def get_count_for_tokens(tokens):
    unique_tokens = set(tokens)
    token_counts = {}
    for token in unique_tokens:
        token_counts[token] = tokens.count(token)

    return token_counts


def get_percentage_weights(attention_matrix, masks):
    token_weights = {}
    for token, mask in masks.items():
        token_weights[token] = np.sum(attention_matrix * mask) / np.sum(mask)

    return token_weights


#function to loop through all layers and heads and get the token_weights for each attenction matrix. ttention matrix is of ghe form num_layers x num_heads x seq_len x seq_len
def get_token_weights(attention, masks):
    token_weights = {}
    for layer in range(attention.size(0)):
        for head in range(attention.size(1)):
            token_weights[(layer, head)] = get_percentage_weights(attention[layer, head, :, :].detach().cpu().numpy(), masks)

    return token_weights


def aggregate_token_weights(data, model, tokenizer):
    # Initialize an empty dictionary to store the aggregate token weights
    aggregate_weights = {}
    limit = len(data['train']['input_ids'])
    #limit = 100

    # Loop over all the data points
    for i in tqdm(range(limit), desc="Processing data points"):
        # Get the current material and mask tokens
        material = data['train']['input_ids'][i]
        mask_tokens = data['train']['analysis'][i]

        # Encode the material
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = tokenizer.encode(material, return_tensors="pt")
        inputs = inputs.to(device)

        # Run the model
        outputs = model(inputs)
        attention_ = outputs.attentions

        # Get the masks
        masks = mask_dict(mask_tokens)

        # Format the attention
        attention = format_attention(attention_, layers=[0,1,2,3], heads=[0,1,2,3,4,5,6,7])

        # Get the token weights
        token_weights = get_token_weights(attention, masks)

        # Add the token weights to the aggregate weights
        for token, weights in token_weights.items():
            if token not in aggregate_weights:
                aggregate_weights[token] = weights
            else:
                for key, weight in weights.items():
                    if key not in aggregate_weights[token]:
                        aggregate_weights[token][key] = weight
                    else:
                        aggregate_weights[token][key] += weight

    # Divide the aggregate weights by the total number of data points to get the average
    for token, weights in aggregate_weights.items():
        for key in weights.keys():
            aggregate_weights[token][key] /= limit

    return aggregate_weights




def get_attention_weights(data_path: str, ckpt:str, representation:str , result_json_name:str):
    """Get the attention weights for a given dataset and model checkpoint."""
    dataset = get_dataset(data_path,representation)
    tokenizer = get_tokenizer_from_rep(representation)
    model = AutoModel.from_pretrained(ckpt, output_attentions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    aggregate_weights = aggregate_token_weights(dataset, model, tokenizer)
    result_name = f"{representation}_{result_json_name}"
    save_json(aggregate_weights, f"{result_name}_aggregate_weights.json")
    _plot_heatmap(aggregate_weights,result_name)
    #return aggregate_weights


def _plot_heatmap(token_weights_,plot_name:str):
    """Plot a heatmap of the token weights."""

    # Get the set of unique tokens
    unique_tokens = set(token for weights in token_weights_.values() for token in weights.keys())

    # Get the global min and max weights
    global_min = min(weight for weights in token_weights_.values() for weight in weights.values())
    global_max = max(weight for weights in token_weights_.values() for weight in weights.values())

    # Create a normalizer
    norm = Normalize(vmin=global_min, vmax=global_max)

    # Create a separate subplot for each token
    fig, axs = plt.subplots(len(unique_tokens), 1, figsize=(10, len(unique_tokens) * 5))

    for ax, token in zip(axs, unique_tokens):
        # Extract the weights for the current token
        weights = {(layer, head): weight[token] for (layer, head), weight in token_weights_.items() if token in weight}

        # Convert the weights to a DataFrame
        df = pd.DataFrame.from_dict(weights, orient='index', columns=[token])
        df.index = pd.MultiIndex.from_tuples(df.index, names=['Layer', 'Head'])

        # Unstack the DataFrame and reset the column names
        df = df.unstack()
        df.columns = df.columns.get_level_values(1)

        # Plot the weights
        sns.heatmap(df, annot=True, fmt=".3f", square=True, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, norm=norm, cbar=True)
        ax.set_title(f'Token: {token}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Set title for the entire plot with plot_name
    plt.suptitle(plot_name)
    #plt.show()
    plot_name = f"{plot_name}_plot.png"
    plt.savefig(plot_name, format='png')
    plot_name = f"{plot_name}_plot.pdf"
    plt.savefig(plot_name, format='pdf')


def main(path: str, ckpt: str, representation:str, result_json_name: str):
    """Main function to be called from the command line."""
    get_attention_weights(path, ckpt, representation,result_json_name)

if __name__ == "__main__":
    fire.Fire(main)
