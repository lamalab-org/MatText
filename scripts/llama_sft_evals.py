import json

import fire
import torch
from datasets import DatasetDict, load_dataset
from peft import (
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
    pipeline,
)

PROPERTY_MAP = {
    "matbench_log_gvrh": "shear modulus (in GPa)",
    "matbench_log_kvrh": "bulk modulus (in GPa)",
    "matbench_dielectric": "refractive index",
    "matbench_perovskites": "formation energy (in eV)",
}

MATERIAL_MAP = {
    "matbench_log_gvrh": "material",
    "matbench_log_kvrh": "material",
    "matbench_dielectric": "dielectric material",
    "matbench_perovskites": "perovskite material",
}

IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


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


def _setup_model():
    pretrained_ckpt = "meta-llama/Llama-2-7b-hf"
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt, trust_remote_code=True)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    print(special_tokens_dict)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=base_model,
    )
    return base_model, tokenizer


def get_model_for_eval(adapter, batch_size, max_length=2048):
    base_model, tokenizer = _setup_model()
    model = PeftModel.from_pretrained(base_model, adapter)
    model = model.merge_and_unload()

    logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        do_sample=True,
        temperature=0.001,
    )

    return pipe


def _prepare_datasets(path: str,
                      matbench_dataset:str,
                      representation:str) -> DatasetDict:
    """
    Prepare training and validation datasets.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data.

    Returns:
        DatasetDict: Dictionary containing training and validation datasets.
    """
    property_ = PROPERTY_MAP[matbench_dataset]
    material_ = MATERIAL_MAP[matbench_dataset]

    def format_qstns(sample):
        question = f"""Question: What is the {property_} of the material {sample[representation]}?\n"""
        response = f"""Answer:{round(float(sample['labels']),3)}###"""
        return "".join([i for i in [question, response] if i is not None])

    def template_dataset(sample):
        sample["text"] = f"{format_qstns(sample)}"
        return sample

    ds = load_dataset("json", data_files=path, split="train")
    return ds.map(template_dataset, remove_columns=list(ds.features))



def main(
    batch_size=8,
    max_length=2048,
    matbench_dataset="matbench_log_kvrh",
    representation="composition",
    testset_path=None,
    adapter_path=None,
):

    if testset_path is None:
        testset_path = f"/work/so87pot/material_db/all_1/test_{matbench_dataset}_0.json"
    testset = _prepare_datasets(testset_path, matbench_dataset, representation)
    pipe = get_model_for_eval(adapter_path, batch_size=batch_size, max_length=max_length)
    responses_dict = {}
    resp = pipe(testset["text"])
    for j, (prompt, responses) in enumerate(zip(testset["text"], resp)):
        # print(prompt)
        generated_text = responses[0]["generated_text"]
        # Extract the response part by removing the prompt part
        complete_response = generated_text.replace(prompt, "").strip()
        parsed_answer = complete_response.replace(
            "The response is ### Response:", ""
        ).strip()
        responses_dict[j] = {
            "prompt": prompt,
            "response": responses[0]["generated_text"],
            "parsed_answer": parsed_answer,
        }

    with open(f"llama_evals_{matbench_dataset}_{representation}.json", "w") as f:
        json.dump(responses_dict, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
