# MatText Tokenizers

MatText comes with custom tokenizers suitable for modelling text representations.
Currently available [`tokenizer`](api.md#mattext.tokenizer) are `SliceTokenizer`, `CompositionTokenizer`, `CifTokenizer`, `RobocrysTokenizer`, `SmilesTokenizer`. All the MatText representations can be translated to tokens using one of these tokenizers.


## Using MatText Tokenizers

By default, the tokenizer is initialized with `[CLS]` and `[SEP]`
tokens. For an example, see the `SliceTokenizer` usage: 

``` python
from mattext.tokenizer import SliceTokenizer

tokenizer = SliceTokenizer(
                model_max_length=512, 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
print(tokenizer.cls_token) 
print(tokenizer.tokenize("Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -"))
```

??? success "output"

    ```bash
    [CLS]
    ['[CLS]', 'Ga', 'Ga', 'P', 'P', '0', '3', '- - o', '0', '2', '- o -', '0', '1', 'o - -', '[SEP]']
    ```


???+ tip "tip"

    You can access the `[CLS]` token using the `cls_token` attribute of the tokenizer. 

During decoding, you can utilize the `skip_special_tokens` parameter to skip these special tokens.

``` python
token_ids = tokenizer.encode("Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -")
print(token_ids)
decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
print(decoded)
```
??? success "output"

    ```bash
    [149, 57, 57, 41, 41, 139, 142, 24, 139, 141, 20, 139, 140, 8, 150]
    Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -
    ```

## Initializing Tokenizers With Custom Special Tokens

In scenarios where the `[CLS]` token is not required, you can initialize
the tokenizer with an empty special_tokens dictionary.

Initialization without `[CLS]` and `[SEP]` tokens:

``` python
tokenizer = SliceTokenizer(
                model_max_length=512, 
                special_tokens={}, 
                truncation=True,
                padding="max_length", 
                max_length=512
            )
```

All `MatText Tokenizer` instances inherit from
[PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and accept arguments compatible with the Hugging Face tokenizer.

## Tokenizers With Special Number Tokenization

The `special_num_token` argument (by default `False`) can be
set to `True`  to tokenize numbers in a special way as designed and
implemented by
[RegressionTransformer](https://www.nature.com/articles/s42256-023-00639-z).


``` python
tokenizer = SliceTokenizer(
                special_num_token=True,
                model_max_length=512, 
                special_tokens={}, 
                truncation=True,
                padding="max_length", 
                max_length=512
            )
tokenizer.tokenize("H2SO4")
```

??? success "output"

    ```python
    ['H', '_2_0_', 'S', 'O', '_4_0_']
    ```



## Updating Vocabulary in the Tokenizers

The MatText tokenizers allow one to update the vocabulary or load a custom vocabulary file.

Default vocabulary can be determined by calling the `vocab` attribute.

``` python
tokenizer = SliceTokenizer()
print(tokenizer.vocab)

```

??? success "output"

    ```python
    {'o o o': 0, 'o o +': 1, 'o o -': 2, 'o + o': 3, 'o + +': 4, 'o + -': 5, 'o - o': 6, 'o - +': 7, 'o - -': 8, '+ o o': 9, '+ o +': 10, '+ o -': 11, '+ + o': 12, '+ + +': 13, '+ + -': 14, '+ - o': 15, '+ - +': 16, '+ - -': 17, '- o o': 18, '- o +': 19, '- o -': 20, '- + o': 21, '- + +': 22, '- + -': 23, '- - o': 24, '- - +': 25, '- - -': 26, 'H': 27, 'He': 28, 'Li': 29, 'Be': 30, 'B': 31, 'C': 32, 'N': 33, 'O': 34, 'F': 35, 'Ne': 36, 'Na': 37, 'Mg': 38, 'Al': 39, 'Si': 40, 'P': 41, 'S': 42, 'Cl': 43, 'K': 44, 'Ar': 45, 'Ca': 46, 'Sc': 47, 'Ti': 48, 'V': 49, 'Cr': 50, 'Mn': 51, 'Fe': 52, 'Ni': 53, 'Co': 54, 'Cu': 55, 'Zn': 56, 'Ga': 57, 'Ge': 58, 'As': 59, 'Se': 60, 'Br': 61, 'Kr': 62, 'Rb': 63, 'Sr': 64, 'Y': 65, 'Zr': 66, 'Nb': 67, 'Mo': 68, 'Tc': 69, 'Ru': 70, 'Rh': 71, 'Pd': 72, 'Ag': 73, 'Cd': 74, 'In': 75, 'Sn': 76, 'Sb': 77, 'Te': 78, 'I': 79, 'Xe': 80, 'Cs': 81, 'Ba': 82, 'La': 83, 'Ce': 84, 'Pr': 85, 'Nd': 86, 'Pm': 87, 'Sm': 88, 'Eu': 89, 'Gd': 90, 'Tb': 91, 'Dy': 92, 'Ho': 93, 'Er': 94, 'Tm': 95, 'Yb': 96, 'Lu': 97, 'Hf': 98, 'Ta': 99, 'W': 100, 'Re': 101, 'Os': 102, 'Ir': 103, 'Pt': 104, 'Au': 105, 'Hg': 106, 'Tl': 107, 'Pb': 108, 'Bi': 109, 'Th': 110, 'Pa': 111, 'U': 112, 'Np': 113, 'Pu': 114, 'Am': 115, 'Cm': 116, 'Bk': 117, 'Cf': 118, 'Es': 119, 'Fm': 120, 'Md': 121, 'No': 122, 'Lr': 123, 'Rf': 124, 'Db': 125, 'Sg': 126, 'Bh': 127, 'Hs': 128, 'Mt': 129, 'Ds': 130, 'Rg': 131, 'Cn': 132, 'Nh': 133, 'Fl': 134, 'Mc': 135, 'Lv': 136, 'Ts': 137, 'Og': 138, '0': 139, '1': 140, '2': 141, '3': 142, '4': 143, '5': 144, '6': 145, '7': 146, '8': 147, '9': 148, '[CLS]': 149, '[SEP]': 150}
    ```


``` python

# Path to your custom vocabulary file (JSON or TXT format)
vocab_file_path = "path/to/your/vocab_file.json"

tokenizer = SliceTokenizer(
                special_num_token=True,
                model_max_length=512, 
                special_tokens={}, 
                truncation=True,
                padding="max_length", 
                max_length=512,
                vocab_file=vocab_file_path
            )
```

here is an example format for the vocabulary json file

   
```python
    import json
    import tempfile # (1)
    new_vocab = {
    "H": 1,
    "He": 2,
    "New_Atom": 3,
    "1":4,
    "2":5,}

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
        # Write the JSON string to the temporary file
        json.dump(my_dict, temp_file)
        temp_file_path = temp_file.name


    tokenizer = SliceTokenizer(
                    model_max_length=512,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    vocab_file=temp_file_path
                )

    print(tokenizer.tokenize("H He Na New_Atom 1  2 9"))
    print(tokenizer.vocab)

```

1.  :man_raising_hand: here notice that we are writing the vocabulary dictionary to a temporary file, since Tokenizer class accepts only file paths.

??? success "output"

    ```python
    ['[CLS]', 'H', 'He', 'New_Atom', '1', '2', '[SEP]']

    #Atoms that are not within the vocabulary are ignored. and newly added atoms are correctly tokenized.

    {'H': 1, 'He': 2, 'New_Atom': 3, '1': 4, '2': 5, '[CLS]': 6, '[SEP]': 7}
    ```

