# MatText Tokenizers

MatText comes with custom tokenizers suitable for modelling text representations.
Currently available [`tokenizer`](api.md#mattext.tokenizer) are `SliceTokenizer`, `CompositionTokenizer`, `CifTokenizer`, `RobocrysTokenizer`, `SmilesTokenizer`. All the MatText representations can be translated to tokens using one of these tokenizers.


## Using MatText Tokenizers

By default, the tokenizer is initialized with `\[CLS\]` and `\[SEP\]`
tokens. For an example, see the `SliceTokenizer` usage: 

``` python
from mattext.tokenizer import SliceTokenizer

tokenizer = SliceTokenizer(
                model_max_length=512, 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
print(tokenizer.cls_token) # returns [CLS]
```
example tokenization output
```python
>>> tokenizer.tokenize("Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -")
['[CLS]', 'Ga', 'Ga', 'P', 'P', '0', '3', '- - o', '0', '2', '- o -', '0', '1', 'o - -', '[SEP]']
```

You can access the `\[CLS\]` token using the [cls_token]{.title-ref}
attribute of the tokenizer. During decoding, you can utilize the
[skip_special_tokens]{.title-ref} parameter to skip these special
tokens.

Decoding with skipping special tokens:

``` python
tokenizer.decode(token_ids, skip_special_tokens=True)
```


## Initializing Tokenizers With Custom Special Tokens

In scenarios where the `\[CLS\]` token is not required, you can initialize
the tokenizer with an empty special_tokens dictionary.

Initialization without `\[CLS\]` and `\[SEP\]` tokens:

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
```

example output
```python
tokenizer.tokenize("H2SO4")
['H', '_2_0_', 'S', 'O', '_4_0_']

```


## Updating Vocabulary in the Tokenizers

The MatText tokenizers allow one to update the vocabulary or load a custom vocabulary file.

Default vocabulary can be determined by calling the `vocab` attribute.

``` python
tokenizer = SliceTokenizer()
print(tokenizer.vocab)

"""
output: {'o o o': 0, 'o o +': 1, 'o o -': 2, 'o + o': 3, 'o + +': 4, ....
"""
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

``` python

import json
import tempfile


new_vocab = {
    "H": 1,
    "He": 2,
    "New_Atom": 3,
    "1":4,
    "2":5,
}

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
"""
output : ['[CLS]', 'H', 'He', 'New_Atom', '1', '2', '[SEP]']
Atoms that are not within the vocabulary are ignored. and newly added atoms are correctly tokenized.

output : {'H': 1, 'He': 2, 'New_Atom': 3, '1': 4, '2': 5, '[CLS]': 6, '[SEP]': 7}
"""

```

>here notice that we are writing the vocabulary dictionary to a temporary file, since Tokenizer class accepts only file paths.



