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


## Initializing tokenizers with custom special tokens

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

## Tokenizers with special number tokenization

The `special_num_token` argument (by default `False`) can be
set to true to tokenize numbers in a special way as designed and
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

