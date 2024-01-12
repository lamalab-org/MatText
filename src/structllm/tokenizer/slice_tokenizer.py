import os
import re
from transformers import PreTrainedTokenizer

class AtomVocabTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, model_max_length=None, **kwargs):
        super(AtomVocabTokenizer, self).__init__(model_max_length=model_max_length, **kwargs)
        
        self.vocab = self.load_vocab(vocab_file)
        

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as file:
            vocab = file.read().splitlines()
        return {token: idx for idx, token in enumerate(vocab)}
        
        
    def tokenize(self, text):
        # List of tokens
        tokens = list(self.vocab.keys())

        # Escape special characters in the vocab to ensure they are treated as literals in the regex
        escaped_tokens = [re.escape(token) for token in tokens]

        # Join the escaped vocab terms into a regex pattern
        pattern_str = '|'.join(escaped_tokens)
        pattern = re.compile(pattern_str)

        # Find all matches in the text
        matches = pattern.findall(text)
        return matches

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def _add_tokens(self, new_tokens, **kwargs):
        # Override _add_tokens to add new tokens to the vocabulary
        for token in new_tokens:
            if token not in self.added_tokens_encoder:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = token

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return list(self.vocab.keys())[index]

    def save_vocabulary(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(self.vocab))

