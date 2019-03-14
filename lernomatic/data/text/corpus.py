"""
CORPUS
Object that wraps a corpus of text

Stefan Wong 2019
"""

import torch
import numpy as np
from lernomatic.data.text import word_map


class Corpus(object):
    def __init__(self, wmap : word_map.WordMap, **kwargs) -> None:
        self.wmap      : word_map.WordMap = wmap
        self.filename  : str = kwargs.pop('filename', None)
        self.end_token : str = kwargs.pop('end_token', '<end>')
        self.body            = None

    def __repr__(self) -> str:
        return 'Corpus'

    def get_body(self):
        return self.body

    def tokenize(self, text : list) -> torch.LongTensor:
        tokens = torch.LongTensor(len(text))
        for n, word in enumerate(text):
            tokens[n] = self.wmap.lookup(word)

        return tokens

    def tokenize_list(self, text : list,
                      update_map : bool = False,
                      return_tensor : bool = False) -> torch.LongTensor:
        if update_map is True:
            self.wmap.update(text)
            self.wmap.generate()

        if return_tensor is True:
            tokens = torch.LongTensor(len(text))
        else:
            tokens = np.zeros(len(text))
        for n, w in enumerate(text):
            tokens[n] = self.wmap.lookup_word(w)

        return tokens

    def tokenize_file(self, filename : str,
                      update_map : bool = False,
                      return_tensor : bool = False) -> torch.LongTensor:
        self.filename = filename
        # find number of tokens in file
        with open(filename, 'r') as fp:
            num_tokens = 0
            for line in fp:
                words = line.split() + [self.end_token]
                num_tokens += len(words)
                if update_map is True:
                    self.wmap.update(words)

        if update_map is True:
            self.wmap.generate()

        # Now go back and tokenize the data
        with open(filename, 'r') as fp:
            if return_tensor is True:
                tokens = torch.LongTensor(num_tokens)
            else:
                tokens = np.zeros(num_tokens)
            tok_ptr = 0
            for line in fp:
                words = line.split() + [self.end_token]
                for w in words:
                    tokens[tok_ptr] = self.wmap.lookup_word(w)
                    tok_ptr += 1

        return tokens

