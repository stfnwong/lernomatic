"""
WORD_MAP
A word map object

Stefan Wong 2018
"""

import json
from collections import Counter

# debug
#from pudb import set_trace; set_trace()

class WordMap(object):
    def __init__(self, **kwargs) -> None:
        self.img_key       = kwargs.pop('img_key', 'images')
        # options
        self.verbose       = kwargs.pop('verbose', False)
        self.max_len       = kwargs.pop('max_len', 100)
        self.min_word_freq = kwargs.pop('min_word_freq', 5)
        # init word map
        self.word_map      = None
        self.map_word      = None            # inverse of a word map
        self.word_freq     = Counter()

    def __repr__(self) -> str:
        return 'WordMap'

    def __str__(self) -> str:
        s = []
        s.append('WordMap (%d words)' % len(self.word_map))
        return ''.join(s)

    def __len__(self) -> int:
        return len(self.word_map)

    def _init_data(self):
        self.train_data.reset()
        self.test_data.reset()
        self.val_data.reset()

    def save(self, fname:str) -> None:
        """
        SAVE
        Commit the word map to disk in JSON format
        """
        with open(fname, 'w') as fp:
            json.dump(self.word_map, fp)

    def load(self, fname:str) -> None:
        """
        LOAD
        Load a word map from a JSON on disk
        """
        with open(fname, 'r') as fp:
            self.word_map = json.load(fp)
            self.gen_map_word()

    def get_vocab_size(self) -> int:
        return len(self.word_map)

    def get_unk(self) -> int:
        return self.word_map['<unk>']

    def get_start(self) -> int:
        return self.word_map['<start>']

    def get_end(self) -> int:
        return self.word_map['<end>']

    def get_pad(self) -> int:
        return self.word_map['<pad>']

    def gen_map_word(self) -> str:
        if self.word_map is None:
            return

        self.map_word = dict()
        for k, v in self.word_map.items():
            self.map_word[v] = k

    def lookup_word(self, word:int) -> str:
        if self.map_word is None:
            self.gen_map_word()
        try:
            return self.map_word[word]
        except:
            return self.map_word[self.word_map['<unk>']]

    def tok2word(self, tok:int) -> str:
        try:
            return self.map_word[tok]
        except:
            return self.map_word[self.word_map['<unk>']]

    def word2tok(self, word:str) -> int:
        try:
            return self.word_map[word]
        except:
            return self.word_map['<unk>']

    def get_word_map(self) -> dict:
        return self.word_map

    def update(self, word_list:list) -> None:
        """
        UPDATE
        Update the word map with new words
        """
        for w in word_list:
            self.word_freq.update(w)

    def generate(self) -> None:
        """
        GENERATE
        Generate a word mapping based on the internal word
        frequency information.
        """
        self.words = [w for w in self.word_freq.keys() \
                      if self.word_freq[w] > self.min_word_freq]
        self.word_map            = {k: v+1 for v, k in enumerate(self.words)}
        self.word_map['<unk>']   = len(self.word_map) + 1
        self.word_map['<start>'] = len(self.word_map) + 1
        self.word_map['<end>']   = len(self.word_map) + 1
        self.word_map['<pad>']   = 0
        self.gen_map_word()
