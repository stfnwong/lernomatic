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
    def __init__(self, **kwargs):
        self.img_key       = kwargs.pop('img_key', 'images')
        # options
        self.verbose       = kwargs.pop('verbose', False)
        self.max_len       = kwargs.pop('max_len', 100)
        self.min_word_freq = kwargs.pop('min_word_freq', 5)
        self.capt_per_img  = kwargs.pop('capt_per_img', 5)
        # init word map
        self.word_map      = None
        self.map_word      = None            # inverts a word map
        self.word_freq     = Counter()

    def __repr__(self):
        return 'WordMap'

    def __str__(self):
        s = []
        s.append('WordMap (%d words)' % len(self.word_map))
        return ''.join(s)

    def __len__(self):
        return len(self.word_map)

    def _init_data(self):
        self.train_data.reset()
        self.test_data.reset()
        self.val_data.reset()

    def save(self, fname):
        """
        SAVE
        Commit the word map to disk in JSON format
        """
        if self.word_map is None:
            if self.verbose:
                print('No word map data, exiting...')
            return

        with open(fname, 'w') as fp:
            json.dump(self.word_map, fp)

    def load(self, fname):
        """
        LOAD
        Load a word map from a JSON on disk
        """
        with open(fname, 'r') as fp:
            self.word_map = json.load(fp)

    def gen_map_word(self):
        if self.word_map is None:
            return

        self.map_word = dict()
        for k, v in self.word_map.items():
            self.map_word[v] = k

    def lookup_word(self, word):
        if self.map_word is None:
            self.gen_map_word()
        try:
            return self.map_word[word]
        except:
            return self.map_word[self.word_map['<unk>']]

    def get_word_map(self):
        return self.word_map

    # TODO : type hints?
    def update(self, word_list):
        """
        UPDATE
        Update the word map with new words
        """
        if type(word_list[0]) is str:
            self.word_freq.update(word_list)
        if type(word_list[0]) is list:
            for w in word_list:
                self.word_freq.update(w)

    def generate(self):
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
