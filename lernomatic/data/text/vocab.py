"""
VOCAB
A new Vocabulary object. Based on the one from (https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)

Stefan Wong 2019
"""

import codecs
import re
import unicodedata


def read_vocs(filename:str, corpus_name:str) -> tuple:
    lines =  open(filename, encoding='utf-8').read().strip().split('\n')
    # split each line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Vocabulary(corpus_name)

    return voc, pairs


def filter_pair(p:list) -> bool:
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs:list) -> list:
    return [pair for pair in pairs if filter_pair(pair)]



class Vocabulary(object):
    def __init__(self, name:str) -> None:
        self.name:str = name
        self.init()

    def __repr__(self) -> str:
        return 'Vocabulary [%s]' % str(self.name)

    def __str__(self) -> str:
        return 'Vocabulary [%s] (%d words)' % (str(self.name), len(self.idx2word))

    def __len__(self) -> int:
        return len(self.idx2word)

    def init(self) -> None:
        """
        Re-initializes the internal state
        """
        self.word2idx   = dict()
        self.word2count = dict()
        self.idx2word   = dict()

        # reserved token constants
        self.pad_tok = '<pad>'
        self.sos_tok = '<sos>'
        self.eos_tok = '<eos>'
        self.unk_tok = '<unk>'
        # placed the reserved words into the index
        self.word2idx[self.pad_tok] = 0
        self.word2idx[self.sos_tok] = 1
        self.word2idx[self.eos_tok] = 2
        self.word2idx[self.unk_tok] = 3
        self.num_words = len(self.word2idx)

    def get_eos(self) -> int:
        return self.word2idx[self.eos_tok]

    def get_eos_str(self) -> str:
        return self.eos_tok

    def get_sos(self) -> int:
        return self.word2idx[self.sos_tok]

    def get_sos_str(self) -> str:
        return self.sos_tok

    def get_pad(self) -> int:
        return self.word2idx[self.pad_tok]

    def get_pad_str(self) -> str:
        return self.pad_tok

    def get_unk(self) -> int:
        return self.word2idx[self.unk_tok]

    def get_unk_str(self) -> str:
        return self.unk_tok

    def lookup_word(self, word:str) -> int:
        try:
            return self.word2idx[word]
        except:
            return self.word2idx[self.unk_tok]

    def lookup_idx(self, idx:int) -> str:
        try:
            return self.idx2word[idx]
        except:
            return self.idx2word[self.word2idx[self.unk_tok]]

    def add_word(self, word:str) -> None:
        if word not in self.word2idx:
            self.word2idx[word]           = self.num_words
            self.word2count[word]         = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence:list, delimiter:str=' ') -> None:
        for word in sentence.split(delimiter):
            self.add_word(word)

    def trim_freq(self, min_word_count:int) -> None:
        """
        trim_freq()

        Remove words from the vocabulary that occur fewer than min_word_count times
        """
        min_freq_words = []
        for k, v in self.word2count.items():
            if v >= min_word_count:
                min_freq_words.append(k)

        self.init()
        for w in min_freq_words:
            self.add_word(w)
