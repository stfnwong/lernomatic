"""
VOCAB
A new Vocabulary object. Based on the one from (https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)

Stefan Wong 2019
"""

import re

# TODO : move this into something that can manage pair lifetime

MAX_LENGTH = 10

def unicode_to_ascii(s:str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s:str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

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



# TODO: do we ever want to compare vocabularies?
class Vocabulary(object):
    def __init__(self, name:str) -> None:
        self.name = name
        self.init()

    def __repr__(self) -> str:
        return 'Vocabulary'

    def __str__(self) -> str:
        return 'Vocabulary (%d words)' % len(self.idx2word)

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
        # placed the reserved words into the index
        self.idx2word[self.pad_tok] = 0
        self.idx2word[self.sos_tok] = 1
        self.idx2word[self.eos_tok] = 2
        self.num_words = len(self.idx2word)

    def add_word(self, word:str) -> None:
        if word not in self.word2idx:
            self.word2idx[word]           = self.num_words
            self.word2count[word]         = 1
            self.idx2word[self.num_words] = word
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence:list) -> None:
        for word in sentence.split(' '):
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
