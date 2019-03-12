"""
TEXT_DATASET
Datasets for text data

Stefan Wong 2019
"""

import h5py
import json
import torch
from torch.utils.data import Dataset

from lernomatic.data.text import corpus
from lernomatic.data.text import word_map


# TODO : even though most text datasets are going to be small enough to fit
# completely into memory, it might be worth writing a batched loader as well
# for completeness
class TextDatasetString(Dataset):
    def __init__(self, filename, wmap : word_map.WordMap, **kwargs) -> None:
        self.wmap           = wmap
        self.seq_len : int  = kwargs.pop('seq_len', 64)
        with open(filename, 'r') as fp:
            self.data = fp.read()

        self.data_ptr : int = 0

    def __repr__(self) -> str:
        return 'TextDatasetString'

    def __str__(self) -> str:
        return 'TextDatasetString (%d chars)' % len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        pass



class CorpusDataset(Dataset):
    def __init__(self, text_corpus : corpus.Corpus, **kwargs) -> None:
        self.text_corpus = text_corpus

    def __repr__(self) -> str:
        return 'TextCorpusDataset'



# TODO : these can stay empty for now
class TextDatasetHDF5(Dataset):
    def __init__(self, filename : str, **kwargs) -> None:
        self.fp = h5py.File(filename, 'r')

        self.seq_len_name  : str = kwargs.pop('seq_len_name', 'seqlen')
        self.sequence_name : str = kwargs.pop('sequence_name', 'sequence')
        self.target_name   : str = kwargs.pop('target_name', 'target')

    def __repr__(self) -> str:
        return 'TextDatasetHDF5'

    def __str__(self) -> str:
        return 'TextDatasetHDF5 (%d items)' % len(self.fp[self.sequence_name])

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.sequence_name])

    def __getitem__(self, idx):
        pass


class TextDatasetJSON(Dataset):
    def __init__(self, filename : str , wmap : word_map.WordMap, **kwargs) -> None:
        self.wmap = wmap
        self.text_key = kwargs.pop('text_key', 'body')

        # read in data
        with open(filename, 'r') as fp:
            self.data = json.load(fp)

    def __repr__(self) -> str:
        return 'TextDatasetJSON'

    def __str__(self) -> str:
        return 'TextDatasetJSON (%d items)' % len(self)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        pass

