"""
TEXT_DATASET
Datasets for text data

Stefan Wong 2019
"""

import h5py
import json
import torch
from torch.utils import data

from lernomatic.data.text import corpus
from lernomatic.data.text import word_map


class TextRawDataset(data.Dataset):
    def __init__(self, fname, **kwargs) -> None:
        self.chars_per_block : int = kwargs.pop('chars_per_block', 24)
        self.pred_window     : int = kwargs.pop('pred_window', 2)
        # read file contents into memory.... incrementally?

    def __repr__(self) -> str:
        return 'TextRawDataset'

    def __del__(self) -> None:
        self.fp.close()

    def __getitem__(self, idx):
        pass


class TextHDF5Dataset(data.Dataset):
    """
    TextHDF5Dataset
    Reads in a text dataset that has been pre-processed into an HDF5 file.
    """
    def __init__(self, fname : str, **kwargs) -> None:
        self.data_key        : str = kwargs.pop('data_key', 'text')
        self.chars_per_block : int = kwargs.pop('chars_per_block', 24)
        self.pred_window     : int = kwargs.pop('pred_window', 2)
        self.fp = h5py.File(fname, 'r')

    def __repr__(self) -> str:
        return 'TextHDF5Dataset'

    def __del__(self):
        self.fp.close()

    def __len__(self):
        return len(self.fp[self.data_key])

    def __getitem__(self, idx):
        if (idx + self.pred_window) > len(self):
            raise ValueError('TODO : need to handle data wraps')
