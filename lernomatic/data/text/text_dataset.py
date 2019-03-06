"""
TEXT_DATASET
Datasets for text data

Stefan Wong 2019
"""

import h5py
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, filename, **kwargs):
        self.fp = h5py.File(filename, 'r')

        self.seq_len_name  = kwargs.pop('seq_len_name', 'seqlen')
        self.sequence_name = kwargs.pop('sequence_name', 'sequence')
        self.target_name   = kwargs.pop('target_name', 'target')

    def __repr__(self):
        return 'TextDataset'

    def __str__(self):
        return 'TextDataset (%d items)' % len(self.fp[self.sequence_name])

    def __del__(self):
        self.fp.close()

    def __len__(self):
        return len(self.fp[self.sequence_name])

    def __getitem__(self, idx):
        pass
