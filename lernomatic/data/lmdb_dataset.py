"""
LMDB_DATASET
Torch Dataset wrapper for an LMDB file

Stefan Wong 2019
"""

import io
import lmdb
import torch
from torch.utils.data import Dataset
import numpy as np
import PIL
from torch.utils.data import Dataset

from typing import Tuple

# debug
from pudb import set_trace; set_trace()


# TODO : check the LSUN dataset source for ideas
class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, filename:str, **kwargs) -> None:
        self.filename = filename
        self.read_ahead:bool = kwargs.pop('read_ahead', False)
        self.lock:bool       = kwargs.pop('lock', False)
        # label keys?
        self.label:str       = kwargs.pop('label', 'labels')
        self.feature:str     = kwargs.pop('feature', 'features')
        self.transform       = kwargs.pop('transform', None)

        self.cur_idx: int = 0
        self.env = lmdb.open(
            self.filename,
            readonly=True,
            readahead = self.read_ahead,
            lock = self.lock
            )
        # cache the size of the dataset
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        # cache all the keys in the dataset
        # TODO : write cache to a file
        with self.env.begin(write=False) as txn:
            self.keys = [key for key, _ in txn.cursor()]

    def __repr__(self) -> str:
        return 'LMDBDataset [%s]' % str(self.filename)

    def __str__(self) -> str:
        s = []
        s.append('LMDBDataset [%s]' % str(self.filename))
        if self.read_ahead:
            s.append(' readahead')
        if self.lock:
            s.append(', lock')
        s.append('\n')

        return ''.join(s)

    def __len__(self) -> int:
        return self.length

    # TODO : also write __iter__ and __next__? This seems the more natural
    # way to use LMDB files generally, but need to check if it works properly
    # with the pytorch data loaders

    def __getitem__(self, idx:int) -> tuple:
        img = None
        target = 0

        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[idx])

        # convert the buf to an image
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = PIL.Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            target = torch.zeros(*img.shape)


        return (img, target)        # TODO : target is always None here...
