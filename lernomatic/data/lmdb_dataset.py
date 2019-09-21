"""
LMDB_DATASET
Torch Dataset wrapper for an LMDB file

Stefan Wong 2019
"""

import lmdb
import torch
from torch.utils.data import Dataset
import numpy as np
import PIL
from torch.utils.data import Dataset

from typing import Tuple

# debug
#from pudb import set_trace; set_trace()


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, filename:str, **kwargs) -> None:
        self.filename = filename
        self.read_ahead:bool = kwargs.pop('read_ahead', False)
        self.lock:bool       = kwargs.pop('lock', False)

        self.cur_idx: int = 0
        self.fp = lmdb.open(
            self.filename,
            readonly=True,
            readahead = self.read_ahead,
            lock = self.lock
            )
        # get a cursor
        self.env  = self.fp.begin()
        self.cursor = self.env.cursor()


    def __del__(self) -> None:
        self.fp.close()

    def __repr__(self) -> str:
        return 'LMDBDataset'

    def __str__(self) -> str:
        s = []
        s.append('LMDBDataset [%s]' % str(self.filename))
        if self.read_ahead:
            s.append(' readahead')
        if self.lock:
            s.append(', lock')
        s.append('\n')

        return ''.join(s)

    def get_num_items(self) -> int:
        num_items = 0
        self.cursor.first()
        while(self.cursor.next()):
            num_items += 1

        return num_items

    # TODO : also write __iter__ and __next__? This seems the more natural
    # way to use LMDB files generally, but need to check if it works properly
    # with the pytorch data loaders

    def __getitem__(self, idx:int) -> tuple:
        # TODO : how to find the size of the dataset (and therefore bounds
        # check idx)?

        data = self.cursor.item()

        # go to the next element
        cur_next = self.cursor.next()
        if cur_next:
            self.cur_idx += 1
        else:
            raise StopIteration


        return data



