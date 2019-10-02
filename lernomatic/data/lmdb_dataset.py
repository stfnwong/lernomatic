"""
LMDB_DATASET
Torch Dataset wrapper for an LMDB file

Stefan Wong 2019
"""

import os
import io
import lmdb
import torch
import string
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
import numpy as np
import PIL
from torch.utils.data import Dataset

from typing import Tuple


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, filename:str, **kwargs) -> None:
        self.filename = filename
        self.read_ahead:bool = kwargs.pop('read_ahead', False)
        self.lock:bool       = kwargs.pop('lock', False)
        # label keys?
        self.label:str       = kwargs.pop('label', 'labels')
        self.feature:str     = kwargs.pop('feature', 'features')
        self.verbose:bool    = kwargs.pop('verbose', False)
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
        cache_file = '__lmdb_dataset_' + ''.join(c for c in self.filename if c in string.ascii_letters) + '.cache'
        if os.path.isfile(cache_file):
            if self.verbose:
                print('Loading cache data from file [%s]' % str(cache_file))
            # load data from cache
            with open(cache_file, 'rb') as fp:
                self.keys = pickle.load(fp)
        else:
            if self.verbose:
                print('Generating LMDB cache file [%s]...' % str(cache_file))

            # cache keys and write to filec
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in tqdm(txn.cursor(), unit='keys', total=self.length)]

            with open(cache_file, 'wb') as fp:
                pickle.dump(self.keys, fp)

        if self.verbose:
            print('\n...OK')

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

        return (img, target)        # TODO : target should be the id?
