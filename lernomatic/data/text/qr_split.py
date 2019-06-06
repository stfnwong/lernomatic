"""
QR_SPLIT
Data splits specialized around Query/Response datasets

Stefan Wong 2019
"""

import json
import numpy as np
from lernomatic.data import data_split
from lernomatic.data.text import qr_pair


class QRDataSplit(object):
    def __init__(self, split_name:str='train') -> None:
        self.split_name:str = split_name
        self.pairs:list     = []
        self.idx:int        = 0

    def __repr__(self) -> str:
        return 'QRDataSplit'

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.pairs):
            raise StopIteration
        pair = self.pairs[self.idx]
        self.idx += 1
        return pair

    def add_pair(self, p:qr_pair.QRPair) -> None:
        self.pairs.append(p)

    def save(self, filename:str) -> None:
        d = {
            'pairs': self.pairs,
            'split_name': self.split_name
        }
        with open(filename, 'w') as fp:
            json.dump(d, fp)

    def load(self, filename:str) -> None:
        with open(filename, 'r') as fp:
            d = json.load(fp)

        self.pairs      = d['pairs']
        self.split_name = d['split_name']


class QRDataSplitter(data_split.DataSplitter):
    def __init__(self, **kwargs) -> None:
        self.verbose = kwargs.pop('verbose', False)
        super(QRDataSplitter, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'QRDataSplitter'

    def __str__(self):
        s = []
        s.append('QRDataSplitter (%d splits) \n' % len(self.split_ratios))
        for n in range(len(self.split_ratios)):
            s.append('Split [%s] : %.2f %%\n' % (self.split_names[n], 100 * self.split_ratios[n]))
        return ''.join(s)

    def gen_splits(self, qr_pairs:list) -> None:
        if self.split_method == 'random':
            split_idxs = np.random.permutation(len(qr_pairs))
        elif self.split_method == 'seq':
            split_idxs = range(len(qr_pairs))
        else:
            raise ValueError('Unknown split method [%s]' % str(self.split_method))

        split_lens, split_offsets = self.gen_split_idx_offset(len(qr_pairs))
        # create split objects
        splits = [QRDataSplit(split_name=self.split_names[n]) for n in range(len(self.split_names))]

        if self.verbose:
            sample_count = 0
        for o in range(len(split_offsets)-1):
            for idx in range(split_offsets[o], split_offsets[o+1]):
                # Show progress
                if self.verbose:
                    print('Split <%s> : adding sample [%d/%d]' %\
                          (self.split_names[o], sample_count, len(split_idxs)),
                          end='\r'
                    )
                splits[o].add_pair(qr_pairs[split_idxs[idx]])

                if self.verbose:
                    sample_count += 1
            if self.verbose:
                print('\n\tdone')

        return splits
