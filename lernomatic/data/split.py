"""
SPLIT
Utils for creating data splits

Stefan Wong 2019
"""

import os
import numpy as np
from lernomatic.data import data_split

# debug
from pudb import set_trace; set_trace()


class DataSplitter(object):
    def __init__(self, **kwargs):
        valid_split_methods = ('random', 'seq')
        self.split_ratios = kwargs.pop('split_ratios', [0.8, 0.1, 0.1])
        self.split_names  = kwargs.pop('split_names', ['train', 'test', 'val'])
        self.split_method = kwargs.pop('split_method', 'random')
        self.split_data_root = kwargs.pop('data_root', None)

        if len(self.split_ratios) != len(self.split_names):
            raise ValueError('Must be same number of split_ratios as split_names')

        if self.split_method not in valid_split_methods:
            raise ValueError('Unknown split method %s, must be one of %s' %\
                             (self.split_method, str(valid_split_methods))
        )

        if self.split_data_root is None:
            self.split_data_root = ['' for n in range(len(self.split_names))]
        elif len(self.split_data_root) == 1:
            self.split_data_root = [self.split_data_root for n in range(len(self.split_names))]

    def __repr__(self):
        return 'DataSplitter'

    def __str__(self):
        s = []
        s.append('Data Splitter (%d splits) \n' % len(self.split_ratios))
        for n in range(len(self.split_ratios)):
            s.append('Split [%s] : %.2f %%\n' % (self.split_names[n], 100 * self.split_ratios[n]))
        return ''.join(s)

    def gen_split_idx_offset(self, df_len):

        split_lens = [int(df_len * r) for r in self.split_ratios]
        split_offsets = [0]
        for n, sp in enumerate(split_lens):
            split_offsets.append(split_offsets[n] + sp)

        return (split_lens, split_offsets)

# TODO : csv splitter

class ListSplitter(DataSplitter):
    """
    Generate splits from a list of filenames
    """
    def __init__(self, **kwargs):
        self.verbose = kwargs.pop('verbose', False)
        super(ListSplitter, self).__init__(**kwargs)

    def __repr__(self):
        return 'ListSplitter'

    def __str__(self):
        s = []
        s.append('ListSplitter (%d splits) \n' % len(self.split_ratios))
        for n in range(len(self.split_ratios)):
            s.append('Split [%s] : %.2f %%\n' % (self.split_names[n], 100 * self.split_ratios[n]))
        return ''.join(s)

    """
    When finding the split indicies, we should
    1) Create a list of indicies ordered the way we like (eq:random, in-order, etc)
    2) Then create the split offsets based on that.
    """

    def gen_splits(self, data_list, label_list):
        if self.split_method == 'random':
            split_idxs = np.random.permutation(len(data_list))
        elif self.split_method == 'seq':
            split_idxs = range(len(data_list))
        else:
            raise ValueError('Unknown split method [%s]' % str(self.split_method))

        split_lens, split_offsets = self.gen_split_idx_offset(len(data_list))

        # create split objects
        splits = [data_split.DataSplit(split_name=self.split_names[n]) for n in range(len(self.split_names))]

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
                path = os.path.join(
                    self.split_data_root[o],
                    data_list[split_idxs[idx]]
                )
                splits[o].add_elem(path, 0, label_list[split_idxs[idx]])

                if self.verbose:
                    sample_count += 1
            if self.verbose:
                print('\n\tdone')

        return splits
