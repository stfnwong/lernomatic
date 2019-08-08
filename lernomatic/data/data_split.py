"""
DATA_SPLIT
Represents a generic split of data

Stefan Wong 2018
"""

import json
import os
import numpy as np


class DataSplit(object):
    """
    DATASPLIT
    Holds information about a datasplit
    """
    def __init__(self, split_name:str='train') -> None:
        self.data_paths  = list()
        self.data_labels = list()
        self.elem_ids    = list()
        self.split_name  = split_name
        self.idx         = 0
        self.has_labels = False
        self.has_ids    = False

    def __len__(self) -> int:
        return len(self.data_paths)

    def __repr__(self) -> str:
        return 'DataSplit <%s> (%d items)' % (self.split_name, len(self))

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> 'DataSplit':
        self.idx = 0
        return self

    def __next__(self) -> tuple:
        if self.idx >= len(self.data_paths):
            raise StopIteration
        path  = self.data_paths[self.idx]
        if self.has_ids:
            elem_id = self.elem_ids[self.idx]
        else:
            elem_id = None
        if self.has_labels:
            label = self.data_labels[self.idx]
        else:
            label = None
        self.idx += 1

        return (path, elem_id, label)

    def get_name(self) -> str:
        return self.split_name

    def get_param_dict(self) -> dict:
        param = dict()
        param['data_paths']  = self.data_paths
        param['data_labels'] = self.data_labels
        param['split_name']  = self.split_name
        param['elem_ids']    = self.elem_ids

        return param

    def set_param_from_dict(self, param: str) -> None:
        self.data_paths  = param['data_paths']
        self.data_labels = param['data_labels']
        self.split_name  = param['split_name']
        self.elem_ids    = param['elem_ids']

    def add_path(self, p) -> None:
        self.data_paths.append(p)

    def add_label(self, l) -> None:
        self.data_labels.append(l)

    def add_id(self, i) -> None:
        self.elem_ids.append(i)

    def add_elem(self, p, i, l) -> None:
        self.data_paths.append(p)
        self.elem_ids.append(i)
        self.data_labels.append(l)

    def init_labels_ids(self) -> None:
        self.elem_ids    = [int(0) for _ in range(len(self.data_paths))]
        self.data_labels = [int(0) for _ in range(len(self.data_paths))]

    def save(self, fname:str) -> None:
        param = self.get_param_dict()
        with open(fname, 'w') as fp:
            json.dump(param, fp)

    def load(self, fname:str) -> None:
        with open(fname, 'r') as fp:
            param = json.load(fp)
        self.set_param_from_dict(param)

    def get_elem(self, idx:int) -> tuple:
        return (self.data_paths[idx], self.elem_ids[idx], self.data_labels[idx])

    # TODO : these
    #def to_csv(self, fname):
    #    pass

    #def from_csv(self, fname):
    #    pass

# ======== Splitters ======== #
class DataSplitter(object):
    def __init__(self, **kwargs) -> None:
        valid_split_methods = ('random', 'seq')
        self.split_ratios    :list = kwargs.pop('split_ratios', [0.8, 0.1, 0.1])
        self.split_names     :list = kwargs.pop('split_names', ['train', 'test', 'val'])
        self.split_method    :str  = kwargs.pop('split_method', 'random')
        self.split_data_root :str  = kwargs.pop('data_root', None)

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

    def __repr__(self) -> str:
        return 'DataSplitter'

    def __str__(self) -> str:
        s = []
        s.append('Data Splitter (%d splits) \n' % len(self.split_ratios))
        for n in range(len(self.split_ratios)):
            s.append('Split [%s] : %.2f %%\n' % (self.split_names[n], 100 * self.split_ratios[n]))
        return ''.join(s)

    def gen_split_idx_offset(self, df_len:int) -> tuple:
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
    def __init__(self, **kwargs) -> None:
        self.verbose = kwargs.pop('verbose', False)
        super(ListSplitter, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'ListSplitter'

    def __str__(self) -> str:
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

    def gen_splits(self, data_list:list , label_list:list, id_list=None) -> list:
        if self.split_method == 'random':
            split_idxs = np.random.permutation(len(data_list))
        elif self.split_method == 'seq':
            split_idxs = range(len(data_list))
        else:
            raise ValueError('Unknown split method [%s]' % str(self.split_method))

        split_lens, split_offsets = self.gen_split_idx_offset(len(data_list))

        # create split objects
        splits = [DataSplit(split_name=self.split_names[n]) for n in range(len(self.split_names))]

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
                if id_list is None:
                    splits[o].add_elem(path, 0, label_list[split_idxs[idx]])
                else:
                    splits[o].add_elem(path, id_list[split_idxs[idx]], label_list[split_idxs[idx]])

                if self.verbose:
                    sample_count += 1
            if self.verbose:
                print('\n\tdone')

        return splits
