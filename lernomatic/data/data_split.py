"""
DATA_SPLIT
Represents a generic split of data

Stefan Wong 2018
"""

import json

class DataSplit(object):
    """
    DATASPLIT
    Holds information about a datasplit
    """
    def __init__(self, split_name='train'):
        self.data_paths  = list()
        self.data_labels = list()
        self.elem_ids    = list()
        self.split_name  = split_name
        self.idx         = 0

    def __len__(self):
        return len(self.data_paths)

    def __repr__(self):
        return 'DataSplit <%s> (%d items)' % (self.split_name, len(self))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.data_paths):
            raise StopIteration
        path  = self.data_paths[self.idx]
        elem_id = self.elem_ids[self.idx]
        label = self.data_labels[self.idx]
        self.idx += 1

        return (path, elem_id, label)

    def get_name(self):
        return self.split_name

    def get_param_dict(self):
        param = dict()
        param['data_paths']  = self.data_paths
        param['data_labels'] = self.data_labels
        param['split_name']  = self.split_name
        param['elem_ids']    = self.elem_ids

        return param

    def set_param_from_dict(self, param):
        self.data_paths  = param['data_paths']
        self.data_labels = param['data_labels']
        self.split_name  = param['split_name']
        self.elem_ids    = param['elem_ids']

    def add_path(self, p):
        self.data_paths.append(p)

    def add_label(self, l):
        self.data_labels.append(l)

    def add_id(self, i):
        self.elem_ids.append(i)

    def add_elem(self, p, i, l):
        self.data_paths.append(p)
        self.elem_ids.append(i)
        self.data_labels.append(l)

    def save(self, fname):
        param = self.get_param_dict()
        # debuig
        print('saving parameters to file [%s]' % fname)
        for k, v in param.items():
            print('[%s] : %s' % (str(k), type(v)))
            print('\t type of [%s][0] : <%s>' % (str(k), type(v[0])))
        with open(fname, 'w') as fp:
            json.dump(param, fp)

    def load(self, fname):
        with open(fname, 'r') as fp:
            param = json.load(fp)
        self.set_param_from_dict(param)

    def get_elem(self, idx):
        return (self.data_paths[idx], self.elem_ids[idx], self.data_labels[idx])

    # TODO : these
    #def to_csv(self, fname):
    #    pass

    #def from_csv(self, fname):
    #    pass

