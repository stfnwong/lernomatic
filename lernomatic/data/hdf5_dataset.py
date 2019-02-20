"""
HDF5_DATASET
Dataset for an HDF5 file
"""

import h5py
import torch
from torch.utils.data import Dataset

# debug
#from pudb import set_trace; set_trace()


class HDF5Dataset(Dataset):
    def __init__(self, filename, **kwargs):
        self.fp = h5py.File(filename, 'r')
        # set dataset names
        self.feature_name  = kwargs.pop('feature_name', 'features')
        self.label_name    = kwargs.pop('label_name', 'labels')
        self.transform     = kwargs.pop('transform', None)
        self.label_max_dim = kwargs.pop('label_max_dim', 0)

    def __repr__(self):
        return 'HDF5Dataset'

    def __del__(self):
        self.fp.close()

    def __len__(self):
        return len(self.fp[self.feature_name])

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        feature = torch.FloatTensor(self.fp[self.feature_name][idx][:])
        label   = torch.LongTensor(self.fp[self.label_name][idx][:])

        if len(feature.shape) == 2:
            feature = feature.unsqueeze(0)

        if self.label_max_dim > 0 and len(label.shape) > self.label_max_dim:
            label = torch.squeeze(label, self.label_max_dim-1)

        # Sort of hack...
        if self.label_max_dim == 1:
            label = torch.squeeze(label, 0)

        if self.transform is not None:
            feature = self.transform(feature)

        return feature, label
