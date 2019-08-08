"""
HDF5_DATASET
Dataset for an HDF5 file
"""

import h5py
import torch
import numpy as np
import PIL
from torch.utils.data import Dataset

from typing import Tuple

# debug
#from pudb import set_trace; set_trace()


class HDF5Dataset(Dataset):
    """
    HDF5Dataset

    Wrapper for an HDF5 file. Provides torch Dataset functionality and converts
    *.h5 file contents to torch.Tensor.

    Arguments:
        filename (str) :
            Name of input file.

        feature_name (str) :
            Name of feature key. (default: features)

        label_name (str) :
            Name of label key (default: labels)

        transform (torchvision.Transform):
            One or more torchvision transforms (default: None)

        label_max_dim (int) :
            Maximum number of dimensions that the label can have. Label elements that
            are larger than this are sequeezed to remove the top label_max_dim-1
            dimensions. (default: 1)
    """
    def __init__(self, filename: str, **kwargs) -> None:
        self.fp = h5py.File(filename, 'r')
        # set dataset names
        self.feature_name  :str = kwargs.pop('feature_name', 'features')
        self.label_name    :str = kwargs.pop('label_name', 'labels')
        self.transform          = kwargs.pop('transform', None)
        self.label_max_dim :int = kwargs.pop('label_max_dim', 0)

    def __repr__(self) -> str:
        return 'HDF5Dataset'

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.feature_name])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx > len(self)-1:
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

        return (feature, label)


# Almost the same thing, but doesn't convert to torch.Tensor
class HDF5RawDataset(Dataset):
    """
    HDF5RawDataset

    Same an an HDF5Dataset, but does not perform conversion to torch.Tensor

    Arguments:
        filename (str) :
            Name of input file.

        feature_name (str) :
            Name of feature key. (default: features)

        label_name (str) :
            Name of label key (default: labels)
    """
    def __init__(self, filename: str, **kwargs) -> None:
        self.fp = h5py.File(filename, 'r')
        # set dataset names
        self.feature_name  = kwargs.pop('feature_name', 'features')
        self.label_name    = kwargs.pop('label_name', 'labels')

    def __repr__(self) -> str:
        return 'HDF5RawDataset'

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.feature_name])

    # NOTE : even thought the type hint says ndarray, this could actually be
    # anything
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx > len(self)-1:
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        feature = self.fp[self.feature_name][idx][:]
        label   = self.fp[self.label_name][idx][:]

        return (feature, label)



# Almost the same thing again, but converts to PIL.Image
class HDF5PILDataset(Dataset):
    """
    HDF5PILDataset

    Same an an HDF5Dataset, but performs a conversion to a PIL.Image for the
    elements of the feature dataset.

    Arguments:
        filename (str) :
            Name of input file.

        feature_name (str) :
            Name of feature key. (default: features)

        label_name (str) :
            Name of label key (default: labels)
    """
    def __init__(self, filename: str, **kwargs) -> None:
        self.fp = h5py.File(filename, 'r')
        # set dataset names
        self.feature_name  = kwargs.pop('feature_name', 'features')
        self.label_name    = kwargs.pop('label_name', 'labels')

    def __repr__(self) -> str:
        return 'HDF5PILDataset'

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.feature_name])

    # NOTE : even thought the type hint says ndarray, this could actually be
    # anything
    def __getitem__(self, idx: int) -> Tuple[PIL.Image.Image, np.ndarray]:
        if idx > len(self)-1:
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        feature = PIL.Image.fromarray(self.fp[self.feature_name][idx][:].astype('uint8'))
        label   = self.fp[self.label_name][idx][:]

        return (feature, label)

