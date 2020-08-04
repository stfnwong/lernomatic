"""
HDF5_UTIL
Utilities for working with HDF5 files

Stefan Wong 2018
"""

import h5py
import numpy as np


class HDF5Data(object):
    def __init__(self, fname: str, **kwargs) -> None:
        """
        HDF5Data
        Wrapper around an HDF5 dataset
        """
        self.filename = fname      # Cache the filename when we call read()
        self.verbose  = kwargs.pop('verbose', False)
        self.mode     = kwargs.pop('mode', 'r')

        if self.mode not in ('r', 'w'):
            raise ValueError('Invalid mode [%s], must be one of [r] or [w]' % str(self.mode))

        # load file pointer
        try:
            self.fp = h5py.File(self.filename, self.mode)
        except:
            raise RuntimeError('Failed to open HDF5 file [%s]' % str(self.filename))

        self._init_data()

    def __del__(self) -> None:
        self.fp.close()

    def __repr__(self) -> str:
        return 'HDF5Data-%s' % str(self.shapes)

    def __str__(self) -> str:
        s = []
        s.append('HDF5Data')
        if self.filename is not None:
            s.append(' (%s)\n' % str(self.filename))
        else:
            s.append('\n')
        s.append('> Datasets :\n')
        #for k, v in self.fp.data.items():
        for k in self.get_datasets():
            s.append('\t[%s] : %d %s <%s>\n' %\
                     (str(k), self.size[k], str(self.shape[k]), str(self.dtype[k]))
                     )
        s.append('\n')
        s.append('> Attributes :\n')
        for k, v in self.attrs.items():
            s.append('\t[%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_data(self) -> None:
        """
        Init meta information about dataset
        """
        self.dtype    = dict()
        self.shape    = dict()
        self.size     = dict()
        self.attrs    = dict()
        self.data_ptr = dict()

        if self.mode == 'r':
            for k in self.fp.keys():
                self.dtype[k]    = self.fp[k].dtype
                self.shape[k]    = self.fp[k].shape
                self.size[k]     = self.fp[k].shape[0]
                self.data_ptr[k] = 0

    def _meta_to_dict(self) -> dict:
        meta = dict()
        meta['dtype'] = self.dtype
        meta['shape'] = self.shape
        meta['size']  = self.size
        meta['attrs'] = self.attrs

        return meta

    def compute_dataset_nbytes(self, k) -> int:
        if k not in self.data.keys():
            return 0
        num_bytes = 0
        for elem in self.fp.data[k]:
            num_bytes += elem.nbytes

        return num_bytes

    def create_dataset(self, k:str, N:int, shape:tuple, dtype=np.float32) -> None:
        """
        CREATE_DATASET
        Create a new dataset.

        Args:
            k     - Name of the dataset. This is used as the key into self.data
            N     - Maximum size of the dataset in elements
            shape - Tuple giving the size of a single element of the dataset
            dtype - Datatype of this dataset
        """
        if k in self.fp.keys():
            if self.verbose:
                print('dataset %s already exists in internal data' % str(k))
            return

        if type(shape) is not tuple:
            raise ValueError('shape must be a tuple giving the size of a single element')
        if N < 1:
            raise ValueError('minimum dataset length is 1')

        # cache metadata
        self.dtype[k] = dtype
        self.shape[k] = shape
        self.size[k]  = N
        # create data on disk
        self.fp.create_dataset(k, (self.size[k],) + self.shape[k], dtype=self.dtype[k])
        self.data_ptr[k] = 0

    def append_data(self, k:str, data) -> None:
        """
        APPEND_DATA
        Append a data element at the end of the dataset with key k
        """
        if k not in self.fp.keys():
            raise ValueError('No dataset %s in internal data' % str(k))
        if self.data_ptr[k] > self.size[k]:
            raise ValueError('Maximum index reached (%d)' % self.size[k])

        self.fp[k][self.data_ptr[k]] = data
        self.data_ptr[k] += 1

    def add_attr(self, attr_name:str, attr_data) -> None:
        self.fp.attrs[attr_name] = attr_data

    def add_attrs(self, attrs) -> None:
        if type(attrs) is not dict:
            raise ValueError('attrs must be a dict of attributes')
        for k, v in attrs.items():
            self.attrs[k] = v

        for k, v in self.attrs.items():
            self.fp.attrs[k] = v

    def dump_meta(self) -> dict:
        """
        DUMP_META
        Returns metadata from file
        """
        for k in self.fp.keys():
            self.dtype[k] = self.fp[k].dtype
            self.shape[k]  = self.fp[k].shape

        # Read attributes, if any
        if len(self.fp.attrs.keys()) > 0:
            for k, v in self.fp.attrs.items():
                self.attrs[k] = v
        else:
            # might be attrs attached to the dataset
            for k in self.fp.keys():
                if hasattr(self.fp[k], 'attrs'):
                    for k, v in self.fp[k].attrs.items():
                        self.attrs[k] = v

        return self._meta_to_dict()

    # various getters
    def get_elem(self, k:str, idx:int):
        if idx > self.size[k]:
            raise IndexError('Max element in dataset %s is %d' % (str(k), self.size[k]))
        return self.fp[k][idx]

    def get_data(self, k:str, asarray=False):
        if k not in self.fp.keys():
            return None
        if asarray is True:
            return np.asarray(self.fp[k][:])
        else:
            return self.fp[k][:]

    def get_dataset(self, k):
        if k not in self.fp.keys():
            return None
        return self.fp[k]

    def get_datasets(self):
        return self.fp.keys()

    def has_dataset(self, k) -> bool:
        if k in self.fp.keys():
            return True
        return False

    def get_shape(self, k:str) -> tuple:
        if k not in self.fp.keys():
            return None
        return self.fp[k].shape

    def get_size(self, k:str) -> tuple:
        if k not in self.fp.keys():
            return None
        return self.fp[k].shape[0]
