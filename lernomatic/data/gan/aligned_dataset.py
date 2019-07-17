"""
ALIGNED_DATASET
Dataset that provides aligned image pairs

Stefan Wong 2019
"""

import cv2
import h5py
import torch
from torch.utils.data import Dataset


# Aligned dataset from Folders/Paths
class AlignedDataset(Dataset):
    def __init__(self, a_data_paths:list, b_data_paths:list, **kwargs) -> None:
        self.a_data_paths :list = a_data_paths
        self.b_data_paths :list = b_data_paths
        self.data_root    :str  = kwargs.pop('data_root', None)
        self.input_nc     :int  = kwargs.pop('input_nc', 3)
        self.output_nc    :int  = kwargs.pop('output_nc', 3)
        self.transform = kwargs.pop('transform', None)

        # Ensure that we have the same number of A paths as B paths
        if len(self.a_data_paths) != len(self.b_data_paths):
            raise ValueError('[%s] num A paths (%d) must equal num B paths (%d)' %\
                    (repr(self), len(self.a_data_paths), len(self.b_data_paths))
            )

    def __repr__(self) -> str:
        return 'AlignedDataset'

    def __len__(self) -> int:
        return len(self.a_data_paths)

    def __getitem__(self, idx:int) -> tuple:
        if idx > len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        if self.data_root is None:
            a_img = cv2.imread(self.a_data_paths[idx])
            b_img = cv2.imread(self.b_data_paths[idx])
        else:
            a_img = cv2.imread(str(self.data_root + self.a_data_paths[idx]))
            b_img = cv2.imread(str(self.data_root + self.b_data_paths[idx]))

        # transpose the image arrays to match the pytorch tensor shape order
        a_img = a_img.transpose(2, 0, 1)
        b_img = b_img.transpose(2, 0, 1)

        if self.transform is not None:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        return (a_img, b_img)


# TODO : subclass from HDF5Dataset?
# Aligned dataset from HDF5
class AlignedDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, h5_filename:str, **kwargs) -> None:
        self.transform              = kwargs.pop('transform', None)
        self.image_dataset_name:str = kwargs.pop('image_dataset_name', 'images')
        self.get_ids:bool           = kwargs.pop('get_ids', False)
        self.a_id_name:str          = kwargs.pop('a_id_name', 'A_ids')
        self.b_id_name:str          = kwargs.pop('b_id_name', 'B_ids')

        self.fp = h5py.File(h5_filename, 'r')
        self.filename = h5_filename

    def __repr__(self) -> str:
        return 'AlignedDatasetHDF5'

    def __str__(self) -> str:
        return 'AlignedDatasetHDF5 <%s> (%d items)' % (self.filename, len(self))

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.image_dataset_name])

    def __getitem__(self, idx:int) -> tuple:
        if idx > len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        aligned_img = torch.FloatTensor(self.fp[self.image_dataset_name][idx][:])

        if self.transform is not None:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        if self.get_ids:
            a_id = self.fp[self.a_id_name][idx][:]
            b_id = self.fp[self.b_id_name][idx][:]

            return (a_img, b_img, a_id, b_id)

        return (a_img, b_img)
