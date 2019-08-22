"""
UNALIGNED_DATASET
Dataset that provides un-aligned image pairs

Stefan Wong 2019
"""

import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class UnalignedDataset(torch.utils.data.Dataset):
    def __init__(self, a_data_paths:list, b_data_paths:list, **kwargs) -> None:
        self.a_data_paths:list   = a_data_paths
        self.b_data_paths:list   = b_data_paths
        self.transform           = kwargs.pop('transform', None)
        self.data_root:str       = kwargs.pop('data_root', None)
        self.input_nc:int        = kwargs.pop('input_nc', 3)
        self.output_nc:int       = kwargs.pop('output_nc', 3)
        self.serial_batches:bool = kwargs.pop('serial_batches', False)
        self.do_transpose:bool   = kwargs.pop('do_transpose', True)
        # TODO: direction?
        self.num_a_elem:int    = len(self.a_data_paths)
        self.num_b_elem:int    = len(self.b_data_paths)

        # Ensure that we have the same number of A paths as B paths
        if len(self.a_data_paths) != len(self.b_data_paths):
            raise ValueError('[%s] num A paths (%d) must equal num B paths (%d)' %\
                    (repr(self), len(self.a_data_paths), len(self.b_data_paths))
            )

    def __repr__(self) -> str:
        return 'UnalignedDataset'

    def __len__(self) -> int:
        return np.max(self.num_a_elem, self.num_b_elem)

    def __getitem__(self, idx:int) -> tuple:
        cur_a_path = self.a_data_paths[idx % self.num_a_elem]
        if self.serial_batches:
            b_idx = idx % self.num_b_elem
        else:
            b_idx = np.random.randint(0, self.num_b_elem-1)

        cur_b_path = self.b_data_paths[b_idx]

        # I don't know when you wouldn't do this, but just in case
        if self.do_transpose:
            a_img = cv2.imread(cur_a_path).transpose(2, 0, 1)
            b_img = cv2.imread(cur_b_path).transpose(2, 0, 1)

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        return (a_img, b_img)



class UnalignedDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, h5_filename:str, **kwargs) -> None:
        self.transform                = kwargs.pop('transform', None)
        self.direction          :int  = kwargs.pop('direction', 0)    # 0: A->B, 1: B->A
        self.a_img_name         :str  = kwargs.pop('a_img_name', 'a_imgs')
        self.b_img_name         :str  = kwargs.pop('b_img_name', 'b_imgs')
        self.get_ids            :bool = kwargs.pop('get_ids', False)
        self.a_id_name          :str  = kwargs.pop('a_id_name', 'A_ids')
        self.b_id_name          :str  = kwargs.pop('b_id_name', 'B_ids')

        self.fp = h5py.File(h5_filename, 'r')
        self.filename = h5_filename

    def __repr__(self) -> str:
        return 'UnalignedDatasetHDF5'

    def __str__(self) -> str:
        return 'UnalignedDatasetHDF5 <%s> (%d items)' % (self.filename, len(self))

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return max(len(self.fp[self.a_img_name]), len(self.fp[self.b_img_name]))

    def __getitem__(self, idx:int) -> tuple:
        if idx >= len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        a_idx = idx % self.get_a_dataset_len()
        b_idx = idx % self.get_b_dataset_len()

        if self.direction == 0:
            a_img = torch.FloatTensor(self.fp[self.a_img_name][a_idx][:])
            b_img = torch.FloatTensor(self.fp[self.b_img_name][b_idx][:])
        else:
            a_img = torch.FloatTensor(self.fp[self.b_img_name][b_idx][:])
            b_img = torch.FloatTensor(self.fp[self.a_img_name][a_idx][:])

        if self.transform is not None:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        if self.get_ids:
            a_id = self.fp[self.a_id_name][a_idx][:]
            b_id = self.fp[self.b_id_name][b_idx][:]

            return (a_img, b_img, a_id, b_id)

        return (a_img, b_img)

    def get_a_datset_len(self) -> int:
        return len(self.fp[self.a_img_name])

    def get_b_dataset_len(self) -> int:
        return len(self.fp[self.b_img_name])
