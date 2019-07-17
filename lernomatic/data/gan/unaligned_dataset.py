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



class UnalignedDataset(Dataset):
    def __init__(self, a_data_paths:list, b_data_paths:list, **kwargs) -> None:
        self.a_data_paths:list   = a_data_paths
        self.b_data_paths:list   = b_data_paths
        self.transform           = kwargs.pop('transform', None)
        self.data_root:str       = kwargs.pop('data_root', None)
        self.input_nc:int        = kwargs.pop('input_nc', 3)
        self.output_nc:int       = kwargs.pop('output_nc', 3)
        self.serial_batches:bool = kwargs.pop('serial_batches', False)
        # TODO: direction?
        self.num_a_elem:int    = len(self.a_data_paths)
        self.num_b_elem:int    = len(self.b_data_paths)

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

        a_img = cv2.imread(cur_a_path).transpose(2, 0, 1)
        b_img = cv2.imread(cur_b_path).transpose(2, 0, 1)

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        return (a_img, b_img)

# TODO: HDF5 version
