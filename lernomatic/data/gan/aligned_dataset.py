"""
ALIGNED_DATASET
Dataset that provides aligned image pairs

Stefan Wong 2019
"""

import h5py
import torch
from torch.utils.data import Dataset


# Aligned dataset from Folders/Paths
class AlignedDataset(Dataset):
    def __init__(self, data_root:str, **kwargs) -> None:
        self.data_root = data_root
        self.transform = kwargs.pop('transform', None)

        # TODO : need a way to get all the paths for A and B datasets

    def __repr__(self) -> str:
        return 'AlignedDataset'

    def __len__(self) -> int:
        return 0        # will these use HDF5?

    def __getitem__(self, idx:int) -> tuple:
        pass



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

    # TODO: actually, I think I want to take the AB image and split it
    # manually into A and B images
    def __getitem(self, idx:int) -> tuple:
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
