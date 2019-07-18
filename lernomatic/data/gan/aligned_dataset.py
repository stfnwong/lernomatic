"""
ALIGNED_DATASET
Dataset that provides aligned image pairs

Stefan Wong 2019
"""

import cv2
import h5py
import torch
from torch.utils.data import Dataset


from pudb import set_trace; set_trace()


# Aligned dataset from Folders/Paths
class AlignedDataset(Dataset):
    def __init__(self, ab_data_paths:list, **kwargs) -> None:
        self.ab_data_paths:list = ab_data_paths
        self.data_root    :str  = kwargs.pop('data_root', None)
        self.input_nc     :int  = kwargs.pop('input_nc', 3)
        self.output_nc    :int  = kwargs.pop('output_nc', 3)
        self.do_transpose:bool  = kwargs.pop('do_transpose', True)
        self.transform = kwargs.pop('transform', None)

    def __repr__(self) -> str:
        return 'AlignedDataset'

    def __len__(self) -> int:
        return len(self.ab_data_paths)

    def __getitem__(self, idx:int) -> tuple:
        if idx > len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        if self.data_root is None:
            ab_img = cv2.imread(self.ab_data_paths[idx])
        else:
            ab_img = cv2.imread(str(self.data_root + self.ab_data_paths[idx]))

        # TODO : could have another thing here for grayscale?
        # transpose the image arrays to match the pytorch tensor shape order
        if self.do_transpose:
            ab_img = ab_img.transpose(2, 0, 1)

        # split into two images
        _, ab_w, ab_h = ab_img.shape
        w2 = int(ab_w / 2)
        a_img = ab_img[:, 0: w2, 0 : ab_h]
        b_img = ab_img[:, w2:ab_w, 0: ab_h]

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
